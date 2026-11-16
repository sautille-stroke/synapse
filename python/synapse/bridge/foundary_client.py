"""Client models and transport for foundary container inference services.

The ``foundary`` container hosts heavyweight GPU inference services
(TRELLIS, Hunyuan3D, SAM2, GVHMR, Visual-JEV) behind a thin HTTP
endpoint.  This module defines the Pydantic request/response envelope
models and two transport implementations:

- :class:`MockFoundaryTransport` — returns dummy success for testing.
- :class:`HttpFoundaryTransport` — placeholder for real HTTP calls;
  raises :class:`~synapse.errors.BackendUnavailableError` when invoked
  without a live endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------


class FoundaryRequest(BaseModel):
    """Base request model for foundary services.

    Attributes:
        service: Name of the target inference service.
        job_id: Opaque job identifier assigned by the upstream scheduler.
        inputs: Arbitrary input payload (paths, arrays, etc.).
        params: Service-specific hyper-parameters.
    """

    service: Literal[
        "trellis", "hunyuan3d", "sam2", "gvhmr", "visual_jev"
    ]
    job_id: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class FoundaryResponse(BaseModel):
    """Base response model from foundary services.

    Attributes:
        status: ``"ok"`` or ``"error"``.
        service: Echo of the service name from the request.
        outputs: Arbitrary output payload.
        error_message: Human-readable error description (empty on success).
        elapsed_seconds: Wall-clock inference time in seconds.
    """

    status: Literal["ok", "error"] = "ok"
    service: str = ""
    outputs: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    elapsed_seconds: float = 0.0


# -----------------------------------------------------------------------
# Abstract transport
# -----------------------------------------------------------------------


class FoundaryTransport(ABC):
    """Abstract transport for communicating with foundary services."""

    @abstractmethod
    def submit(self, request: FoundaryRequest) -> FoundaryResponse:
        """Submit a request and return the response.

        Args:
            request: Validated :class:`FoundaryRequest`.

        Returns:
            A :class:`FoundaryResponse` with status and outputs.
        """


# -----------------------------------------------------------------------
# Mock transport — testing only
# -----------------------------------------------------------------------


class MockFoundaryTransport(FoundaryTransport):
    """Mock transport that returns dummy successful responses.

    Args:
        latency_seconds: Simulated latency echoed in
            :attr:`FoundaryResponse.elapsed_seconds`.
    """

    def __init__(self, latency_seconds: float = 0.0) -> None:
        self._latency = latency_seconds

    def submit(self, request: FoundaryRequest) -> FoundaryResponse:
        """Return a synthetic ``ok`` response with ``{"mock": True}``."""
        return FoundaryResponse(
            status="ok",
            service=request.service,
            outputs={"mock": True},
            elapsed_seconds=self._latency,
        )


# -----------------------------------------------------------------------
# HTTP transport — production placeholder
# -----------------------------------------------------------------------


class HttpFoundaryTransport(FoundaryTransport):
    """HTTP transport for foundary container endpoint.

    Supports unary binary payload, /healthz health checking, W3C traceparent
    forwarding, and exponential backoff retry on 502/503/504 errors.

    Args:
        base_url: Base URL of the foundary HTTP service.
        timeout_seconds: HTTP request timeout in seconds.
        max_retries: Maximum attempts for transient server errors (502, 503, 504).
    """

    def __init__(
        self,
        base_url: str = "",
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    def is_available(self) -> bool:
        """Check if the foundary endpoint is responding."""
        if not self._base_url:
            return False
        import urllib.error
        import urllib.request

        health_url = f"{self._base_url}/healthz"
        for url in (health_url, self._base_url):
            try:
                with urllib.request.urlopen(url, timeout=3.0) as resp:
                    if resp.status in (200, 204):
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code in (200, 204, 405):
                    return True
            except Exception:
                continue
        return False

    def submit(self, request: FoundaryRequest) -> FoundaryResponse:
        """Submit a request to the foundary service endpoint.

        Raises:
            BackendUnavailableError: If base_url is empty, connection refused,
                or server times out.
        """
        import json
        import time
        import urllib.error
        import urllib.request

        from synapse.errors import BackendUnavailableError
        from synapse.telemetry import TraceContext

        if not self._base_url:
            raise BackendUnavailableError(
                "Foundary HTTP endpoint not configured",
                agent_fix_hint=(
                    "Set FOUNDARY_URL environment variable or use "
                    "MockFoundaryTransport for testing."
                ),
            )

        url = f"{self._base_url}/v1/{request.service}"
        payload = request.model_dump_json().encode("utf-8")
        trace_headers = TraceContext.current().to_headers()
        headers = {
            "Content-Type": "application/json",
            **trace_headers,
        }

        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return FoundaryResponse.model_validate(data)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (502, 503, 504) and attempt < self._max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise BackendUnavailableError(
                    f"Foundary service '{request.service}' HTTP {exc.code}: {exc.reason}",
                    agent_fix_hint="Check foundary container logs and GPU health.",
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise BackendUnavailableError(
                    f"Cannot reach foundary service at {url}: {exc}",
                    agent_fix_hint="Ensure the foundary container is running and port is forwarded.",
                ) from exc

        raise BackendUnavailableError(
            f"Failed to submit to foundary after {self._max_retries} attempts: {last_error}",
            agent_fix_hint="Verify foundary endpoint availability and network connectivity.",
        )
