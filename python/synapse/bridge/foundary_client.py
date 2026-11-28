"""Client models and transport for foundary container inference services.

The ``foundary`` container hosts heavyweight GPU inference services
(TRELLIS, Hunyuan3D, SAM2, GVHMR, Visual-JEV) behind a thin HTTP
endpoint.  This module defines the Pydantic request/response envelope
models and two transport implementations:

- :class:`MockFoundaryTransport` — returns dummy success for testing.
- :class:`HttpFoundaryTransport` — live HTTP transport for foundary container
  endpoints; raises :class:`~synapse.errors.BackendUnavailableError` when
  invoked without a configured/reachable endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------


FOUNDARY_SERVICE_NAMES: tuple[str, ...] = (
    "cotracker",
    "megasam",
    "depth_anything",
    "trellis",
    "hunyuan3d",
    "sam2",
    "gvhmr",
    "visual_jev",
    "delight",
    "wan",
    "flux",
    "sdxl",
    "dust3r",
    "metric3d",
    "wham",
    "intrinsic_image",
    "diffusion_renderer",
)

FoundaryServiceName = Literal[
    "cotracker",
    "megasam",
    "depth_anything",
    "trellis",
    "hunyuan3d",
    "sam2",
    "gvhmr",
    "visual_jev",
    "delight",
    "wan",
    "flux",
    "sdxl",
    "dust3r",
    "metric3d",
    "wham",
    "intrinsic_image",
    "diffusion_renderer",
]


class FoundaryRequest(BaseModel):
    """Base request model for foundary services.

    Attributes:
        service: Name of the target inference service.
        job_id: Opaque job identifier assigned by the upstream scheduler.
        inputs: Arbitrary input payload (paths, arrays, etc.).
        params: Service-specific hyper-parameters.
    """

    service: FoundaryServiceName
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


class FoundaryJobStatus(BaseModel):
    """Async job status descriptor for foundary container inference."""

    job_id: str
    service: str
    state: Literal["queued", "running", "succeeded", "failed"] = "queued"
    progress: float = 0.0
    outputs: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""


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

    def submit_job(self, request: FoundaryRequest) -> FoundaryJobStatus:
        """Submit an asynchronous inference job (default wraps synchronous :meth:`submit`)."""
        resp = self.submit(request)
        jid = request.job_id or f"job_{request.service}"
        if resp.status == "error":
            return FoundaryJobStatus(
                job_id=jid,
                service=request.service,
                state="failed",
                progress=1.0,
                outputs=resp.outputs,
                error_message=resp.error_message,
            )
        return FoundaryJobStatus(
            job_id=jid,
            service=request.service,
            state="succeeded",
            progress=1.0,
            outputs=resp.outputs,
        )

    def poll_job(self, job_id: str, service: str = "") -> FoundaryJobStatus:
        """Query current state of an asynchronous inference job."""
        return FoundaryJobStatus(
            job_id=job_id,
            service=service,
            state="succeeded",
            progress=1.0,
            outputs={},
        )

    def wait_for_job(
        self,
        job_id: str,
        service: str = "",
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 0.5,
    ) -> FoundaryResponse:
        """Poll an async job until completion or raise ``BackendUnavailableError`` (exit 30)."""
        import time

        from synapse.errors import BackendUnavailableError

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        delay = max(0.01, poll_interval_seconds)
        while True:
            status = self.poll_job(job_id, service=service)
            if status.state == "succeeded":
                return FoundaryResponse(
                    status="ok",
                    service=status.service or service,
                    outputs=status.outputs,
                )
            if status.state == "failed":
                raise BackendUnavailableError(
                    f"Foundary async job '{job_id}' ({service}) failed: "
                    f"{status.error_message or 'unknown error'}",
                    agent_fix_hint="Inspect foundary container job logs and GPU VRAM.",
                )
            if time.monotonic() >= deadline:
                raise BackendUnavailableError(
                    f"Timed out waiting for foundary job '{job_id}' ({service}) "
                    f"after {timeout_seconds:.1f}s.",
                    agent_fix_hint="Increase timeout_seconds or inspect foundary worker queue.",
                )
            time.sleep(min(delay, max(0.01, deadline - time.monotonic())))
            delay = min(delay * 1.5, 5.0)

    def list_serving_recipes(
        self,
        rfc: str = "RFC-0060",
        role: str = "",
    ) -> list[dict[str, Any]]:
        """Query registered serving recipes filtered by optional RFC anchor and role."""
        _ = (rfc, role)
        return []


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
        self._jobs: dict[str, FoundaryJobStatus] = {}

    def submit(self, request: FoundaryRequest) -> FoundaryResponse:
        """Return a synthetic ``ok`` response with ``{"mock": True}``."""
        return FoundaryResponse(
            status="ok",
            service=request.service,
            outputs={"mock": True},
            elapsed_seconds=self._latency,
        )

    def submit_job(self, request: FoundaryRequest) -> FoundaryJobStatus:
        """Simulate async job submission and cache completion payload."""
        resp = self.submit(request)
        jid = request.job_id or f"mock_{request.service}_{len(self._jobs) + 1}"
        st = FoundaryJobStatus(
            job_id=jid,
            service=request.service,
            state="succeeded" if resp.status == "ok" else "failed",
            progress=1.0,
            outputs=resp.outputs,
            error_message=resp.error_message,
        )
        self._jobs[jid] = st
        return st

    def poll_job(self, job_id: str, service: str = "") -> FoundaryJobStatus:
        """Return cached mock job status."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        return super().poll_job(job_id, service=service)

    def list_serving_recipes(
        self,
        rfc: str = "RFC-0060",
        role: str = "",
    ) -> list[dict[str, Any]]:
        """Return built-in Foundary serving recipe metadata for offline tests."""
        catalog: list[dict[str, Any]] = [
            {"id": "sam2", "role": "segmentation", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "dust3r", "role": "sfm", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "metric3d-v2", "role": "depth", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "cotracker3", "role": "tracking", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "wham", "role": "mocap", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "trellis", "role": "gen3d", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "intrinsic-image", "role": "intrinsic", "rfc_anchor": "RFC-0060", "engine": "container"},
            {"id": "deepseek-r1", "role": "planner", "rfc_anchor": "RFC-0053", "engine": "vllm"},
            {"id": "deepseek-v3", "role": "actor", "rfc_anchor": "RFC-0053", "engine": "vllm"},
            {"id": "qwen2.5-vl-72b", "role": "critic", "rfc_anchor": "RFC-0053", "engine": "vllm"},
            {"id": "qwen2.5-vl-7b", "role": "critic_lite", "rfc_anchor": "RFC-0053", "engine": "vllm"},
        ]
        rfc_norm = rfc.strip().upper()
        role_norm = role.strip().lower()
        return [
            item
            for item in catalog
            if (not rfc_norm or item["rfc_anchor"].upper() == rfc_norm)
            and (not role_norm or item["role"].lower() == role_norm)
        ]


# -----------------------------------------------------------------------
# HTTP transport — live foundary container endpoint
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
        allow_loopback: bool = False,
    ) -> None:
        self._allow_loopback = allow_loopback
        if base_url:
            from synapse.bridge.http_security import validate_safe_url

            if allow_loopback:
                validate_safe_url(base_url, allow_loopback=True)
            else:
                validate_safe_url(base_url)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    def is_available(self) -> bool:
        """Check if the foundary endpoint is responding."""
        if not self._base_url:
            return False
        import urllib.error

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url

        try:
            if self._allow_loopback:
                validate_safe_url(self._base_url, allow_loopback=True)
            else:
                validate_safe_url(self._base_url)
        except Exception:
            return False

        health_url = f"{self._base_url}/healthz"
        for url in (health_url, self._base_url):
            try:
                open_ctx = (
                    safe_urlopen(url, timeout=3.0, allow_loopback=True)
                    if self._allow_loopback
                    else safe_urlopen(url, timeout=3.0)
                )
                with open_ctx as resp:
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
        import hashlib
        import http.client
        import json
        import time
        import urllib.error
        import urllib.request

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url
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

        if self._allow_loopback:
            validate_safe_url(self._base_url, allow_loopback=True)
        else:
            validate_safe_url(self._base_url)
        url = f"{self._base_url}/v1/{request.service}"
        payload = request.model_dump_json().encode("utf-8")
        idempotency_key = hashlib.sha256(payload).hexdigest()
        trace_headers = TraceContext.current().to_headers()
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
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
                open_ctx = (
                    safe_urlopen(req, timeout=self._timeout, allow_loopback=True)
                    if self._allow_loopback
                    else safe_urlopen(req, timeout=self._timeout)
                )
                with open_ctx as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    resp_obj = FoundaryResponse.model_validate(data)
                    if resp_obj.status == "error":
                        raise BackendUnavailableError(
                            f"Foundary service '{request.service}' reported error: "
                            f"{resp_obj.error_message or 'unknown error'}",
                            agent_fix_hint="Check foundary container logs and GPU health.",
                        )
                    return resp_obj
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (502, 503, 504) and attempt < self._max_retries - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise BackendUnavailableError(
                    f"Foundary service '{request.service}' HTTP {exc.code}: {exc.reason}",
                    agent_fix_hint="Check foundary container logs and GPU health.",
                ) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.HTTPException,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise BackendUnavailableError(
                    f"Cannot reach foundary service at {url}: {exc}",
                    agent_fix_hint="Ensure the foundary container is running and port is forwarded.",
                ) from exc

        raise BackendUnavailableError(
            f"Failed to submit to foundary after {self._max_retries} attempts: {last_error}",
            agent_fix_hint="Verify foundary endpoint availability and network connectivity.",
        )

    def submit_job(self, request: FoundaryRequest) -> FoundaryJobStatus:
        """Submit an async job to ``POST /v1/{service}/jobs``."""
        import json
        import urllib.request

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url
        from synapse.errors import BackendUnavailableError

        if not self._base_url:
            raise BackendUnavailableError(
                "Foundary HTTP endpoint not configured",
                agent_fix_hint="Set FOUNDARY_URL or use MockFoundaryTransport.",
            )
        if self._allow_loopback:
            validate_safe_url(self._base_url, allow_loopback=True)
        else:
            validate_safe_url(self._base_url)

        url = f"{self._base_url}/v1/{request.service}/jobs"
        req = urllib.request.Request(
            url,
            data=request.model_dump_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            open_ctx = (
                safe_urlopen(req, timeout=self._timeout, allow_loopback=True)
                if self._allow_loopback
                else safe_urlopen(req, timeout=self._timeout)
            )
            with open_ctx as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return FoundaryJobStatus.model_validate(data)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to submit async foundary job to {url}: {exc}",
                agent_fix_hint="Verify foundary endpoint availability and async jobs route.",
            ) from exc

    def poll_job(self, job_id: str, service: str = "") -> FoundaryJobStatus:
        """Query async job status from ``GET /v1/jobs/{job_id}``."""
        import json
        import urllib.request

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url
        from synapse.errors import BackendUnavailableError

        if not self._base_url:
            raise BackendUnavailableError(
                "Foundary HTTP endpoint not configured",
                agent_fix_hint="Set FOUNDARY_URL or use MockFoundaryTransport.",
            )
        if self._allow_loopback:
            validate_safe_url(self._base_url, allow_loopback=True)
        else:
            validate_safe_url(self._base_url)

        url = f"{self._base_url}/v1/jobs/{job_id}"
        req = urllib.request.Request(url, method="GET")
        try:
            open_ctx = (
                safe_urlopen(req, timeout=self._timeout, allow_loopback=True)
                if self._allow_loopback
                else safe_urlopen(req, timeout=self._timeout)
            )
            with open_ctx as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return FoundaryJobStatus.model_validate(data)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to poll async foundary job '{job_id}': {exc}",
                agent_fix_hint="Check foundary container health and network connectivity.",
            ) from exc

    def list_serving_recipes(
        self,
        rfc: str = "RFC-0060",
        role: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch serving recipes from ``GET /api/v1/serving/recipes``."""
        import json
        import urllib.parse
        import urllib.request

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url
        from synapse.errors import BackendUnavailableError

        if not self._base_url:
            raise BackendUnavailableError(
                "Foundary HTTP endpoint not configured",
                agent_fix_hint="Set FOUNDARY_URL or use MockFoundaryTransport.",
            )
        if self._allow_loopback:
            validate_safe_url(self._base_url, allow_loopback=True)
        else:
            validate_safe_url(self._base_url)

        params: dict[str, str] = {}
        if rfc.strip():
            params["rfc"] = rfc.strip()
        if role.strip():
            params["role"] = role.strip()
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self._base_url}/api/v1/serving/recipes{query}"
        req = urllib.request.Request(url, method="GET")
        try:
            open_ctx = (
                safe_urlopen(req, timeout=self._timeout, allow_loopback=True)
                if self._allow_loopback
                else safe_urlopen(req, timeout=self._timeout)
            )
            with open_ctx as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                data = payload.get("data", []) if isinstance(payload, dict) else []
                return [item for item in data if isinstance(item, dict)]
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to query serving recipes at {url}: {exc}",
                agent_fix_hint="Check foundary server health and /api/v1/serving/recipes route.",
            ) from exc


class FoundaryClient:
    """High-level ergonomic client for foundary container and LLM inference services."""

    def __init__(self, transport: FoundaryTransport | None = None) -> None:
        self.transport: FoundaryTransport = transport or HttpFoundaryTransport()

    def list_serving_recipes(
        self,
        rfc: str = "RFC-0060",
        role: str = "",
    ) -> list[dict[str, Any]]:
        """List Foundary serving recipes filtered by optional RFC anchor and role."""
        return self.transport.list_serving_recipes(rfc=rfc, role=role)

    def infer(
        self,
        service: FoundaryServiceName,
        inputs: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        *,
        job_id: str = "",
        async_mode: bool = False,
        timeout_seconds: float = 300.0,
    ) -> FoundaryResponse:
        """Execute synchronous or polled-async inference against *service*."""
        req = FoundaryRequest(
            service=service,
            job_id=job_id,
            inputs=inputs or {},
            params=params or {},
        )
        if async_mode:
            status = self.transport.submit_job(req)
            if status.state == "succeeded":
                return FoundaryResponse(
                    status="ok",
                    service=status.service or service,
                    outputs=status.outputs,
                )
            return self.transport.wait_for_job(
                status.job_id,
                service=service,
                timeout_seconds=timeout_seconds,
            )
        return self.transport.submit(req)
