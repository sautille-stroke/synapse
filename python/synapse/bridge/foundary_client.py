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

    In production this would make real HTTP POST requests to the
    ``foundary`` container.  The current implementation raises
    :class:`~synapse.errors.BackendUnavailableError` because no
    endpoint is configured.

    Args:
        base_url: Base URL of the foundary HTTP service.
        timeout_seconds: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout_seconds

    def submit(self, request: FoundaryRequest) -> FoundaryResponse:
        """Raise — no live endpoint configured yet.

        Raises:
            BackendUnavailableError: Always, until a real endpoint is
                deployed.
        """
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            f"Foundary HTTP endpoint not configured: {self._base_url}",
            agent_fix_hint=(
                "Set FOUNDARY_URL environment variable or use "
                "MockFoundaryTransport for testing."
            ),
        )
