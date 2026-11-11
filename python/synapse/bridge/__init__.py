"""Bridge subpackage: vg-core FFI, foundary transport, and HITL circuit breaker.

This package provides:

- :mod:`synapse.bridge.vg_bridge` — thin bridge to ``vg-core`` native
  pybind11 extensions with mock and native backends.
- :mod:`synapse.bridge.foundary_client` — Pydantic request/response models
  and abstract transport for the ``foundary`` container inference services.
- :mod:`synapse.bridge.hitl` — HITL (Human-In-The-Loop) 15-minute circuit
  breaker and ``CommandIR`` data collection records.
"""

from __future__ import annotations

from synapse.bridge.foundary_client import (
    FoundaryRequest,
    FoundaryResponse,
    FoundaryTransport,
    HttpFoundaryTransport,
    MockFoundaryTransport,
)
from synapse.bridge.hitl import (
    CommandIRRecord,
    FallbackAction,
    OperatorIdentity,
    evaluate_hitl_circuit_breaker,
)
from synapse.bridge.vg_bridge import (
    MockVgBackend,
    NativeVgBackend,
    SynapseReconBackend,
    VgBackend,
    enforce_distillation_geometry_gate,
    get_recon_backend,
    get_vg_backend,
)

__all__ = [
    "CommandIRRecord",
    "FallbackAction",
    "FoundaryRequest",
    "FoundaryResponse",
    "FoundaryTransport",
    "HttpFoundaryTransport",
    "MockFoundaryTransport",
    "MockVgBackend",
    "NativeVgBackend",
    "OperatorIdentity",
    "SynapseReconBackend",
    "VgBackend",
    "enforce_distillation_geometry_gate",
    "evaluate_hitl_circuit_breaker",
    "get_recon_backend",
    "get_vg_backend",
]
