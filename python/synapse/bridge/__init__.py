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
    VgBackend,
    enforce_distillation_geometry_gate,
    get_vg_backend,
    register_recon_backend,
)

# SynapseReconBackend and get_recon_backend live in synapse.recon.backend
# (Tier 2).  bridge is Tier 0, so they are lazy re-exported via __getattr__.

_LAZY_REEXPORTS = {"SynapseReconBackend", "get_recon_backend"}


def __getattr__(name: str) -> object:
    if name in _LAZY_REEXPORTS:
        from synapse.bridge.vg_bridge import __getattr__ as _vg_getattr

        return _vg_getattr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CommandIRRecord",
    "FallbackAction",
    "FoundaryRequest",
    "FoundaryResponse",
    "FoundaryTransport",
    "HttpFoundaryTransport",
    "MockFoundaryTransport",
    "MockVgBackend",
    "OperatorIdentity",
    "SynapseReconBackend",
    "VgBackend",
    "enforce_distillation_geometry_gate",
    "evaluate_hitl_circuit_breaker",
    "get_recon_backend",
    "get_vg_backend",
    "register_recon_backend",
]
