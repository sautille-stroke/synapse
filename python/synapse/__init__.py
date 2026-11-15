"""Synapse --- 3D CG Rendering & Neural Video Generation Interoperability Hub.

Implements the Hybrid Cinema Pipeline (RFC 0044 & RFC 0059).

Submodules:
    fgpp        FGPP (Feature Guidance Package Protocol) SDK
    adapters    Video model adapter framework
    pipeline    5-stage hybrid cinema pipeline orchestration
    style       Style extraction & alignment system
    composite   Selective re-rendering & surgical compositing
    metrics     Cross-modal temporal, fidelity, and style metrics
    bench       Synapse-Bench evaluation dataset & scenarios
    inspect     3D asset inspection schemas & pose alignment
    training    LoRA training pipeline
    rsi         Recursive Self-Improvement flywheel
    recon       Video-to-3D inverse reconstruction & asset distillation
    refine      Local anatomical refinement & multi-model fusion
    render_core Tier-0 shared rendering primitives, rasterizer, and conditions
    bridge      Cross-package thin bridge layer (foundary, hitl, vg)
    cli         CLI tools
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# PEP 562 lazy loading — all subpackages are imported on first access only.
# This eliminates pulling Tier 4 cli/pipeline and all 19 subpackages when
# only importing e.g. ``synapse.fgpp.schema``.
# ---------------------------------------------------------------------------

_SUBMODULES = frozenset({
    "adapters",
    "bench",
    "bridge",
    "cli",
    "composite",
    "fgpp",
    "generators",
    "inspect",
    "metrics",
    "models",
    "pipeline",
    "recon",
    "refine",
    "render_core",
    "rsi",
    "style",
    "training",
    "viewer",
})

if TYPE_CHECKING:  # pragma: no cover — static analysis only
    from synapse import (
        adapters,
        bench,
        bridge,
        cli,
        composite,
        fgpp,
        generators,
        inspect,
        metrics,
        models,
        pipeline,
        recon,
        refine,
        render_core,
        rsi,
        style,
        training,
        viewer,
    )


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        return importlib.import_module(f"synapse.{name}")
    raise AttributeError(f"module 'synapse' has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*sorted(_SUBMODULES), "__version__"]


__all__ = [
    "__version__",
    "adapters",
    "bench",
    "bridge",
    "cli",
    "composite",
    "fgpp",
    "generators",
    "inspect",
    "metrics",
    "models",
    "pipeline",
    "recon",
    "refine",
    "render_core",
    "rsi",
    "style",
    "training",
    "viewer",
]
