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
    bridge      Cross-package thin bridge layer (foundary, hitl, vg)
    cli         CLI tools
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Lightweight submodules that have no heavy third-party deps (cv2, numba,
# trimesh) are imported eagerly so that ``import synapse`` stays fast in
# minimal / CI environments.
from synapse import (
    adapters,
    bench,
    bridge,
    cli,
    composite,
    fgpp,
    inspect,
    metrics,
    pipeline,
    recon,
    rsi,
    style,
    training,
)

# Heavy submodules are lazy-loaded: they pull in cv2 (generators/dataset.py),
# numba (generators/rasterizer.py), or trimesh (models/loader.py) at import
# time.  Deferring them avoids ImportError in environments that only need the
# lightweight core (e.g. FGPP SDK, bench scenarios, CLI list).
_LAZY_SUBMODULES = {"generators", "models", "viewer"}

if TYPE_CHECKING:  # pragma: no cover — static analysis only
    from synapse import generators, models, viewer


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"synapse.{name}")
    raise AttributeError(f"module 'synapse' has no attribute {name!r}")


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
    "rsi",
    "style",
    "training",
    "viewer",
]
