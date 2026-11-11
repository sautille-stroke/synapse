"""Synapse — 3D CG Rendering & Neural Video Generation Interoperability Hub.

Implements the Hybrid Cinema Pipeline (RFC 0044 & RFC 0059).

Submodules:
    fgpp        FGPP (Feature Guidance Package Protocol) SDK
    adapters    Video model adapter framework
    pipeline    5-stage hybrid cinema pipeline orchestration
    style       Style extraction & alignment system
    composite   Selective re-rendering & surgical compositing
    metrics     Cross-modal temporal, fidelity, and style metrics
    bench       Synapse-Bench evaluation dataset & scenarios
    cli         CLI tools
"""

from synapse import (
    adapters,
    bench,
    cli,
    composite,
    fgpp,
    generators,
    metrics,
    models,
    pipeline,
    style,
    viewer,
)

__version__ = "0.1.0"

__all__ = [
    "fgpp",
    "models",
    "generators",
    "viewer",
    "adapters",
    "pipeline",
    "style",
    "composite",
    "metrics",
    "bench",
    "cli",
    "__version__",
]
