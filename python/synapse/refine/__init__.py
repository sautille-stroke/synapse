# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""synapse.refine — Anatomical local refinement subpackage (RFC 0061)."""

from __future__ import annotations

from synapse.refine.face import CanonicalAnchorGallery, FaceRefiner
from synapse.refine.fusion import MultiModelRefineFusion
from synapse.refine.gates import (
    GateThresholds,
    GateVerdict,
    evaluate_refinement_gates,
    export_hard_negatives,
)
from synapse.refine.hand import HandRefiner
from synapse.refine.roi_tracker import KalmanBBoxTracker, compute_affine_align
from synapse.refine.stitcher import ScalpelStitcher, StitchConfig, StitchReport
from synapse.refine.tiled_executor import TiledBypassExecutor
from synapse.refine.types import (
    AnatomicalRegion,
    FusionMethod,
    RefinementResult,
    RefinementROI,
)

__all__ = [
    "AnatomicalRegion",
    "CanonicalAnchorGallery",
    "FaceRefiner",
    "FusionMethod",
    "GateThresholds",
    "GateVerdict",
    "HandRefiner",
    "KalmanBBoxTracker",
    "MultiModelRefineFusion",
    "RefinementResult",
    "RefinementROI",
    "ScalpelStitcher",
    "StitchConfig",
    "StitchReport",
    "TiledBypassExecutor",
    "compute_affine_align",
    "evaluate_refinement_gates",
    "export_hard_negatives",
]
