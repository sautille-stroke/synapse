"""synapse.recon.loop — Forward guide types, radiometric conversion, VAE
phase-locking, and closed-loop inverse rendering runner.

This sub-package implements both the forward and inverse halves of the
inverse rendering closed loop (RFC 0062).  It enforces the T = 4k+1
frame-count constraint required by the 3-D Causal VAE, provides sRGB ↔
ACEScg radiometric conversion utilities, exposes κ-reanchoring, and
orchestrates the full 5-step closed-loop cycle via
:class:`ClosedLoopRunner`.

Import explicitly::

    from synapse.recon.loop import ForwardGuide, LoopConfig, ClosedLoopRunner
    from synapse.recon.loop.radiometric import linear_to_acescg
"""

from __future__ import annotations

from synapse.recon.loop.closed_loop_runner import ClosedLoopResult, ClosedLoopRunner
from synapse.recon.loop.forward_guide import ForwardGuide
from synapse.recon.loop.mitsuba_optimizer import (
    AnalyticUVInverseOptimizer,
    InverseOptimConfig,
    InverseOptimResult,
)
from synapse.recon.loop.radiometric import (
    acescg_to_linear,
    apply_kappa_reanchor,
    enforce_phase_lock,
    linear_to_acescg,
    validate_frame_count_phase,
)
from synapse.recon.loop.texture_extractor import (
    PBRTextureSet,
    extract_and_pack_pbr_textures,
)
from synapse.recon.loop.types import ForwardGuideResult, LoopConfig

__all__ = [
    "AnalyticUVInverseOptimizer",
    "ClosedLoopResult",
    "ClosedLoopRunner",
    "ForwardGuide",
    "ForwardGuideResult",
    "InverseOptimConfig",
    "InverseOptimResult",
    "LoopConfig",
    "PBRTextureSet",
    "acescg_to_linear",
    "apply_kappa_reanchor",
    "enforce_phase_lock",
    "extract_and_pack_pbr_textures",
    "linear_to_acescg",
    "validate_frame_count_phase",
]
