"""synapse.recon.loop — Forward guide types, radiometric conversion, and VAE phase-locking.

This sub-package implements the forward half of the inverse rendering closed
loop (RFC 0062).  It enforces the T = 4k+1 frame-count constraint required by
the 3-D Causal VAE, provides sRGB ↔ ACEScg radiometric conversion utilities,
and exposes κ-reanchoring to prevent studio-lighting contamination from preroll
frames leaking into the production sequence.

Import explicitly::

    from synapse.recon.loop import ForwardGuide, LoopConfig
    from synapse.recon.loop.radiometric import linear_to_acescg
"""

from __future__ import annotations

from synapse.recon.loop.forward_guide import ForwardGuide
from synapse.recon.loop.radiometric import (
    acescg_to_linear,
    apply_kappa_reanchor,
    enforce_phase_lock,
    linear_to_acescg,
    validate_frame_count_phase,
)
from synapse.recon.loop.types import ForwardGuideResult, LoopConfig

__all__ = [
    "ForwardGuide",
    "ForwardGuideResult",
    "LoopConfig",
    "acescg_to_linear",
    "apply_kappa_reanchor",
    "enforce_phase_lock",
    "linear_to_acescg",
    "validate_frame_count_phase",
]
