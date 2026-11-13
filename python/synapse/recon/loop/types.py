"""Core data-types for the inverse rendering closed loop (RFC 0062).

All types are plain ``dataclass`` es with full type-annotations so that
``mypy`` can validate call-sites without importing heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Valid preroll frame counts — matches the 3-D Causal VAE block structure.
# ---------------------------------------------------------------------------
VALID_PREROLL_COUNTS: frozenset[int] = frozenset({0, 4, 8})


@dataclass
class LoopConfig:
    """Configuration for the inverse rendering closed loop.

    Attributes:
        n_preroll: Number of preroll (warm-up) frames prepended before the
            production sequence.  Must be one of ``{0, 4, 8}``.
        kappa: Hard-cut re-anchoring strength at the first production frame.
            ``1.0`` fully replaces the last preroll latent with the production
            key-frame, preventing 5500 K studio-lighting contamination.
        max_iterations: Maximum optimisation iterations in the inverse loop.
        convergence_threshold: L2 residual threshold for early-stop convergence.
    """

    n_preroll: int = 0
    kappa: float = 1.0
    max_iterations: int = 5
    convergence_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.n_preroll not in VALID_PREROLL_COUNTS:
            from synapse.errors import ContractValidationError

            raise ContractValidationError(
                f"n_preroll must be one of {sorted(VALID_PREROLL_COUNTS)}, "
                f"got {self.n_preroll}.",
                agent_fix_hint=(
                    f"Set n_preroll to one of {sorted(VALID_PREROLL_COUNTS)}.  "
                    "These values align with the 3-D Causal VAE block "
                    "boundaries."
                ),
            )


@dataclass
class ForwardGuideResult:
    """Result of forward 3D-guided video generation.

    Attributes:
        frames: Generated video frames with shape ``(T, H, W, 3)`` in linear
            sRGB, where *T* satisfies the ``4k+1`` constraint.
        fgpp_sequence: Per-frame FGPP metadata dictionaries.
        preroll_count: Actual number of preroll frames that were used.
        phase_valid: ``True`` when *T* satisfies the ``T = 4k+1`` constraint.
    """

    frames: np.ndarray
    fgpp_sequence: list[dict[str, Any]] = field(default_factory=list)
    preroll_count: int = 0
    phase_valid: bool = True
