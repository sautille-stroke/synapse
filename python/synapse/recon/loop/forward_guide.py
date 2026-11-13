"""ForwardGuide — orchestrates 3D-guided video generation with VAE phase-locking.

This module is the entry-point for the *forward* half of the inverse rendering
closed loop (RFC 0062 §3).  It ensures that:

1. The total frame count satisfies ``T = 4k + 1`` for the 3-D Causal VAE.
2. ``n_preroll ∈ {0, 4, 8}`` only.
3. κ = 1.0 hard-cut re-anchoring is applied at the preroll → production
   boundary to prevent 5500 K studio-lighting contamination.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synapse.errors import ContractValidationError
from synapse.recon.loop.radiometric import (
    apply_kappa_reanchor,
    enforce_phase_lock,
    validate_frame_count_phase,
)
from synapse.recon.loop.types import (
    VALID_PREROLL_COUNTS,
    ForwardGuideResult,
    LoopConfig,
)


class ForwardGuide:
    """Orchestrates 3D-guided video generation with phase-locked frame counts.

    Enforces:
      - T = 4k+1 for 3-D Causal VAE compatibility
      - n_preroll ∈ {0, 4, 8} only
      - κ = 1.0 hard-cut re-anchoring at production start

    Example::

        guide = ForwardGuide(LoopConfig(n_preroll=4, kappa=1.0))
        result = guide.generate(conditioning=fgpp_channels, n_production_frames=97)
    """

    def __init__(self, config: LoopConfig | None = None) -> None:
        self._config = config or LoopConfig()

    # -- public API -----------------------------------------------------------

    @property
    def config(self) -> LoopConfig:
        """Return the active loop configuration (read-only copy)."""
        return self._config

    def generate(
        self,
        conditioning: dict[str, np.ndarray],
        adapter: Any = None,
        n_production_frames: int = 97,
    ) -> ForwardGuideResult:
        """Generate a forward-guided video sequence.

        Args:
            conditioning: FGPP channel arrays keyed by channel name
                (e.g. ``"CameraLinearDepth.Z"``, ``"WorldNormal.X"``).
            adapter: Optional :class:`~synapse.adapters.VideoAdapter` instance.
                When ``None``, a CPU-only mock pass-through is used (useful
                for contract validation without GPU).
            n_production_frames: Number of *production* frames to generate.
                The total frame count (preroll + production) will be rounded
                up to satisfy ``T = 4k + 1`` if necessary.

        Returns:
            :class:`ForwardGuideResult` containing the generated frames,
            per-frame FGPP metadata, preroll count, and phase validity flag.

        Raises:
            ContractValidationError:
                ``exit_code = 10`` when ``n_preroll`` is not in ``{0, 4, 8}``
                or the total frame count cannot satisfy ``T = 4k + 1``.
        """
        cfg = self._config
        self._validate_preroll(cfg.n_preroll)

        total_raw = cfg.n_preroll + n_production_frames
        total_locked = enforce_phase_lock(total_raw)
        phase_valid = validate_frame_count_phase(total_locked)

        if not phase_valid:
            raise ContractValidationError(
                f"Phase-lock enforcement failed: {total_locked} does not "
                f"satisfy T = 4k+1.",
                agent_fix_hint=(
                    "This should not happen after enforce_phase_lock; "
                    "file a bug against synapse.recon.loop.radiometric."
                ),
            )

        # Determine spatial dimensions from conditioning channels.
        h, w = self._spatial_dims(conditioning)

        # --- Synthesise frames (mock / adapter) ------------------------------
        if adapter is not None:
            frames = self._run_adapter(
                adapter, conditioning, total_locked, h, w
            )
        else:
            frames = self._mock_generate(conditioning, total_locked, h, w)

        # --- κ re-anchoring --------------------------------------------------
        actual_preroll = cfg.n_preroll
        if actual_preroll > 0 and actual_preroll < total_locked:
            preroll_segment = frames[:actual_preroll]
            production_first = frames[actual_preroll]
            frames[:actual_preroll] = apply_kappa_reanchor(
                preroll_segment, production_first, kappa=cfg.kappa
            )

        fgpp_seq: list[dict[str, Any]] = [
            {"frame_index": i} for i in range(total_locked)
        ]

        return ForwardGuideResult(
            frames=frames,
            fgpp_sequence=fgpp_seq,
            preroll_count=actual_preroll,
            phase_valid=phase_valid,
        )

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _validate_preroll(n_preroll: int) -> None:
        """Raise if *n_preroll* is not in the valid set."""
        if n_preroll not in VALID_PREROLL_COUNTS:
            raise ContractValidationError(
                f"n_preroll must be one of {sorted(VALID_PREROLL_COUNTS)}, "
                f"got {n_preroll}.",
                agent_fix_hint=(
                    f"Set n_preroll to one of {sorted(VALID_PREROLL_COUNTS)}.  "
                    "These values align with the 3-D Causal VAE block "
                    "boundaries."
                ),
            )

    @staticmethod
    def _spatial_dims(
        conditioning: dict[str, np.ndarray],
    ) -> tuple[int, int]:
        """Infer ``(H, W)`` from the first conditioning array."""
        for arr in conditioning.values():
            if arr.ndim >= 3:
                return int(arr.shape[-3]), int(arr.shape[-2])
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])
        return 64, 64  # Fallback for empty / scalar conditioning.

    @staticmethod
    def _mock_generate(
        conditioning: dict[str, np.ndarray],
        total_frames: int,
        h: int,
        w: int,
    ) -> np.ndarray:
        """CPU-only pass-through that produces a neutral grey sequence.

        Preroll frames are tinted warm (5500 K bias) so that tests can verify
        that κ re-anchoring removes this contamination.
        """
        frames = np.full((total_frames, h, w, 3), 0.5, dtype=np.float32)

        # Apply a slight warm tint to the first few frames to simulate
        # preroll studio-lighting contamination (5500 K).
        warm_tint = np.array([0.08, 0.02, -0.04], dtype=np.float32)
        # Determine how many frames to tint — use the first conditioning
        # channel's temporal extent or default to 8.
        n_tinted = min(8, total_frames)
        frames[:n_tinted] += warm_tint

        return frames

    @staticmethod
    def _run_adapter(
        adapter: Any,
        conditioning: dict[str, np.ndarray],
        total_frames: int,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Delegate frame generation to the supplied VideoAdapter."""
        if hasattr(adapter, "generate"):
            result = adapter.generate(
                conditioning=conditioning,
                num_frames=total_frames,
                height=h,
                width=w,
            )
            if isinstance(result, np.ndarray):
                return result
            # Adapter may return a dict with a "frames" key.
            if isinstance(result, dict) and "frames" in result:
                return np.asarray(result["frames"])
        # Ultimate fallback — treat adapter as a callable.
        return np.asarray(
            adapter(conditioning, total_frames, h, w),
            dtype=np.float32,
        )
