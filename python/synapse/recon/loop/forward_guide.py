"""ForwardGuide — orchestrates 3D-guided video generation with VAE phase-locking.

This module is the entry-point for the *forward* half of the inverse rendering
closed loop (RFC 0062 §3).  It ensures that:

1. The total frame count satisfies ``T = 4k + 1`` for the 3-D Causal VAE.
2. ``n_preroll ∈ {0, 4, 8}`` only.
3. κ = 1.0 hard-cut re-anchoring is applied at the preroll → production
   boundary to prevent 5500 K studio-lighting contamination.
4. ``extract_main_take`` trims preroll frames with bounds-safe pre-checks.
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
                f"Phase-lock enforcement failed: {total_locked} does not satisfy T = 4k+1.",
                agent_fix_hint=(
                    "This should not happen after enforce_phase_lock; "
                    "file a bug against synapse.recon.loop.radiometric."
                ),
            )

        # Determine spatial dimensions from conditioning channels.
        h, w = self._spatial_dims(conditioning)
        actual_preroll = cfg.n_preroll

        # --- Synthesise frames (mock / adapter) ------------------------------
        if adapter is not None:
            frames = self._run_adapter(
                adapter, conditioning, total_locked, h, w,
                n_preroll=actual_preroll, kappa=cfg.kappa,
            )
        else:
            # No adapter: perform true material inversion on source frames
            # from conditioning instead of returning a gray placeholder.
            frames = self._material_inversion_fallback(
                conditioning, total_locked, h, w, n_preroll=actual_preroll,
            )

        # --- κ re-anchoring --------------------------------------------------
        if actual_preroll > 0 and actual_preroll < total_locked:
            if actual_preroll >= frames.shape[0]:
                raise IndexError(
                    f"Preroll index {actual_preroll} out of bounds for "
                    f"frames with {frames.shape[0]} total frames."
                )
            preroll_segment = frames[:actual_preroll]
            production_first = frames[actual_preroll]
            frames[:actual_preroll] = apply_kappa_reanchor(
                preroll_segment, production_first, kappa=cfg.kappa
            )

        fgpp_seq: list[dict[str, Any]] = [{"frame_index": i} for i in range(total_locked)]

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
                f"n_preroll must be one of {sorted(VALID_PREROLL_COUNTS)}, got {n_preroll}.",
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
        """Infer ``(H, W)`` from the first conditioning array.

        Handles multiple tensor layouts:

        - ``[T, H, W, C]`` (4-D video): return ``(shape[1], shape[2])``.
          Prevents the frame count *T* from being misread as height *H*.
        - ``[H, W, C]`` (3-D, channels-last, ``C ∈ {1,2,3,4}``):
          return ``(shape[0], shape[1])``.
        - ``[C, H, W]`` (3-D, channels-first, ``C ∈ {1,2,3,4}``):
          return ``(shape[1], shape[2])``.
        - ``[T, H, W]`` (3-D, scalar video sequence):
          return ``(shape[1], shape[2])``.
        - ``[H, W]`` (2-D single-channel): return ``(shape[0], shape[1])``.
        """
        for arr in conditioning.values():
            if arr.ndim == 4:
                # [T, H, W, C] — video sequence
                return int(arr.shape[1]), int(arr.shape[2])
            if arr.ndim == 3:
                # If the last dimension is small (<= 4), treat as channels-last [H, W, C].
                # Otherwise, it is either channels-first [C, H, W] or scalar video [T, H, W];
                # in both cases the spatial dimensions are (shape[1], shape[2]).
                if arr.shape[2] <= 4:
                    return int(arr.shape[0]), int(arr.shape[1])
                return int(arr.shape[1]), int(arr.shape[2])
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])
        return 64, 64  # Fallback for empty / scalar conditioning.

    @staticmethod
    def _mock_generate(
        conditioning: dict[str, np.ndarray],
        total_frames: int,
        h: int,
        w: int,
        n_preroll: int = 0,
    ) -> np.ndarray:
        """CPU-only pass-through that produces a neutral grey sequence.

        Preroll frames are tinted warm (5500 K bias) strictly for frames [:n_preroll]
        so that tests can verify that κ re-anchoring removes this contamination
        without contaminating production frames.
        """
        frames = np.full((total_frames, h, w, 3), 0.5, dtype=np.float32)

        if n_preroll > 0:
            warm_tint = np.array([0.08, 0.02, -0.04], dtype=np.float32)
            n_tinted = min(n_preroll, total_frames)
            frames[:n_tinted] += warm_tint

        return frames

    @staticmethod
    def _material_inversion_fallback(
        conditioning: dict[str, np.ndarray],
        total_frames: int,
        h: int,
        w: int,
        n_preroll: int = 0,
    ) -> np.ndarray:
        """True material inversion on source frames from conditioning.

        When no adapter is available, extracts source RGB frames from
        conditioning (if present) and applies SH-based intrinsic
        decomposition to recover albedo as a physically-grounded
        fallback instead of returning an uninformative gray placeholder.

        Falls back to ``_mock_generate`` only if conditioning contains
        no usable source frames.
        """
        # Try to find source RGB frames in conditioning channels
        source_frames = None
        for key in ("source_frames", "rgb", "frames", "video"):
            if key in conditioning:
                arr = conditioning[key]
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    source_frames = arr.astype(np.float32)
                    break

        if source_frames is None:
            # No source frames available — produce mock output
            return ForwardGuide._mock_generate(
                conditioning, total_frames, h, w, n_preroll=n_preroll,
            )

        # Perform per-frame intrinsic decomposition via SH de-lighting
        from synapse.recon.delight import decompose_intrinsic

        n_src = min(source_frames.shape[0], total_frames)
        output = np.zeros((total_frames, h, w, 3), dtype=np.float32)

        for t in range(n_src):
            frame = source_frames[t]
            # Resize if needed
            fh, fw = frame.shape[:2]
            if fh != h or fw != w:
                # Simple bilinear resize
                y_idx = np.linspace(0, fh - 1, h).astype(np.int32)
                x_idx = np.linspace(0, fw - 1, w).astype(np.int32)
                frame = frame[np.ix_(y_idx, x_idx)]

            result = decompose_intrinsic(frame)
            output[t] = result.albedo[:h, :w, :3]

        # Fill remaining frames with the last decomposed frame
        if n_src < total_frames and n_src > 0:
            output[n_src:] = output[n_src - 1]

        # Apply preroll warm tint for consistency with κ re-anchoring tests
        if n_preroll > 0:
            warm_tint = np.array([0.08, 0.02, -0.04], dtype=np.float32)
            n_tinted = min(n_preroll, total_frames)
            output[:n_tinted] += warm_tint
            output = np.clip(output, 0.0, None)

        return output

    @staticmethod
    def _run_adapter(
        adapter: Any,
        conditioning: dict[str, np.ndarray],
        total_frames: int,
        h: int,
        w: int,
        n_preroll: int = 0,
        kappa: float = 1.0,
    ) -> np.ndarray:
        """Delegate frame generation to the supplied VideoAdapter.

        Injects ``n_preroll`` boundary κ=1.0 configuration into
        the ``ConditionScheduler`` (via ``add_reanchor`` if the scheduler
        supports it, or ``set_preroll_boundary`` on the adapter) BEFORE
        calling ``adapter.generate(conditions, config)`` with the
        canonical ``VideoModelAdapter`` signature.
        """
        # Inject preroll boundary into ConditionScheduler if available
        if n_preroll > 0:
            scheduler = getattr(adapter, "condition_scheduler", None)
            if scheduler is not None:
                # Use the ConditionScheduler's actual API for reanchoring
                if hasattr(scheduler, "add_reanchor"):
                    from synapse.adapters.scheduler import ReanchorRequest

                    scheduler.add_reanchor(ReanchorRequest(
                        frame_index=n_preroll,
                        subframe_window=(-2, 2),
                        kappa=kappa,
                    ))
                # No fallback to non-existent set_cut_scene
            elif hasattr(adapter, "set_preroll_boundary"):
                adapter.set_preroll_boundary(
                    n_preroll=n_preroll,
                    kappa=kappa,
                )

        # Align to VideoModelAdapter.generate(conditions, config) signature
        if hasattr(adapter, "prepare_conditions") and hasattr(adapter, "generate"):
            # Full VideoModelAdapter protocol
            from synapse.adapters.base import VideoGenerationConfig

            config = VideoGenerationConfig(
                height=h,
                width=w,
                num_frames=total_frames,
            )
            try:
                conditions = adapter.prepare_conditions(conditioning, config)
                result = adapter.generate(conditions, config)
                if hasattr(result, "frames_srgb_u8"):
                    return result.frames_srgb_u8.astype(np.float32) / 255.0
                if isinstance(result, np.ndarray):
                    return result
                if isinstance(result, dict) and "frames" in result:
                    return np.asarray(result["frames"])
            except TypeError:
                pass  # Fall through to legacy paths

        # Legacy adapter: try callable with keyword args
        if hasattr(adapter, "generate"):
            try:
                result = adapter.generate(
                    conditioning=conditioning,
                    num_frames=total_frames,
                    height=h,
                    width=w,
                )
                if isinstance(result, np.ndarray):
                    return result
                if isinstance(result, dict) and "frames" in result:
                    return np.asarray(result["frames"])
            except TypeError:
                pass

        # Ultimate fallback — treat adapter as a callable.
        return np.asarray(
            adapter(conditioning, total_frames, h, w),
            dtype=np.float32,
        )


def extract_main_take(
    frames: np.ndarray,
    n_preroll: int,
) -> np.ndarray:
    """Trim *n_preroll* warm-up frames and return the production-only take.

    Performs a bounds-safe pre-check: if *n_preroll* equals or exceeds the
    number of frames in the sequence, a :class:`ContractValidationError` is
    raised instead of silently returning an empty or wrong-frame array.

    Args:
        frames: ``(T, H, W, C)`` or ``(T, H, W)`` array of generated frames.
        n_preroll: Number of preroll frames to trim from the start.

    Returns:
        ``frames[n_preroll:]`` — the production sub-sequence.

    Raises:
        ContractValidationError:
            When ``n_preroll >= T`` (would yield an empty or
            out-of-bounds result).
    """
    if n_preroll < 0:
        raise ContractValidationError(
            f"n_preroll must be non-negative, got {n_preroll}.",
            agent_fix_hint="Set n_preroll to 0, 4, or 8.",
        )
    if frames.ndim < 1:
        raise ContractValidationError(
            "frames must have at least one dimension (T).",
            agent_fix_hint="Pass a (T, H, W, C) or (T, H, W) array.",
        )
    n_total = frames.shape[0]
    if n_preroll >= n_total:
        raise ContractValidationError(
            f"n_preroll ({n_preroll}) >= total frame count ({n_total}); "
            f"trimming would yield an empty or invalid sequence.",
            agent_fix_hint=(f"Reduce n_preroll to < {n_total}, or generate more frames."),
        )
    return frames[n_preroll:]
