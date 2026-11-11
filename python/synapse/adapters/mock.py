"""Deterministic CPU-only mock video adapter for testing.

``MockVideoAdapter`` produces reproducible sRGB frames from FGPP inputs
using simple Lambertian shading, optional film grain, and controllable
defect injection (silhouette shift, flicker, colour-temperature shift).

No GPU or heavy dependencies are required — only NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.adapters.base import (
    AdapterTier,
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoModelAdapter,
    VideoResult,
    register_adapter,
)

# ---------------------------------------------------------------------------
# Defect injection configuration
# ---------------------------------------------------------------------------

_FIXED_LIGHT_DIR: np.ndarray = np.array(
    [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
    dtype=np.float32,
)
"""Normalised (1/√3, 1/√3, 1/√3) — light coming from upper-right-front."""


@dataclass
class DefectInjectionConfig:
    """Controlled defect injection for testing quality metrics.

    Attributes:
        frame_range: ``(start, end)`` inclusive frame indices to apply
            the defect.  ``-1`` as *end* means the last frame.
        entity_id: Cryptomatte entity ID to restrict the defect to.
            ``0`` means apply to *all* pixels.
        silhouette_shift_px: Horizontal pixel shift applied to the
            target entity region (positive = rightward).
        flicker_amplitude: Peak brightness modulation amplitude
            applied as ``sin(2π · t / T) * amplitude``.
        kelvin_shift: Colour-temperature shift in Kelvin.
            Positive warms (boost red, reduce blue); negative cools.
    """

    frame_range: tuple[int, int] = (0, -1)
    entity_id: int = 0
    silhouette_shift_px: int = 0
    flicker_amplitude: float = 0.0
    kelvin_shift: float = 0.0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _kelvin_rgb_ratio(kelvin_shift: float) -> np.ndarray:
    """Return an ``[3]`` float32 multiplicative RGB ratio for *kelvin_shift*.

    Uses a simplified model:
      * Positive kelvin_shift → warmer → boost R, reduce B.
      * Negative kelvin_shift → cooler → reduce R, boost B.
    """
    # Simple linear model clamped to a safe range
    k = np.clip(kelvin_shift / 6500.0, -0.8, 0.8)
    return np.array([1.0 + k * 0.3, 1.0, 1.0 - k * 0.3], dtype=np.float32)


def _apply_silhouette_shift(
    frame: np.ndarray,
    mask: np.ndarray,
    shift_px: int,
) -> np.ndarray:
    """Shift pixels covered by *mask* horizontally by *shift_px*.

    Args:
        frame: ``[H, W, 3]`` float32 image to modify (mutated in-place).
        mask: ``[H, W]`` bool mask of the target region.
        shift_px: Number of pixels to shift right (positive) or left.

    Returns:
        The mutated *frame*.
    """
    if shift_px == 0 or not np.any(mask):
        return frame

    h, w, _ = frame.shape
    shifted = np.zeros_like(frame)

    if shift_px > 0:
        shifted[:, shift_px:, :] = frame[:, :w - shift_px, :]
    else:
        s = -shift_px
        shifted[:, :w - s, :] = frame[:, s:, :]

    frame[mask] = shifted[mask]
    return frame


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light float32 image to sRGB float32.

    Uses the standard piecewise sRGB transfer function.
    """
    out = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0031308), 1.0 / 2.4) - 0.055,
    )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Mock adapter implementation
# ---------------------------------------------------------------------------


@register_adapter
class MockVideoAdapter(VideoModelAdapter):
    """Deterministic CPU-only mock adapter for Synapse-Bench testing.

    Produces simple Lambertian-shaded frames from FGPP ``base_color``
    and ``world_normal`` channels, with optional film grain and
    defect injection.
    """

    def __init__(
        self,
        defects: list[DefectInjectionConfig] | None = None,
    ) -> None:
        self._defects: list[DefectInjectionConfig] = defects or []

    # -- properties ----------------------------------------------------------

    @property
    def adapter_id(self) -> str:  # noqa: D102
        return "mock"

    @property
    def tier(self) -> AdapterTier:  # noqa: D102
        return "mock"

    # -- public interface ----------------------------------------------------

    def prepare_conditions(
        self,
        fgpp_sequence: Any,  # FGPPSequence
        config: VideoGenerationConfig,
    ) -> PreparedConditionPayload:
        """Extract per-frame FGPP channels into a condition payload.

        Args:
            fgpp_sequence: An ``FGPPSequence`` with at least ``depth``.
            config: Generation configuration (unused in mock).

        Returns:
            ``PreparedConditionPayload`` populated with numpy arrays.
        """
        n_frames = len(fgpp_sequence.frames)
        if n_frames == 0:
            return PreparedConditionPayload()

        first = fgpp_sequence.frames[0]
        h, w = first.height, first.width

        # Allocate stacked arrays
        base_color = np.zeros((n_frames, h, w, 3), dtype=np.float32)
        world_normal = np.zeros((n_frames, h, w, 3), dtype=np.float32)
        direct_radiance = np.zeros((n_frames, h, w, 3), dtype=np.float32)
        depth = np.zeros((n_frames, h, w), dtype=np.float32)
        cryptomatte_id = np.zeros((n_frames, h, w), dtype=np.uint32)

        for i, frame in enumerate(fgpp_sequence.frames):
            depth[i] = frame.depth.astype(np.float32)

            if frame.base_color is not None:
                base_color[i] = frame.base_color.astype(np.float32)

            if frame.world_normal is not None:
                world_normal[i] = frame.world_normal.astype(np.float32)

            if frame.direct_radiance is not None:
                direct_radiance[i] = frame.direct_radiance.astype(
                    np.float32
                )

            if frame.cryptomatte_id is not None:
                cryptomatte_id[i] = frame.cryptomatte_id

        return PreparedConditionPayload(
            base_color=base_color,
            world_normal=world_normal,
            direct_radiance=direct_radiance,
            depth=depth,
            cryptomatte_id=cryptomatte_id,
        )

    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> VideoResult:
        """Generate deterministic mock video frames.

        Pipeline:
        1. Lambertian diffuse shading from ``world_normal``.
        2. Modulate by ``base_color``.
        3. Add film grain (RNG-seeded Gaussian noise).
        4. Apply any configured defect injections.
        5. Linear → sRGB → uint8.

        Args:
            conditions: Prepared payload from ``prepare_conditions``.
            config: Generation config (``seed`` is used for RNG).

        Returns:
            ``VideoResult`` with ``[T, H, W, 3]`` uint8 sRGB frames.
        """
        rng = np.random.Generator(np.random.PCG64(config.seed))

        # Determine output dimensions
        if conditions.base_color is not None:
            n_frames, h, w, _ = conditions.base_color.shape
        elif conditions.depth is not None:
            n_frames, h, w = conditions.depth.shape
        else:
            n_frames = config.num_frames
            h, w = config.height, config.width

        # --- Step 1-2: Lambertian diffuse shading --------------------------
        frames_linear = np.zeros((n_frames, h, w, 3), dtype=np.float32)

        if conditions.world_normal is not None:
            normals = conditions.world_normal  # [T, H, W, 3]
            # Lambertian N·L clamped to [0, 1]
            ndotl = np.einsum("thwc,c->thw", normals, _FIXED_LIGHT_DIR)
            ndotl = np.clip(ndotl, 0.0, 1.0)
            # Ambient + diffuse
            shading = 0.15 + 0.85 * ndotl  # [T, H, W]
            frames_linear = shading[..., np.newaxis] * np.ones(
                (1, 1, 1, 3), dtype=np.float32
            )
        else:
            frames_linear[:] = 0.5

        if conditions.base_color is not None:
            frames_linear *= conditions.base_color

        # --- Step 3: Film grain noise --------------------------------------
        grain = rng.normal(
            0.0, 0.02, size=(n_frames, h, w, 3)
        ).astype(np.float32)
        frames_linear = frames_linear + grain

        # --- Step 4: Defect injections -------------------------------------
        applied_defects: list[dict[str, Any]] = []
        for defect in self._defects:
            t0 = defect.frame_range[0]
            t1 = defect.frame_range[1]
            if t1 < 0:
                t1 = n_frames - 1

            affected_frames = range(
                max(0, t0), min(n_frames, t1 + 1)
            )

            # Build per-frame entity mask
            for t in affected_frames:
                if defect.entity_id == 0:
                    mask = np.ones((h, w), dtype=bool)
                elif conditions.cryptomatte_id is not None:
                    mask = conditions.cryptomatte_id[t] == defect.entity_id
                else:
                    mask = np.ones((h, w), dtype=bool)

                # Silhouette shift
                if defect.silhouette_shift_px != 0:
                    frames_linear[t] = _apply_silhouette_shift(
                        frames_linear[t], mask, defect.silhouette_shift_px
                    )

                # Flicker (brightness oscillation)
                if defect.flicker_amplitude != 0.0:
                    phase = math.sin(
                        2.0 * math.pi * t / max(n_frames, 1)
                    )
                    flicker = 1.0 + defect.flicker_amplitude * phase
                    frames_linear[t][mask] *= flicker

                # Kelvin shift (colour temperature)
                if defect.kelvin_shift != 0.0:
                    ratio = _kelvin_rgb_ratio(defect.kelvin_shift)
                    frames_linear[t][mask] *= ratio

            applied_defects.append({
                "silhouette_shift_px": defect.silhouette_shift_px,
                "flicker_amplitude": defect.flicker_amplitude,
                "kelvin_shift": defect.kelvin_shift,
                "entity_id": defect.entity_id,
                "frame_range": [t0, t1],
            })

        # --- Step 5: Linear → sRGB → uint8 --------------------------------
        frames_linear = np.clip(frames_linear, 0.0, 1.0)
        frames_srgb = _linear_to_srgb(frames_linear)
        frames_u8 = np.clip(
            np.round(frames_srgb * 255.0), 0, 255
        ).astype(np.uint8)

        return VideoResult(
            frames_srgb_u8=frames_u8,
            fps=config.fps,
            seed=config.seed,
            adapter_id=self.adapter_id,
            metadata={"defects_applied": applied_defects},
        )
