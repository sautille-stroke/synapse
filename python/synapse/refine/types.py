# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Core data contracts for anatomical local refinement (RFC 0061).

All image data flows through this module in **linear ACEScg** colour space.
Conversion to sRGB is provided solely for display / model-input purposes and
must never feed back into the main pipeline computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class AnatomicalRegion(Enum):
    """Anatomical body region for targeted refinement."""

    HAND = auto()
    FACE = auto()
    FOOT = auto()
    HAIR = auto()
    FULL_BODY = auto()


class FusionMethod(Enum):
    """Multi-model fusion strategy."""

    LAPLACIAN_FREQUENCY = auto()  # Frequency-domain Laplacian pyramid blend
    CASCADE = auto()  # Cascaded multi-resolution refinement
    LATENT_TIMESTEP_SWITCH = auto()  # Latent-space diffusion timestep switching
    UNCERTAINTY_WEIGHTED = auto()  # Uncertainty-based pixel-wise weighting
    POISSON_GRADIENT = auto()  # Poisson gradient-domain seamless clone


@dataclass
class RefinementROI:
    """Region of Interest for local refinement, always in linear ACEScg.

    Attributes:
        bbox_xyxy: Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.
        crop_acescg: Cropped image in linear ACEScg float32, shape ``(H, W, 3)``.
        region: Anatomical region type.
        confidence: Detection confidence score in ``[0, 1]``.
        frame_index: Source frame index.
    """

    bbox_xyxy: np.ndarray  # shape (4,) float32
    crop_acescg: np.ndarray  # shape (H, W, 3) float32 linear ACEScg
    region: AnatomicalRegion
    confidence: float = 0.0
    frame_index: int = 0

    def to_display_srgb_u8(self) -> np.ndarray:
        """Convert *crop_acescg* to sRGB uint8 for display / model input ONLY.

        Never use the returned array for main pipeline computation — it is
        gamma-encoded and quantised to 8 bit.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 3)`` with dtype ``uint8``.
        """
        linear = np.clip(self.crop_acescg, 0.0, 1.0)
        srgb = np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
        )
        return (srgb * 255.0).astype(np.uint8)


@dataclass
class RefinementResult:
    """Result of a local refinement pass.

    Attributes:
        refined_acescg: Refined image patch in linear ACEScg float32,
            shape ``(H, W, 3)``.
        region: Anatomical region that was refined.
        fusion_method: Fusion strategy used for this refinement.
        quality_score: Quality assessment score in ``[0, 1]``.
        frame_index: Source frame index.
        bbox_xyxy: Bounding box ``[x1, y1, x2, y2]`` in the original frame.
    """

    refined_acescg: np.ndarray  # (H, W, 3) float32 in linear ACEScg
    region: AnatomicalRegion
    fusion_method: FusionMethod
    quality_score: float = 0.0  # [0, 1] quality assessment
    frame_index: int = 0
    bbox_xyxy: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
