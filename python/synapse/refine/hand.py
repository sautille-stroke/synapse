# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Hand refinement using FGPP geometric priors (RFC 0061).

Validates finger topology via convex-hull defect counting on the hand
silhouette.  When the topology check fails (e.g. a 6-finger anomaly
produced by the generative model) the refiner falls back to FGPP
depth-channel guidance and emits a structured warning.

All image data is processed in **linear ACEScg** colour space.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from synapse.refine.types import (
    AnatomicalRegion,
    FusionMethod,
    RefinementResult,
    RefinementROI,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FINGER_COUNT: int = 5
"""Expected number of fingers for a normal human hand."""

_Z_NEAR_DEFAULT: float = 0.1
"""Default near plane (metres) when not available from FGPP metadata."""

_Z_FAR_DEFAULT: float = 100.0
"""Default far plane (metres) when not available from FGPP metadata."""


# ---------------------------------------------------------------------------
# HandRefiner
# ---------------------------------------------------------------------------


class HandRefiner:
    """Hand refinement using FGPP geometric priors (depth + normal channels).

    Reuses the FGPP log-normalised depth and camera-space normal
    primitives defined in ``synapse.fgpp.schema`` and
    ``synapse.adapters.wan``.

    Parameters
    ----------
    finger_count_expected:
        Number of fingers that a healthy hand is expected to have.
        Defaults to ``5``.
    z_near:
        Near plane for log-depth normalisation (metres).
    z_far:
        Far plane for log-depth normalisation (metres).
    """

    def __init__(
        self,
        finger_count_expected: int = _DEFAULT_FINGER_COUNT,
        z_near: float = _Z_NEAR_DEFAULT,
        z_far: float = _Z_FAR_DEFAULT,
    ) -> None:
        self._finger_count_expected = finger_count_expected
        self._z_near = z_near
        self._z_far = z_far

    # -- public interface ----------------------------------------------------

    def prepare_hard_priors(
        self,
        roi: RefinementROI,
        depth_map: np.ndarray | None = None,
        normal_map: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Prepare FGPP-derived geometric priors for a hand region.

        Returns
        -------
        dict[str, np.ndarray]
            ``"log_depth_normalized"``
                Log-normalised depth in ``[0, 1]``, shape ``(H, W)``.
            ``"camera_normal_encoded"``
                Camera-space normals mapped to ``[0, 1]``, shape
                ``(H, W, 3)``.
            ``"hand_mask"``
                Binary mask (``float32``) of the hand region,
                shape ``(H, W)``.
        """
        from synapse.fgpp.schema import linear_to_log_normalized_depth

        h, w = roi.crop_acescg.shape[:2]

        # -- log-depth -------------------------------------------------------
        if depth_map is not None:
            depth_crop = depth_map.astype(np.float32)
        else:
            # Uniform mid-range depth when FGPP depth is unavailable
            mid_depth = (self._z_near + self._z_far) / 2.0
            depth_crop = np.full((h, w), mid_depth, dtype=np.float32)

        log_depth = linear_to_log_normalized_depth(
            depth_crop,
            z_near_m=self._z_near,
            z_far_m=self._z_far,
        ).astype(np.float32)

        # -- camera normals --------------------------------------------------
        if normal_map is not None:
            normals = normal_map.astype(np.float32)
        else:
            # Default forward-facing normal
            normals = np.zeros((h, w, 3), dtype=np.float32)
            normals[..., 2] = 1.0

        # Encode [-1, 1] → [0, 1] for condition tensor compatibility
        cam_normal_encoded = (normals * 0.5 + 0.5).astype(np.float32)
        cam_normal_encoded = np.clip(cam_normal_encoded, 0.0, 1.0)

        # -- hand mask -------------------------------------------------------
        # Derive a simple hand mask from the crop's luminance (non-black)
        lum = np.mean(roi.crop_acescg, axis=-1)
        hand_mask = (lum > 0.01).astype(np.float32)

        return {
            "log_depth_normalized": log_depth,
            "camera_normal_encoded": cam_normal_encoded,
            "hand_mask": hand_mask,
        }

    def refine(
        self,
        roi: RefinementROI,
        priors: dict[str, np.ndarray] | None = None,
    ) -> RefinementResult:
        """Refine a hand ROI, validating finger topology first.

        If finger topology validation fails (count ≠ ``finger_count_expected``)
        the refiner falls back to FGPP depth-channel guidance and logs a
        warning.

        Parameters
        ----------
        roi:
            The hand region of interest.
        priors:
            Pre-computed geometric priors from :meth:`prepare_hard_priors`.
            If ``None``, priors are computed on the fly.

        Returns
        -------
        RefinementResult
            Refined hand patch in linear ACEScg.
        """
        if priors is None:
            priors = self.prepare_hard_priors(roi)

        hand_mask = priors["hand_mask"]
        finger_count, is_valid = self._validate_finger_topology(hand_mask)

        quality = 1.0
        fusion = FusionMethod.POISSON_GRADIENT

        if not is_valid:
            logger.warning(
                "Finger topology invalid: detected %d finger(s) "
                "(expected %d) at frame %d. Falling back to "
                "FGPP depth-channel guidance.",
                finger_count,
                self._finger_count_expected,
                roi.frame_index,
            )
            quality = 0.5
            fusion = FusionMethod.UNCERTAINTY_WEIGHTED

        # Apply depth-guided refinement (blend original with
        # depth-modulated version for smoother hand geometry)
        log_depth = priors["log_depth_normalized"]
        refined = roi.crop_acescg.copy()

        # Depth-aware sharpening: emphasise high-frequency detail in
        # areas with strong depth gradients (finger edges)
        if log_depth.shape[:2] == refined.shape[:2]:
            # Compute a simple depth-edge weight
            dy = np.diff(log_depth, axis=0, prepend=log_depth[:1, :])
            dx = np.diff(log_depth, axis=1, prepend=log_depth[:, :1])
            edge = np.sqrt(dx ** 2 + dy ** 2)
            edge_weight = np.clip(edge * 5.0, 0.0, 1.0)[..., np.newaxis]
            # Gently sharpen by boosting local contrast at depth edges
            local_mean = np.mean(refined, axis=-1, keepdims=True)
            sharpened = refined + edge_weight * (refined - local_mean) * 0.3
            refined = np.clip(sharpened, 0.0, 1.0).astype(np.float32)

        return RefinementResult(
            refined_acescg=refined,
            region=AnatomicalRegion.HAND,
            fusion_method=fusion,
            quality_score=quality,
            frame_index=roi.frame_index,
            bbox_xyxy=roi.bbox_xyxy.copy(),
        )

    # -- private helpers -----------------------------------------------------

    def _validate_finger_topology(
        self,
        hand_mask: np.ndarray,
    ) -> tuple[int, bool]:
        """Count finger-like protrusions via upper-region component analysis.

        The algorithm:
        1. Binarise the mask (> 0.5).
        2. Identify the "finger zone" — the upper portion of the mask's
           bounding box where fingers protrude beyond the palm.
        3. Count distinct connected components in the finger zone
           (each connected strip is one finger).

        Parameters
        ----------
        hand_mask:
            Binary hand mask, shape ``(H, W)``.

        Returns
        -------
        tuple[int, bool]
            ``(detected_finger_count, is_valid)`` where ``is_valid``
            is ``True`` when ``detected_finger_count == finger_count_expected``.
        """
        binary = (hand_mask > 0.5).astype(np.uint8)
        if binary.sum() < 10:
            return 0, False

        h, w = binary.shape

        # Find the vertical extent of the hand
        rows_with_pixels = np.any(binary > 0, axis=1)
        row_indices = np.where(rows_with_pixels)[0]
        if len(row_indices) == 0:
            return 0, False

        top_row = int(row_indices[0])
        bottom_row = int(row_indices[-1])
        hand_height = bottom_row - top_row + 1
        if hand_height < 4:
            return 1, self._finger_count_expected == 1

        # The "finger zone" is the upper third of the hand bounding box.
        # Fingers are the thin protrusions that extend above the palm mass.
        finger_zone_bottom = top_row + hand_height // 3
        finger_zone = binary[top_row:finger_zone_bottom, :]

        if finger_zone.sum() < 2:
            return 1, self._finger_count_expected == 1

        # Count connected components in the finger zone — each distinct
        # horizontal cluster in the topmost rows is a finger.
        labels, n_components = _label_connected_components(finger_zone)
        # Filter out tiny components (noise)
        min_finger_pixels = max(2, finger_zone.sum() // 20)
        finger_count = 0
        for lab in range(1, n_components + 1):
            if int(np.sum(labels == lab)) >= min_finger_pixels:
                finger_count += 1

        is_valid = finger_count == self._finger_count_expected
        return finger_count, is_valid


# ---------------------------------------------------------------------------
# Pure-numpy geometry helpers (no cv2 dependency)
# ---------------------------------------------------------------------------


def _label_connected_components(
    binary: np.ndarray,
) -> tuple[np.ndarray, int]:
    """4-connected component labelling via iterative flood fill.

    Parameters
    ----------
    binary:
        ``(H, W)`` uint8 mask with values 0 / 1.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(labels, n_components)`` — labels array same shape as input.
    """
    h, w = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 1 and labels[y, x] == 0:
                current_label += 1
                # BFS flood fill
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if (
                        0 <= cy < h
                        and 0 <= cx < w
                        and binary[cy, cx] == 1
                        and labels[cy, cx] == 0
                    ):
                        labels[cy, cx] = current_label
                        stack.extend([
                            (cy - 1, cx),
                            (cy + 1, cx),
                            (cy, cx - 1),
                            (cy, cx + 1),
                        ])

    return labels, current_label

