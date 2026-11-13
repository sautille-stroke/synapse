# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for ``synapse.refine`` — types, Kalman tracker, and tiled executor."""

from __future__ import annotations

import numpy as np
import pytest

from synapse.refine.roi_tracker import KalmanBBoxTracker, compute_affine_align
from synapse.refine.tiled_executor import TiledBypassExecutor
from synapse.refine.types import (
    AnatomicalRegion,
    FusionMethod,
    RefinementROI,
)


# ---------------------------------------------------------------------------
# KalmanBBoxTracker
# ---------------------------------------------------------------------------


class TestKalmanReducesJitter:
    """Generate 50 noisy bbox measurements around a fixed centre and verify
    that the Kalman filter output variance is ≥ 70 % lower than input."""

    def test_kalman_reduces_jitter(self) -> None:
        rng = np.random.default_rng(42)
        n_steps = 50
        true_bbox = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)
        noise_std = 8.0  # pixels

        raw_bboxes = np.tile(true_bbox, (n_steps, 1)) + rng.normal(
            0, noise_std, size=(n_steps, 4)
        ).astype(np.float32)

        tracker = KalmanBBoxTracker(process_noise=1e-4, measurement_noise=1.0)
        filtered: list[np.ndarray] = []

        for bbox in raw_bboxes:
            tracker.predict()
            filtered.append(tracker.update(bbox))

        raw_var = np.var(raw_bboxes, axis=0).mean()
        filt_var = np.var(np.array(filtered), axis=0).mean()

        reduction = 1.0 - filt_var / raw_var
        assert reduction >= 0.70, (
            f"Kalman filter reduced variance by only {reduction * 100:.1f}% "
            f"(need ≥ 70%)"
        )


# ---------------------------------------------------------------------------
# Affine round-trip
# ---------------------------------------------------------------------------


class TestAffineRoundtripPrecision:
    """Random bboxes → forward/inverse affine → round-trip corner points.
    Max error must be < 1e-4 px."""

    def test_affine_roundtrip_precision(self) -> None:
        rng = np.random.default_rng(123)

        for _ in range(20):
            x1, y1 = rng.uniform(0, 500, size=2)
            w, h = rng.uniform(20, 300, size=2)
            bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float64)

            M_fwd, M_inv = compute_affine_align(bbox, target_size=512)

            # Corners of the bbox
            corners = np.array(
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ],
                dtype=np.float64,
            )

            for pt in corners:
                src = np.append(pt, 1.0)  # homogeneous
                dst = M_fwd @ src
                reconstructed = M_inv @ np.append(dst, 1.0)
                error = np.max(np.abs(reconstructed - pt))
                assert error < 1e-4, (
                    f"Round-trip error {error} exceeds 1e-4 for bbox={bbox}"
                )


# ---------------------------------------------------------------------------
# TiledBypassExecutor
# ---------------------------------------------------------------------------


class TestTiledExecutorBypassCount:
    """10 ROIs, 6 healthy (confidence > 0.8) → bypass 6, defective 4."""

    def test_tiled_executor_bypass_count(self) -> None:
        dummy_crop = np.zeros((64, 64, 3), dtype=np.float32)
        dummy_bbox = np.array([0, 0, 64, 64], dtype=np.float32)

        rois: list[RefinementROI] = []
        # 6 healthy (confidence > 0.8)
        for i in range(6):
            rois.append(
                RefinementROI(
                    bbox_xyxy=dummy_bbox.copy(),
                    crop_acescg=dummy_crop.copy(),
                    region=AnatomicalRegion.HAND,
                    confidence=0.85 + i * 0.01,
                    frame_index=0,
                )
            )
        # 4 defective (confidence ≤ 0.8)
        for i in range(4):
            rois.append(
                RefinementROI(
                    bbox_xyxy=dummy_bbox.copy(),
                    crop_acescg=dummy_crop.copy(),
                    region=AnatomicalRegion.FACE,
                    confidence=0.3 + i * 0.1,
                    frame_index=0,
                )
            )

        frames = np.zeros((1, 256, 256, 3), dtype=np.float32)
        executor = TiledBypassExecutor(tile_size=512, health_threshold=0.8)
        defective, bypass_count = executor.extract_and_filter(frames, rois)

        assert bypass_count == 6, f"Expected 6 bypassed, got {bypass_count}"
        assert len(defective) == 4, f"Expected 4 defective, got {len(defective)}"


# ---------------------------------------------------------------------------
# RefinementROI display conversion
# ---------------------------------------------------------------------------


class TestROIToDisplaySRGB:
    """Known ACEScg values → sRGB uint8 in expected range."""

    def test_roi_to_display_srgb(self) -> None:
        # Mid-grey in linear ≈ 0.2 → sRGB ≈ 0.484 → uint8 ≈ 123
        linear_val = 0.2
        crop = np.full((8, 8, 3), linear_val, dtype=np.float32)
        roi = RefinementROI(
            bbox_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
            crop_acescg=crop,
            region=AnatomicalRegion.FACE,
            confidence=0.9,
        )
        srgb = roi.to_display_srgb_u8()

        assert srgb.dtype == np.uint8
        assert srgb.shape == (8, 8, 3)

        # Verify the sRGB transfer function:
        # 0.2 > 0.0031308 → 1.055 * 0.2^(1/2.4) - 0.055
        expected_f = 1.055 * (0.2 ** (1.0 / 2.4)) - 0.055
        expected_u8 = int(expected_f * 255.0)
        # Allow ±1 for rounding
        assert abs(int(srgb[0, 0, 0]) - expected_u8) <= 1, (
            f"Expected ~{expected_u8}, got {srgb[0, 0, 0]}"
        )

    def test_roi_srgb_clamps_oob(self) -> None:
        """Values outside [0, 1] should be clamped before conversion."""
        crop = np.array([[[1.5, -0.1, 0.5]]], dtype=np.float32)
        roi = RefinementROI(
            bbox_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
            crop_acescg=crop,
            region=AnatomicalRegion.HAIR,
        )
        srgb = roi.to_display_srgb_u8()
        assert srgb[0, 0, 0] >= 254  # clamped 1.5 → 1.0 → ~255
        assert srgb[0, 0, 1] == 0  # clamped -0.1 → 0.0 → 0


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestFusionMethodEnumCount:
    """Assert FusionMethod has exactly 5 members."""

    def test_fusion_method_enum_count(self) -> None:
        assert len(FusionMethod) == 5, (
            f"Expected 5 FusionMethod members, got {len(FusionMethod)}"
        )
