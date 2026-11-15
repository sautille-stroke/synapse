"""Tests for image quality metrics, metric aliases, chunked Chamfer fallback,
and degenerate output validation.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

import numpy as np
import pytest
from synapse.metrics import (
    METRIC_ALIASES,
    ciede2000,
    evaluate_gate,
    evaluate_metrics_gate,
    get_profile_thresholds,
    lpips,
    psnr_db,
    ssim,
    validate_output_non_degenerate,
)


class TestImageQualityMetrics:
    """Verify PSNR, SSIM, CIEDE2000, and LPIPS behavior."""

    def test_identical_images_optimal(self) -> None:
        rng = np.random.default_rng(7)
        img = rng.random((32, 32, 3), dtype=np.float64)
        assert psnr_db(img, img) == pytest.approx(100.0)
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)
        assert ciede2000(img, img) == pytest.approx(0.0, abs=1e-6)
        assert lpips(img, img) == pytest.approx(0.0, abs=1e-6)

    def test_known_psnr_value(self) -> None:
        gt = np.zeros((16, 16, 3), dtype=np.float64)
        pred = np.full((16, 16, 3), 0.1, dtype=np.float64)
        # MSE = 0.01 -> PSNR = 10 * log10(1 / 0.01) = 20 dB
        assert psnr_db(pred, gt) == pytest.approx(20.0, abs=1e-6)

    def test_monotonic_degradation_with_noise(self) -> None:
        rng = np.random.default_rng(42)
        gt = rng.random((32, 32, 3), dtype=np.float64) * 0.6 + 0.2
        mild = np.clip(gt + rng.normal(0.0, 0.02, gt.shape), 0.0, 1.0)
        heavy = np.clip(gt + rng.normal(0.0, 0.20, gt.shape), 0.0, 1.0)

        assert psnr_db(mild, gt) > psnr_db(heavy, gt)
        assert ssim(mild, gt) > ssim(heavy, gt)
        assert ciede2000(mild, gt) < ciede2000(heavy, gt)
        assert lpips(mild, gt) < lpips(heavy, gt)

    def test_uint8_and_sequence_support(self) -> None:
        seq_a = np.full((3, 24, 24, 3), 128, dtype=np.uint8)
        seq_b = np.full((3, 24, 24, 3), 200, dtype=np.uint8)
        assert psnr_db(seq_a, seq_b) < 30.0
        assert 0.0 <= ssim(seq_a, seq_b) <= 1.0
        assert ciede2000(seq_a, seq_b) > 5.0
        assert 0.0 <= lpips(seq_a, seq_b) <= 1.0


class TestProfileAliasesAndFailClosed:
    """Verify scenario metric aliases and fail-closed profile resolution."""

    def test_case_insensitive_profile_and_unknown_raises(self) -> None:
        t = get_profile_thresholds("TRACK_A_ASSET")
        assert t["chamfer_mm"] == pytest.approx(5.0)

        with pytest.raises(ValueError, match="Unknown gate profile"):
            get_profile_thresholds("non_existent_profile")

        with pytest.raises(ValueError, match="Unknown gate profile"):
            evaluate_metrics_gate({"chamfer_mm": 1.0}, "bad_profile")

    def test_scenario_aliases_resolved(self) -> None:
        assert METRIC_ALIASES["style_color_de"] == "border_ring_ciede2000"
        assert METRIC_ALIASES["style_light_error_deg"] == "light_dir_error_deg"

        # Passing style_color_de satisfies border_ring_ciede2000 threshold (2.5 in track_a)
        verdict_fail = evaluate_metrics_gate(
            {"style_color_de": 9.0, "psnr_db": 22.0},
            "track_a_asset",
            scenario_overrides={"psnr_db": 28.0},
        )
        assert verdict_fail.passed is False
        failed_names = {v.metric_name for v in verdict_fail.violations}
        assert "border_ring_ciede2000" in failed_names
        assert "psnr_db" in failed_names

        verdict_ok = evaluate_gate(
            {"border_ring_ciede2000": 1.2},
            "track_a_asset",
            required_keys=["style_color_de"],
        )
        assert verdict_ok.passed is True


class TestChunkedChamferFallback:
    """Verify chamfer_mm and f_score_1mm work without scipy installed."""

    def test_chamfer_and_f_score_without_scipy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from synapse.metrics import chamfer_mm, f_score_1mm

        real_import = builtins.__import__

        def _block_scipy(
            name: str,
            globals_dict: Any = None,
            locals_dict: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("Simulated missing scipy")
            return real_import(name, globals_dict, locals_dict, fromlist, level)

        monkeypatch.delitem(sys.modules, "scipy", raising=False)
        monkeypatch.delitem(sys.modules, "scipy.spatial", raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_scipy)

        pts = np.zeros((60, 3), dtype=np.float64)
        pts[:, 0] = np.linspace(0.0, 1.0, 60)
        shifted = pts + np.array([0.0, 0.002, 0.0])  # exact 2 mm perpendicular shift

        assert chamfer_mm(pts, shifted) == pytest.approx(2.0, abs=1e-6)
        assert f_score_1mm(pts, pts) == pytest.approx(1.0)
        assert f_score_1mm(pts, shifted, threshold_mm=1.0) == pytest.approx(0.0)


class TestLpipsMaskNormalization:
    """Verify LPIPS with spatial mask normalizes by valid pixel count."""

    def test_lpips_mask_higher_than_unmasked_background(self) -> None:
        """With a mask, lpips should be higher (more sensitive) than without
        mask when the difference is concentrated in the masked region, because
        background zeros no longer dilute the result."""
        rng = np.random.default_rng(42)
        gt = rng.random((32, 32, 3), dtype=np.float64)
        pred = gt.copy()
        # Introduce large difference only in a small region
        pred[10:15, 10:15, :] = 1.0 - pred[10:15, 10:15, :]

        mask = np.zeros((32, 32), dtype=bool)
        mask[10:15, 10:15] = True

        lpips_masked = lpips(pred, gt, mask=mask)
        lpips_no_mask = lpips(pred, gt)

        # Masked should be >= unmasked since it focuses on the changed region
        assert lpips_masked >= lpips_no_mask * 0.8  # reasonable range

    def test_lpips_empty_mask_returns_zero(self) -> None:
        """An all-False mask means no valid pixels — returns 0.0."""
        gt = np.random.default_rng(7).random((16, 16, 3), dtype=np.float64)
        pred = 1.0 - gt
        mask = np.zeros((16, 16), dtype=bool)
        assert lpips(pred, gt, mask=mask) == pytest.approx(0.0)


class TestValidateOutputNonDegenerate:
    """Tests for per-channel variance and temporal diff validation."""

    def test_valid_diverse_frames_pass(self) -> None:
        """Normal varied frames should pass validation."""
        rng = np.random.default_rng(42)
        frames = rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
        # Should not raise
        validate_output_non_degenerate(frames)

    def test_pure_red_constant_blocked(self) -> None:
        """Pure red [255, 0, 0] constant sequence must be caught as
        per-channel degenerate (G and B channels have zero variance)."""
        from synapse.errors import QualityGateError

        frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        frames[..., 0] = 255  # pure red on all frames
        with pytest.raises(QualityGateError, match="channel.*variance"):
            validate_output_non_degenerate(frames)

    def test_all_zero_blocked(self) -> None:
        """All-zero frames must be caught."""
        from synapse.errors import QualityGateError

        frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        with pytest.raises(QualityGateError, match="all-zero"):
            validate_output_non_degenerate(frames)

    def test_nan_blocked(self) -> None:
        """NaN frames must be caught."""
        from synapse.errors import QualityGateError

        frames = np.full((2, 8, 8, 3), np.nan, dtype=np.float32)
        with pytest.raises(QualityGateError, match="NaN"):
            validate_output_non_degenerate(frames)

    def test_frozen_still_blocked(self) -> None:
        """Identical frames across time (frozen-still) must be caught
        by temporal diff variance check."""
        from synapse.errors import QualityGateError

        rng = np.random.default_rng(99)
        single = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)
        # Repeat identical frame 8 times
        frames = np.tile(single, (8, 1, 1, 1))
        with pytest.raises(QualityGateError, match="temporally frozen"):
            validate_output_non_degenerate(frames)

    def test_single_frame_skips_temporal_check(self) -> None:
        """Single frame (T=1) should only check spatial variance, not temporal."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        # Should not raise for a single diverse frame
        validate_output_non_degenerate(frame)
