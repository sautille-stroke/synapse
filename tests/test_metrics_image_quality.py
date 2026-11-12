"""Tests for image quality metrics, metric aliases, and chunked Chamfer fallback."""

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
