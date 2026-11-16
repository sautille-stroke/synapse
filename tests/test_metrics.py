"""Tests for synapse.metrics — 4-pillar 16-metric evaluation suite."""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pillar 0: Profiles & Gate Evaluation
# ---------------------------------------------------------------------------


class TestProfiles:
    """Tests for gate profiles and the evaluate_metrics_gate function."""

    def test_profiles_exist(self) -> None:
        from synapse.metrics.profiles import (
            RSI_PROMOTION,
            TRACK_A_ASSET,
            TRACK_B_VIDEO,
        )

        assert len(TRACK_B_VIDEO) == 16
        assert len(TRACK_A_ASSET) >= 16
        assert len(RSI_PROMOTION) >= 16
        assert "non_manifold_edges" in TRACK_A_ASSET
        assert "f_score_5mm" in TRACK_A_ASSET
        assert "watertight_pct" in TRACK_A_ASSET

    def test_all_profile_keys_registered_in_metric_directions(self) -> None:
        """Every metric key in every tier MUST be explicitly registered in
        _METRIC_DIRECTIONS — no silent fallback to _LOWER allowed."""
        from synapse.metrics.profiles import (
            _METRIC_DIRECTIONS,
            RSI_PROMOTION,
            RSI_PROMOTION_TARGET,
            TRACK_A_ASSET,
            TRACK_B_VIDEO,
        )

        all_profiles = {
            "track_b_video": TRACK_B_VIDEO,
            "track_a_asset": TRACK_A_ASSET,
            "rsi_promotion": RSI_PROMOTION,
            "rsi_promotion_target": RSI_PROMOTION_TARGET,
        }
        for profile_name, profile in all_profiles.items():
            for key in profile:
                assert key in _METRIC_DIRECTIONS, (
                    f"Metric key {key!r} in profile {profile_name!r} is NOT "
                    f"registered in _METRIC_DIRECTIONS. All keys must be "
                    f"explicitly registered to prevent silent direction fallback."
                )

    def test_f_score_5mm_and_watertight_directions(self) -> None:
        """f_score_5mm and watertight_pct must be higher_is_better."""
        from synapse.metrics.profiles import _METRIC_DIRECTIONS

        assert _METRIC_DIRECTIONS["f_score_5mm"] == "higher_is_better"
        assert _METRIC_DIRECTIONS["watertight_pct"] == "higher_is_better"

    def test_watertight_pct_100_passes_gate(self) -> None:
        """watertight_pct=100.0 with higher_is_better must pass gate."""
        from synapse.metrics import evaluate_metrics_gate

        verdict = evaluate_metrics_gate(
            {"watertight_pct": 100.0},
            "track_b_video",
            scenario_overrides={"watertight_pct": 50.0},
        )
        assert verdict.passed is True

    def test_watertight_pct_0_fails_gate(self) -> None:
        """watertight_pct=0.0 with higher_is_better threshold must fail."""
        from synapse.metrics import evaluate_metrics_gate

        verdict = evaluate_metrics_gate(
            {"watertight_pct": 0.0},
            "track_b_video",
            scenario_overrides={"watertight_pct": 50.0},
        )
        assert verdict.passed is False
        assert any(
            v.metric_name == "watertight_pct" for v in verdict.violations
        )

    def test_profile_strictness_ordering(self) -> None:
        """RSI thresholds must be equal or stricter than Track-A."""
        from synapse.metrics.profiles import (
            _METRIC_DIRECTIONS,
            RSI_PROMOTION,
            TRACK_A_ASSET,
            TRACK_B_VIDEO,
        )

        for name in TRACK_B_VIDEO:
            direction = _METRIC_DIRECTIONS.get(name, "lower_is_better")
            b = TRACK_B_VIDEO[name]
            a = TRACK_A_ASSET[name]
            r = RSI_PROMOTION[name]
            if direction == "lower_is_better":
                assert b >= a >= r, (
                    f"{name}: track_b={b} >= track_a={a} >= rsi={r}"
                )
            else:
                assert b <= a <= r, (
                    f"{name}: track_b={b} <= track_a={a} <= rsi={r}"
                )

    def test_gate_all_pass(self) -> None:
        from synapse.metrics import evaluate_metrics_gate

        # All metrics far within track_b_video thresholds
        measured = {
            "temporal_warp_error": 0.001,
            "flicker_score": 0.001,
            "fvd": 10.0,
            "depth_rmse_m": 0.01,
            "scale_invariant_depth_error": 0.01,
            "normal_angular_error_deg": 1.0,
            "flow_endpoint_error_px": 0.5,
            "silhouette_iou": 0.99,
            "chamfer_mm": 0.5,
            "f_score_1mm": 0.95,
            "volume_breathing_pct": 0.5,
            "foot_sliding_cm": 0.1,
            "bone_length_drift_pct": 0.1,
            "albedo_delta_e00": 0.5,
            "border_ring_ciede2000": 0.3,
            "light_dir_error_deg": 1.0,
        }
        verdict = evaluate_metrics_gate(measured, "track_b_video")
        assert verdict.passed is True
        assert verdict.profile == "track_b_video"
        assert len(verdict.violations) == 0

    def test_gate_some_fail(self) -> None:
        from synapse.metrics import evaluate_metrics_gate

        measured = {
            "temporal_warp_error": 999.0,  # far above threshold
            "silhouette_iou": 0.01,  # far below threshold
        }
        verdict = evaluate_metrics_gate(measured, "rsi_promotion")
        assert verdict.passed is False
        assert len(verdict.violations) == 2

        names = {v.metric_name for v in verdict.violations}
        assert "temporal_warp_error" in names
        assert "silhouette_iou" in names

    def test_gate_scenario_overrides(self) -> None:
        from synapse.metrics import evaluate_metrics_gate

        measured = {"temporal_warp_error": 0.010}
        # Default track_b_video threshold is 0.020 → should pass
        v1 = evaluate_metrics_gate(measured, "track_b_video")
        assert v1.passed is True

        # Override to tighter threshold → should fail
        v2 = evaluate_metrics_gate(
            measured, "track_b_video",
            scenario_overrides={"temporal_warp_error": 0.005},
        )
        assert v2.passed is False

    def test_gate_missing_metrics_ignored(self) -> None:
        from synapse.metrics import evaluate_metrics_gate

        # Only supply one metric; others should be silently skipped
        verdict = evaluate_metrics_gate(
            {"depth_rmse_m": 0.01}, "track_a_asset",
        )
        assert verdict.passed is True


# ---------------------------------------------------------------------------
# Pillar 1: Temporal Consistency
# ---------------------------------------------------------------------------


class TestTemporal:
    """Tests for temporal consistency metrics."""

    def test_static_frames_warp_error_near_zero(self) -> None:
        from synapse.metrics import temporal_warp_error

        T, H, W = 5, 8, 8
        frames = np.ones((T, H, W, 3), dtype=np.float32) * 0.5
        mvs = np.zeros((T - 1, H, W, 2), dtype=np.float32)
        err = temporal_warp_error(frames, mvs)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_warp_error_with_occlusion(self) -> None:
        from synapse.metrics import temporal_warp_error

        T, H, W = 3, 4, 4
        frames = np.random.default_rng(42).random(
            (T, H, W, 3), dtype=np.float32,
        )
        mvs = np.zeros((T - 1, H, W, 2), dtype=np.float32)
        # Fully occluded → error should be 0
        occ = np.ones((T - 1, H, W), dtype=np.float32)
        err = temporal_warp_error(frames, mvs, occlusion_masks=occ)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_flicker_static_zero(self) -> None:
        from synapse.metrics import flicker_score

        frames = np.ones((10, 4, 4, 3), dtype=np.float32) * 0.3
        assert flicker_score(frames) == pytest.approx(0.0, abs=1e-8)

    def test_flicker_nonzero_for_alternating(self) -> None:
        from synapse.metrics import flicker_score

        T, H, W = 10, 4, 4
        frames = np.zeros((T, H, W, 3), dtype=np.float32)
        for t in range(T):
            frames[t] = 0.0 if t % 2 == 0 else 1.0
        score = flicker_score(frames)
        assert score > 0.0

    def test_fvd_identical_near_zero(self) -> None:
        from synapse.metrics import fvd_feature_distance

        rng = np.random.default_rng(123)
        feats = rng.standard_normal((50, 16)).astype(np.float64)
        fvd = fvd_feature_distance(feats, feats)
        assert fvd == pytest.approx(0.0, abs=1e-3)

    def test_fvd_different_positive(self) -> None:
        from synapse.metrics import fvd_feature_distance

        rng = np.random.default_rng(7)
        real = rng.standard_normal((100, 8))
        gen = rng.standard_normal((100, 8)) + 5.0
        fvd = fvd_feature_distance(real, gen)
        assert fvd > 0.0

    def test_fvd_singular_covariance_no_warning(self) -> None:
        """Singular covariance (constant features) must NOT trigger
        scipy.linalg.LinAlgWarning thanks to Tikhonov regularization."""
        from synapse.metrics import fvd_feature_distance

        # All rows identical -> covariance is zero matrix -> singular
        feats_real = np.ones((10, 4), dtype=np.float64)
        feats_gen = np.ones((10, 4), dtype=np.float64) * 2.0
        # With -W error::scipy.linalg.LinAlgWarning, this would raise
        # if regularization is missing.
        fvd = fvd_feature_distance(feats_real, feats_gen)
        assert fvd >= 0.0
        assert np.isfinite(fvd)

    def test_fvd_near_singular_covariance_stable(self) -> None:
        """Near-singular covariance (low-rank features) remains stable."""
        from synapse.metrics import fvd_feature_distance

        rng = np.random.default_rng(99)
        # 50 samples of 16-D but only rank 2
        basis = rng.standard_normal((2, 16))
        coeffs_real = rng.standard_normal((50, 2))
        coeffs_gen = rng.standard_normal((50, 2)) + 3.0
        feats_real = coeffs_real @ basis
        feats_gen = coeffs_gen @ basis
        fvd = fvd_feature_distance(feats_real, feats_gen)
        assert fvd > 0.0
        assert np.isfinite(fvd)

    def test_compute_fvd_from_videos_proxy(self) -> None:
        """compute_fvd_from_videos with allow_statistical_proxy returns FVDResult."""
        from synapse.metrics import compute_fvd_from_videos

        rng = np.random.default_rng(42)
        real = rng.random((3, 4, 8, 8, 3)).astype(np.float32)
        gen = rng.random((3, 4, 8, 8, 3)).astype(np.float32) + 0.5

        result = compute_fvd_from_videos(
            real, gen, allow_statistical_proxy=True,
        )
        assert result.is_proxy_metric is True
        assert result.value >= 0.0
        assert np.isfinite(result.value)

    def test_compute_fvd_from_videos_no_proxy_raises(self) -> None:
        """compute_fvd_from_videos with require_pretrained=True raises
        BackendUnavailableError when torch is not available."""
        from synapse.metrics import compute_fvd_from_videos

        rng = np.random.default_rng(42)
        real = rng.random((3, 4, 8, 8, 3)).astype(np.float32)
        gen = rng.random((3, 4, 8, 8, 3)).astype(np.float32)

        # This test may pass or raise depending on torch availability,
        # but at minimum it should not crash with unhandled exception.
        try:
            result = compute_fvd_from_videos(
                real, gen, require_pretrained=True,
            )
            # If torch is available, we get a real result
            assert result.is_proxy_metric is False
        except Exception as exc:
            from synapse.errors import BackendUnavailableError
            assert isinstance(exc, BackendUnavailableError)
            assert exc.exit_code == 30


# ---------------------------------------------------------------------------
# Pillar 2: FGPP Fidelity
# ---------------------------------------------------------------------------


class TestFGPPFidelity:
    """Tests for FGPP channel fidelity metrics."""

    def test_depth_rmse_gt_vs_gt(self) -> None:
        from synapse.metrics import depth_rmse_m

        d = np.random.default_rng(1).random((32, 32)).astype(np.float64)
        assert depth_rmse_m(d, d) == pytest.approx(0.0, abs=1e-10)

    def test_depth_rmse_known_offset(self) -> None:
        from synapse.metrics import depth_rmse_m

        gt = np.ones((10, 10), dtype=np.float64) * 5.0
        pred = gt + 0.1  # constant offset of 0.1 m
        assert depth_rmse_m(pred, gt) == pytest.approx(0.1, abs=1e-10)

    def test_depth_rmse_with_mask(self) -> None:
        from synapse.metrics import depth_rmse_m

        gt = np.ones((4, 4), dtype=np.float64)
        pred = np.ones((4, 4), dtype=np.float64) * 2.0
        mask = np.zeros((4, 4), dtype=bool)
        # No valid pixels → 0
        assert depth_rmse_m(pred, gt, valid_mask=mask) == 0.0

    def test_scale_invariant_depth_gt_vs_gt(self) -> None:
        from synapse.metrics import scale_invariant_depth_error

        d = np.ones((8, 8), dtype=np.float64) * 3.0
        assert scale_invariant_depth_error(d, d) == pytest.approx(
            0.0, abs=1e-10,
        )

    def test_normal_angular_error_gt_vs_gt(self) -> None:
        from synapse.metrics import normal_angular_error_deg

        n = np.zeros((8, 8, 3), dtype=np.float64)
        n[..., 2] = 1.0  # all pointing +Z
        assert normal_angular_error_deg(n, n) == pytest.approx(
            0.0, abs=1e-10,
        )

    def test_normal_angular_error_known_rotation(self) -> None:
        from synapse.metrics import normal_angular_error_deg

        gt = np.zeros((1, 1, 3), dtype=np.float64)
        gt[..., 2] = 1.0  # [0, 0, 1]
        # Rotate 10 deg around X axis
        angle = np.radians(10.0)
        pred = np.zeros((1, 1, 3), dtype=np.float64)
        pred[0, 0] = [0.0, np.sin(angle), np.cos(angle)]
        err = normal_angular_error_deg(pred, gt)
        assert err == pytest.approx(10.0, abs=0.01)

    def test_flow_epe_gt_vs_gt(self) -> None:
        from synapse.metrics import flow_endpoint_error_px

        f = np.random.default_rng(2).random((16, 16, 2))
        assert flow_endpoint_error_px(f, f) == pytest.approx(
            0.0, abs=1e-10,
        )

    def test_flow_epe_known_offset(self) -> None:
        from synapse.metrics import flow_endpoint_error_px

        gt = np.zeros((4, 4, 2), dtype=np.float64)
        pred = np.ones((4, 4, 2), dtype=np.float64)  # offset (1,1)
        epe = flow_endpoint_error_px(pred, gt)
        assert epe == pytest.approx(math.sqrt(2.0), abs=1e-10)

    def test_silhouette_iou_identical(self) -> None:
        from synapse.metrics import silhouette_iou

        m = np.ones((8, 8), dtype=bool)
        assert silhouette_iou(m, m) == pytest.approx(1.0)

    def test_silhouette_iou_disjoint(self) -> None:
        from synapse.metrics import silhouette_iou

        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[:2] = True
        b[2:] = True
        assert silhouette_iou(a, b) == pytest.approx(0.0)

    def test_silhouette_iou_both_empty(self) -> None:
        from synapse.metrics import silhouette_iou

        a = np.zeros((4, 4), dtype=bool)
        assert silhouette_iou(a, a) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pillar 3: Roundtrip Geometry & Appearance
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Tests for roundtrip geometry and appearance metrics."""

    def test_chamfer_identical(self) -> None:
        from synapse.metrics import chamfer_mm

        pts = np.random.default_rng(3).random((50, 3))
        assert chamfer_mm(pts, pts) == pytest.approx(0.0, abs=1e-8)

    def test_chamfer_known_translation(self) -> None:
        from synapse.metrics import chamfer_mm

        pts = np.zeros((100, 3), dtype=np.float64)
        pts[:, 0] = np.linspace(0, 1, 100)
        shifted = pts + np.array([0.003, 0.0, 0.0])  # 3 mm shift
        cd = chamfer_mm(pts, shifted)
        assert cd == pytest.approx(3.0, abs=0.1)

    def test_f_score_identical(self) -> None:
        from synapse.metrics import f_score_1mm

        pts = np.random.default_rng(4).random((80, 3))
        assert f_score_1mm(pts, pts) == pytest.approx(1.0)

    def test_f_score_far_apart(self) -> None:
        from synapse.metrics import f_score_1mm

        a = np.zeros((10, 3))
        b = np.ones((10, 3)) * 100.0  # 100 m apart
        assert f_score_1mm(a, b) == pytest.approx(0.0)

    def test_volume_breathing_constant(self) -> None:
        from synapse.metrics import volume_breathing_pct

        # Unit tetrahedron with constant geometry
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        meshes = [(verts, faces)] * 5
        assert volume_breathing_pct(meshes) == pytest.approx(0.0, abs=1e-8)

    def test_volume_breathing_positive_on_scale(self) -> None:
        from synapse.metrics import volume_breathing_pct

        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        meshes = [
            (verts * scale, faces)
            for scale in [1.0, 1.0, 2.0, 1.0, 1.0]
        ]
        pct = volume_breathing_pct(meshes)
        assert pct > 0.0

    def test_bone_length_drift_constant(self) -> None:
        from synapse.metrics import bone_length_drift_pct

        bl = np.ones((10, 5), dtype=np.float64) * 0.5
        assert bone_length_drift_pct(bl) == pytest.approx(0.0, abs=1e-10)

    def test_bone_length_drift_positive(self) -> None:
        from synapse.metrics import bone_length_drift_pct

        bl = np.ones((10, 3), dtype=np.float64) * 0.5
        bl[5, 0] = 0.6  # 20% drift on bone 0
        drift = bone_length_drift_pct(bl)
        assert drift > 0.0

    def test_foot_sliding_no_contact(self) -> None:
        from synapse.metrics import foot_sliding_cm

        pos = np.random.default_rng(5).random((10, 2, 3))
        contact = np.zeros((10, 2), dtype=bool)
        assert foot_sliding_cm(pos, contact) == pytest.approx(0.0)

    def test_foot_sliding_moving_contact(self) -> None:
        from synapse.metrics import foot_sliding_cm

        T = 10
        pos = np.zeros((T, 1, 3), dtype=np.float64)
        for t in range(T):
            pos[t, 0, 0] = t * 0.01  # 1 cm per frame in X
        contact = np.ones((T, 1), dtype=bool)
        slide = foot_sliding_cm(pos, contact)
        assert slide > 0.0

    def test_albedo_delta_e00_identical(self) -> None:
        from synapse.metrics import albedo_delta_e00

        img = np.full((4, 4, 3), 0.5, dtype=np.float64)
        assert albedo_delta_e00(img, img) == pytest.approx(0.0, abs=1e-6)

    def test_albedo_delta_e00_different(self) -> None:
        from synapse.metrics import albedo_delta_e00

        a = np.full((4, 4, 3), 0.2, dtype=np.float64)
        b = np.full((4, 4, 3), 0.8, dtype=np.float64)
        de = albedo_delta_e00(a, b)
        assert de > 0.0


# ---------------------------------------------------------------------------
# Pillar 4: Style Consistency
# ---------------------------------------------------------------------------


class TestStyleConsistency:
    """Tests for style-consistency metrics."""

    def test_border_ring_identical(self) -> None:
        from synapse.metrics import border_ring_ciede2000

        img = np.full((32, 32, 3), 0.5, dtype=np.float64)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:22, 10:22] = True
        de = border_ring_ciede2000(img, img, mask, ring_width=4)
        assert de == pytest.approx(0.0, abs=1e-6)

    def test_border_ring_different(self) -> None:
        from synapse.metrics import border_ring_ciede2000

        comp = np.full((32, 32, 3), 0.8, dtype=np.float64)
        bg = np.full((32, 32, 3), 0.2, dtype=np.float64)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:22, 10:22] = True
        de = border_ring_ciede2000(comp, bg, mask, ring_width=4)
        assert de > 0.0

    def test_light_dir_identical(self) -> None:
        from synapse.metrics import light_dir_error_deg

        d = np.array([0.0, 0.0, 1.0])
        assert light_dir_error_deg(d, d) == pytest.approx(0.0, abs=1e-10)

    def test_light_dir_90_deg(self) -> None:
        from synapse.metrics import light_dir_error_deg

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert light_dir_error_deg(a, b) == pytest.approx(90.0, abs=0.01)

    def test_tone_curve_identical(self) -> None:
        from synapse.metrics import tone_curve_rmse

        c = np.linspace(0, 1, 256)
        assert tone_curve_rmse(c, c) == pytest.approx(0.0, abs=1e-12)

    def test_tone_curve_offset(self) -> None:
        from synapse.metrics import tone_curve_rmse

        c = np.linspace(0, 1, 256)
        rmse = tone_curve_rmse(c + 0.1, c)
        assert rmse == pytest.approx(0.1, abs=1e-6)

    def test_histogram_identical(self) -> None:
        from synapse.metrics import histogram_intersection

        h = np.array([10, 20, 30, 40], dtype=np.float64)
        assert histogram_intersection(h, h) == pytest.approx(1.0)

    def test_histogram_disjoint(self) -> None:
        from synapse.metrics import histogram_intersection

        a = np.array([1, 0, 0, 0], dtype=np.float64)
        b = np.array([0, 0, 0, 1], dtype=np.float64)
        assert histogram_intersection(a, b) == pytest.approx(0.0)

    def test_histogram_partial(self) -> None:
        from synapse.metrics import histogram_intersection

        a = np.array([10, 20, 30, 40], dtype=np.float64)
        b = np.array([5, 10, 30, 50], dtype=np.float64)
        # min: [5, 10, 30, 40] sum=85; sum(a)=100
        assert histogram_intersection(a, b) == pytest.approx(0.85, abs=1e-6)
