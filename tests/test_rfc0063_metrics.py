"""Unit tests for RFC-0063 / Track 1 & Track 2 Metrics Specifications (PR-67).

Validates:
1. Temporal warp error L1 definition vs legacy RMSE option.
2. FVD video feature bridge with custom and default feature extractors.
3. 3D geometry metrics: f_score_5mm, watertight_pct, and chamfer_mm reduction conventions.
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.metrics.roundtrip import (
    chamfer_mm,
    f_score_1mm,
    f_score_5mm,
    watertight_pct,
)
from synapse.metrics.temporal import (
    compute_fvd_from_videos,
    temporal_warp_error,
)


class TestTemporalWarpErrorL1:
    """Tests for L1 vs RMSE temporal warp error per RFC-0063 §4.4."""

    def test_static_frames_zero_for_both(self) -> None:
        frames = np.ones((4, 16, 16, 3), dtype=np.float32) * 0.4
        mvs = np.zeros((3, 16, 16, 2), dtype=np.float32)

        err_l1 = temporal_warp_error(frames, mvs, metric="l1")
        err_rmse = temporal_warp_error(frames, mvs, metric="rmse")
        assert err_l1 == pytest.approx(0.0, abs=1e-6)
        assert err_rmse == pytest.approx(0.0, abs=1e-6)

    def test_l1_vs_rmse_mathematical_distinction(self) -> None:
        """With non-uniform delta, RMSE > L1 by Jensen's inequality."""
        _T, H, W = 2, 2, 2
        f0 = np.zeros((H, W, 3), dtype=np.float32)
        # Create non-uniform differences: half pixels differ by 0.2, half by 0.8
        f1 = np.zeros((H, W, 3), dtype=np.float32)
        f1[0, 0] = 0.2
        f1[0, 1] = 0.8
        f1[1, 0] = 0.2
        f1[1, 1] = 0.8

        frames = np.stack([f0, f1], axis=0)  # [2, 2, 2, 3]
        mvs = np.zeros((1, 2, 2, 2), dtype=np.float32)

        err_l1 = temporal_warp_error(frames, mvs, metric="l1")
        err_rmse = temporal_warp_error(frames, mvs, metric="rmse")

        # Mean absolute error = (0.2 + 0.8 + 0.2 + 0.8) / 4 = 0.5
        assert err_l1 == pytest.approx(0.5, abs=1e-5)
        # Root mean square error = sqrt((0.04 + 0.64 + 0.04 + 0.64) / 4) = sqrt(0.34) ~= 0.5831
        expected_rmse = float(np.sqrt(0.34))
        assert err_rmse == pytest.approx(expected_rmse, abs=1e-4)
        assert err_rmse > err_l1

    def test_invalid_metric_raises(self) -> None:
        frames = np.ones((2, 4, 4, 3), dtype=np.float32)
        mvs = np.zeros((1, 4, 4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported warp error metric"):
            temporal_warp_error(frames, mvs, metric="invalid_metric")  # type: ignore[arg-type]


class TestFvdVideoBridge:
    """Tests for compute_fvd_from_videos feature bridge."""

    def test_custom_extractor_identical_distributions(self) -> None:
        def mock_extractor(vids: list[np.ndarray]) -> np.ndarray:
            # Output deterministic 4-dimensional embeddings per video
            return np.array([
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ], dtype=np.float64)

        dummy_videos = [np.zeros((5, 16, 16, 3), dtype=np.float32) for _ in range(3)]
        fvd = compute_fvd_from_videos(dummy_videos, dummy_videos, feature_extractor=mock_extractor)
        assert fvd == pytest.approx(0.0, abs=1e-5)

    def test_custom_extractor_distinct_distributions(self) -> None:
        def mock_extractor_real(vids: list[np.ndarray]) -> np.ndarray:
            return np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)

        def mock_extractor_gen(vids: list[np.ndarray]) -> np.ndarray:
            return np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float64)

        dummy_real = [np.zeros((3, 8, 8, 3)) for _ in range(2)]
        dummy_gen = [np.ones((3, 8, 8, 3)) for _ in range(2)]

        # Combined mock extractor checking video content
        def extractor(vids: list[np.ndarray]) -> np.ndarray:
            if np.mean(vids[0]) == 0.0:
                return mock_extractor_real(vids)
            return mock_extractor_gen(vids)

        fvd = compute_fvd_from_videos(dummy_real, dummy_gen, feature_extractor=extractor)
        assert fvd > 10.0  # type: ignore[operator]

    def test_insufficient_samples_raises(self) -> None:
        def mock_extractor(vids: list[np.ndarray]) -> np.ndarray:
            return np.array([[1.0, 2.0]], dtype=np.float64)

        dummy_videos = [np.zeros((5, 8, 8, 3))]
        with pytest.raises(ValueError, match="at least 2 video samples"):
            compute_fvd_from_videos(dummy_videos, dummy_videos, feature_extractor=mock_extractor)


class TestRoundtripMetrics3D:
    """Tests for 3D geometric metrics (f_score_5mm, watertight_pct, chamfer_mm reduction)."""

    def test_chamfer_mm_reduction_mean_vs_sum(self) -> None:
        # Two point sets separated along X by 0.01 m (10 mm)
        pts_a = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        pts_b = np.array([[0.01, 0.0, 0.0], [0.01, 1.0, 0.0]], dtype=np.float64)

        cd_mean = chamfer_mm(pts_a, pts_b, reduction="mean")
        cd_sum = chamfer_mm(pts_a, pts_b, reduction="sum")

        assert cd_mean == pytest.approx(10.0, abs=1e-4)
        assert cd_sum == pytest.approx(20.0, abs=1e-4)
        assert cd_sum == pytest.approx(2.0 * cd_mean, abs=1e-6)

    def test_chamfer_mm_invalid_reduction_raises(self) -> None:
        pts = np.zeros((2, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="Unsupported reduction"):
            chamfer_mm(pts, pts, reduction="max")  # type: ignore[arg-type]

    def test_f_score_5mm_calibration(self) -> None:
        # Ground truth point cloud and reconstructed point cloud
        # Point 1 distance = 2 mm (<= 5mm -> inlier for both 1mm and 5mm? No, 2mm > 1mm, but <= 5mm)
        pts_gt = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        pts_rec = np.array([[0.003, 0.0, 0.0]], dtype=np.float64)  # 3 mm distance

        f_1mm = f_score_1mm(pts_gt, pts_rec)
        f_5mm = f_score_5mm(pts_gt, pts_rec)

        assert f_1mm == 0.0  # 3 mm > 1 mm threshold
        assert f_5mm == 1.0  # 3 mm <= 5 mm threshold

    def test_watertight_pct(self) -> None:
        # 1. Closed watertight cube (8 vertices, 12 faces)
        cube_v = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float64)
        cube_f = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 6, 2], [3, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int64)

        # 2. Open non-watertight single quad (4 vertices, 2 faces with boundary)
        open_v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype=np.float64)
        open_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        # Batch with 1 watertight cube -> 100%
        assert watertight_pct([(cube_v, cube_f)]) == 100.0

        # Batch with 1 open mesh -> 0%
        assert watertight_pct([(open_v, open_f)]) == 0.0

        # Mixed batch of 2 meshes -> 50%
        assert watertight_pct([(cube_v, cube_f), (open_v, open_f)]) == 50.0

        # Empty collection -> 0.0%
        assert watertight_pct([]) == 0.0
