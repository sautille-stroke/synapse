"""Tests for synapse.adapters.spatial_memory — Spatial Memory point cloud."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from synapse.adapters.spatial_memory import (
    SpatialMemory,
    SpatialMemoryConfig,
    SpatialPointCloud,
    compute_reference_consistency_score,
    depth_to_world_points,
)

# ---------------------------------------------------------------------------
# Lightweight camera stub
# ---------------------------------------------------------------------------


@dataclass
class _StubCamera:
    """Minimal camera stub matching CameraParams interface."""

    fx: float = 500.0
    fy: float = 500.0
    cx: float = 160.0
    cy: float = 120.0
    width: int = 320
    height: int = 240
    view_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64),
    )
    world_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64),
    )


@dataclass
class _StubFrame:
    """Minimal FGPP frame stub."""

    depth: np.ndarray
    camera: _StubCamera


def _make_frame(
    depth_value: float = 5.0,
    h: int = 240,
    w: int = 320,
    camera: _StubCamera | None = None,
) -> _StubFrame:
    """Create a stub frame with uniform depth."""
    cam = camera or _StubCamera(width=w, height=h, cx=w / 2, cy=h / 2)
    depth = np.full((h, w), depth_value, dtype=np.float32)
    return _StubFrame(depth=depth, camera=cam)


def _make_translated_camera(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> _StubCamera:
    """Camera translated in world space."""
    world = np.eye(4, dtype=np.float64)
    world[0, 3] = tx
    world[1, 3] = ty
    world[2, 3] = tz
    view = np.linalg.inv(world)
    return _StubCamera(
        view_matrix=view,
        world_matrix=world,
    )


# ===========================================================================
# Tests: SpatialPointCloud
# ===========================================================================


class TestSpatialPointCloud:
    """Basic point cloud data structure tests."""

    def test_empty_cloud(self) -> None:
        cloud = SpatialPointCloud()
        assert cloud.count == 0
        assert cloud.memory_bytes >= 0

    def test_count_matches_positions(self) -> None:
        pts = np.random.randn(100, 3).astype(np.float32)
        fids = np.zeros(100, dtype=np.int32)
        cloud = SpatialPointCloud(positions=pts, frame_ids=fids)
        assert cloud.count == 100

    def test_memory_includes_normals(self) -> None:
        pts = np.zeros((50, 3), dtype=np.float32)
        fids = np.zeros(50, dtype=np.int32)
        normals = np.zeros((50, 3), dtype=np.float32)
        cloud_no_normals = SpatialPointCloud(positions=pts, frame_ids=fids)
        cloud_with = SpatialPointCloud(
            positions=pts, normals=normals, frame_ids=fids,
        )
        assert cloud_with.memory_bytes > cloud_no_normals.memory_bytes


# ===========================================================================
# Tests: SpatialMemoryConfig
# ===========================================================================


class TestSpatialMemoryConfig:
    """Config defaults and construction."""

    def test_defaults(self) -> None:
        cfg = SpatialMemoryConfig()
        assert cfg.voxel_size == pytest.approx(0.05)
        assert cfg.max_points == 4_000_000
        assert cfg.subsample_stride == 4
        assert cfg.depth_min_m == pytest.approx(0.1)
        assert cfg.depth_max_m == pytest.approx(100.0)
        assert cfg.normal_estimation is True

    def test_custom_values(self) -> None:
        cfg = SpatialMemoryConfig(voxel_size=0.1, max_points=1000)
        assert cfg.voxel_size == pytest.approx(0.1)
        assert cfg.max_points == 1000


# ===========================================================================
# Tests: SpatialMemory
# ===========================================================================


class TestSpatialMemory:
    """Core spatial memory functionality."""

    def test_construction_defaults(self) -> None:
        mem = SpatialMemory()
        assert mem.num_points == 0
        assert mem.ingested_frame_count == 0

    def test_ingest_single_frame(self) -> None:
        cfg = SpatialMemoryConfig(subsample_stride=8, normal_estimation=False)
        mem = SpatialMemory(cfg)
        frame = _make_frame(depth_value=5.0, h=64, w=64)
        n_added = mem.ingest_frame(0, frame)
        assert n_added > 0
        assert mem.num_points == n_added
        assert mem.ingested_frame_count == 1

    def test_ingest_duplicate_frame_raises(self) -> None:
        mem = SpatialMemory()
        frame = _make_frame(depth_value=5.0, h=16, w=16)
        mem.ingest_frame(0, frame)
        with pytest.raises(ValueError, match="already ingested"):
            mem.ingest_frame(0, frame)

    def test_ingest_no_depth_raises(self) -> None:
        mem = SpatialMemory()

        @dataclass
        class _NoDepth:
            camera: _StubCamera = field(default_factory=_StubCamera)

        with pytest.raises(ValueError, match="depth"):
            mem.ingest_frame(0, _NoDepth())

    def test_ingest_invalid_depth_shape_raises(self) -> None:
        mem = SpatialMemory()
        frame = _StubFrame(
            depth=np.ones((2, 3, 4), dtype=np.float32),
            camera=_StubCamera(),
        )
        with pytest.raises(ValueError, match="2-D"):
            mem.ingest_frame(0, frame)

    def test_multiple_frames_accumulate(self) -> None:
        cfg = SpatialMemoryConfig(
            subsample_stride=16, voxel_size=1.0, normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        # Two frames from different camera positions → different voxels
        cam1 = _make_translated_camera(tx=0.0)
        cam2 = _make_translated_camera(tx=100.0)  # Far apart
        f1 = _make_frame(depth_value=5.0, h=32, w=32, camera=cam1)
        f2 = _make_frame(depth_value=5.0, h=32, w=32, camera=cam2)
        n1 = mem.ingest_frame(0, f1)
        n2 = mem.ingest_frame(1, f2)
        assert n1 > 0
        assert n2 > 0
        assert mem.num_points == n1 + n2
        assert mem.ingested_frame_count == 2

    def test_voxel_dedup_reduces_points(self) -> None:
        """Same camera position → overlapping voxels → fewer new points."""
        cfg = SpatialMemoryConfig(
            subsample_stride=16, voxel_size=0.5, normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        cam = _StubCamera()
        f1 = _make_frame(depth_value=5.0, h=32, w=32, camera=cam)
        f2 = _make_frame(depth_value=5.0, h=32, w=32, camera=cam)
        n1 = mem.ingest_frame(0, f1)
        n2 = mem.ingest_frame(1, f2)
        assert n1 > 0
        assert n2 == 0  # Identical camera → same voxels

    def test_eviction_respects_max_points(self) -> None:
        cfg = SpatialMemoryConfig(
            max_points=50,
            subsample_stride=4,
            voxel_size=0.001,  # Very fine → many unique voxels
            normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        for i in range(5):
            cam = _make_translated_camera(tx=float(i * 1000))
            frame = _make_frame(depth_value=5.0, h=32, w=32, camera=cam)
            mem.ingest_frame(i, frame)
        assert mem.num_points <= cfg.max_points

    def test_reset_clears_all(self) -> None:
        mem = SpatialMemory()
        frame = _make_frame(depth_value=5.0, h=16, w=16)
        mem.ingest_frame(0, frame)
        assert mem.num_points > 0
        mem.reset()
        assert mem.num_points == 0
        assert mem.ingested_frame_count == 0

    def test_stats_keys(self) -> None:
        mem = SpatialMemory()
        s = mem.stats()
        assert "num_points" in s
        assert "num_frames_ingested" in s
        assert "num_voxels" in s
        assert "memory_bytes" in s
        assert "memory_mb" in s

    def test_zero_depth_frame_adds_no_points(self) -> None:
        """Frame with all-zero depth should add 0 points (below min)."""
        cfg = SpatialMemoryConfig(
            subsample_stride=4, normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        frame = _make_frame(depth_value=0.0, h=32, w=32)
        n = mem.ingest_frame(0, frame)
        assert n == 0

    def test_deterministic(self) -> None:
        """Identical inputs → identical point counts."""
        cfg = SpatialMemoryConfig(
            subsample_stride=8, voxel_size=0.5, normal_estimation=False,
        )
        mem1 = SpatialMemory(cfg)
        mem2 = SpatialMemory(cfg)
        frame = _make_frame(depth_value=5.0, h=64, w=64)
        n1 = mem1.ingest_frame(0, frame)
        n2 = mem2.ingest_frame(0, frame)
        assert n1 == n2
        np.testing.assert_array_equal(mem1.cloud.positions, mem2.cloud.positions)

    def test_partial_invalid_depth_with_normals_does_not_crash(self) -> None:
        """Mixed valid/invalid depth with normal_estimation=True aligns arrays."""
        cfg = SpatialMemoryConfig(
            subsample_stride=4, voxel_size=0.05, normal_estimation=True,
        )
        mem = SpatialMemory(cfg)
        cam = _StubCamera(width=32, height=32, cx=16, cy=16)
        depth = np.full((32, 32), 5.0, dtype=np.float32)
        depth[:10, :] = 0.0
        depth[20:24, :] = np.nan
        n_added = mem.ingest_frame(0, _StubFrame(depth=depth, camera=cam))
        assert n_added > 0
        assert mem.cloud.normals is not None
        assert mem.cloud.normals.shape == (mem.num_points, 3)

    def test_evicted_voxel_can_be_readded(self) -> None:
        """Eviction removes evicted voxel keys so revisited regions re-enter."""
        cfg = SpatialMemoryConfig(
            max_points=5,
            subsample_stride=16,
            voxel_size=0.05,
            normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        cam0 = _make_translated_camera(tx=0.0)
        cam1 = _make_translated_camera(tx=100.0)
        f0 = _make_frame(depth_value=5.0, h=32, w=32, camera=cam0)
        f1 = _make_frame(depth_value=5.0, h=32, w=32, camera=cam1)
        f0_revisit = _make_frame(depth_value=5.0, h=32, w=32, camera=cam0)

        n0 = mem.ingest_frame(0, f0)
        n1 = mem.ingest_frame(1, f1)
        assert n0 == 4
        assert n1 == 4
        assert mem.num_points == 5
        readded = mem.ingest_frame(2, f0_revisit)
        assert readded == 3

    def test_opengl_y_up_backprojection_convention(self) -> None:
        """Top image pixels (y < cy) must backproject to positive Y_cam."""
        cam = _StubCamera(width=32, height=32, cx=16.0, cy=16.0)
        depth = np.full((32, 32), 5.0, dtype=np.float32)
        pts = depth_to_world_points(depth, cam, stride=16)
        # Top row sampled at y=0.5 (< cy=16.0) -> +Y_cam; bottom at y=16.5 (> cy) -> -Y_cam
        assert pts[0, 1] > 0.0
        assert pts[-1, 1] < 0.0

    def test_render_reference_roundtrip_depth_accuracy(self) -> None:
        """Rasterised reference depth matches original planar depth for hit pixels."""
        cfg = SpatialMemoryConfig(
            subsample_stride=1, voxel_size=0.005, normal_estimation=True,
        )
        mem = SpatialMemory(cfg)
        cam = _StubCamera(width=32, height=32, cx=16.0, cy=16.0)
        frame = _make_frame(depth_value=5.0, h=32, w=32, camera=cam)
        mem.ingest_frame(0, frame)
        ref_depth, ref_normal = mem.render_reference(cam, width=32, height=32)
        hit = ref_depth > 0
        assert np.sum(hit) >= 0.9 * 32 * 32
        np.testing.assert_allclose(ref_depth[hit], 5.0, atol=1e-4)
        assert ref_normal is not None
        # Front-parallel plane normal should point toward +Z_cam = (0, 0, 1)
        np.testing.assert_allclose(ref_normal[hit][0], [0.0, 0.0, 1.0], atol=1e-4)


# ===========================================================================
# Tests: Normal estimation
# ===========================================================================


class TestNormalEstimation:
    """Test that normal estimation produces valid results."""

    def test_normals_computed_when_enabled(self) -> None:
        cfg = SpatialMemoryConfig(
            subsample_stride=8, normal_estimation=True,
        )
        mem = SpatialMemory(cfg)
        frame = _make_frame(depth_value=5.0, h=32, w=32)
        mem.ingest_frame(0, frame)
        assert mem.cloud.normals is not None
        assert mem.cloud.normals.shape == (mem.num_points, 3)

    def test_normals_not_computed_when_disabled(self) -> None:
        cfg = SpatialMemoryConfig(
            subsample_stride=8, normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        frame = _make_frame(depth_value=5.0, h=32, w=32)
        mem.ingest_frame(0, frame)
        assert mem.cloud.normals is None

    def test_normals_are_unit_length(self) -> None:
        cfg = SpatialMemoryConfig(
            subsample_stride=8, normal_estimation=True,
        )
        mem = SpatialMemory(cfg)
        frame = _make_frame(depth_value=5.0, h=32, w=32)
        mem.ingest_frame(0, frame)
        assert mem.cloud.normals is not None
        norms = np.linalg.norm(mem.cloud.normals, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ===========================================================================
# Tests: Reference rendering
# ===========================================================================


class TestRenderReference:
    """Test render_reference produces valid depth maps."""

    def test_empty_memory_returns_zeros(self) -> None:
        mem = SpatialMemory()
        cam = _StubCamera()
        depth, normal = mem.render_reference(cam, width=64, height=48)
        assert depth.shape == (48, 64)
        assert np.all(depth == 0.0)
        assert normal is None  # No normals accumulated

    def test_renders_nonzero_after_ingest(self) -> None:
        cfg = SpatialMemoryConfig(
            subsample_stride=4, normal_estimation=False,
        )
        mem = SpatialMemory(cfg)
        cam = _StubCamera(width=64, height=48, cx=32, cy=24)
        frame = _make_frame(depth_value=5.0, h=48, w=64, camera=cam)
        mem.ingest_frame(0, frame)
        depth, _ = mem.render_reference(cam, width=64, height=48)
        assert depth.shape == (48, 64)
        # At least some pixels should have depth
        assert np.any(depth > 0)

    def test_render_with_normals(self) -> None:
        cfg = SpatialMemoryConfig(
            subsample_stride=4, normal_estimation=True,
        )
        mem = SpatialMemory(cfg)
        cam = _StubCamera(width=64, height=48, cx=32, cy=24)
        frame = _make_frame(depth_value=5.0, h=48, w=64, camera=cam)
        mem.ingest_frame(0, frame)
        depth, normal = mem.render_reference(cam, width=64, height=48)
        assert depth.shape == (48, 64)
        assert normal is not None
        assert normal.shape == (48, 64, 3)

    def test_render_uses_camera_width_height(self) -> None:
        mem = SpatialMemory()
        cam = _StubCamera(width=100, height=50)
        depth, _ = mem.render_reference(cam)
        assert depth.shape == (50, 100)


# ===========================================================================
# Tests: depth_to_world_points utility
# ===========================================================================


class TestDepthToWorldPoints:
    """Test the standalone back-projection utility."""

    def test_basic_backproject(self) -> None:
        depth = np.full((32, 32), 5.0, dtype=np.float32)
        cam = _StubCamera(width=32, height=32, cx=16, cy=16)
        pts = depth_to_world_points(depth, cam, stride=4)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert len(pts) > 0

    def test_all_zero_depth_returns_empty(self) -> None:
        depth = np.zeros((16, 16), dtype=np.float32)
        cam = _StubCamera(width=16, height=16)
        pts = depth_to_world_points(depth, cam)
        assert len(pts) == 0

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            depth_to_world_points(np.zeros((2, 3, 4)), _StubCamera())

    def test_stride_reduces_points(self) -> None:
        depth = np.full((64, 64), 5.0, dtype=np.float32)
        cam = _StubCamera(width=64, height=64, cx=32, cy=32)
        pts_s1 = depth_to_world_points(depth, cam, stride=1)
        pts_s4 = depth_to_world_points(depth, cam, stride=4)
        assert len(pts_s1) > len(pts_s4)


# ===========================================================================
# Tests: compute_reference_consistency_score
# ===========================================================================


class TestConsistencyScore:
    """Test reference-vs-actual depth consistency metric."""

    def test_identical_depth_returns_one(self) -> None:
        d = np.full((16, 16), 5.0, dtype=np.float32)
        score = compute_reference_consistency_score(d, d)
        assert score == pytest.approx(1.0)

    def test_disjoint_depth_returns_zero(self) -> None:
        ref = np.full((16, 16), 5.0, dtype=np.float32)
        act = np.full((16, 16), 500.0, dtype=np.float32)  # Way off
        score = compute_reference_consistency_score(ref, act, threshold_m=0.1)
        assert score == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        ref = np.full((10, 10), 5.0, dtype=np.float32)
        act = ref.copy()
        act[5:, :] = 100.0  # Half the pixels are way off
        score = compute_reference_consistency_score(ref, act, threshold_m=0.1)
        assert 0.3 < score < 0.7  # Roughly half match

    def test_empty_valid_returns_zero(self) -> None:
        ref = np.zeros((8, 8), dtype=np.float32)
        act = np.zeros((8, 8), dtype=np.float32)
        score = compute_reference_consistency_score(ref, act)
        assert score == pytest.approx(0.0)
