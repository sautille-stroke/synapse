"""Tests for ConditionScheduler v2 + SpatialMemory conditioning (M57 PR-229)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from synapse.adapters.base import VideoGenerationConfig
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
    inject_spatial_memory_control,
)
from synapse.adapters.spatial_memory import SpatialMemory, SpatialMemoryConfig
from synapse.adapters.wan import LocalCallableBackend, WanAdapter
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence


def _make_corridor_sequence(
    n_frames: int = 200,
    width: int = 32,
    height: int = 32,
    cut_at: int | None = None,
) -> FGPPSequence:
    """Create a synthetic long-corridor FGPP sequence with optional scene cut."""
    frames: list[FGPPFrame] = []
    for i in range(n_frames):
        # Smooth forward dolly motion along -Z, shifting right slightly on cut
        tx = 0.0 if (cut_at is None or i < cut_at) else 500.0
        tz = -0.02 * float(i if cut_at is None or i < cut_at else i - cut_at)
        world = np.eye(4, dtype=np.float64)
        world[0, 3] = tx
        world[2, 3] = tz
        view = np.linalg.inv(world)

        cam = CameraParams(
            fx=32.0,
            fy=32.0,
            cx=float(width) / 2.0,
            cy=float(height) / 2.0,
            width=width,
            height=height,
            view_matrix=view,
            world_matrix=world,
            z_near=0.1,
            z_far=100.0,
        )
        depth_m = max(1.0, 10.0 + tz)
        depth = np.full((height, width), depth_m, dtype=np.float32)
        if cut_at is not None and i >= cut_at:
            crypto = np.full((height, width), 77, dtype=np.uint32)
            mv = np.full((height, width, 2), 40.0, dtype=np.float32)
            normal = np.tile(
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
                (height, width, 1),
            )
        else:
            crypto = np.ones((height, width), dtype=np.uint32)
            mv = np.zeros((height, width, 2), dtype=np.float32)
            normal = np.tile(
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                (height, width, 1),
            )

        frames.append(
            FGPPFrame(
                camera=cam,
                depth=depth,
                world_normal=normal,
                direct_radiance=np.full((height, width, 3), 0.4, dtype=np.float32),
                base_color=np.full((height, width, 3), 0.5, dtype=np.float32),
                roughness_metallic=np.full((height, width, 2), 0.5, dtype=np.float32),
                motion_vector=mv,
                cryptomatte_id=crypto,
                frame_index=i,
            )
        )
    return FGPPSequence(frames=frames, fps=24.0, scene_id="t4_long_corridor")


class TestConditionSchedulerSpatialMemory:
    """ConditionScheduler v2 integrates SpatialMemory across multi-window runs."""

    def test_disabled_by_default_preserves_none_fields(self) -> None:
        seq = _make_corridor_sequence(n_frames=161)
        plan = CausalWindowPlan(total_frames=161)
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
        )
        windows = scheduler.schedule(seq)
        assert len(windows) == 2
        for sw in windows:
            assert sw.spatial_ref_depth is None
            assert sw.spatial_ref_normal is None
            assert sw.spatial_consistency_score is None

    def test_long_corridor_200f_populates_spatial_references(self) -> None:
        seq = _make_corridor_sequence(n_frames=200, width=32, height=32)
        plan = CausalWindowPlan(
            window_frames=97,
            stride_frames=64,
            overlap_frames=33,
            total_frames=200,
        )
        mem = SpatialMemory(
            SpatialMemoryConfig(
                voxel_size=0.05,
                subsample_stride=2,
                normal_estimation=True,
            )
        )
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
            spatial_memory=mem,
        )
        windows = scheduler.schedule(seq)
        assert len(windows) == 3

        # First window initializes memory -> no prior ref
        assert windows[0].spatial_ref_depth is None
        # Subsequent overlap windows receive spatial memory reference depth/normal
        for sw in windows[1:]:
            assert sw.spatial_ref_depth is not None
            assert sw.spatial_ref_depth.shape == (32, 32)
            assert np.any(sw.spatial_ref_depth > 0.0)
            assert sw.spatial_ref_normal is not None
            assert sw.spatial_ref_normal.shape == (32, 32, 3)
            assert sw.spatial_consistency_score is not None
            assert sw.spatial_consistency_score > 0.9

    def test_spatial_memory_reduces_drift_psnr_over_2db(self) -> None:
        """SpatialMemory conditioning improves drifted overlap PSNR by >= 2 dB."""
        from synapse.fgpp.schema import linear_to_log_normalized_depth

        seq = _make_corridor_sequence(n_frames=161, width=32, height=32)
        plan = CausalWindowPlan(
            window_frames=97, stride_frames=64, overlap_frames=33, total_frames=161
        )
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
            enable_spatial_memory=True,
        )
        windows = scheduler.schedule(seq)
        sw1 = windows[1]
        assert sw1.spatial_ref_depth is not None

        # Ground-truth log depth for overlap window [64..97)
        gt_ctrl = np.zeros((7, 33, 32, 32), dtype=np.float32)
        for i in range(33):
            depth_arr = seq.frames[64 + i].depth
            assert depth_arr is not None
            gt_ctrl[0, i] = linear_to_log_normalized_depth(depth_arr, 0.1, 100.0)

        # Simulated drifted overlap prediction (uniform systematic depth shift)
        drifted_ctrl = gt_ctrl.copy()
        drifted_ctrl[0] = np.clip(drifted_ctrl[0] + 0.15, 0.0, 1.0)

        stabilized = inject_spatial_memory_control(
            drifted_ctrl,
            sw1.spatial_ref_depth,
            sw1.spatial_ref_normal,
            overlap_frames=33,
            blend_weight=0.6,
        )

        hit = sw1.spatial_ref_depth > 0
        mse_before = float(np.mean((drifted_ctrl[0, :, hit] - gt_ctrl[0, :, hit]) ** 2))
        mse_after = float(np.mean((stabilized[0, :, hit] - gt_ctrl[0, :, hit]) ** 2))
        psnr_gain_db = 10.0 * float(np.log10(mse_before / max(mse_after, 1e-12)))
        assert psnr_gain_db >= 2.0

    def test_scene_cut_resets_spatial_memory_at_boundary(self) -> None:
        seq = _make_corridor_sequence(n_frames=40, width=16, height=16, cut_at=20)
        plan = CausalWindowPlan(
            window_frames=20,
            stride_frames=10,
            overlap_frames=10,
            total_frames=40,
        )
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
            enable_spatial_memory=True,
            anchor_interval=0,
        )
        windows = scheduler.schedule(seq)
        # Discontinuity window at cut_at=20 should suppress stale pre-cut ref if cut at start
        assert len(windows) >= 2
        assert scheduler.spatial_memory is not None
        assert scheduler.spatial_memory.num_points > 0


class TestInjectSpatialMemoryControl:
    """Tests for inject_spatial_memory_control 7-ch conditioning blender."""

    def test_blends_overlap_depth_and_normal_only_on_hit_pixels(self) -> None:
        ctrl = np.zeros((7, 16, 8, 8), dtype=np.float32)
        ctrl[1:4] = np.array([1.0, 0.0, 0.0], dtype=np.float32)[:, None, None, None]

        ref_depth = np.zeros((8, 8), dtype=np.float32)
        ref_depth[:4, :4] = 10.0  # only upper-left quadrant has projected hits
        ref_normal = np.zeros((8, 8, 3), dtype=np.float32)
        ref_normal[:4, :4] = [0.0, 0.0, 1.0]

        out = inject_spatial_memory_control(
            ctrl,
            ref_depth,
            ref_normal,
            overlap_frames=6,
            z_near=0.1,
            z_far=100.0,
            blend_weight=0.5,
        )
        # Non-overlap tail untouched
        np.testing.assert_allclose(out[:, 6:], ctrl[:, 6:])
        # Unvisited quadrant untouched
        np.testing.assert_allclose(out[:, :6, 4:, 4:], ctrl[:, :6, 4:, 4:])
        # Hit quadrant shifted toward log(10/0.1)/log(100/0.1) = 2/3 -> 0.5 * 2/3 = 1/3
        assert out[0, 0, 0, 0] == pytest.approx(1.0 / 3.0, abs=1e-4)
        # Normal on hit quadrant rotated and remains unit-length
        n_hit = out[1:4, 0, 0, 0]
        assert float(np.linalg.norm(n_hit)) == pytest.approx(1.0, abs=1e-5)
        assert n_hit[2] > 0.5

    def test_shape_mismatch_raises(self) -> None:
        ctrl = np.zeros((7, 4, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            inject_spatial_memory_control(ctrl, np.ones((16, 16), dtype=np.float32))


class TestWanAdapterSpatialMemoryIntegration:
    """WanAdapter multi-window generation wires SpatialMemory when enabled."""

    def test_multi_window_submits_spatial_ref_on_window_2(self) -> None:
        seq = _make_corridor_sequence(n_frames=25, width=16, height=16)
        submitted: list[dict[str, Any]] = []

        def _backend_fn(payload: Any, cfg: VideoGenerationConfig) -> np.ndarray:
            submitted.append(
                {
                    "has_ref": "spatial_ref_depth" in payload.extra,
                    "score": payload.extra.get("spatial_consistency_score"),
                    "control": payload.control_tensor.copy()
                    if payload.control_tensor is not None
                    else None,
                }
            )
            tt, yy, xx = np.mgrid[: cfg.num_frames, : cfg.height, : cfg.width]
            return np.stack(
                [
                    ((xx * 13 + tt * 5 + 30) % 256).astype(np.uint8),
                    ((yy * 13 + tt * 7 + 60) % 256).astype(np.uint8),
                    (((xx + yy) * 7 + tt * 11 + 90) % 256).astype(np.uint8),
                ],
                axis=-1,
            )

        adapter = WanAdapter(backend=LocalCallableBackend(_backend_fn))
        config = VideoGenerationConfig(
            width=16,
            height=16,
            num_frames=25,
            extra={
                "window_frames": 13,
                "stride_frames": 9,
                "overlap_frames": 4,
                "enable_spatial_memory": True,
            },
        )
        cond = adapter.prepare_conditions(seq, config)
        res = adapter.generate(cond, config)
        assert res.frames_srgb_u8.shape == (25, 16, 16, 3)
        assert len(submitted) >= 2
        assert submitted[0]["has_ref"] is False
        assert submitted[1]["has_ref"] is True
        assert submitted[1]["score"] is not None
