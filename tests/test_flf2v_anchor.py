"""Tests for FLF2V (First-Last-Frame-to-Video) anchor fields (PR-218).

Covers:
- anchor_first_frame / anchor_last_frame fields on ScheduledWindow
- ConditionScheduler populates FLF2V anchors at discontinuity windows
- Non-discontinuity windows have None anchors
"""

from __future__ import annotations

import numpy as np
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
    ScheduledWindow,
    WindowSlice,
)

# ---------------------------------------------------------------------------
# ScheduledWindow FLF2V field existence
# ---------------------------------------------------------------------------


class TestScheduledWindowFLF2VFields:
    """anchor_first_frame and anchor_last_frame fields exist."""

    def test_defaults_to_none(self) -> None:
        sw = ScheduledWindow(window=WindowSlice(start=0, end=97))
        assert sw.anchor_first_frame is None
        assert sw.anchor_last_frame is None

    def test_explicit_set(self) -> None:
        sw = ScheduledWindow(
            window=WindowSlice(start=0, end=97),
            anchor_first_frame=10,
            anchor_last_frame=9,
        )
        assert sw.anchor_first_frame == 10
        assert sw.anchor_last_frame == 9


# ---------------------------------------------------------------------------
# ConditionScheduler populates FLF2V anchors
# ---------------------------------------------------------------------------


def _make_fgpp_sequence_with_cut(
    n_frames: int = 20,
    height: int = 8,
    width: int = 8,
    cut_at: int = 10,
) -> object:
    """Create an FGPPSequence with a sharp discontinuity at frame `cut_at`."""
    from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

    frames = []
    for i in range(n_frames):
        cam = CameraParams(
            fx=500.0,
            fy=500.0,
            cx=float(width // 2),
            cy=float(height // 2),
            width=width,
            height=height,
            view_matrix=np.eye(4, dtype=np.float64),
            z_near=0.1,
            z_far=100.0,
        )

        # Before cut: entity 1 filling foreground
        # After cut: entity 99 (completely new entity) + big MV jump
        if i < cut_at:
            crypto = np.full((height, width), 1, dtype=np.uint32)
            mv = np.zeros((height, width, 2), dtype=np.float32)
            normal = np.tile(
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                (height, width, 1),
            )
        else:
            crypto = np.full((height, width), 99, dtype=np.uint32)
            # Large MV jump to trigger norm_motion_jump
            mv = np.full((height, width, 2), 50.0, dtype=np.float32)
            # Flipped normals to trigger normal_cosine_drop
            normal = np.tile(
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
                (height, width, 1),
            )

        frame = FGPPFrame(
            camera=cam,
            depth=np.full((height, width), 5.0, dtype=np.float32),
            world_normal=normal,
            direct_radiance=np.full(
                (height, width, 3), 0.5, dtype=np.float32,
            ),
            base_color=np.full(
                (height, width, 3), 0.7, dtype=np.float32,
            ),
            roughness_metallic=np.zeros(
                (height, width, 2), dtype=np.float32,
            ),
            motion_vector=mv,
            cryptomatte_id=crypto,
            frame_index=i,
        )
        frames.append(frame)

    return FGPPSequence(frames=frames, fps=24.0)


class TestConditionSchedulerFLF2V:
    """ConditionScheduler populates FLF2V anchors at cuts."""

    def test_no_cut_no_anchors(self) -> None:
        """Without discontinuities, all windows have None anchors."""
        from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

        # Smooth 20-frame sequence
        frames = []
        for _ in range(20):
            cam = CameraParams(
                fx=500.0,
                fy=500.0,
                cx=4.0,
                cy=4.0,
                width=8,
                height=8,
                view_matrix=np.eye(4, dtype=np.float64),
                z_near=0.1,
                z_far=100.0,
            )
            frame = FGPPFrame(
                camera=cam,
                depth=np.full((8, 8), 5.0, dtype=np.float32),
                world_normal=np.tile(
                    np.array([0.0, 1.0, 0.0], dtype=np.float32), (8, 8, 1),
                ),
                direct_radiance=np.full((8, 8, 3), 0.5, dtype=np.float32),
                base_color=np.full((8, 8, 3), 0.7, dtype=np.float32),
                roughness_metallic=np.zeros((8, 8, 2), dtype=np.float32),
                motion_vector=np.zeros((8, 8, 2), dtype=np.float32),
                cryptomatte_id=np.full((8, 8), 1, dtype=np.uint32),
            )
            frames.append(frame)
        seq = FGPPSequence(frames=frames, fps=24.0)

        plan = CausalWindowPlan(
            window_frames=10, stride_frames=7, overlap_frames=3, total_frames=20,
        )
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
            anchor_interval=0,  # Disable physical anchors
        )
        windows = scheduler.schedule(seq)
        for w in windows:
            assert w.anchor_first_frame is None
            assert w.anchor_last_frame is None

    def test_cut_produces_anchors(self) -> None:
        """A sharp scene cut should populate FLF2V anchor fields."""
        seq = _make_fgpp_sequence_with_cut(n_frames=20, cut_at=10)

        plan = CausalWindowPlan(
            window_frames=20, stride_frames=20, overlap_frames=0, total_frames=20,
        )
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
            anchor_interval=0,
        )
        windows = scheduler.schedule(seq)

        # There should be exactly 1 window covering the whole sequence
        assert len(windows) == 1
        sw = windows[0]
        # The cut is at frame 10, so it should be flagged
        assert sw.needs_reanchor is True
        assert sw.anchor_first_frame == 10
        assert sw.anchor_last_frame == 9

    def test_multi_window_cut_at_boundary(self) -> None:
        """Cut at window boundary: second window gets FLF2V anchors."""
        seq = _make_fgpp_sequence_with_cut(n_frames=20, cut_at=10)

        plan = CausalWindowPlan(
            window_frames=12, stride_frames=10, overlap_frames=2, total_frames=20,
        )
        scheduler = ConditionScheduler(
            plan=plan,
            detector=DiscontinuityDetector(threshold=0.65),
            anchor_interval=0,
        )
        windows = scheduler.schedule(seq)

        # First window [0, 12) contains cut at 10
        assert windows[0].needs_reanchor is True
        assert windows[0].anchor_first_frame == 10
        assert windows[0].anchor_last_frame == 9
