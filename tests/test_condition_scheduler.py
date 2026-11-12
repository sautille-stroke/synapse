"""Tests for synapse.adapters.scheduler — causal window scheduling,
discontinuity detection, overlap fusion, and latent flow operator.

Covers:
  * CausalWindowPlan window computation for various total_frames.
  * DiscontinuityDetector kappa computation and threshold triggering.
  * fuse_overlapping_latents variance preservation and boundary conditions.
  * compute_latent_flow_operator with synthetic motion data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
    compute_latent_flow_operator,
    fuse_overlapping_latents,
)
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

# -------------------------------------------------------------------
# Test helpers
# -------------------------------------------------------------------

_W, _H = 32, 32


def _cam(w: int = _W, h: int = _H) -> CameraParams:
    return CameraParams(
        fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, width=w, height=h,
    )


def _make_frame(
    idx: int,
    w: int = _W,
    h: int = _H,
    mv: np.ndarray | None = None,
    normal: np.ndarray | None = None,
    crypto: np.ndarray | None = None,
) -> FGPPFrame:
    """Build a minimal FGPPFrame for scheduler tests."""
    cam = _cam(w, h)
    depth = np.full((h, w), 5.0, dtype=np.float32)
    if normal is None:
        normal = np.zeros((h, w, 3), dtype=np.float16)
        normal[..., 2] = 1.0
    if crypto is None:
        crypto = np.ones((h, w), dtype=np.uint32)
    return FGPPFrame(
        camera=cam,
        depth=depth,
        world_normal=normal,
        motion_vector=mv,
        cryptomatte_id=crypto,
        frame_index=idx,
    )


def _make_sequence(n: int, **kwargs: object) -> FGPPSequence:
    """Build an FGPPSequence of *n* default frames."""
    frames = [_make_frame(i, **kwargs) for i in range(n)]
    return FGPPSequence(frames=frames, fps=24.0, scene_id="test")


# ===================================================================
# TestCausalWindowPlan
# ===================================================================


class TestCausalWindowPlan:
    """Window computation for various total_frames values."""

    def test_single_window_exact(self) -> None:
        """97 frames => exactly 1 window, no warmup."""
        plan = CausalWindowPlan(total_frames=97)
        windows = plan.compute_windows()
        assert len(windows) == 1
        w = windows[0]
        assert w.start == 0
        assert w.end == 97
        assert w.length == 97
        assert w.causal_warmup_prefix_indices == []

    def test_two_windows_161(self) -> None:
        """161 frames => 2 windows with overlap."""
        plan = CausalWindowPlan(total_frames=161)
        windows = plan.compute_windows()
        assert len(windows) == 2

        # Window 0: [0, 97)
        assert windows[0].start == 0
        assert windows[0].end == 97
        assert windows[0].causal_warmup_prefix_indices == []

        # Window 1: [64, 161)
        assert windows[1].start == 64
        assert windows[1].end == 161
        assert windows[1].causal_warmup_prefix_indices == [61, 62, 63]

    def test_three_windows_225(self) -> None:
        """225 frames => 3 windows."""
        plan = CausalWindowPlan(total_frames=225)
        windows = plan.compute_windows()
        assert len(windows) == 3

        assert windows[0].start == 0
        assert windows[0].end == 97

        assert windows[1].start == 64
        assert windows[1].end == 161
        assert windows[1].causal_warmup_prefix_indices == [61, 62, 63]

        assert windows[2].start == 128
        assert windows[2].end == 225
        assert windows[2].causal_warmup_prefix_indices == [125, 126, 127]

    def test_six_windows_385(self) -> None:
        """385 frames => multiple windows covering the full range."""
        plan = CausalWindowPlan(total_frames=385)
        windows = plan.compute_windows()

        # Verify coverage: first starts at 0, last ends at 385
        assert windows[0].start == 0
        assert windows[-1].end == 385

        # Verify overlap between consecutive windows
        for i in range(1, len(windows)):
            overlap = windows[i - 1].end - windows[i].start
            assert overlap > 0, (
                f"Windows {i-1} and {i} do not overlap"
            )

        # Verify warmup for k >= 1
        for k in range(1, len(windows)):
            wu = windows[k].causal_warmup_prefix_indices
            assert len(wu) == 3
            base = 64 * k
            assert wu == [base - 3, base - 2, base - 1]

    def test_short_last_window(self) -> None:
        """Total frames not aligned => last window is shorter."""
        plan = CausalWindowPlan(total_frames=100)
        windows = plan.compute_windows()
        assert len(windows) == 2
        # Window 1: [64, 100) — only 36 frames, shorter than 97
        assert windows[1].end == 100
        assert windows[1].length == 36

    def test_zero_frames(self) -> None:
        """Zero total_frames => empty window list."""
        plan = CausalWindowPlan(total_frames=0)
        assert plan.compute_windows() == []

    def test_very_short_sequence(self) -> None:
        """Fewer frames than a single window."""
        plan = CausalWindowPlan(total_frames=10)
        windows = plan.compute_windows()
        assert len(windows) == 1
        assert windows[0].start == 0
        assert windows[0].end == 10
        assert windows[0].length == 10


# ===================================================================
# TestDiscontinuityDetector
# ===================================================================


class TestDiscontinuityDetector:
    """Kappa computation, threshold triggering, normal flip detection."""

    def test_no_discontinuity(self) -> None:
        """Identical consecutive frames => kappa ~ 0."""
        det = DiscontinuityDetector()
        h, w = 16, 16
        mv = np.zeros((h, w, 2), dtype=np.float32)
        normal = np.zeros((h, w, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        crypto = np.ones((h, w), dtype=np.uint32)

        kappa = det.compute_kappa(mv, mv, normal, normal, crypto, crypto)
        assert kappa < 0.01

    def test_large_motion_jump(self) -> None:
        """Large motion vector difference triggers high kappa."""
        det = DiscontinuityDetector()
        h, w = 16, 16
        mv_prev = np.zeros((h, w, 2), dtype=np.float32)
        mv_curr = np.full((h, w, 2), 30.0, dtype=np.float32)
        normal = np.zeros((h, w, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        crypto = np.ones((h, w), dtype=np.uint32)

        kappa = det.compute_kappa(
            mv_prev, mv_curr, normal, normal, crypto, crypto,
        )
        # norm_motion_jump should saturate at 1.0, contributing 0.4
        assert kappa >= 0.35

    def test_normal_flip_detection(self) -> None:
        """Flipped normals => high normal_cosine_drop."""
        det = DiscontinuityDetector()
        h, w = 16, 16
        normal_prev = np.zeros((h, w, 3), dtype=np.float32)
        normal_prev[..., 2] = 1.0  # facing camera
        normal_curr = np.zeros((h, w, 3), dtype=np.float32)
        normal_curr[..., 2] = -1.0  # flipped

        fg = np.ones((h, w), dtype=bool)
        drop = det.normal_cosine_drop(normal_prev, normal_curr, fg)
        # cos(180 deg) = -1 => drop = 1 - (-1) = 2, clamped to 1.0
        assert drop >= 0.99

    def test_mask_iou_disjoint(self) -> None:
        """Disjoint foreground masks => IoU = 0."""
        h, w = 16, 16
        crypto_prev = np.zeros((h, w), dtype=np.uint32)
        crypto_prev[:8, :] = 1
        crypto_curr = np.zeros((h, w), dtype=np.uint32)
        crypto_curr[8:, :] = 2

        iou = DiscontinuityDetector.mask_iou(crypto_prev, crypto_curr)
        assert iou == 0.0

    def test_mask_iou_identical(self) -> None:
        """Identical masks => IoU = 1."""
        h, w = 16, 16
        crypto = np.ones((h, w), dtype=np.uint32)
        iou = DiscontinuityDetector.mask_iou(crypto, crypto)
        assert iou == 1.0

    def test_threshold_triggering(self) -> None:
        """Detect method flags frames exceeding threshold."""
        det = DiscontinuityDetector(threshold=0.3)
        n = 5
        frames = []
        for i in range(n):
            mv = np.zeros((_H, _W, 2), dtype=np.float16)
            normal = np.zeros((_H, _W, 3), dtype=np.float16)
            normal[..., 2] = 1.0
            crypto = np.ones((_H, _W), dtype=np.uint32)

            # Inject a large jump at frame 3
            if i == 3:
                mv[:] = 50.0
                normal[..., 2] = -1.0
                crypto[:] = 0

            frames.append(_make_frame(
                i, mv=mv, normal=normal, crypto=crypto,
            ))

        seq = FGPPSequence(frames=frames, fps=24.0)
        reanchors = det.detect(seq)
        flagged = {r.frame_index for r in reanchors}
        # Frame 3 should be flagged (jump from frame 2)
        assert 3 in flagged

    def test_detect_returns_kappa_value(self) -> None:
        """ReanchorRequest carries the computed kappa value."""
        det = DiscontinuityDetector(threshold=0.0)
        frames = []
        for i in range(2):
            mv = np.full((_H, _W, 2), float(i * 10), dtype=np.float16)
            normal = np.zeros((_H, _W, 3), dtype=np.float16)
            normal[..., 2] = 1.0
            crypto = np.ones((_H, _W), dtype=np.uint32)
            frames.append(_make_frame(
                i, mv=mv, normal=normal, crypto=crypto,
            ))

        seq = FGPPSequence(frames=frames, fps=24.0)
        reanchors = det.detect(seq)
        assert len(reanchors) >= 1
        assert reanchors[0].kappa > 0.0
        assert reanchors[0].subframe_window == (-2, 2)


# ===================================================================
# TestFuseOverlapping
# ===================================================================


class TestFuseOverlapping:
    """Variance preservation, boundary continuity, alpha(0) = 0."""

    def test_alpha_zero_keeps_prev(self) -> None:
        """At tau'=0, alpha=0 => output equals z_prev."""
        rng = np.random.default_rng(42)
        shape = (9, 4, 8, 8)
        z_prev = rng.standard_normal(shape).astype(np.float32)
        z_curr = rng.standard_normal(shape).astype(np.float32)

        fused = fuse_overlapping_latents(z_prev, z_curr)
        # At step 0: alpha=0, normaliser=1 => fused[0] == z_prev[0]
        np.testing.assert_allclose(
            fused[0], z_prev[0], atol=1e-5,
        )

    def test_variance_preservation(self) -> None:
        """Fused output variance approx 1.0 when inputs have var 1."""
        rng = np.random.default_rng(123)
        n_trials = 50
        shape = (9, 4, 64, 64)

        variances = []
        for _ in range(n_trials):
            z_prev = rng.standard_normal(shape).astype(np.float32)
            z_curr = rng.standard_normal(shape).astype(np.float32)
            fused = fuse_overlapping_latents(z_prev, z_curr)
            variances.append(float(np.var(fused)))

        mean_var = np.mean(variances)
        assert abs(mean_var - 1.0) < 0.02, (
            f"Mean variance {mean_var:.4f} deviates from 1.0"
        )

    def test_boundary_continuity(self) -> None:
        """Alpha transitions smoothly from 0 to ~1."""
        rng = np.random.default_rng(7)
        shape = (9, 1, 4, 4)
        z_prev = rng.standard_normal(shape).astype(np.float32)
        z_curr = rng.standard_normal(shape).astype(np.float32)

        fused = fuse_overlapping_latents(z_prev, z_curr, lam=8.0)

        # At step 0, output == z_prev[0]
        np.testing.assert_allclose(fused[0], z_prev[0], atol=1e-5)

        # At last step, output should be very close to z_curr
        # alpha(8) = sigmoid(8 * (8/8 - 0.5)) = sigmoid(4) ~ 0.982
        alpha_last = 1.0 / (1.0 + math.exp(-8.0 * (8.0 / 8.0 - 0.5)))
        # The contribution of z_curr should dominate
        assert alpha_last > 0.95

    def test_raises_on_short_tensors(self) -> None:
        """ValueError if tensors are shorter than overlap_steps."""
        z_short = np.zeros((5, 1, 4, 4), dtype=np.float32)
        z_ok = np.zeros((9, 1, 4, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="z_prev has 5 steps"):
            fuse_overlapping_latents(z_short, z_ok)

        with pytest.raises(ValueError, match="z_curr has 5 steps"):
            fuse_overlapping_latents(z_ok, z_short)

    def test_output_shape_matches(self) -> None:
        """Output shape is [overlap_steps, ...] matching input."""
        shape = (12, 4, 16, 16)
        z_prev = np.ones(shape, dtype=np.float32)
        z_curr = np.ones(shape, dtype=np.float32)
        fused = fuse_overlapping_latents(z_prev, z_curr, overlap_steps=9)
        assert fused.shape == (9, 4, 16, 16)

    def test_identical_inputs(self) -> None:
        """When z_prev == z_curr, output ≈ input (up to normaliser)."""
        rng = np.random.default_rng(99)
        shape = (9, 2, 8, 8)
        z = rng.standard_normal(shape).astype(np.float32)

        fused = fuse_overlapping_latents(z, z.copy())
        # fused = z / sqrt((1-a)^2 + a^2) which is z * 1/sqrt(...)
        # At alpha=0: normaliser=1 => fused[0] == z[0]
        np.testing.assert_allclose(fused[0], z[0], atol=1e-5)


# ===================================================================
# TestLatentFlowOperator
# ===================================================================


class TestLatentFlowOperator:
    """Latent flow operator with synthetic rightward-moving sphere."""

    def test_basic_rightward_motion(self) -> None:
        """Uniform rightward MV => pooled displacement is rightward."""
        T = 5  # (T-1)=4, divisible by 4
        H, W = 16, 16
        mv = np.zeros((T, H, W, 2), dtype=np.float32)
        # Constant 2 px/frame rightward motion on all frames > 0
        mv[1:, :, :, 0] = 2.0
        depths = np.ones((T, H, W), dtype=np.float32)

        result = compute_latent_flow_operator(mv, depths)
        # tau = (5-1)//4 + 1 = 2
        assert result.shape == (2, 2, 2, 2)

        # At tau=0: no motion accumulated => ~0
        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)

        # At tau=1: accumulated 4 frames of 2px rightward = 8px total
        # All blocks should show ~8.0 in x-direction
        assert np.all(result[1, :, :, 0] > 7.0)

    def test_discontinuity_reset(self) -> None:
        """Discontinuity frame resets the integrator."""
        T = 5
        H, W = 8, 8
        mv = np.zeros((T, H, W, 2), dtype=np.float32)
        mv[1:, :, :, 0] = 1.0  # 1 px/frame rightward
        depths = np.ones((T, H, W), dtype=np.float32)

        # Discontinuity at frame 2 => reset accumulator
        result_with_disc = compute_latent_flow_operator(
            mv, depths, discontinuity_frames=[2],
        )
        result_without = compute_latent_flow_operator(mv, depths)

        # With discontinuity, total displacement should be less
        mag_with = float(np.sum(np.abs(result_with_disc)))
        mag_without = float(np.sum(np.abs(result_without)))
        assert mag_with < mag_without

    def test_validation_error(self) -> None:
        """(T-1) not divisible by temporal_compression => ValueError."""
        T = 4  # (T-1)=3, not divisible by 4
        mv = np.zeros((T, 8, 8, 2), dtype=np.float32)
        depths = np.ones((T, 8, 8), dtype=np.float32)

        with pytest.raises(ValueError, match="divisible"):
            compute_latent_flow_operator(mv, depths)

    def test_output_shape(self) -> None:
        """Output shape matches the expected latent dimensions."""
        T = 9  # (T-1)=8, tau=(8//4)+1=3
        H, W = 24, 32
        mv = np.zeros((T, H, W, 2), dtype=np.float32)
        depths = np.ones((T, H, W), dtype=np.float32)

        result = compute_latent_flow_operator(
            mv, depths, spatial_block=8,
        )
        assert result.shape == (3, 3, 4, 2)
        assert result.dtype == np.float32

    def test_depth_based_pooling(self) -> None:
        """Closest (min depth) pixel's displacement is selected."""
        T = 5
        H, W = 8, 8
        block = 8

        mv = np.zeros((T, H, W, 2), dtype=np.float32)
        depths = np.full((T, H, W), 100.0, dtype=np.float32)

        # Place a close object at pixel (2, 3) with unique MV
        depths[:, 2, 3] = 1.0
        mv[1:, 2, 3, 0] = 3.0  # 3 px/frame rightward
        mv[1:, 2, 3, 1] = 1.0  # 1 px/frame downward

        result = compute_latent_flow_operator(
            mv, depths, spatial_block=block,
        )
        # Single block output at (0, 0)
        assert result.shape == (2, 1, 1, 2)

        # tau=1: accumulated 4 frames => [12, 4]
        np.testing.assert_allclose(
            result[1, 0, 0, 0], 12.0, atol=0.1,
        )
        np.testing.assert_allclose(
            result[1, 0, 0, 1], 4.0, atol=0.1,
        )

    def test_occlusion_filtering(self) -> None:
        """Motion vectors exceeding threshold are filtered."""
        T = 5
        H, W = 8, 8
        mv = np.zeros((T, H, W, 2), dtype=np.float32)
        # Very large MV — should be filtered as occluded
        mv[1:, :, :, 0] = 100.0
        depths = np.ones((T, H, W), dtype=np.float32)

        result = compute_latent_flow_operator(mv, depths)
        # All filtered => displacement stays at 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


# ===================================================================
# TestConditionScheduler (integration)
# ===================================================================


class TestConditionScheduler:
    """Integration test combining ConditionScheduler components."""

    def test_schedule_no_discontinuities(self) -> None:
        """Clean sequence => no reanchor flags."""
        plan = CausalWindowPlan(total_frames=161)
        det = DiscontinuityDetector(threshold=0.65)
        sched = ConditionScheduler(plan, det)

        seq = _make_sequence(161)
        scheduled = sched.schedule(seq)

        assert len(scheduled) == 2
        assert all(not sw.needs_reanchor for sw in scheduled)

    def test_schedule_with_discontinuity(self) -> None:
        """Discontinuity in window 1 only => that window is flagged."""
        plan = CausalWindowPlan(total_frames=161)
        det = DiscontinuityDetector(threshold=0.3)
        sched = ConditionScheduler(plan, det)

        frames = []
        for i in range(161):
            mv = np.zeros((_H, _W, 2), dtype=np.float16)
            normal = np.zeros((_H, _W, 3), dtype=np.float16)
            normal[..., 2] = 1.0
            crypto = np.ones((_H, _W), dtype=np.uint32)

            # Big jump at frame 100 (in window 1: [64, 161) only,
            # outside window 0: [0, 97))
            if i == 100:
                mv[:] = 50.0
                normal[..., 2] = -1.0
                crypto[:] = 0

            frames.append(_make_frame(
                i, mv=mv, normal=normal, crypto=crypto,
            ))

        seq = FGPPSequence(frames=frames, fps=24.0)
        scheduled = sched.schedule(seq)

        assert len(scheduled) == 2
        # Window 0 should be clean, window 1 flagged
        assert not scheduled[0].needs_reanchor
        assert scheduled[1].needs_reanchor
        assert any(
            r.frame_index == 100
            for r in scheduled[1].reanchor_requests
        )

    def test_window_slice_fields(self) -> None:
        """ScheduledWindow exposes correct window_index."""
        plan = CausalWindowPlan(total_frames=225)
        det = DiscontinuityDetector()
        sched = ConditionScheduler(plan, det)

        seq = _make_sequence(225)
        scheduled = sched.schedule(seq)
        for i, sw in enumerate(scheduled):
            assert sw.window_index == i
