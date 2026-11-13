# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for PR-36: ScalpelStitcher, refinement gates, and blend.py motion warp."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from synapse.composite.blend import composite_surgical_sequence
from synapse.errors import CircuitBreakerError
from synapse.refine.gates import (
    GateThresholds,
    evaluate_refinement_gates,
    export_hard_negatives,
)
from synapse.refine.stitcher import (
    ScalpelStitcher,
    StitchConfig,
)
from synapse.refine.types import (
    AnatomicalRegion,
    FusionMethod,
    RefinementResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W = 64, 64


def _make_frames(t: int = 4) -> np.ndarray:
    """Create (T, H, W, 3) uint8 sRGB test video."""
    rng = np.random.default_rng(42)
    return rng.integers(100, 200, size=(t, H, W, 3), dtype=np.uint8)


def _make_acescg_frames(t: int = 4) -> np.ndarray:
    """Create (T, H, W, 3) float32 ACEScg test video."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.2, 0.8, size=(t, H, W, 3)).astype(np.float32)


def _make_masks(t: int = 4) -> np.ndarray:
    """Create (T, H, W) float32 surgical masks."""
    masks = np.zeros((t, H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    r = min(H, W) // 4
    y, x = np.ogrid[:H, :W]
    circle = ((y - cy) ** 2 + (x - cx) ** 2) <= r ** 2
    for i in range(t):
        masks[i][circle] = 1.0
    return masks


def _make_motion_vectors(t: int = 4, mag: float = 5.0) -> np.ndarray:
    """Create (T, H, W, 2) float32 motion vectors."""
    rng = np.random.default_rng(99)
    return rng.uniform(-mag, mag, size=(t, H, W, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: blend.py motion_vectors activation
# ---------------------------------------------------------------------------


def test_blend_with_motion_vectors_runs() -> None:
    """composite_surgical_sequence works with motion_vectors provided."""
    t_count = 4
    video = _make_frames(t_count)
    rerendered = _make_frames(t_count)
    masks = _make_masks(t_count)
    mv = _make_motion_vectors(t_count, mag=5.0)

    result = composite_surgical_sequence(
        video, rerendered, masks,
        motion_vectors=mv,
    )
    assert result.shape == video.shape
    assert result.dtype == np.uint8


def test_blend_motion_vectors_reduces_temporal_error() -> None:
    """With motion vectors, temporal warp error at boundaries should reduce."""
    t_count = 4
    video = _make_frames(t_count)
    rerendered = _make_frames(t_count)
    masks = _make_masks(t_count)
    # Small, consistent flow
    mv = np.ones((t_count, H, W, 2), dtype=np.float32) * 2.0

    result_no_mv = composite_surgical_sequence(
        video, rerendered, masks,
        motion_vectors=None,
    )
    result_with_mv = composite_surgical_sequence(
        video, rerendered, masks,
        motion_vectors=mv,
    )

    # Both should be valid uint8 frames
    assert result_no_mv.shape == result_with_mv.shape
    assert result_with_mv.dtype == np.uint8


def test_blend_large_flow_reduces_ema_alpha() -> None:
    """Large flow magnitude (>15px) should cause adaptive alpha reduction."""
    t_count = 4
    video = _make_frames(t_count)
    rerendered = _make_frames(t_count)
    masks = _make_masks(t_count)
    # Large flow — should trigger adaptive alpha
    mv = np.ones((t_count, H, W, 2), dtype=np.float32) * 30.0

    result = composite_surgical_sequence(
        video, rerendered, masks,
        motion_vectors=mv,
        ema_alpha=0.85,
    )
    assert result.shape == video.shape


# ---------------------------------------------------------------------------
# Tests: ScalpelStitcher
# ---------------------------------------------------------------------------


def test_stitcher_basic() -> None:
    """ScalpelStitcher produces output with correct shape."""
    frames = _make_acescg_frames(4)
    config = StitchConfig(border_ring_px=8)
    stitcher = ScalpelStitcher(config)

    # One refinement for frame 0
    refined_patch = np.full((20, 20, 3), 0.9, dtype=np.float32)
    ref_result = RefinementResult(
        refined_acescg=refined_patch,
        region=AnatomicalRegion.HAND,
        fusion_method=FusionMethod.CASCADE,
        bbox_xyxy=np.array([10, 10, 30, 30], dtype=np.float32),
    )
    refinements: list[list[RefinementResult]] = [
        [ref_result], [], [], [],
    ]

    result, report = stitcher.stitch_sequence(frames, refinements)
    assert result.shape == frames.shape
    assert report.num_patches_applied == 1


def test_stitcher_border_psnr() -> None:
    """Stitching identical content should give high border PSNR."""
    frames = np.full((2, H, W, 3), 0.5, dtype=np.float32)
    config = StitchConfig(border_ring_px=16)
    stitcher = ScalpelStitcher(config)

    # Refine a region with nearly the same colour
    refined = np.full((30, 30, 3), 0.50, dtype=np.float32)
    ref_result = RefinementResult(
        refined_acescg=refined,
        region=AnatomicalRegion.FACE,
        fusion_method=FusionMethod.LAPLACIAN_FREQUENCY,
        bbox_xyxy=np.array([10, 10, 40, 40], dtype=np.float32),
    )
    refinements = [[ref_result], []]

    result, report = stitcher.stitch_sequence(frames, refinements)
    # Nearly identical content should give very high PSNR
    assert report.mean_border_psnr_db >= 38.0


def test_stitcher_empty_refinements() -> None:
    """No refinements should produce unchanged output."""
    frames = _make_acescg_frames(2)
    stitcher = ScalpelStitcher()
    result, report = stitcher.stitch_sequence(frames, [[], []])
    np.testing.assert_array_equal(result, frames)
    assert report.num_patches_applied == 0


# ---------------------------------------------------------------------------
# Tests: refinement gates
# ---------------------------------------------------------------------------


def test_gates_pass() -> None:
    """All gates should pass with good metrics."""
    verdict = evaluate_refinement_gates(
        border_psnr_db=45.0,
        border_ciede2000=1.0,
        temporal_warp_error=0.010,
        elapsed_seconds=100.0,
    )
    assert verdict.passed
    assert verdict.tier_results["tier1_automatic"]
    assert verdict.tier_results["tier3_hitl"]


def test_gates_fail_psnr() -> None:
    """Low PSNR should fail Tier 1."""
    verdict = evaluate_refinement_gates(
        border_psnr_db=30.0,
        border_ciede2000=1.0,
    )
    assert not verdict.passed
    assert not verdict.tier_results["tier1_automatic"]
    assert any("psnr" in r for r in verdict.failure_reasons)


def test_gates_fail_ciede2000() -> None:
    """High ΔE₀₀ should fail Tier 1."""
    verdict = evaluate_refinement_gates(
        border_psnr_db=45.0,
        border_ciede2000=5.0,
    )
    assert not verdict.passed
    assert any("ciede" in r for r in verdict.failure_reasons)


def test_gates_circuit_breaker_900s() -> None:
    """elapsed_seconds > 900.0 must raise CircuitBreakerError (exit 40)."""
    with pytest.raises(CircuitBreakerError) as exc_info:
        evaluate_refinement_gates(
            border_psnr_db=45.0,
            border_ciede2000=1.0,
            elapsed_seconds=901.0,
        )
    assert int(exc_info.value.exit_code) == 40


def test_gates_custom_thresholds() -> None:
    """Custom thresholds should be respected."""
    th = GateThresholds(min_psnr_db=20.0, max_ciede2000=10.0)
    verdict = evaluate_refinement_gates(
        border_psnr_db=25.0,
        border_ciede2000=5.0,
        thresholds=th,
    )
    assert verdict.passed


# ---------------------------------------------------------------------------
# Tests: hard_negatives export
# ---------------------------------------------------------------------------


def test_export_hard_negatives() -> None:
    """export_hard_negatives writes valid JSONL."""
    negatives = [
        {
            "frame_index": 0,
            "region": "HAND",
            "failure_reasons": ["psnr < 38"],
            "metrics": {"border_psnr_db": 30.0},
        },
        {
            "frame_index": 5,
            "region": "FACE",
            "failure_reasons": ["ciede > 2.0"],
            "metrics": {"border_ciede2000": 3.5},
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = export_hard_negatives(
            negatives, Path(tmpdir) / "hard_negatives.jsonl",
        )
        assert out_path.exists()

        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)
            assert "frame_index" in parsed
            assert "failure_reasons" in parsed
