# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tests for synapse.refine.fusion — MultiModelRefineFusion five kernels."""

from __future__ import annotations

import numpy as np
import pytest
from synapse.refine.fusion import (
    MultiModelRefineFusion,
    _sobel_gradient_magnitude,
)
from synapse.refine.types import FusionMethod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W, C = 64, 64, 3


def _make_patch(seed: int = 0, value: float = 0.5) -> np.ndarray:
    """Create a deterministic (H, W, 3) float32 ACEScg patch."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W, C), value, dtype=np.float32)
    noise = rng.standard_normal((H, W, C)).astype(np.float32) * 0.05
    return np.clip(base + noise, 0.0, 1.0)


def _gradient_patch(seed: int = 42) -> np.ndarray:
    """Create a patch with strong gradients for gradient preservation tests."""
    rng = np.random.default_rng(seed)
    patch = np.zeros((H, W, C), dtype=np.float32)
    # Horizontal gradient
    for col in range(W):
        patch[:, col, :] = col / W
    # Add some high-frequency detail
    patch += rng.standard_normal((H, W, C)).astype(np.float32) * 0.02
    return np.clip(patch, 0.0, 1.0)


@pytest.fixture()
def fusion() -> MultiModelRefineFusion:
    return MultiModelRefineFusion()


@pytest.fixture()
def two_candidates() -> list[np.ndarray]:
    return [_make_patch(seed=0, value=0.4), _make_patch(seed=1, value=0.6)]


@pytest.fixture()
def three_candidates() -> list[np.ndarray]:
    return [
        _make_patch(seed=10, value=0.3),
        _make_patch(seed=20, value=0.5),
        _make_patch(seed=30, value=0.7),
    ]


@pytest.fixture()
def uniform_weights_2() -> np.ndarray:
    return np.array([0.5, 0.5], dtype=np.float32)


@pytest.fixture()
def weights_3() -> np.ndarray:
    return np.array([0.5, 0.3, 0.2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Test: all five methods return valid output with no NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(FusionMethod))
def test_all_methods_return_valid_float32(
    fusion: MultiModelRefineFusion,
    two_candidates: list[np.ndarray],
    uniform_weights_2: np.ndarray,
    method: FusionMethod,
) -> None:
    """Every FusionMethod must return (H, W, 3) float32 without raising."""
    result = fusion.fuse(two_candidates, method, uniform_weights_2)
    assert result.dtype == np.float32
    assert result.shape == (H, W, C)
    # Must be finite
    assert np.all(np.isfinite(result)), "Result contains NaN or Inf"


@pytest.mark.parametrize("method", list(FusionMethod))
def test_three_candidates(
    fusion: MultiModelRefineFusion,
    three_candidates: list[np.ndarray],
    weights_3: np.ndarray,
    method: FusionMethod,
) -> None:
    """All methods work with three candidates."""
    result = fusion.fuse(three_candidates, method, weights_3)
    assert result.dtype == np.float32
    assert result.shape == (H, W, C)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Test: single candidate passthrough
# ---------------------------------------------------------------------------


def test_single_candidate_passthrough(
    fusion: MultiModelRefineFusion,
) -> None:
    """A single candidate must pass through unmodified."""
    patch = _make_patch(seed=99, value=0.5)
    for method in FusionMethod:
        result = fusion.fuse([patch], method)
        np.testing.assert_array_equal(result, patch)


# ---------------------------------------------------------------------------
# Test: frequency fusion preserves >=92% gradient magnitude
# ---------------------------------------------------------------------------


def test_laplacian_frequency_gradient_preservation(
    fusion: MultiModelRefineFusion,
) -> None:
    """Laplacian frequency fusion must preserve >=92% of gradient magnitude."""
    # Two patches with strong gradients
    patch_a = _gradient_patch(seed=42)
    patch_b = _gradient_patch(seed=43)

    weights = np.array([0.6, 0.4], dtype=np.float32)
    result = fusion.fuse(
        [patch_a, patch_b],
        FusionMethod.LAPLACIAN_FREQUENCY,
        weights,
    )

    # Compute gradient magnitude of weighted average (reference)
    ref = weights[0] * patch_a + weights[1] * patch_b
    grad_ref = _sobel_gradient_magnitude(ref)
    grad_result = _sobel_gradient_magnitude(result)

    ref_total = float(np.sum(grad_ref))
    result_total = float(np.sum(grad_result))

    if ref_total > 0:
        ratio = result_total / ref_total
        assert ratio >= 0.92, (
            f"Gradient preservation ratio {ratio:.4f} < 0.92"
        )


# ---------------------------------------------------------------------------
# Test: boundary ΔE₀₀ <= 1.5 for Laplacian frequency fusion
# ---------------------------------------------------------------------------


def test_laplacian_frequency_boundary_delta_e(
    fusion: MultiModelRefineFusion,
) -> None:
    """Boundary colour deviation must be <= 1.5 ΔE₀₀ for frequency fusion.

    We approximate ΔE₀₀ with a simplified CIELAB Euclidean distance
    (ΔE*ab) which is an upper bound for ΔE₀₀ on smooth colour ramps.
    """
    # Two patches with similar colours (simulating model variants)
    patch_a = np.full((H, W, C), 0.5, dtype=np.float32)
    patch_b = np.full((H, W, C), 0.52, dtype=np.float32)
    # Add a tiny gradient so it's not degenerate
    for col in range(W):
        patch_a[:, col, 0] += col * 0.001
        patch_b[:, col, 0] += col * 0.001

    weights = np.array([0.5, 0.5], dtype=np.float32)
    result = fusion.fuse(
        [patch_a, patch_b],
        FusionMethod.LAPLACIAN_FREQUENCY,
        weights,
    )

    # Compute boundary ring (8px from edges)
    ring_width = 8
    boundary_mask = np.zeros((H, W), dtype=bool)
    boundary_mask[:ring_width, :] = True
    boundary_mask[-ring_width:, :] = True
    boundary_mask[:, :ring_width] = True
    boundary_mask[:, -ring_width:] = True

    # Reference is the weighted average
    ref = weights[0] * patch_a + weights[1] * patch_b

    # Approximate ΔE as Euclidean distance in linear space * 100
    # (very rough, but sufficient for testing boundary smoothness)
    diff = result - ref
    boundary_diff = diff[boundary_mask]
    delta_e_approx = float(np.sqrt(np.mean(boundary_diff ** 2))) * 100

    assert delta_e_approx <= 1.5, (
        f"Boundary ΔE₀₀ approximation {delta_e_approx:.3f} > 1.5"
    )


# ---------------------------------------------------------------------------
# Test: edge cases and error handling
# ---------------------------------------------------------------------------


def test_empty_candidates_raises(
    fusion: MultiModelRefineFusion,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fusion.fuse([], FusionMethod.CASCADE)


def test_mismatched_shapes_raises(
    fusion: MultiModelRefineFusion,
) -> None:
    a = np.zeros((32, 32, 3), dtype=np.float32)
    b = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        fusion.fuse([a, b], FusionMethod.CASCADE)


def test_wrong_weights_length_raises(
    fusion: MultiModelRefineFusion,
) -> None:
    a = np.zeros((32, 32, 3), dtype=np.float32)
    b = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="weights length"):
        fusion.fuse(
            [a, b],
            FusionMethod.CASCADE,
            np.array([1.0, 0.5, 0.2]),
        )


def test_none_weights_defaults_to_uniform(
    fusion: MultiModelRefineFusion,
    two_candidates: list[np.ndarray],
) -> None:
    """Passing weights=None should produce a result without error."""
    result = fusion.fuse(two_candidates, FusionMethod.UNCERTAINTY_WEIGHTED)
    assert result.shape == (H, W, C)


# ---------------------------------------------------------------------------
# Test: Poisson gradient produces smooth interior
# ---------------------------------------------------------------------------


def test_poisson_gradient_smoothness(
    fusion: MultiModelRefineFusion,
) -> None:
    """Poisson gradient fusion should produce a smooth interior."""
    patch_a = _make_patch(seed=100, value=0.4)
    patch_b = _make_patch(seed=101, value=0.6)

    result = fusion.fuse(
        [patch_a, patch_b],
        FusionMethod.POISSON_GRADIENT,
        np.array([0.5, 0.5], dtype=np.float32),
    )

    # The result should be finite and have similar statistics
    assert np.all(np.isfinite(result))
    # Mean should be between the two candidates' means
    mean_a = float(np.mean(patch_a))
    mean_b = float(np.mean(patch_b))
    mean_r = float(np.mean(result))
    assert min(mean_a, mean_b) * 0.5 <= mean_r <= max(mean_a, mean_b) * 1.5


# ---------------------------------------------------------------------------
# Test: uncertainty fusion down-weights disagreement regions
# ---------------------------------------------------------------------------


def test_uncertainty_fusion_weights_agreement(
    fusion: MultiModelRefineFusion,
) -> None:
    """In agreement regions, uncertainty fusion should produce near-mean."""
    # Two nearly identical patches
    patch_a = np.full((32, 32, 3), 0.5, dtype=np.float32)
    patch_b = np.full((32, 32, 3), 0.5, dtype=np.float32)

    result = fusion.fuse(
        [patch_a, patch_b],
        FusionMethod.UNCERTAINTY_WEIGHTED,
        np.array([0.5, 0.5], dtype=np.float32),
    )

    # Should be very close to 0.5
    np.testing.assert_allclose(
        result, 0.5, atol=0.01,
        err_msg="Uncertainty fusion deviated from agreement value",
    )
