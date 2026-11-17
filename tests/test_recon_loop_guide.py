"""Tests for synapse.recon.loop — forward guide, radiometric conversion, VAE phase-locking.

Covers:
- T = 4k+1 phase-lock validation and enforcement
- n_preroll ∈ {0, 4, 8} contract validation
- κ = 1.0 hard-cut re-anchoring
- sRGB ↔ ACEScg round-trip accuracy
- Forward guide light-leak prevention
- _spatial_dims 4-D / 3-D layout detection (H6-P1-04)
- extract_main_take preroll trimming bounds check
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.errors import ContractValidationError
from synapse.recon.loop import (
    ForwardGuide,
    ForwardGuideResult,
    LoopConfig,
    extract_main_take,
)
from synapse.recon.loop.radiometric import (
    acescg_to_linear,
    apply_kappa_reanchor,
    enforce_phase_lock,
    linear_to_acescg,
    validate_frame_count_phase,
)

# ---------------------------------------------------------------------------
# Phase-lock validation
# ---------------------------------------------------------------------------

_VALID_COUNTS = [
    5,
    9,
    13,
    17,
    21,
    25,
    29,
    33,
    37,
    41,
    45,
    49,
    53,
    57,
    61,
    65,
    69,
    73,
    77,
    81,
    85,
    89,
    93,
    97,
]

_INVALID_COUNTS = [4, 6, 10, 96, 98, 100]


@pytest.mark.parametrize("t", _VALID_COUNTS)
def test_phase_lock_valid_counts(t: int) -> None:
    """All T = 4k+1 frame counts must be accepted."""
    assert validate_frame_count_phase(t) is True


@pytest.mark.parametrize("t", _INVALID_COUNTS)
def test_phase_lock_invalid_counts(t: int) -> None:
    """Non-4k+1 frame counts must be rejected."""
    assert validate_frame_count_phase(t) is False


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(96, 97), (10, 13), (4, 5)],
)
def test_enforce_phase_lock_rounds_up(requested: int, expected: int) -> None:
    """enforce_phase_lock must round up to the nearest valid T = 4k+1."""
    assert enforce_phase_lock(requested) == expected


# ---------------------------------------------------------------------------
# Preroll contract validation
# ---------------------------------------------------------------------------


def test_preroll_must_be_0_4_8() -> None:
    """n_preroll=3 must raise ContractValidationError with exit_code=10."""
    with pytest.raises(ContractValidationError) as exc_info:
        LoopConfig(n_preroll=3)
    assert exc_info.value.exit_code == 10


@pytest.mark.parametrize("n", [0, 4, 8])
def test_preroll_valid_values(n: int) -> None:
    """All valid preroll values must be accepted without error."""
    cfg = LoopConfig(n_preroll=n)
    assert cfg.n_preroll == n


# ---------------------------------------------------------------------------
# κ re-anchoring
# ---------------------------------------------------------------------------


def test_kappa_reanchor_replaces_at_1_0() -> None:
    """With kappa=1.0, the last preroll frame must exactly equal the production first frame."""
    rng = np.random.default_rng(42)
    preroll = rng.random((4, 16, 16, 3), dtype=np.float64)
    prod_first = rng.random((16, 16, 3), dtype=np.float64)

    result = apply_kappa_reanchor(preroll, prod_first, kappa=1.0)

    # Last preroll frame should be identical to production_first_frame.
    np.testing.assert_array_equal(result[-1], prod_first)
    # Earlier preroll frames should be unchanged.
    np.testing.assert_array_equal(result[:-1], preroll[:-1])


def test_kappa_reanchor_blends_at_half() -> None:
    """With kappa=0.5, the last preroll frame should be a 50/50 blend."""
    preroll = np.ones((2, 8, 8, 3), dtype=np.float64) * 0.2
    prod = np.ones((8, 8, 3), dtype=np.float64) * 0.8

    result = apply_kappa_reanchor(preroll, prod, kappa=0.5)

    expected = 0.5 * 0.2 + 0.5 * 0.8  # 0.5
    np.testing.assert_allclose(result[-1], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# ACEScg round-trip
# ---------------------------------------------------------------------------


def test_acescg_roundtrip() -> None:
    """linear_to_acescg → acescg_to_linear round-trip error must be < 1e-5."""
    rng = np.random.default_rng(123)
    linear_rgb = rng.random((32, 32, 3), dtype=np.float64)

    roundtrip = acescg_to_linear(linear_to_acescg(linear_rgb))

    np.testing.assert_allclose(roundtrip, linear_rgb, atol=1e-5)


def test_acescg_known_values() -> None:
    """Verify known colour transforms (pure red, green, blue, white)."""
    primaries = np.array(
        [
            [1.0, 0.0, 0.0],  # sRGB red
            [0.0, 1.0, 0.0],  # sRGB green
            [0.0, 0.0, 1.0],  # sRGB blue
            [1.0, 1.0, 1.0],  # D65 white
        ],
        dtype=np.float64,
    )

    acescg = linear_to_acescg(primaries)

    # White should be [1, 1, 1] in ACEScg (both share D65 adaptation).
    np.testing.assert_allclose(acescg[3], [1.0, 1.0, 1.0], atol=1e-4)

    # Round-trip for all.
    np.testing.assert_allclose(acescg_to_linear(acescg), primaries, atol=1e-5)


# ---------------------------------------------------------------------------
# ForwardGuide integration
# ---------------------------------------------------------------------------


def test_forward_guide_basic_generate() -> None:
    """ForwardGuide.generate must return a valid ForwardGuideResult."""
    guide = ForwardGuide(LoopConfig(n_preroll=0))
    conditioning = {"CameraLinearDepth.Z": np.zeros((64, 64), dtype=np.float32)}

    result = guide.generate(conditioning=conditioning, n_production_frames=97)

    assert isinstance(result, ForwardGuideResult)
    assert result.phase_valid is True
    assert validate_frame_count_phase(result.frames.shape[0])


def test_forward_guide_with_preroll() -> None:
    """ForwardGuide with n_preroll=4 must phase-lock the total frame count."""
    guide = ForwardGuide(LoopConfig(n_preroll=4, kappa=1.0))
    conditioning = {"WorldNormal.X": np.zeros((32, 32), dtype=np.float32)}

    result = guide.generate(conditioning=conditioning, n_production_frames=93)

    assert result.phase_valid is True
    assert result.preroll_count == 4
    total = result.frames.shape[0]
    assert validate_frame_count_phase(total)


def test_forward_guide_prevents_light_leak() -> None:
    """Production frames must not contain the preroll's 5500 K warm colour bias.

    The mock generator adds a warm tint [+0.08, +0.02, -0.04] to the first
    8 preroll frames.  After κ = 1.0 re-anchoring, the last preroll frame
    (index n_preroll - 1) must be replaced with the (neutral) first
    production frame, so it should NOT carry the warm tint.
    """
    guide = ForwardGuide(LoopConfig(n_preroll=8, kappa=1.0))
    conditioning = {"CameraLinearDepth.Z": np.zeros((16, 16), dtype=np.float32)}

    result = guide.generate(conditioning=conditioning, n_production_frames=89)

    # The re-anchored last preroll frame (index 7) should now equal the
    # production first frame (index 8), which is neutral grey (0.5).
    last_preroll = result.frames[7]
    production_first = result.frames[8]
    np.testing.assert_array_equal(last_preroll, production_first)

    # Verify that the production first frame is close to neutral 0.5 (no warm tint).
    mean_rgb = production_first.mean(axis=(0, 1))
    assert abs(mean_rgb[0] - 0.5) < 0.01, f"Red channel biased: {mean_rgb[0]}"
    assert abs(mean_rgb[2] - 0.5) < 0.01, f"Blue channel biased: {mean_rgb[2]}"


def test_forward_guide_invalid_preroll_raises() -> None:
    """n_preroll=3 via generate must raise ContractValidationError."""
    with pytest.raises(ContractValidationError) as exc_info:
        ForwardGuide(LoopConfig(n_preroll=3))
    assert exc_info.value.exit_code == 10


# ---------------------------------------------------------------------------
# _spatial_dims layout detection (H6-P1-04)
# ---------------------------------------------------------------------------


def test_spatial_dims_4d_non_square() -> None:
    """4-D [T, H, W, C] conditioning: _spatial_dims returns (H, W) not (T, H).

    The frame count T=17 must NOT be misread as image height H.
    """
    conditioning = {
        "CameraLinearDepth.Z": np.zeros((17, 64, 96, 3), dtype=np.float32),
    }
    h, w = ForwardGuide._spatial_dims(conditioning)
    assert (h, w) == (64, 96), f"Expected (64, 96), got ({h}, {w})"


def test_spatial_dims_3d_channels_last() -> None:
    """3-D [H, W, C] conditioning (channels-last): returns (H, W)."""
    conditioning = {
        "WorldNormal.X": np.zeros((128, 64, 3), dtype=np.float32),
    }
    h, w = ForwardGuide._spatial_dims(conditioning)
    assert (h, w) == (128, 64), f"Expected (128, 64), got ({h}, {w})"


def test_spatial_dims_3d_channels_first() -> None:
    """3-D [C, H, W] conditioning (channels-first): returns (H, W)."""
    conditioning = {
        "CameraLinearDepth.Z": np.zeros((1, 48, 72), dtype=np.float32),
    }
    h, w = ForwardGuide._spatial_dims(conditioning)
    assert (h, w) == (48, 72), f"Expected (48, 72), got ({h}, {w})"


def test_spatial_dims_2d() -> None:
    """2-D [H, W] single-channel conditioning: returns (H, W)."""
    conditioning = {
        "Mask": np.zeros((32, 48), dtype=np.float32),
    }
    h, w = ForwardGuide._spatial_dims(conditioning)
    assert (h, w) == (32, 48), f"Expected (32, 48), got ({h}, {w})"


def test_spatial_dims_empty_fallback() -> None:
    """Empty conditioning dict falls back to (64, 64)."""
    h, w = ForwardGuide._spatial_dims({})
    assert (h, w) == (64, 64)


def test_forward_guide_4d_conditioning_shape() -> None:
    """ForwardGuide with 4-D [T, H, W, C] conditioning produces correct spatial dims."""
    guide = ForwardGuide(LoopConfig(n_preroll=0))
    conditioning = {
        "CameraLinearDepth.Z": np.zeros((17, 64, 96, 3), dtype=np.float32),
    }
    result = guide.generate(conditioning=conditioning, n_production_frames=97)
    # frames should have the spatial dims from conditioning: (64, 96), not (17, 64)
    assert result.frames.shape[1] == 64, f"Height should be 64, got {result.frames.shape[1]}"
    assert result.frames.shape[2] == 96, f"Width should be 96, got {result.frames.shape[2]}"


# ---------------------------------------------------------------------------
# extract_main_take — preroll trimming with bounds checks
# ---------------------------------------------------------------------------


def test_extract_main_take_basic() -> None:
    """extract_main_take trims n_preroll frames from the start."""
    frames = np.arange(97 * 3).reshape(97, 1, 1, 3).astype(np.float32)
    result = extract_main_take(frames, n_preroll=4)
    assert result.shape[0] == 93
    np.testing.assert_array_equal(result[0], frames[4])


def test_extract_main_take_zero_preroll() -> None:
    """With n_preroll=0, extract_main_take returns the full sequence."""
    frames = np.zeros((97, 8, 8, 3), dtype=np.float32)
    result = extract_main_take(frames, n_preroll=0)
    assert result.shape[0] == 97
    np.testing.assert_array_equal(result, frames)


def test_extract_main_take_oob_raises() -> None:
    """n_preroll >= T must raise ContractValidationError, not silently return empty."""
    frames = np.zeros((10, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ContractValidationError) as exc_info:
        extract_main_take(frames, n_preroll=10)
    assert exc_info.value.exit_code == 10


def test_extract_main_take_exceeds_raises() -> None:
    """n_preroll > T must raise ContractValidationError."""
    frames = np.zeros((5, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ContractValidationError):
        extract_main_take(frames, n_preroll=100)


def test_extract_main_take_negative_raises() -> None:
    """Negative n_preroll must raise ContractValidationError."""
    frames = np.zeros((10, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ContractValidationError):
        extract_main_take(frames, n_preroll=-1)
