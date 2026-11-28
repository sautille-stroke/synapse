"""Tests for the DiffusionRenderer neural inverse/forward rendering transport.

Covers:
- GBufferResult / RelitResult dataclass construction
- gbuffer_to_fgpp_channels mapping and validation
- fgpp_channels_to_gbuffer_inputs round-trip
- MockDiffusionRendererBackend inverse and forward modes
- FoundaryDiffusionRendererBackend raises when no endpoint
- DualTrackComparisonResult and compare_inverse_rendering_tracks
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.bridge.diffusion_renderer import (
    DiffRendererMode,
    DualTrackComparisonResult,
    FoundaryDiffusionRendererBackend,
    GBufferResult,
    MockDiffusionRendererBackend,
    RelitResult,
    compare_inverse_rendering_tracks,
    fgpp_channels_to_gbuffer_inputs,
    gbuffer_to_fgpp_channels,
)
from synapse.errors import BackendUnavailableError, ContractValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T, H, W = 4, 16, 16


def _make_input_video() -> np.ndarray:
    """Create a synthetic test video [T, H, W, 3] uint8."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)


def _make_gbuffer() -> GBufferResult:
    """Create a synthetic G-buffer for testing."""
    rng = np.random.RandomState(99)
    normals = np.zeros((T, H, W, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    return GBufferResult(
        depth=rng.rand(T, H, W).astype(np.float32) * 10.0,
        normals=normals,
        albedo=rng.rand(T, H, W, 3).astype(np.float32),
        roughness=np.full((T, H, W), 0.5, dtype=np.float32),
        metallic=np.zeros((T, H, W), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# DiffRendererMode enum
# ---------------------------------------------------------------------------


class TestDiffRendererMode:
    """Mode enum values."""

    def test_values(self) -> None:
        assert DiffRendererMode.INVERSE.value == "inverse"
        assert DiffRendererMode.FORWARD.value == "forward"


# ---------------------------------------------------------------------------
# GBufferResult / RelitResult
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Dataclass construction and field access."""

    def test_gbuffer_construction(self) -> None:
        gbuf = _make_gbuffer()
        assert gbuf.depth.shape == (T, H, W)
        assert gbuf.normals.shape == (T, H, W, 3)
        assert gbuf.albedo.shape == (T, H, W, 3)
        assert gbuf.roughness.shape == (T, H, W)
        assert gbuf.metallic.shape == (T, H, W)
        assert gbuf.metadata == {}

    def test_gbuffer_with_metadata(self) -> None:
        gbuf = GBufferResult(
            depth=np.zeros((1, 2, 2), dtype=np.float32),
            normals=np.zeros((1, 2, 2, 3), dtype=np.float32),
            albedo=np.zeros((1, 2, 2, 3), dtype=np.float32),
            roughness=np.zeros((1, 2, 2), dtype=np.float32),
            metallic=np.zeros((1, 2, 2), dtype=np.float32),
            metadata={"model": "cosmos-7b"},
        )
        assert gbuf.metadata["model"] == "cosmos-7b"

    def test_relit_result(self) -> None:
        result = RelitResult(
            frames_srgb_u8=np.zeros((T, H, W, 3), dtype=np.uint8),
            metadata={"light": "hdri_studio"},
        )
        assert result.frames_srgb_u8.shape == (T, H, W, 3)
        assert result.metadata["light"] == "hdri_studio"


# ---------------------------------------------------------------------------
# FGPP ↔ G-buffer mapping
# ---------------------------------------------------------------------------


class TestGBufferFGPPMapping:
    """G-buffer ↔ FGPP channel mapping functions."""

    def test_gbuffer_to_fgpp_channels(self) -> None:
        gbuf = _make_gbuffer()
        channels = gbuffer_to_fgpp_channels(gbuf)
        assert "depth" in channels
        assert "world_normal" in channels
        assert "base_color" in channels
        assert "roughness_metallic" in channels
        assert channels["depth"].shape == (T, H, W)
        assert channels["world_normal"].shape == (T, H, W, 3)
        assert channels["base_color"].shape == (T, H, W, 3)
        assert channels["roughness_metallic"].shape == (T, H, W, 2)
        # Verify roughness is first channel
        np.testing.assert_array_equal(
            channels["roughness_metallic"][..., 0], gbuf.roughness,
        )
        # Verify metallic is second channel
        np.testing.assert_array_equal(
            channels["roughness_metallic"][..., 1], gbuf.metallic,
        )

    def test_gbuffer_to_fgpp_validation_normals_mismatch(self) -> None:
        gbuf = _make_gbuffer()
        gbuf.normals = np.zeros((T, H + 1, W, 3), dtype=np.float32)
        with pytest.raises(ContractValidationError, match="normals shape"):
            gbuffer_to_fgpp_channels(gbuf)

    def test_gbuffer_to_fgpp_validation_roughness_mismatch(self) -> None:
        gbuf = _make_gbuffer()
        gbuf.roughness = np.zeros((T, H + 1, W), dtype=np.float32)
        with pytest.raises(ContractValidationError, match="roughness shape"):
            gbuffer_to_fgpp_channels(gbuf)

    def test_roundtrip_fgpp_mapping(self) -> None:
        """G-buffer → FGPP → G-buffer inputs preserves data."""
        gbuf = _make_gbuffer()
        fgpp = gbuffer_to_fgpp_channels(gbuf)
        back = fgpp_channels_to_gbuffer_inputs(
            depth=fgpp["depth"],
            world_normal=fgpp["world_normal"],
            base_color=fgpp["base_color"],
            roughness_metallic=fgpp["roughness_metallic"],
        )
        np.testing.assert_allclose(back["depth"], gbuf.depth, atol=1e-6)
        np.testing.assert_allclose(back["normals"], gbuf.normals, atol=1e-6)
        np.testing.assert_allclose(back["albedo"], gbuf.albedo, atol=1e-6)
        np.testing.assert_allclose(back["roughness"], gbuf.roughness, atol=1e-6)
        np.testing.assert_allclose(back["metallic"], gbuf.metallic, atol=1e-6)


# ---------------------------------------------------------------------------
# MockDiffusionRendererBackend
# ---------------------------------------------------------------------------


class TestMockBackend:
    """MockDiffusionRendererBackend produces deterministic results."""

    def test_always_available(self) -> None:
        assert MockDiffusionRendererBackend().is_available()

    def test_inverse_render_shapes(self) -> None:
        backend = MockDiffusionRendererBackend()
        video = _make_input_video()
        gbuf = backend.inverse_render(video)
        assert gbuf.depth.shape == (T, H, W)
        assert gbuf.normals.shape == (T, H, W, 3)
        assert gbuf.albedo.shape == (T, H, W, 3)
        assert gbuf.roughness.shape == (T, H, W)
        assert gbuf.metallic.shape == (T, H, W)

    def test_inverse_render_deterministic(self) -> None:
        backend = MockDiffusionRendererBackend()
        video = _make_input_video()
        g1 = backend.inverse_render(video)
        g2 = backend.inverse_render(video)
        np.testing.assert_array_equal(g1.depth, g2.depth)
        np.testing.assert_array_equal(g1.albedo, g2.albedo)

    def test_inverse_render_normals_front_facing(self) -> None:
        backend = MockDiffusionRendererBackend()
        video = _make_input_video()
        gbuf = backend.inverse_render(video)
        # All normals should be [0, 0, 1]
        np.testing.assert_array_equal(gbuf.normals[..., 0], 0.0)
        np.testing.assert_array_equal(gbuf.normals[..., 1], 0.0)
        np.testing.assert_array_equal(gbuf.normals[..., 2], 1.0)

    def test_inverse_render_metadata(self) -> None:
        backend = MockDiffusionRendererBackend()
        gbuf = backend.inverse_render(_make_input_video())
        assert gbuf.metadata["backend"] == "mock"
        assert gbuf.metadata["mode"] == "inverse"

    def test_forward_render_shapes(self) -> None:
        backend = MockDiffusionRendererBackend()
        gbuf = _make_gbuffer()
        relit = backend.forward_render(gbuf)
        assert relit.frames_srgb_u8.shape == (T, H, W, 3)
        assert relit.frames_srgb_u8.dtype == np.uint8

    def test_forward_render_metadata(self) -> None:
        backend = MockDiffusionRendererBackend()
        relit = backend.forward_render(_make_gbuffer())
        assert relit.metadata["backend"] == "mock"
        assert relit.metadata["mode"] == "forward"

    def test_forward_render_nonzero_output(self) -> None:
        """Forward render with non-zero albedo produces non-zero output."""
        backend = MockDiffusionRendererBackend()
        gbuf = _make_gbuffer()
        relit = backend.forward_render(gbuf)
        assert np.any(relit.frames_srgb_u8 > 0)

    def test_roundtrip_inverse_forward(self) -> None:
        """Inverse → forward produces valid uint8 output."""
        backend = MockDiffusionRendererBackend()
        video = _make_input_video()
        gbuf = backend.inverse_render(video)
        relit = backend.forward_render(gbuf)
        assert relit.frames_srgb_u8.shape == video.shape
        assert relit.frames_srgb_u8.dtype == np.uint8


# ---------------------------------------------------------------------------
# FoundaryDiffusionRendererBackend
# ---------------------------------------------------------------------------


class TestFoundaryBackend:
    """FoundaryDiffusionRendererBackend raises when unconnected."""

    def test_not_available(self) -> None:
        assert not FoundaryDiffusionRendererBackend().is_available()

    def test_inverse_raises(self) -> None:
        backend = FoundaryDiffusionRendererBackend()
        with pytest.raises(BackendUnavailableError):
            backend.inverse_render(_make_input_video())

    def test_forward_raises(self) -> None:
        backend = FoundaryDiffusionRendererBackend()
        with pytest.raises(BackendUnavailableError):
            backend.forward_render(_make_gbuffer())


# ---------------------------------------------------------------------------
# Dual-track comparison
# ---------------------------------------------------------------------------


class TestDualTrackComparison:
    """compare_inverse_rendering_tracks returns valid metrics."""

    def test_identical_tracks_perfect_score(self) -> None:
        """Identical inputs should give ~100 dB PSNR and 1.0 cosine sim."""
        gbuf = _make_gbuffer()
        result = compare_inverse_rendering_tracks(
            classical_albedo=gbuf.albedo,
            classical_normals=gbuf.normals,
            classical_depth=gbuf.depth,
            neural_gbuffer=gbuf,
        )
        assert result.albedo_psnr == 100.0
        assert abs(result.normal_cosine_sim - 1.0) < 1e-6
        assert result.depth_rmse < 1e-10
        assert result.agreement_score > 0.95

    def test_different_tracks_lower_score(self) -> None:
        """Different inputs should give lower agreement."""
        gbuf1 = _make_gbuffer()
        gbuf2 = _make_gbuffer()
        # Modify neural result
        gbuf2.albedo = np.ones_like(gbuf2.albedo) * 0.5
        gbuf2.depth = gbuf2.depth + 5.0

        result = compare_inverse_rendering_tracks(
            classical_albedo=gbuf1.albedo,
            classical_normals=gbuf1.normals,
            classical_depth=gbuf1.depth,
            neural_gbuffer=gbuf2,
        )
        assert result.albedo_psnr < 50.0
        assert result.depth_rmse > 1.0
        assert result.agreement_score < 0.9

    def test_result_dataclass_fields(self) -> None:
        result = DualTrackComparisonResult(
            albedo_psnr=35.0,
            normal_cosine_sim=0.95,
            depth_rmse=0.5,
            agreement_score=0.85,
        )
        assert result.albedo_psnr == 35.0
        assert result.metadata == {}
