"""Tests for Traceboard instrumentation in adapters, style, and composite modules.

Verifies that Traceboard probe logging fires correctly when a mock probe
is active.  All tests are CPU-only and do not require GPU or real model
weights.

> **Operated by**: `opus4.6`
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from synapse.adapters.wan import WanAdapter

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_PATCH_TARGET = "synapse.observability.get_probe"


@pytest.fixture
def mock_probe() -> MagicMock:
    """Return a mock Traceboard probe with active=True."""
    probe = MagicMock()
    probe.active = True
    return probe


# ---------------------------------------------------------------------------
# 1. WanAdapter.generate() instrumentation
# ---------------------------------------------------------------------------


class TestWanAdapterTraceboard:
    """Verify Traceboard logging in WanAdapter.generate()."""

    def _make_adapter(self) -> WanAdapter:
        from synapse.adapters.wan import LocalCallableBackend, WanAdapter

        def fake_backend(payload: object, config: object) -> np.ndarray:
            num_frames = getattr(config, "num_frames", 4)
            height = getattr(config, "height", 32)
            width = getattr(config, "width", 32)
            # Non-zero frames to pass validate_output_non_degenerate
            rng = np.random.default_rng(99)
            return rng.integers(10, 240, (num_frames, height, width, 3), dtype=np.uint8)

        backend = LocalCallableBackend(pipeline_fn=fake_backend)
        return WanAdapter(backend=backend)

    def test_generate_logs_scalars(self, mock_probe: MagicMock) -> None:
        """WanAdapter.generate logs output dimensions and config."""
        from synapse.adapters.base import PreparedConditionPayload, VideoGenerationConfig

        adapter = self._make_adapter()
        config = VideoGenerationConfig(
            seed=42,
            fps=24.0,
            width=32,
            height=32,
            num_frames=4,
        )
        conditions = PreparedConditionPayload(
            control_tensor=np.zeros((7, 4, 32, 32), dtype=np.float32),
        )

        with patch(_PATCH_TARGET, return_value=mock_probe):
            result = adapter.generate(conditions, config)

        assert result.frames_srgb_u8.shape[0] == 4
        scalar_tags = [c.args[0] for c in mock_probe.log_scalar.call_args_list]
        assert "adapter/wan/num_output_frames" in scalar_tags
        assert "adapter/wan/guidance_scale" in scalar_tags
        assert "adapter/wan/gamma_flow_bias" in scalar_tags
        assert "adapter/wan/has_lora" in scalar_tags

    def test_generate_silent_without_probe(self) -> None:
        """Generate works when probe is None."""
        from synapse.adapters.base import PreparedConditionPayload, VideoGenerationConfig

        adapter = self._make_adapter()
        config = VideoGenerationConfig(
            seed=42,
            fps=24.0,
            width=32,
            height=32,
            num_frames=4,
        )
        conditions = PreparedConditionPayload(
            control_tensor=np.zeros((7, 4, 32, 32), dtype=np.float32),
        )

        with patch(_PATCH_TARGET, return_value=None):
            result = adapter.generate(conditions, config)

        assert result.frames_srgb_u8 is not None


# ---------------------------------------------------------------------------
# 2. Style extraction instrumentation
# ---------------------------------------------------------------------------


class TestStyleExtractorTraceboard:
    """Verify Traceboard logging in style extraction functions."""

    def test_extract_optical_logs_descriptors(self, mock_probe: MagicMock) -> None:
        """extract_style_optical logs CCT, EV, grain, contrast, saturation."""
        from synapse.style.extractor import extract_style_optical

        rng = np.random.default_rng(42)
        frames = rng.integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)

        with patch(_PATCH_TARGET, return_value=mock_probe):
            desc = extract_style_optical(frames)

        assert desc.color_temp_kelvin > 0
        scalar_tags = [c.args[0] for c in mock_probe.log_scalar.call_args_list]
        assert "style/optical/color_temp_kelvin" in scalar_tags
        assert "style/optical/exposure_ev" in scalar_tags
        assert "style/optical/grain_sigma" in scalar_tags
        assert "style/optical/contrast" in scalar_tags
        assert "style/optical/saturation" in scalar_tags
        assert "style/optical/num_sampled_frames" in scalar_tags

    def test_extract_optical_silent_without_probe(self) -> None:
        """Extraction works without active probe."""
        from synapse.style.extractor import extract_style_optical

        rng = np.random.default_rng(42)
        frames = rng.integers(0, 255, (2, 8, 8, 3), dtype=np.uint8)

        with patch(_PATCH_TARGET, return_value=None):
            desc = extract_style_optical(frames)

        assert desc.color_temp_kelvin > 0

    def test_extract_fgpp_logs_light_params(self, mock_probe: MagicMock) -> None:
        """extract_style_fgpp_intrinsic logs light direction and ambient."""
        from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
        from synapse.style.extractor import extract_style_fgpp_intrinsic

        rng = np.random.default_rng(42)
        h, w = 8, 8
        frames_srgb = rng.integers(50, 200, (2, h, w, 3), dtype=np.uint8)

        fgpp_frames = []
        for _ in range(2):
            cam = CameraParams(
                fx=500.0,
                fy=500.0,
                cx=float(w) / 2,
                cy=float(h) / 2,
                width=w,
                height=h,
                view_matrix=np.eye(4, dtype=np.float64),
                z_near=0.1,
                z_far=100.0,
            )
            fgpp_frames.append(
                FGPPFrame(
                    camera=cam,
                    depth=rng.random((h, w), dtype=np.float32) * 10.0 + 0.5,
                    world_normal=np.tile(
                        np.array([0.0, 1.0, 0.0], dtype=np.float32), (h, w, 1),
                    ),
                    base_color=rng.random((h, w, 3)).astype(np.float32) * 0.8 + 0.1,
                ),
            )
        seq = FGPPSequence(frames=fgpp_frames)

        with patch(_PATCH_TARGET, return_value=mock_probe):
            extract_style_fgpp_intrinsic(frames_srgb, seq)

        scalar_tags = [c.args[0] for c in mock_probe.log_scalar.call_args_list]
        assert "style/fgpp/num_light_samples" in scalar_tags
        # Optical metrics should also be present
        assert "style/optical/color_temp_kelvin" in scalar_tags


# ---------------------------------------------------------------------------
# 3. Composite surgical sequence instrumentation
# ---------------------------------------------------------------------------


class TestCompositeSurgicalTraceboard:
    """Verify Traceboard logging in composite_surgical_sequence."""

    def test_surgical_sequence_logs_metrics(self, mock_probe: MagicMock) -> None:
        """composite_surgical_sequence logs frame count and mask coverage."""
        from synapse.composite.blend import composite_surgical_sequence

        rng = np.random.default_rng(42)
        t, h, w = 3, 16, 16
        video = rng.integers(0, 255, (t, h, w, 3), dtype=np.uint8)
        rerendered = rng.integers(0, 255, (t, h, w, 3), dtype=np.uint8)
        masks = np.zeros((t, h, w), dtype=np.float32)
        masks[:, 4:12, 4:12] = 1.0

        with patch(_PATCH_TARGET, return_value=mock_probe):
            result = composite_surgical_sequence(video, rerendered, masks)

        assert result.shape == video.shape
        scalar_tags = [c.args[0] for c in mock_probe.log_scalar.call_args_list]
        assert "composite/surgical/num_frames" in scalar_tags
        assert "composite/surgical/height" in scalar_tags
        assert "composite/surgical/ema_alpha" in scalar_tags
        assert "composite/surgical/mean_mask_coverage" in scalar_tags

    def test_surgical_silent_without_probe(self) -> None:
        """composite_surgical_sequence works without probe."""
        from synapse.composite.blend import composite_surgical_sequence

        rng = np.random.default_rng(42)
        t, h, w = 2, 8, 8
        video = rng.integers(0, 255, (t, h, w, 3), dtype=np.uint8)
        rerendered = rng.integers(0, 255, (t, h, w, 3), dtype=np.uint8)
        masks = np.ones((t, h, w), dtype=np.float32)

        with patch(_PATCH_TARGET, return_value=None):
            result = composite_surgical_sequence(video, rerendered, masks)

        assert result.shape == video.shape
