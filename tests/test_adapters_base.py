"""Tests for synapse.adapters — registry and MockVideoAdapter.

Covers:
  * Plugin registry (register / create / list).
  * MockVideoAdapter deterministic output, shapes, and ranges.
  * Defect injection (silhouette shift, flicker, kelvin shift).
"""

from __future__ import annotations

import numpy as np
import pytest

from synapse.adapters import (
    MockVideoAdapter,
    VideoGenerationConfig,
    create_adapter,
    list_adapters,
)
from synapse.adapters.mock import DefectInjectionConfig
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_W, _H, _N = 32, 32, 5


def _make_test_sequence(
    n_frames: int = _N,
    w: int = _W,
    h: int = _H,
    entity_id: int = 42,
) -> FGPPSequence:
    """Build a minimal ``FGPPSequence`` suitable for adapter tests."""
    frames: list[FGPPFrame] = []
    for i in range(n_frames):
        cam = CameraParams(
            fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, width=w, height=h,
        )
        depth = np.full((h, w), 5.0, dtype=np.float32)

        normal = np.zeros((h, w, 3), dtype=np.float16)
        normal[..., 2] = 1.0  # facing camera

        bc = np.full((h, w, 3), 0.5, dtype=np.float16)
        dr = np.full((h, w, 3), 1.0, dtype=np.float16)
        crypto = np.full((h, w), entity_id, dtype=np.uint32)

        frames.append(
            FGPPFrame(
                camera=cam,
                depth=depth,
                world_normal=normal,
                base_color=bc,
                direct_radiance=dr,
                cryptomatte_id=crypto,
                frame_index=i,
            )
        )
    return FGPPSequence(frames=frames, fps=24.0, scene_id="test")


def _default_config(**overrides: object) -> VideoGenerationConfig:
    """Return a ``VideoGenerationConfig`` with small defaults for speed."""
    kw: dict = {
        "seed": 42,
        "fps": 24.0,
        "width": _W,
        "height": _H,
        "num_frames": _N,
    }
    kw.update(overrides)
    return VideoGenerationConfig(**kw)


# ===================================================================
# TestRegistry
# ===================================================================


class TestRegistry:
    """Plugin registry smoke tests."""

    def test_mock_registered(self) -> None:
        """``'mock'`` must be present in the adapter list."""
        assert "mock" in list_adapters()

    def test_create_mock_adapter(self) -> None:
        """``create_adapter('mock')`` returns a ``MockVideoAdapter``."""
        adapter = create_adapter("mock")
        assert isinstance(adapter, MockVideoAdapter)
        assert adapter.adapter_id == "mock"
        assert adapter.tier == "mock"

    def test_create_unknown_raises(self) -> None:
        """Unknown adapter name must raise ``KeyError``."""
        with pytest.raises(KeyError, match="nonexistent"):
            create_adapter("nonexistent")

    def test_list_adapters_nonempty(self) -> None:
        """Registry must contain at least one adapter."""
        adapters = list_adapters()
        assert len(adapters) > 0
        assert isinstance(adapters, list)
        assert all(isinstance(a, str) for a in adapters)


# ===================================================================
# TestMockAdapter
# ===================================================================


class TestMockAdapter:
    """Core MockVideoAdapter behaviour tests."""

    def test_deterministic_output(self) -> None:
        """Same seed → bit-identical output."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()

        cfg = _default_config(seed=123)
        cond = adapter.prepare_conditions(seq, cfg)
        r1 = adapter.generate(cond, cfg)

        cond2 = adapter.prepare_conditions(seq, cfg)
        r2 = adapter.generate(cond2, cfg)

        np.testing.assert_array_equal(r1.frames_srgb_u8, r2.frames_srgb_u8)

    def test_different_seed_different(self) -> None:
        """Different seeds must produce different outputs."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()

        cond_a = adapter.prepare_conditions(seq, _default_config(seed=1))
        r_a = adapter.generate(cond_a, _default_config(seed=1))

        cond_b = adapter.prepare_conditions(seq, _default_config(seed=999))
        r_b = adapter.generate(cond_b, _default_config(seed=999))

        assert not np.array_equal(r_a.frames_srgb_u8, r_b.frames_srgb_u8)

    def test_output_shape(self) -> None:
        """Output shape must be ``[T, H, W, 3]`` uint8."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()
        cfg = _default_config()
        cond = adapter.prepare_conditions(seq, cfg)
        result = adapter.generate(cond, cfg)

        assert result.frames_srgb_u8.shape == (_N, _H, _W, 3)
        assert result.frames_srgb_u8.dtype == np.uint8

    def test_output_range(self) -> None:
        """All pixel values must lie in ``[0, 255]``."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()
        cfg = _default_config()
        cond = adapter.prepare_conditions(seq, cfg)
        result = adapter.generate(cond, cfg)

        assert result.frames_srgb_u8.min() >= 0
        assert result.frames_srgb_u8.max() <= 255

    def test_video_result_metadata(self) -> None:
        """``VideoResult`` must carry correct ``adapter_id`` and ``seed``."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()
        cfg = _default_config(seed=77)
        cond = adapter.prepare_conditions(seq, cfg)
        result = adapter.generate(cond, cfg)

        assert result.adapter_id == "mock"
        assert result.seed == 77
        assert result.fps == cfg.fps

    def test_prepare_conditions_channels(self) -> None:
        """``prepare_conditions`` must populate expected channel arrays."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()
        cfg = _default_config()
        cond = adapter.prepare_conditions(seq, cfg)

        assert cond.base_color is not None
        assert cond.base_color.shape == (_N, _H, _W, 3)
        assert cond.world_normal is not None
        assert cond.world_normal.shape == (_N, _H, _W, 3)
        assert cond.depth is not None
        assert cond.depth.shape == (_N, _H, _W)
        assert cond.cryptomatte_id is not None
        assert cond.cryptomatte_id.shape == (_N, _H, _W)

    def test_empty_sequence(self) -> None:
        """Empty ``FGPPSequence`` must produce an empty payload."""
        seq = FGPPSequence(frames=[], fps=24.0, scene_id="empty")
        adapter = MockVideoAdapter()
        cfg = _default_config()
        cond = adapter.prepare_conditions(seq, cfg)

        assert cond.base_color is None
        assert cond.depth is None


# ===================================================================
# TestDefectInjection
# ===================================================================


class TestDefectInjection:
    """Defect injection correctness tests."""

    def test_silhouette_shift(self) -> None:
        """Silhouette shift must move target entity pixels horizontally."""
        entity_id = 42
        shift_px = 12
        seq = _make_test_sequence(entity_id=entity_id)

        # Generate without defect
        adapter_clean = MockVideoAdapter()
        cfg = _default_config(seed=42)
        cond_clean = adapter_clean.prepare_conditions(seq, cfg)
        result_clean = adapter_clean.generate(cond_clean, cfg)

        # Generate with silhouette shift
        defect = DefectInjectionConfig(
            entity_id=entity_id,
            silhouette_shift_px=shift_px,
        )
        adapter_defect = MockVideoAdapter(defects=[defect])
        cond_defect = adapter_defect.prepare_conditions(seq, cfg)
        result_defect = adapter_defect.generate(cond_defect, cfg)

        # Outputs must differ
        assert not np.array_equal(
            result_clean.frames_srgb_u8,
            result_defect.frames_srgb_u8,
        )

        # Verify defect metadata is recorded
        assert len(result_defect.metadata["defects_applied"]) == 1
        assert (
            result_defect.metadata["defects_applied"][0][
                "silhouette_shift_px"
            ]
            == shift_px
        )

    def test_flicker_amplitude(self) -> None:
        """Flicker injection must cause frame-to-frame brightness changes."""
        seq = _make_test_sequence(n_frames=8)

        defect = DefectInjectionConfig(
            flicker_amplitude=0.3,
        )
        adapter = MockVideoAdapter(defects=[defect])
        cfg = _default_config(seed=42, num_frames=8)
        cond = adapter.prepare_conditions(seq, cfg)
        result = adapter.generate(cond, cfg)

        # Compute per-frame mean brightness
        means = result.frames_srgb_u8.astype(np.float64).mean(axis=(1, 2, 3))

        # With flicker, the mean brightness should vary across frames
        brightness_range = means.max() - means.min()
        assert brightness_range > 1.0, (
            f"Flicker should cause visible brightness variation, "
            f"but range was only {brightness_range:.4f}"
        )

    def test_kelvin_shift(self) -> None:
        """Colour temperature shift must change R/B channel ratio."""
        seq = _make_test_sequence()

        # Generate neutral
        adapter_neutral = MockVideoAdapter()
        cfg = _default_config(seed=42)
        cond_n = adapter_neutral.prepare_conditions(seq, cfg)
        result_neutral = adapter_neutral.generate(cond_n, cfg)

        # Generate with warm kelvin shift
        defect = DefectInjectionConfig(kelvin_shift=3000.0)
        adapter_warm = MockVideoAdapter(defects=[defect])
        cond_w = adapter_warm.prepare_conditions(seq, cfg)
        result_warm = adapter_warm.generate(cond_w, cfg)

        # Compute R/B ratios
        neutral_frames = result_neutral.frames_srgb_u8.astype(np.float64)
        warm_frames = result_warm.frames_srgb_u8.astype(np.float64)

        r_neutral = neutral_frames[..., 0].mean()
        b_neutral = neutral_frames[..., 2].mean()
        rb_ratio_neutral = r_neutral / max(b_neutral, 1e-9)

        r_warm = warm_frames[..., 0].mean()
        b_warm = warm_frames[..., 2].mean()
        rb_ratio_warm = r_warm / max(b_warm, 1e-9)

        # Warm shift should increase R/B ratio
        assert rb_ratio_warm > rb_ratio_neutral, (
            f"Warm kelvin shift should increase R/B ratio: "
            f"neutral={rb_ratio_neutral:.4f}, warm={rb_ratio_warm:.4f}"
        )

    def test_defect_frame_range(self) -> None:
        """Defect limited to a sub-range must leave other frames intact."""
        seq = _make_test_sequence(n_frames=8)

        defect = DefectInjectionConfig(
            frame_range=(2, 4),
            flicker_amplitude=0.5,
        )
        adapter_defect = MockVideoAdapter(defects=[defect])
        adapter_clean = MockVideoAdapter()

        cfg = _default_config(seed=42, num_frames=8)
        cond_d = adapter_defect.prepare_conditions(seq, cfg)
        result_d = adapter_defect.generate(cond_d, cfg)

        cond_c = adapter_clean.prepare_conditions(seq, cfg)
        result_c = adapter_clean.generate(cond_c, cfg)

        # Frame 0 should be identical (outside defect range)
        np.testing.assert_array_equal(
            result_d.frames_srgb_u8[0],
            result_c.frames_srgb_u8[0],
        )

        # Frame 3 (inside range) should differ
        assert not np.array_equal(
            result_d.frames_srgb_u8[3],
            result_c.frames_srgb_u8[3],
        )

    def test_no_defects_metadata(self) -> None:
        """With no defects, metadata should contain an empty list."""
        seq = _make_test_sequence()
        adapter = MockVideoAdapter()
        cfg = _default_config()
        cond = adapter.prepare_conditions(seq, cfg)
        result = adapter.generate(cond, cfg)

        assert result.metadata["defects_applied"] == []
