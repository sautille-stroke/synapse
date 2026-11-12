"""Tests for synapse.style — ACEScg color science, extractor, and profile mapper."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from synapse.style.aces_color import (
    acescg_to_linear_srgb,
    ciede2000_delta_e,
    linear_srgb_to_acescg,
    linear_to_srgb,
    rgb_to_cielab,
    soft_knee_inverse_tonemap,
    srgb_to_linear,
)
from synapse.style.descriptor import StyleDescriptor
from synapse.style.extractor import (
    extract_style_fgpp_intrinsic,
    extract_style_optical,
)
from synapse.style.profile_mapper import (
    save_style_profile_yaml,
    to_engine_light_overrides,
    to_style_profile_dict,
)

# ===================================================================
# TestAcesColor
# ===================================================================


class TestAcesColor:
    """Tests for ACEScg colour science utilities."""

    def test_srgb_linear_roundtrip(self) -> None:
        """sRGB → linear → sRGB roundtrip error < 1e-6."""
        rng = np.random.default_rng(42)
        srgb = rng.uniform(0.0, 1.0, size=(100, 3))
        linear = srgb_to_linear(srgb)
        roundtrip = linear_to_srgb(linear)
        assert np.allclose(srgb, roundtrip, atol=1e-6), (
            f"Max roundtrip error: {np.max(np.abs(srgb - roundtrip))}"
        )

    def test_srgb_to_linear_known_values(self) -> None:
        """Check known sRGB → linear boundary values."""
        # sRGB 0 → linear 0
        assert srgb_to_linear(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-10)
        # sRGB 1 → linear 1
        assert srgb_to_linear(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-10)
        # sRGB at threshold 0.04045 should be continuous
        below = srgb_to_linear(np.array([0.04045]))
        above = srgb_to_linear(np.array([0.04046]))
        assert abs(float(above[0]) - float(below[0])) < 1e-4

    def test_acescg_roundtrip(self) -> None:
        """linear_srgb → ACEScg → linear_srgb roundtrip error < 1e-6."""
        rng = np.random.default_rng(123)
        rgb = rng.uniform(0.0, 1.0, size=(50, 3))
        acescg = linear_srgb_to_acescg(rgb)
        roundtrip = acescg_to_linear_srgb(acescg)
        assert np.allclose(rgb, roundtrip, atol=1e-6), (
            f"Max roundtrip error: {np.max(np.abs(rgb - roundtrip))}"
        )

    def test_acescg_preserves_white(self) -> None:
        """Pure white [1,1,1] maps close to [1,1,1] in ACEScg."""
        white = np.array([[1.0, 1.0, 1.0]])
        acescg_white = linear_srgb_to_acescg(white)
        assert np.allclose(acescg_white, [[1.0, 1.0, 1.0]], atol=0.01)

    def test_soft_knee_identity_below_threshold(self) -> None:
        """L <= 4.0 → identity (passthrough)."""
        values = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
        result = soft_knee_inverse_tonemap(values)
        np.testing.assert_allclose(result, values, atol=1e-12)

    def test_soft_knee_bounded(self) -> None:
        """Max output <= 12.0 for any input including extreme values."""
        values = np.array([0.0, 5.0, 10.0, 100.0, 1e6, 1e20])
        result = soft_knee_inverse_tonemap(values)
        assert np.all(result <= 12.0), f"Max output: {result.max()}"
        assert np.all(result >= 0.0)

    def test_soft_knee_c1_continuous(self) -> None:
        """Derivative is continuous at L = 4.0 (finite difference check)."""
        eps = 1e-7
        x = np.array([4.0 - eps, 4.0, 4.0 + eps])
        y = soft_knee_inverse_tonemap(x)
        # Left derivative
        d_left = (y[1] - y[0]) / eps
        # Right derivative
        d_right = (y[2] - y[1]) / eps
        # C1 continuity: left and right derivatives should match
        assert abs(d_left - d_right) < 1e-3, (
            f"Left deriv: {d_left}, Right deriv: {d_right}"
        )

    def test_soft_knee_no_nan(self) -> None:
        """No NaN/Inf for any finite input (including negatives)."""
        values = np.array([-100.0, -1.0, 0.0, 0.5, 4.0, 100.0, 1e10])
        result = soft_knee_inverse_tonemap(values)
        assert np.all(np.isfinite(result)), f"Non-finite values: {result}"

    def test_soft_knee_nan_inf_input(self) -> None:
        """NaN/Inf inputs are safely handled (mapped to 0)."""
        values = np.array([np.nan, np.inf, -np.inf])
        result = soft_knee_inverse_tonemap(values)
        assert np.all(np.isfinite(result))

    def test_ciede2000_identical(self) -> None:
        """ΔE = 0 for identical colours."""
        lab = np.array([[50.0, 20.0, -10.0]])
        de = ciede2000_delta_e(lab, lab)
        assert de[0] == pytest.approx(0.0, abs=1e-10)

    def test_ciede2000_known_pair(self) -> None:
        """Red vs green should have large ΔE."""
        red_linear = np.array([[1.0, 0.0, 0.0]])
        green_linear = np.array([[0.0, 1.0, 0.0]])
        lab_red = rgb_to_cielab(red_linear)
        lab_green = rgb_to_cielab(green_linear)
        de = ciede2000_delta_e(lab_red, lab_green)
        # Red vs green should be very perceptually different
        assert de[0] > 30.0, f"ΔE(red, green) = {de[0]}, expected > 30"

    def test_ciede2000_symmetry(self) -> None:
        """ΔE(a, b) == ΔE(b, a)."""
        lab1 = np.array([[50.0, 25.0, -15.0]])
        lab2 = np.array([[60.0, -10.0, 30.0]])
        de_ab = ciede2000_delta_e(lab1, lab2)
        de_ba = ciede2000_delta_e(lab2, lab1)
        assert de_ab[0] == pytest.approx(de_ba[0], abs=1e-10)

    def test_ciede2000_batch(self) -> None:
        """Vectorised batch computation produces correct shapes."""
        rng = np.random.default_rng(99)
        lab1 = rng.uniform(-50, 100, size=(20, 3))
        lab2 = rng.uniform(-50, 100, size=(20, 3))
        de = ciede2000_delta_e(lab1, lab2)
        assert de.shape == (20,)
        assert np.all(de >= 0)

    def test_rgb_to_cielab_white(self) -> None:
        """D65 white [1,1,1] should map to L≈100, a≈0, b≈0."""
        white = np.array([[1.0, 1.0, 1.0]])
        lab = rgb_to_cielab(white)
        assert lab[0, 0] == pytest.approx(100.0, abs=0.5)
        assert lab[0, 1] == pytest.approx(0.0, abs=1.0)
        assert lab[0, 2] == pytest.approx(0.0, abs=1.0)

    def test_rgb_to_cielab_black(self) -> None:
        """Black [0,0,0] should map to L≈0."""
        black = np.array([[0.0, 0.0, 0.0]])
        lab = rgb_to_cielab(black)
        assert lab[0, 0] == pytest.approx(0.0, abs=0.5)


# ===================================================================
# TestStyleExtractor
# ===================================================================


class TestStyleExtractor:
    """Tests for style extraction functions."""

    def _make_synthetic_frames(
        self,
        T: int = 4,
        H: int = 64,
        W: int = 64,
        base_color: tuple[int, int, int] = (128, 128, 128),
    ) -> np.ndarray:
        """Create synthetic sRGB frames for testing."""
        frames = np.full((T, H, W, 3), base_color, dtype=np.uint8)
        return frames

    def test_extract_optical_returns_descriptor(self) -> None:
        """extract_style_optical returns a StyleDescriptor instance."""
        frames = self._make_synthetic_frames()
        desc = extract_style_optical(frames)
        assert isinstance(desc, StyleDescriptor)

    def test_extract_optical_color_temp_range(self) -> None:
        """Estimated CCT should be in [2000, 20000]K for reasonable input."""
        frames = self._make_synthetic_frames()
        desc = extract_style_optical(frames)
        assert 2000.0 <= desc.color_temp_kelvin <= 20000.0

    def test_extract_optical_warm_frames(self) -> None:
        """Warm-toned frames should yield lower CCT than cold-toned."""
        warm = self._make_synthetic_frames(base_color=(200, 150, 100))
        cold = self._make_synthetic_frames(base_color=(100, 150, 200))
        desc_warm = extract_style_optical(warm)
        desc_cold = extract_style_optical(cold)
        # Warm tones → lower CCT, cold tones → higher CCT
        assert desc_warm.color_temp_kelvin < desc_cold.color_temp_kelvin

    def test_extract_optical_grain_nonnegative(self) -> None:
        """Grain sigma should always be non-negative."""
        frames = self._make_synthetic_frames()
        desc = extract_style_optical(frames)
        assert desc.grain_sigma >= 0.0

    def test_extract_fgpp_light_direction(self) -> None:
        """Known light direction (az=45°, el=30°) recovered within 8°."""
        T, H, W = 4, 32, 32

        # Ground truth light direction: azimuth 45°, elevation 30°
        az = math.radians(45.0)
        el = math.radians(30.0)
        light_dir_gt = np.array([
            math.cos(el) * math.cos(az),
            math.sin(el),
            math.cos(el) * math.sin(az),
        ], dtype=np.float64)
        light_dir_gt /= np.linalg.norm(light_dir_gt)

        # Create synthetic FGPP data
        from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

        rng = np.random.default_rng(777)
        frames_list = []
        srgb_frames = np.zeros((T, H, W, 3), dtype=np.uint8)

        for t in range(T):
            cam = CameraParams(
                fx=32.0, fy=32.0, cx=16.0, cy=16.0,
                width=W, height=H,
            )

            # Random normals on the hemisphere
            normals = rng.standard_normal((H, W, 3)).astype(np.float16)
            norms = np.linalg.norm(normals.astype(np.float64), axis=-1,
                                   keepdims=True)
            norms = np.clip(norms, 1e-6, None)
            normals = (normals.astype(np.float64) / norms).astype(np.float16)

            # Albedo
            base_color = rng.uniform(0.3, 0.8, (H, W, 3)).astype(np.float16)

            # Simulated radiance: albedo * (ambient + max(0, dot(n, L)))
            ambient = 0.2
            n_f64 = normals.astype(np.float64)
            ndotl = np.clip(
                n_f64[..., 0] * light_dir_gt[0]
                + n_f64[..., 1] * light_dir_gt[1]
                + n_f64[..., 2] * light_dir_gt[2],
                0.0, 1.0,
            )
            bc_f64 = base_color.astype(np.float64)
            radiance = bc_f64 * (ambient + ndotl[..., np.newaxis])

            # Convert to sRGB uint8
            from synapse.style.aces_color import linear_to_srgb
            srgb_float = linear_to_srgb(np.clip(radiance, 0, 1))
            srgb_frames[t] = np.clip(srgb_float * 255, 0, 255).astype(
                np.uint8,
            )

            crypto = np.ones((H, W), dtype=np.uint32)

            frame = FGPPFrame(
                camera=cam,
                depth=np.ones((H, W), dtype=np.float32),
                world_normal=normals,
                base_color=base_color,
                cryptomatte_id=crypto,
                frame_index=t,
            )
            frames_list.append(frame)

        seq = FGPPSequence(frames=frames_list, fps=24.0, scene_id="test")
        desc = extract_style_fgpp_intrinsic(srgb_frames, seq)

        # Check recovered light direction within 8° of ground truth
        recovered = np.array(desc.key_light_dir_world, dtype=np.float64)
        recovered /= np.linalg.norm(recovered) + 1e-12
        cos_angle = np.clip(np.dot(recovered, light_dir_gt), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        assert angle_deg < 8.0, (
            f"Light direction error: {angle_deg:.1f}° (expected < 8°)"
        )

    def test_extract_fgpp_ambient_ratio(self) -> None:
        """Ambient ratio should be in [0, 1]."""
        T, H, W = 2, 16, 16

        from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence

        frames_list = []
        for t in range(T):
            cam = CameraParams(
                fx=16.0, fy=16.0, cx=8.0, cy=8.0,
                width=W, height=H,
            )
            frame = FGPPFrame(
                camera=cam,
                depth=np.ones((H, W), dtype=np.float32),
                world_normal=np.full((H, W, 3), 0.577, dtype=np.float16),
                base_color=np.full((H, W, 3), 0.5, dtype=np.float16),
                cryptomatte_id=np.ones((H, W), dtype=np.uint32),
                frame_index=t,
            )
            frames_list.append(frame)

        srgb = np.full((T, H, W, 3), 128, dtype=np.uint8)
        seq = FGPPSequence(frames=frames_list, fps=24.0)
        desc = extract_style_fgpp_intrinsic(srgb, seq)
        assert 0.0 <= desc.ambient_ratio <= 1.0


# ===================================================================
# TestProfileMapper
# ===================================================================


class TestProfileMapper:
    """Tests for profile_mapper functions."""

    def test_to_profile_dict_schema_version(self) -> None:
        """Output dict has correct schema version string."""
        desc = StyleDescriptor()
        d = to_style_profile_dict(desc)
        assert d["schema_version"] == "style_profile.v2.0.0"

    def test_to_profile_dict_structure(self) -> None:
        """Output dict has expected top-level keys."""
        desc = StyleDescriptor(
            color_temp_kelvin=5000.0,
            exposure_ev=-0.5,
        )
        d = to_style_profile_dict(desc)
        assert "metadata" in d
        assert "shading_config" in d
        assert "post_process_config" in d
        assert d["metadata"]["working_colorspace"] == "ACEScg"
        assert d["post_process_config"]["color_temp_kelvin"] == 5000.0

    def test_save_profile_json_fallback(self) -> None:
        """save_style_profile_yaml produces a readable file (YAML or JSON)."""
        desc = StyleDescriptor(exposure_ev=1.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "style.yaml"
            result = save_style_profile_yaml(desc, path)
            # Result should exist
            assert result.exists()
            # Read back — might be YAML (if pyyaml installed) or JSON
            raw = result.read_text(encoding="utf-8")
            try:
                content = json.loads(raw)
            except json.JSONDecodeError:
                import yaml  # type: ignore[import-untyped]
                content = yaml.safe_load(raw)
            assert content["schema_version"] == "style_profile.v2.0.0"
            assert content["post_process_config"]["exposure_ev"] == 1.5

    def test_engine_light_overrides_keys(self) -> None:
        """Engine light overrides have expected keys."""
        desc = StyleDescriptor(
            key_light_dir_world=[0.5, 0.7, 0.5],
            ambient_ratio=0.3,
            exposure_ev=0.5,
        )
        overrides = to_engine_light_overrides(desc)
        assert "directional_light" in overrides
        assert "ambient_light" in overrides
        assert "exposure" in overrides
        assert overrides["directional_light"]["direction"] == [0.5, 0.7, 0.5]
        assert overrides["directional_light"]["intensity"] == pytest.approx(0.7)
        assert overrides["ambient_light"]["intensity"] == pytest.approx(0.3)
        assert overrides["exposure"]["ev_compensation"] == 0.5

    def test_engine_light_overrides_default(self) -> None:
        """Default StyleDescriptor produces valid overrides."""
        desc = StyleDescriptor()
        overrides = to_engine_light_overrides(desc)
        assert overrides["directional_light"]["intensity"] == pytest.approx(0.8)
        assert overrides["ambient_light"]["intensity"] == pytest.approx(0.2)
