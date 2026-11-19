"""Tests for FGPP wire format contract — PR-02.

Covers:
- schema.py: channel specs, constants, depth conversion, guards
- frame.py: CameraParams.projection_matrix, FGPPSequence save/load roundtrip
- wire_guard.py: internal consistency mode
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from synapse.errors import ContractValidationError
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.schema import (
    DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
    FGPP_BYTES_PER_PIXEL,
    FGPP_CHANNEL_SPECS,
    FGPP_NUM_CHANNELS,
    FGPP_SCHEMA_VERSION,
    MIN_CHARACTER_WHITEBOX_VERTICES,
    PixelType,
    computed_bytes_per_pixel,
    get_channel_names,
    get_logical_groups,
    is_physical_anchor_frame,
    linear_to_log_normalized_depth,
    log_normalized_to_linear_depth,
    validate_whitebox_coarseness_floor,
)

# =========================================================================
# Schema constants
# =========================================================================


class TestSchemaConstants:
    """Verify FGPP schema constants and channel definitions."""

    def test_channel_count(self) -> None:
        assert len(FGPP_CHANNEL_SPECS) == 15
        assert FGPP_NUM_CHANNELS == 15

    def test_channels_ascii_sorted(self) -> None:
        names = get_channel_names()
        assert names == sorted(names), "Channel names must be ASCII sorted"

    def test_bytes_per_pixel(self) -> None:
        assert FGPP_BYTES_PER_PIXEL == 34

    def test_bytes_per_pixel_computed(self) -> None:
        assert computed_bytes_per_pixel() == FGPP_BYTES_PER_PIXEL

    def test_schema_version(self) -> None:
        assert FGPP_SCHEMA_VERSION == "fgpp.v1.0"

    def test_pixel_type_values(self) -> None:
        assert PixelType.UINT32.value == 0
        assert PixelType.HALF.value == 1
        assert PixelType.FLOAT.value == 2

    def test_whitebox_constant(self) -> None:
        assert MIN_CHARACTER_WHITEBOX_VERTICES == 1500

    def test_anchor_interval_constant(self) -> None:
        assert DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES == 48

    def test_logical_groups(self) -> None:
        groups = get_logical_groups()
        assert len(groups) == 7
        assert set(groups.keys()) == {
            "base_color",
            "depth",
            "cryptomatte_id",
            "direct_radiance",
            "motion_vector",
            "roughness_metallic",
            "world_normal",
        }
        assert len(groups["base_color"]) == 3
        assert len(groups["depth"]) == 1
        assert len(groups["cryptomatte_id"]) == 1
        assert len(groups["direct_radiance"]) == 3
        assert len(groups["motion_vector"]) == 2
        assert len(groups["roughness_metallic"]) == 2
        assert len(groups["world_normal"]) == 3

    def test_channel_pixel_types(self) -> None:
        """Verify expected pixel type distribution."""
        type_counts = {pt: 0 for pt in PixelType}
        for _, pt, _ in FGPP_CHANNEL_SPECS:
            type_counts[pt] += 1
        assert type_counts[PixelType.HALF] == 13
        assert type_counts[PixelType.FLOAT] == 1
        assert type_counts[PixelType.UINT32] == 1


# =========================================================================
# Depth conversion
# =========================================================================


class TestDepthConversion:
    """Test log-normalised depth roundtrip and edge cases."""

    def test_roundtrip_arbitrary_depths(self) -> None:
        """Forward then inverse must match within 1e-12."""
        z_near, z_far = 0.1, 1000.0
        z_values = np.linspace(0.1, 1000.0, 500)
        normalised = linear_to_log_normalized_depth(z_values, z_near, z_far)
        recovered = log_normalized_to_linear_depth(normalised, z_near, z_far)
        assert np.max(np.abs(z_values - recovered)) < 1e-12

    def test_roundtrip_single_values(self) -> None:
        z_near, z_far = 0.5, 200.0
        for z in [0.5, 1.0, 10.0, 100.0, 200.0]:
            n = linear_to_log_normalized_depth(z, z_near, z_far)
            r = log_normalized_to_linear_depth(n, z_near, z_far)
            assert abs(float(r) - z) < 1e-12

    def test_near_maps_to_zero(self) -> None:
        result = linear_to_log_normalized_depth(0.1, 0.1, 1000.0)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_far_maps_to_one(self) -> None:
        result = linear_to_log_normalized_depth(1000.0, 0.1, 1000.0)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_z_near_clamped_if_non_positive(self) -> None:
        """z_near <= 0 should be clamped to epsilon, not crash."""
        result = linear_to_log_normalized_depth(1.0, -5.0, 100.0)
        assert np.all(np.isfinite(result))
        result2 = linear_to_log_normalized_depth(1.0, 0.0, 100.0)
        assert np.all(np.isfinite(result2))

    def test_z_far_clamped_if_lte_z_near(self) -> None:
        """z_far <= z_near should be clamped."""
        result = linear_to_log_normalized_depth(1.0, 5.0, 3.0)
        assert np.all(np.isfinite(result))

    def test_nan_input_clamped(self) -> None:
        z = np.array([1.0, float("nan"), 5.0])
        result = linear_to_log_normalized_depth(z, 0.1, 100.0)
        assert np.all(np.isfinite(result))

    def test_inf_input_clamped(self) -> None:
        z = np.array([float("inf"), 1.0, float("-inf")])
        result = linear_to_log_normalized_depth(z, 0.1, 100.0)
        assert np.all(np.isfinite(result))

    def test_output_range(self) -> None:
        z = np.array([0.001, 0.1, 10.0, 500.0, 1000.0, 5000.0])
        result = linear_to_log_normalized_depth(z, 0.1, 1000.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# =========================================================================
# Whitebox & anchor guards
# =========================================================================


class TestWhiteboxGuard:
    """Test whitebox coarseness floor validation."""

    def test_character_passes(self) -> None:
        assert validate_whitebox_coarseness_floor(1500, True, True) is True

    def test_character_above_minimum(self) -> None:
        assert validate_whitebox_coarseness_floor(5000, True, True) is True

    def test_character_below_minimum(self) -> None:
        assert validate_whitebox_coarseness_floor(1499, True, True) is False

    def test_character_no_limbs(self) -> None:
        assert validate_whitebox_coarseness_floor(2000, False, True) is False

    def test_non_character_always_passes(self) -> None:
        assert validate_whitebox_coarseness_floor(10, False, False) is True
        assert validate_whitebox_coarseness_floor(0, False, False) is True


class TestPhysicalAnchor:
    """Test physical anchor frame detection."""

    def test_anchor_frames(self) -> None:
        assert is_physical_anchor_frame(0) is True
        assert is_physical_anchor_frame(48) is True
        assert is_physical_anchor_frame(96) is True

    def test_non_anchor_frames(self) -> None:
        assert is_physical_anchor_frame(1) is False
        assert is_physical_anchor_frame(47) is False
        assert is_physical_anchor_frame(49) is False

    def test_custom_interval(self) -> None:
        assert is_physical_anchor_frame(10, interval=10) is True
        assert is_physical_anchor_frame(11, interval=10) is False


# =========================================================================
# CameraParams projection
# =========================================================================


class TestProjectionMatrix:
    """Test CameraParams.projection_matrix()."""

    def test_projection_shape(self) -> None:
        cam = CameraParams(fx=500, fy=500, cx=256, cy=256, width=512, height=512)
        proj = cam.projection_matrix()
        assert proj.shape == (4, 4)
        assert proj.dtype == np.float64

    def test_projection_properties(self) -> None:
        cam = CameraParams(
            fx=500,
            fy=500,
            cx=256,
            cy=256,
            width=1024,
            height=768,
            fov_degrees=60.0,
            z_near=0.1,
            z_far=100.0,
        )
        proj = cam.projection_matrix()
        # proj[3, 2] should be -1 for perspective
        assert proj[3, 2] == pytest.approx(-1.0)
        # proj[3, 3] should be 0
        assert proj[3, 3] == pytest.approx(0.0)
        # proj[0, 0] > 0 and proj[1, 1] > 0
        assert proj[0, 0] > 0
        assert proj[1, 1] > 0

    def test_new_camera_fields(self) -> None:
        cam = CameraParams(
            fx=500,
            fy=500,
            cx=256,
            cy=256,
            width=512,
            height=512,
            focal_length_mm=50.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
        )
        assert cam.focal_length_mm == 50.0
        assert cam.sensor_width_mm == 36.0
        assert cam.sensor_height_mm == 24.0


# =========================================================================
# Metadata roundtrip — save / load with column-major encoding
# =========================================================================


def _make_rotation_matrix(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Build a 3×3 rotation from pitch (X) then yaw (Y)."""
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    # R = Ry · Rx
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return ry @ rx


class TestMetadataRoundtrip:
    """Test FGPPSequence save_metadata / load_metadata roundtrip."""

    def test_basic_roundtrip(self, tmp_path: Any) -> None:
        """Non-trivial camera pose survives save→load with < 1e-12 error."""
        # Build a view matrix with translation [1.2, -3.4, 5.6] and rotation
        t = np.array([1.2, -3.4, 5.6], dtype=np.float64)
        r = _make_rotation_matrix(pitch_rad=0.3, yaw_rad=0.7)

        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = r
        view[:3, 3] = -r @ t  # world-to-camera translation
        world = np.linalg.inv(view)

        cam = CameraParams(
            fx=500,
            fy=500,
            cx=512,
            cy=384,
            width=1024,
            height=768,
            fov_degrees=55.0,
            view_matrix=view,
            world_matrix=world,
            z_near=0.1,
            z_far=500.0,
        )

        frames = [
            FGPPFrame(
                camera=cam,
                depth=np.ones((768, 1024), dtype=np.float32) * 3.0,
                frame_index=i,
            )
            for i in range(5)
        ]
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="roundtrip_test")

        meta_path = seq.save_metadata(tmp_path / "fgpp_rt")
        assert meta_path.exists()

        # Verify JSON schema version
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == "fgpp.v1.0"
        assert raw["total_frames"] == 5
        assert raw["resolution"] == [1024, 768]
        assert raw["anchor_interval_frames"] == 48

        # Load back
        loaded = FGPPSequence.load_metadata(tmp_path / "fgpp_rt")
        assert loaded.num_frames == 5
        assert loaded.fps == 24.0
        assert loaded.scene_id == "roundtrip_test"

        for orig, loaded_frame in zip(seq.frames, loaded.frames, strict=False):
            np.testing.assert_allclose(
                loaded_frame.camera.view_matrix,
                orig.camera.view_matrix,
                atol=1e-12,
            )
            assert loaded_frame.camera.fov_degrees == pytest.approx(55.0)
            assert loaded_frame.camera.z_near == pytest.approx(0.1)
            assert loaded_frame.camera.z_far == pytest.approx(500.0)

    def test_column_major_encoding(self, tmp_path: Any) -> None:
        """Verify column-major layout matches OpenGL/fulgur memory order."""
        # Simple known matrix
        view = np.array(
            [
                [1, 0, 0, 10],
                [0, 1, 0, 20],
                [0, 0, 1, 30],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        cam = CameraParams(
            fx=500,
            fy=500,
            cx=256,
            cy=256,
            width=512,
            height=512,
            view_matrix=view,
            world_matrix=np.linalg.inv(view),
        )
        frame = FGPPFrame(
            camera=cam,
            depth=np.ones((512, 512), dtype=np.float32),
            frame_index=0,
        )
        seq = FGPPSequence(frames=[frame], fps=24.0)
        seq.save_metadata(tmp_path / "col_test")

        raw = json.loads((tmp_path / "col_test" / "metadata.json").read_text())
        col_major = raw["frames"][0]["view_matrix_col_major"]

        # Column-major of [[1,0,0,10],[0,1,0,20],[0,0,1,30],[0,0,0,1]]
        # = columns flattened: [1,0,0,0, 0,1,0,0, 0,0,1,0, 10,20,30,1]
        expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]
        np.testing.assert_allclose(col_major, expected, atol=1e-14)

    def test_anchor_flags_in_metadata(self, tmp_path: Any) -> None:
        """Physical anchor flags are correct in saved metadata."""
        cam = CameraParams(fx=500, fy=500, cx=256, cy=256, width=512, height=512)
        frames = [
            FGPPFrame(
                camera=cam,
                depth=np.ones((512, 512), dtype=np.float32),
                frame_index=i,
            )
            for i in range(100)
        ]
        seq = FGPPSequence(frames=frames, fps=24.0)
        seq.save_metadata(tmp_path / "anchor_test")

        raw = json.loads((tmp_path / "anchor_test" / "metadata.json").read_text())
        for fr in raw["frames"]:
            idx = fr["frame_index"]
            expected_anchor = idx % 48 == 0
            assert fr["is_physical_anchor"] == expected_anchor, (
                f"Frame {idx}: expected anchor={expected_anchor}"
            )

    def test_invalid_rotation_det_raises(self, tmp_path: Any) -> None:
        """Loading a metadata.json with bad rotation det should raise."""
        # Create metadata with a bad view matrix (scale, not rotation)
        bad_meta = {
            "schema_version": "fgpp.v1.0",
            "scene_id": "bad",
            "fps": 24.0,
            "total_frames": 1,
            "resolution": [512, 512],
            "z_near_meters": 0.1,
            "z_far_meters": 100.0,
            "anchor_interval_frames": 48,
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "is_physical_anchor": True,
                    "fov_y_deg": 60.0,
                    # Scale matrix (det=8, not 1)
                    "view_matrix_col_major": [
                        2,
                        0,
                        0,
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        2,
                        0,
                        0,
                        0,
                        0,
                        1,
                    ],
                    "proj_matrix_col_major": list(np.eye(4).T.flatten()),
                    "channels": ["depth"],
                }
            ],
        }
        out_dir = tmp_path / "bad_rot"
        out_dir.mkdir()
        (out_dir / "metadata.json").write_text(json.dumps(bad_meta))

        with pytest.raises(ContractValidationError, match="determinant"):
            FGPPSequence.load_metadata(out_dir)

    def test_invalid_schema_version_raises(self, tmp_path: Any) -> None:
        """Loading a metadata.json with invalid schema_version must raise ContractValidationError."""
        bad_meta = {
            "schema_version": "fgpp.v0.9",
            "scene_id": "bad_ver",
            "fps": 24.0,
            "total_frames": 1,
            "resolution": [512, 512],
            "frames": [],
        }
        out_dir = tmp_path / "bad_ver"
        out_dir.mkdir()
        (out_dir / "metadata.json").write_text(json.dumps(bad_meta))

        with pytest.raises(ContractValidationError, match="schema_version"):
            FGPPSequence.load_metadata(out_dir)

    def test_flat_proj_matrix_intrinsics_reconstruction(self, tmp_path: Any) -> None:
        """Verify CameraParams reconstruction from flat proj_matrix_col_major."""
        w, h = 640, 480
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        near, far = 0.5, 200.0

        p00 = 2.0 * fx / w
        p11 = 2.0 * fy / h
        p02 = (2.0 * cx - w) / w
        p12 = (2.0 * cy - h) / h
        p22 = -(far + near) / (far - near)
        p23 = -2.0 * far * near / (far - near)

        proj_row = np.array(
            [
                [p00, 0.0, p02, 0.0],
                [0.0, p11, p12, 0.0],
                [0.0, 0.0, p22, p23],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

        proj_col = list(proj_row.T.flatten())
        view_col = list(np.eye(4, dtype=np.float64).T.flatten())

        meta = {
            "schema_version": "fgpp.v1.0",
            "scene_id": "flat_proj",
            "fps": 24.0,
            "total_frames": 1,
            "resolution": [w, h],
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "is_physical_anchor": True,
                    "proj_matrix_col_major": proj_col,
                    "view_matrix_col_major": view_col,
                }
            ],
        }
        out_dir = tmp_path / "flat_proj_test"
        out_dir.mkdir()
        (out_dir / "metadata.json").write_text(json.dumps(meta))

        seq = FGPPSequence.load_metadata(out_dir)
        cam = seq.frames[0].camera
        assert cam.width == w
        assert cam.height == h
        assert cam.fx == pytest.approx(fx, abs=1e-3)
        assert cam.fy == pytest.approx(fy, abs=1e-3)
        assert cam.cx == pytest.approx(cx, abs=1e-3)
        assert cam.cy == pytest.approx(cy, abs=1e-3)
        assert cam.z_near == pytest.approx(near, abs=1e-3)
        assert cam.z_far == pytest.approx(far, abs=1e-3)

    def test_schema_edge_cases_align_aether(self) -> None:
        """Verify is_physical_anchor_frame and depth conversion boundary edge cases."""
        assert is_physical_anchor_frame(0, interval=0) is True
        assert is_physical_anchor_frame(1, interval=0) is False
        assert is_physical_anchor_frame(0, interval=-1) is True

        # z_near <= 0 returns 0 safely
        res = linear_to_log_normalized_depth(np.array([1.0, 2.0]), z_near_m=0.0, z_far_m=10.0)
        np.testing.assert_allclose(res, 0.0)


# =========================================================================
# CameraParams backward compat
# =========================================================================


class TestCameraParamsBackwardCompat:
    """Existing CameraParams tests still pass with new fields."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        cam = CameraParams(
            fx=500,
            fy=500,
            cx=256,
            cy=256,
            width=512,
            height=512,
            fov_degrees=60.0,
            focal_length_mm=50.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
        )
        d = cam.to_dict()
        cam2 = CameraParams.from_dict(d)
        assert cam2.fx == cam.fx
        assert cam2.focal_length_mm == 50.0
        assert cam2.sensor_width_mm == 36.0
        assert cam2.sensor_height_mm == 24.0
        np.testing.assert_allclose(cam2.view_matrix, cam.view_matrix)

    def test_from_dict_defaults(self) -> None:
        """Old dicts without new fields should use defaults."""
        d = {
            "fx": 500,
            "fy": 500,
            "cx": 256,
            "cy": 256,
            "width": 512,
            "height": 512,
            "view_matrix": np.eye(4).tolist(),
            "world_matrix": np.eye(4).tolist(),
        }
        cam = CameraParams.from_dict(d)
        assert cam.focal_length_mm == 35.0
        assert cam.sensor_width_mm == 36.0
        assert cam.sensor_height_mm == 20.25


# =========================================================================
# Wire guard tool
# =========================================================================


@pytest.mark.slow
class TestWireGuard:
    """Test tools/wire_guard.py internal consistency mode."""

    def test_internal_consistency_passes(self) -> None:
        """wire_guard --check with no aether header should pass internal checks."""
        import importlib.util
        import io
        from contextlib import redirect_stdout

        wire_guard_path = Path(__file__).resolve().parent.parent / "tools" / "wire_guard.py"
        spec = importlib.util.spec_from_file_location("wire_guard", wire_guard_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            ret = mod.main(["--check", "--aether-header", "/nonexistent/path/fgpp_types.h"])
        assert ret == 0, f"wire_guard --check failed:\nstdout: {stdout_buf.getvalue()}"
        assert "All checks passed" in stdout_buf.getvalue()
