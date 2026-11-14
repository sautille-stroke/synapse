"""Tests for synapse.fgpp.io and synapse.fgpp.torch_bridge (PR-03)."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.io import (
    resize_fgpp_frame,
    sanitize_fgpp_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(w: int = 64, h: int = 64) -> CameraParams:
    return CameraParams(
        fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0,
        width=w, height=h, fov_degrees=60.0,
        z_near=0.1, z_far=100.0,
    )


def _make_frame(w: int = 64, h: int = 64, idx: int = 0) -> FGPPFrame:
    rng = np.random.default_rng(42 + idx)
    cam = _make_camera(w, h)
    depth = rng.uniform(0.5, 50.0, (h, w)).astype(np.float32)
    normals = rng.standard_normal((h, w, 3)).astype(np.float16)
    # Normalise
    n32 = normals.astype(np.float32)
    norms = np.linalg.norm(n32, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-6, norms, 1.0)
    normals = (n32 / norms).astype(np.float16)

    return FGPPFrame(
        camera=cam,
        depth=depth,
        direct_radiance=rng.uniform(0, 2, (h, w, 3)).astype(np.float16),
        world_normal=normals,
        motion_vector=rng.uniform(-5, 5, (h, w, 2)).astype(np.float16),
        base_color=rng.uniform(0, 1, (h, w, 3)).astype(np.float16),
        roughness_metallic=rng.uniform(0, 1, (h, w, 2)).astype(np.float16),
        cryptomatte_id=rng.integers(0, 2**32, (h, w), dtype=np.uint32),
        frame_index=idx,
    )


def _make_sequence(n: int = 3, w: int = 64, h: int = 64) -> FGPPSequence:
    return FGPPSequence(
        frames=[_make_frame(w, h, i) for i in range(n)],
        fps=24.0,
        scene_id="test_scene",
    )


# ===================================================================
# TestSanitizer
# ===================================================================


class TestSanitizer:
    def test_sanitize_nan_depth(self) -> None:
        frame = _make_frame()
        frame.depth[0, 0] = np.nan
        frame.depth[0, 1] = np.nan
        out = sanitize_fgpp_frame(frame)
        assert np.isfinite(out.depth[0, 0])
        assert out.depth[0, 0] == frame.camera.z_far

    def test_sanitize_inf_depth(self) -> None:
        frame = _make_frame()
        frame.depth[0, 0] = np.inf
        frame.depth[1, 0] = -np.inf
        out = sanitize_fgpp_frame(frame)
        assert out.depth[0, 0] == frame.camera.z_far
        assert out.depth[1, 0] == frame.camera.z_far

    def test_sanitize_negative_depth(self) -> None:
        frame = _make_frame()
        frame.depth[2, 2] = -1.0
        frame.depth[3, 3] = 0.0
        out = sanitize_fgpp_frame(frame)
        assert out.depth[2, 2] == frame.camera.z_far
        assert out.depth[3, 3] == frame.camera.z_far

    def test_sanitize_cryptomatte_cleared(self) -> None:
        frame = _make_frame()
        frame.depth[0, 0] = np.nan
        assert frame.cryptomatte_id is not None
        frame.cryptomatte_id[0, 0] = np.uint32(0xDEADBEEF)
        out = sanitize_fgpp_frame(frame)
        assert out.cryptomatte_id is not None
        assert out.cryptomatte_id[0, 0] == 0

    def test_sanitize_normal_renormalized(self) -> None:
        frame = _make_frame()
        assert frame.world_normal is not None
        # Set a non-unit normal
        frame.world_normal[5, 5] = np.array([2.0, 0.0, 0.0], dtype=np.float16)
        out = sanitize_fgpp_frame(frame)
        assert out.world_normal is not None
        n = out.world_normal[5, 5].astype(np.float32)
        length = np.linalg.norm(n)
        assert abs(length - 1.0) < 0.02  # float16 precision

    def test_sanitize_zero_normal(self) -> None:
        frame = _make_frame()
        assert frame.world_normal is not None
        frame.world_normal[6, 6] = np.array([0.0, 0.0, 0.0], dtype=np.float16)
        out = sanitize_fgpp_frame(frame)
        assert out.world_normal is not None
        n = out.world_normal[6, 6].astype(np.float32)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-3)

    def test_sanitize_motion_vector_nan(self) -> None:
        frame = _make_frame()
        assert frame.motion_vector is not None
        frame.motion_vector[0, 0] = np.array([np.inf, np.nan], dtype=np.float16)
        out = sanitize_fgpp_frame(frame)
        assert out.motion_vector is not None
        assert np.all(np.isfinite(out.motion_vector[0, 0].astype(np.float32)))

    def test_sanitize_returns_new_frame(self) -> None:
        frame = _make_frame()
        frame.depth[0, 0] = np.nan
        out = sanitize_fgpp_frame(frame)
        # Original should NOT be mutated (except the nan we set)
        assert np.isnan(frame.depth[0, 0])
        assert not np.isnan(out.depth[0, 0])


# ===================================================================
# TestResize
# ===================================================================


class TestResize:
    def test_resize_downscale(self) -> None:
        frame = _make_frame(64, 64)
        out = resize_fgpp_frame(frame, (32, 32))
        assert out.depth.shape == (32, 32)
        assert out.camera.width == 32
        assert out.camera.height == 32
        if out.world_normal is not None:
            assert out.world_normal.shape == (32, 32, 3)
        if out.cryptomatte_id is not None:
            assert out.cryptomatte_id.shape == (32, 32)

    def test_resize_cryptomatte_nearest(self) -> None:
        """Verify no interpolated IDs appear after resize."""
        frame = _make_frame(8, 8)
        assert frame.cryptomatte_id is not None
        # Fill with two distinct IDs in a checkerboard
        frame.cryptomatte_id[:] = np.uint32(100)
        frame.cryptomatte_id[0::2, 0::2] = np.uint32(200)
        frame.cryptomatte_id[1::2, 1::2] = np.uint32(200)
        out = resize_fgpp_frame(frame, (4, 4))
        assert out.cryptomatte_id is not None
        unique = set(out.cryptomatte_id.ravel().tolist())
        # Only 100 and 200 should exist — no interpolated values
        assert unique.issubset({100, 200})

    def test_resize_camera_updated(self) -> None:
        frame = _make_frame(64, 64)
        out = resize_fgpp_frame(frame, (32, 32))
        assert out.camera.width == 32
        assert out.camera.height == 32
        assert abs(out.camera.cx - frame.camera.cx * 0.5) < 1e-6
        assert abs(out.camera.cy - frame.camera.cy * 0.5) < 1e-6


# ===================================================================
# TestExportImport (requires OpenEXR)
# ===================================================================


class TestExportImport:
    def test_roundtrip_basic(self) -> None:
        pytest.importorskip("OpenEXR")
        from synapse.fgpp.io import export_fgpp, import_fgpp

        seq = _make_sequence(2, 32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_fgpp(seq, tmpdir)
            loaded = import_fgpp(tmpdir, lazy=False)
            assert isinstance(loaded, FGPPSequence)
            assert loaded.num_frames == 2
            # Depth should round-trip exactly (float32)
            np.testing.assert_array_equal(
                loaded.frames[0].depth, seq.frames[0].depth
            )

    def test_roundtrip_cryptomatte_exact(self) -> None:
        """Large uint32 IDs (>2^24) must survive EXR round-trip exactly."""
        pytest.importorskip("OpenEXR")
        from synapse.fgpp.io import export_fgpp, import_fgpp

        seq = _make_sequence(1, 16, 16)
        assert seq.frames[0].cryptomatte_id is not None
        # Plant specific large IDs
        seq.frames[0].cryptomatte_id[0, 0] = np.uint32(0xDEADBEEF)
        seq.frames[0].cryptomatte_id[0, 1] = np.uint32(0xFFFFFFFF)
        seq.frames[0].cryptomatte_id[0, 2] = np.uint32(0x80000001)
        seq.frames[0].cryptomatte_id[1, 0] = np.uint32(0)
        seq.frames[0].cryptomatte_id[1, 1] = np.uint32(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_fgpp(seq, tmpdir)
            loaded = import_fgpp(tmpdir, lazy=False)
            assert isinstance(loaded, FGPPSequence)
            gt = seq.frames[0].cryptomatte_id
            out = loaded.frames[0].cryptomatte_id
            assert out is not None
            np.testing.assert_array_equal(out, gt)

    def test_roundtrip_half_precision(self) -> None:
        pytest.importorskip("OpenEXR")
        from synapse.fgpp.io import export_fgpp, import_fgpp

        seq = _make_sequence(1, 16, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_fgpp(seq, tmpdir)
            loaded = import_fgpp(tmpdir, lazy=False)
            assert isinstance(loaded, FGPPSequence)
            assert loaded.frames[0].base_color is not None
            assert seq.frames[0].base_color is not None
            max_err = np.max(np.abs(
                loaded.frames[0].base_color.astype(np.float32)
                - seq.frames[0].base_color.astype(np.float32)
            ))
            assert max_err <= 5e-4

    def test_lazy_import_yields_frames(self) -> None:
        pytest.importorskip("OpenEXR")
        from synapse.fgpp.io import export_fgpp, import_fgpp

        seq = _make_sequence(3, 16, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_fgpp(seq, tmpdir)
            gen = import_fgpp(tmpdir, lazy=True)
            frames = list(gen)
            assert len(frames) == 3
            assert all(isinstance(f, FGPPFrame) for f in frames)

    def test_selective_channels(self) -> None:
        pytest.importorskip("OpenEXR")
        from synapse.fgpp.io import export_fgpp, import_fgpp

        seq = _make_sequence(1, 16, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_fgpp(seq, tmpdir)
            loaded = import_fgpp(tmpdir, lazy=False, channels=["depth"])
            assert isinstance(loaded, FGPPSequence)
            f = loaded.frames[0]
            assert f.depth is not None
            assert f.world_normal is None
            assert f.base_color is None


# ===================================================================
# TestTorchBridge (requires torch)
# ===================================================================


class TestTorchBridge:
    def test_frame_to_tensors_keys(self) -> None:
        pytest.importorskip("torch")
        from synapse.fgpp.torch_bridge import frame_to_tensors

        frame = _make_frame()
        tensors = frame_to_tensors(frame)
        expected = {"depth", "direct_radiance", "world_normal",
                    "motion_vector", "base_color", "roughness_metallic",
                    "cryptomatte_id"}
        assert set(tensors.keys()) == expected

    def test_frame_to_tensors_zero_copy(self) -> None:
        pytest.importorskip("torch")
        from synapse.fgpp.torch_bridge import frame_to_tensors

        frame = _make_frame()
        tensors = frame_to_tensors(frame, device="cpu")
        # Modify tensor and check numpy array reflects it
        depth_np = np.ascontiguousarray(frame.depth)
        depth_t = tensors["depth"]
        # They share storage (from_numpy zero-copy)
        assert depth_t.data_ptr() == depth_np.ctypes.data

    def test_cryptomatte_int32_roundtrip(self) -> None:
        pytest.importorskip("torch")
        from synapse.fgpp.torch_bridge import frame_to_tensors, tensors_to_frame

        frame = _make_frame()
        assert frame.cryptomatte_id is not None
        frame.cryptomatte_id[0, 0] = np.uint32(0xDEADBEEF)
        frame.cryptomatte_id[0, 1] = np.uint32(0xFFFFFFFF)

        tensors = frame_to_tensors(frame)
        out = tensors_to_frame(tensors, frame.camera, frame.frame_index)
        assert out.cryptomatte_id is not None
        assert out.cryptomatte_id[0, 0] == 0xDEADBEEF
        assert out.cryptomatte_id[0, 1] == 0xFFFFFFFF

    def test_tensors_to_frame_roundtrip(self) -> None:
        pytest.importorskip("torch")
        from synapse.fgpp.torch_bridge import frame_to_tensors, tensors_to_frame

        frame = _make_frame()
        tensors = frame_to_tensors(frame)
        out = tensors_to_frame(tensors, frame.camera, frame.frame_index)
        np.testing.assert_array_equal(out.depth, frame.depth)
        assert out.world_normal is not None and frame.world_normal is not None
        np.testing.assert_array_equal(out.world_normal, frame.world_normal)
