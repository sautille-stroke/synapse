"""End-to-end cross-repo verification for fulgur C++ FGPP exporter -> synapse.

Validates:
1. Always-on wire-contract fixture reproducing ``fulgur::fgpp_export::FgppPackager``
   on-disk layout (15 ASCII-sorted EXR channels, 34 B/px, ``Cryptomatte.ID`` UINT,
   ``CameraLinearDepth.Z`` FLOAT, 13 HALF channels, and ``metadata.json`` v1.0).
2. Real GPU-exported FGPP sequences produced by ``fulgur`` (auto-discovered at
   ``$SYNAPSE_FULGUR_FGPP_DIR``, ``/tmp/fgpp_102``, or
   ``data/golden/fgpp/course_102_pbr`` when available).
3. Full downstream conditioning via ``WanAdapter.prepare_conditions()``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from synapse.adapters.base import VideoGenerationConfig
from synapse.adapters.wan import WanAdapter
from synapse.fgpp.frame import FGPPSequence
from synapse.fgpp.io import import_fgpp
from synapse.fgpp.schema import (
    FGPP_BYTES_PER_PIXEL,
    FGPP_CHANNEL_SPECS,
    FGPP_NUM_CHANNELS,
    FGPP_SCHEMA_VERSION,
    PixelType,
    computed_bytes_per_pixel,
)


def _discover_fulgur_export_dir() -> Path | None:
    """Locate a real fulgur-exported FGPP directory if present on this host."""
    candidates: list[Path] = []
    env_dir = os.environ.get("SYNAPSE_FULGUR_FGPP_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/tmp/fgpp_102_crypto"),
            Path("/tmp/fgpp_102"),
            Path("/root/work/datasets/synapse/track_b_fulgur/course_102_pbr_17f"),
            Path("data/golden/fgpp/course_102_pbr"),
        ]
    )
    for cand in candidates:
        if (cand / "metadata.json").is_file() and (cand / "frame_000000.exr").is_file():
            return cand
    return None


def _write_cpp_layout_fgpp_package(out_dir: Path, num_frames: int = 3) -> Path:
    """Synthesise an FGPP directory matching fulgur's C++ tinyexr writer byte-for-byte.

    Specifically writes ``Cryptomatte.ID`` using native OpenEXR ``PixelType.UINT``
    (matching ``TINYEXR_PIXELTYPE_UINT`` in ``fgpp_packager.cc``) and writes the
    exact ``metadata.json`` schema emitted by ``FgppPackager::writeMetadata``.
    """
    Imath = pytest.importorskip("Imath")
    OpenEXR = pytest.importorskip("OpenEXR")

    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = 32, 48

    frames_meta: list[dict[str, object]] = []
    for idx in range(num_frames):
        header = OpenEXR.Header(w, h)
        header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

        rng = np.random.default_rng(1000 + idx)
        base_color = rng.uniform(0.02, 0.95, (h, w, 3)).astype(np.float16)
        depth = rng.uniform(1.5, 40.0, (h, w)).astype(np.float32)
        crypto = rng.integers(0, 2**32, (h, w), dtype=np.uint32)
        crypto[0, 0] = np.uint32(0xDEADBEEF)
        crypto[0, 1] = np.uint32(0xFFFFFFFF)
        radiance = rng.uniform(0.0, 4.0, (h, w, 3)).astype(np.float16)
        motion = rng.uniform(-0.05, 0.05, (h, w, 2)).astype(np.float16)
        rough_met = rng.uniform(0.04, 1.0, (h, w, 2)).astype(np.float16)

        raw_n = rng.standard_normal((h, w, 3)).astype(np.float32)
        raw_n /= np.maximum(np.linalg.norm(raw_n, axis=-1, keepdims=True), 1e-6)
        normals = raw_n.astype(np.float16)

        chan_types = {
            "BaseColor.B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "BaseColor.G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "BaseColor.R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "CameraLinearDepth.Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "Cryptomatte.ID": Imath.Channel(Imath.PixelType(Imath.PixelType.UINT)),
            "DirectRadiance.B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "DirectRadiance.G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "DirectRadiance.R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "MotionVector.X": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "MotionVector.Y": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "RoughnessMetallic.G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "RoughnessMetallic.R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "WorldNormal.X": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "WorldNormal.Y": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "WorldNormal.Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        }
        header["channels"] = chan_types

        chan_bytes = {
            "BaseColor.B": base_color[..., 2].tobytes(),
            "BaseColor.G": base_color[..., 1].tobytes(),
            "BaseColor.R": base_color[..., 0].tobytes(),
            "CameraLinearDepth.Z": depth.tobytes(),
            "Cryptomatte.ID": crypto.tobytes(),
            "DirectRadiance.B": radiance[..., 2].tobytes(),
            "DirectRadiance.G": radiance[..., 1].tobytes(),
            "DirectRadiance.R": radiance[..., 0].tobytes(),
            "MotionVector.X": motion[..., 0].tobytes(),
            "MotionVector.Y": motion[..., 1].tobytes(),
            "RoughnessMetallic.G": rough_met[..., 1].tobytes(),
            "RoughnessMetallic.R": rough_met[..., 0].tobytes(),
            "WorldNormal.X": normals[..., 0].tobytes(),
            "WorldNormal.Y": normals[..., 1].tobytes(),
            "WorldNormal.Z": normals[..., 2].tobytes(),
        }

        exr_path = out_dir / f"frame_{idx:06d}.exr"
        out_file = OpenEXR.OutputFile(str(exr_path), header)
        out_file.writePixels(chan_bytes)
        out_file.close()

        eye4 = np.eye(4, dtype=np.float64)
        frames_meta.append(
            {
                "frame_index": idx,
                "timestamp_sec": idx / 24.0,
                "is_physical_anchor": (idx % 48 == 0),
                "fov_y_deg": 60.0,
                "file": f"frame_{idx:06d}.exr",
                "view_matrix_col_major": eye4.T.reshape(-1).tolist(),
                "proj_matrix_col_major": eye4.T.reshape(-1).tolist(),
                "camera": {
                    "fx": 41.569,
                    "fy": 41.569,
                    "cx": w / 2.0,
                    "cy": h / 2.0,
                    "width": w,
                    "height": h,
                    "fov_degrees": 60.0,
                    "focal_length_mm": 35.0,
                    "sensor_width_mm": 36.0,
                    "sensor_height_mm": 20.25,
                    "z_near": 0.1,
                    "z_far": 1000.0,
                    "aperture_fstop": 0.0,
                    "shutter_angle_deg": 180.0,
                    "view_matrix": eye4.tolist(),
                    "world_matrix": eye4.tolist(),
                },
                "channels": [
                    "depth",
                    "direct_radiance",
                    "world_normal",
                    "motion_vector",
                    "base_color",
                    "roughness_metallic",
                    "cryptomatte_id",
                ],
            }
        )

    metadata = {
        "schema_version": FGPP_SCHEMA_VERSION,
        "schema": FGPP_SCHEMA_VERSION,
        "generator": "fulgur",
        "generator_version": "0.1.0",
        "scene_id": "fulgur_export",
        "width": w,
        "height": h,
        "resolution": [w, h],
        "total_frames": num_frames,
        "num_frames": num_frames,
        "fps": 24.0,
        "z_near_meters": 0.1,
        "z_far_meters": 1000.0,
        "anchor_interval_frames": 48,
        "coordinate_system": {
            "handedness": "right",
            "up": "+Y",
            "forward": "-Z",
            "units": "meters",
        },
        "matrix_layout": "column_major",
        "channels": {
            "count": FGPP_NUM_CHANNELS,
            "bytes_per_pixel": FGPP_BYTES_PER_PIXEL,
        },
        "frames": frames_meta,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return out_dir


def _assert_fgpp_package_valid(pkg_dir: Path) -> FGPPSequence:
    """Shared 15-channel + metadata + WanAdapter verification suite."""
    OpenEXR = pytest.importorskip("OpenEXR")

    meta = json.loads((pkg_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta.get("schema_version") == FGPP_SCHEMA_VERSION
    assert meta.get("schema") == FGPP_SCHEMA_VERSION
    assert meta["channels"]["count"] == 15
    assert meta["channels"]["bytes_per_pixel"] == 34
    assert computed_bytes_per_pixel() == 34
    assert meta["coordinate_system"]["handedness"] == "right"
    assert meta["matrix_layout"] == "column_major"

    # Inspect raw OpenEXR header channel names and native pixel types on frame 0
    exr = OpenEXR.InputFile(str(pkg_dir / "frame_000000.exr"))
    exr_channels = exr.header()["channels"]
    exr.close()
    expected_names = [name for name, _, _ in FGPP_CHANNEL_SPECS]
    assert sorted(exr_channels.keys()) == expected_names
    for name, expected_pt, _ in FGPP_CHANNEL_SPECS:
        actual_type_v = int(exr_channels[name].type.v)
        assert actual_type_v == int(expected_pt), (
            f"Channel {name}: expected pixel_type={int(expected_pt)}, "
            f"got {actual_type_v}"
        )

    seq = import_fgpp(pkg_dir, lazy=False)
    assert isinstance(seq, FGPPSequence)
    assert seq.num_frames >= 1

    frame = seq.frames[0]
    assert frame.base_color is not None
    assert frame.depth is not None
    assert frame.world_normal is not None
    assert frame.motion_vector is not None
    assert frame.roughness_metallic is not None
    assert frame.direct_radiance is not None
    assert frame.cryptomatte_id is not None

    # 1. Depth: linear metres, finite, strictly positive
    depth = frame.depth
    assert depth.dtype == np.float32
    assert np.all(np.isfinite(depth))
    assert float(np.min(depth)) > 0.0
    foreground_depth = depth[depth < frame.camera.z_far * 0.99]
    if foreground_depth.size > 0:
        assert float(np.median(foreground_depth)) < 100.0

    # 2. World normals: unit length on foreground geometry
    normals = frame.world_normal.astype(np.float32)
    lengths = np.linalg.norm(normals, axis=-1)
    fg_mask = lengths > 0.1
    assert np.any(fg_mask), "Expected non-empty foreground normals"
    np.testing.assert_allclose(lengths[fg_mask], 1.0, atol=0.02)

    # 3. Cryptomatte: exact uint32 dtype
    assert frame.cryptomatte_id.dtype == np.uint32
    assert frame.cryptomatte_id.shape == (frame.height, frame.width)
    assert PixelType.UINT32 == 0

    # 4. BaseColor: linear [0, 1]
    bc = frame.base_color.astype(np.float32)
    assert float(np.min(bc)) >= -1e-3
    assert float(np.max(bc)) <= 1.0 + 2e-3

    # 5. RoughnessMetallic: [0, 1]^2
    rm = frame.roughness_metallic.astype(np.float32)
    assert float(np.min(rm)) >= -1e-3
    assert float(np.max(rm)) <= 1.0 + 2e-3

    # 6. DirectRadiance: finite non-negative HDR
    rad = frame.direct_radiance.astype(np.float32)
    assert np.all(np.isfinite(rad))
    assert float(np.min(rad)) >= -1e-3

    # 7. Downstream WanAdapter conditioning tensor preparation
    adapter = WanAdapter()
    cfg = VideoGenerationConfig(
        width=frame.width,
        height=frame.height,
        num_frames=seq.num_frames,
        fps=seq.fps,
    )
    payload = adapter.prepare_conditions(seq, cfg)
    assert payload.control_tensor is not None
    assert payload.control_tensor.shape == (
        7,
        seq.num_frames,
        frame.height,
        frame.width,
    )
    assert np.all(np.isfinite(payload.control_tensor))
    return seq


class TestFgppFulgurWireParity:
    """Always-on verification against the exact C++ tinyexr wire layout."""

    def test_cpp_uint32_exr_and_metadata_roundtrip(self, tmp_path: Path) -> None:
        pkg_dir = _write_cpp_layout_fgpp_package(tmp_path / "cpp_fgpp", num_frames=3)
        seq = _assert_fgpp_package_valid(pkg_dir)
        assert seq.num_frames == 3
        crypto = seq.frames[0].cryptomatte_id
        assert crypto is not None
        assert int(crypto[0, 0]) == 0xDEADBEEF
        assert int(crypto[0, 1]) == 0xFFFFFFFF


class TestFgppFulgurRealGpuExport:
    """End-to-end verification against real fulgur GPU-rendered EXR sequences."""

    def test_real_fulgur_course102_export(self) -> None:
        fulgur_dir = _discover_fulgur_export_dir()
        if fulgur_dir is None:
            pytest.skip(
                "Real fulgur FGPP export not found at $SYNAPSE_FULGUR_FGPP_DIR, "
                "/tmp/fgpp_102, or data/golden/fgpp/course_102_pbr"
            )
        seq = _assert_fgpp_package_valid(fulgur_dir)
        # Verify real rendered scene has non-trivial foreground geometry, lighting, and Cryptomatte IDs
        f0 = seq.frames[0]
        assert f0.direct_radiance is not None
        assert float(np.max(f0.direct_radiance.astype(np.float32))) > 0.01
        assert float(np.std(f0.depth)) > 0.05
        assert f0.cryptomatte_id is not None
        assert len(np.unique(f0.cryptomatte_id)) >= 2
