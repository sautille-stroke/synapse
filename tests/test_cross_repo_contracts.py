"""Cross-repository golden contract test suite.

Verifies end-to-end parity and contract compliance across:
1. @aether//fgpp <-> synapse.fgpp (15ch, 34 B/px, metadata JSON, eccentric principal point projection).
2. @aether//color <-> synapse.style.aces_color (D65, kSrgbToAcesCg, negative gamut, bounded soft-knee).
3. render-farm / grand-studio RFC 0039 JobBundle <-> synapse.pipeline.JobBundleWorker.
4. tools/data-inspector/native <-> Electron ShmFrameReader (64B Seqlock header, --shm-file mapping).
"""

from __future__ import annotations

import mmap
import os
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
from synapse.errors import AssetNotFoundError
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.io import export_fgpp, import_fgpp
from synapse.fgpp.schema import (
    FGPP_BYTES_PER_PIXEL,
    FGPP_CHANNEL_SPECS,
    FGPP_NUM_CHANNELS,
    MIN_CHARACTER_WHITEBOX_VERTICES,
    PixelType,
)
from synapse.pipeline.bundle_worker import (
    CLOSURE_TYPE_MAP,
    BundleManifest,
    ClosureType,
    _compute_sha256,
)
from synapse.style.aces_color import (
    _M_SRGB_TO_ACESCG,
    linear_srgb_to_acescg,
    linear_to_srgb,
    soft_knee_forward_tonemap,
    soft_knee_inverse_tonemap,
    srgb_to_linear,
)

# ---------------------------------------------------------------------------
# Path resolution & strict dependency helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_strict_deps() -> bool:
    """Return True if SYNAPSE_CI_STRICT_DEPS is active."""
    val = os.environ.get("SYNAPSE_CI_STRICT_DEPS", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _resolve_aether_root() -> Path | None:
    """Find the sibling @aether repository."""
    candidates = [
        _REPO_ROOT.parent / "aether",
        Path("/root/work/dervgen/aether"),
        Path("/Volumes/work/dervgen/aether"),
        Path("H:/dervgen/aether"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "fgpp").is_dir():
            return c
    return None


# ===========================================================================
# Contract 1: @aether//fgpp <-> synapse.fgpp
# ===========================================================================


class TestAetherFgppContract:
    """Cross-language parity between @aether//fgpp and synapse.fgpp."""

    def test_fgpp_types_header_parity(self) -> None:
        """Parse aether/fgpp/include/aether/fgpp/fgpp_types.h and verify 15ch spec."""
        aether_root = _resolve_aether_root()
        if aether_root is None or not (aether_root / "fgpp").exists():
            if _is_strict_deps():
                pytest.fail("SYNAPSE_CI_STRICT_DEPS=1: sibling @aether repo not found.")
            pytest.skip("Sibling @aether repo not found; skipping C++ header parity.")

        header_path = aether_root / "fgpp" / "include" / "aether" / "fgpp" / "fgpp_types.h"
        assert header_path.is_file(), f"fgpp_types.h missing at {header_path}"
        header_text = header_path.read_text(encoding="utf-8")

        # 1. Parse getAlphabeticalEXRChannelSpecs array
        specs_body = re.search(
            r"getAlphabeticalEXRChannelSpecs\b.*?\{\{\s*(.*?)\s*\}\}\s*;\s*\}",
            header_text,
            re.DOTALL,
        )
        assert specs_body is not None, "getAlphabeticalEXRChannelSpecs not found in header"
        cpp_specs = re.findall(
            r'\{\s*"([^"]+)"\s*,\s*[^,]+,\s*(\d+)u\s*,\s*PixelType::(\w+)\s*,\s*(true|false)\s*\}',
            specs_body.group(1),
        )
        assert len(cpp_specs) == 15, f"Expected 15 channel specs in C++, got {len(cpp_specs)}"
        assert len(cpp_specs) == FGPP_NUM_CHANNELS

        pixel_type_map = {
            "Uint32": PixelType.UINT32,
            "Half": PixelType.HALF,
            "Float": PixelType.FLOAT,
        }
        byte_size_map = {
            "Uint32": 4,
            "Half": 2,
            "Float": 4,
        }

        total_bytes = 0
        for i, (name, _sub_idx, ptype_str, _required) in enumerate(cpp_specs):
            py_name, py_ptype, _group = FGPP_CHANNEL_SPECS[i]
            assert name == py_name, f"Channel {i} name mismatch: C++ '{name}' != Python '{py_name}'"
            assert pixel_type_map[ptype_str] == py_ptype, (
                f"Channel {name} PixelType mismatch: C++ {ptype_str} != Python {py_ptype}"
            )
            total_bytes += byte_size_map[ptype_str]

        # 2. Verify total byte size per pixel == 34 B/px
        assert total_bytes == 34
        assert FGPP_BYTES_PER_PIXEL == total_bytes

        # 3. Verify min character whitebox vertices == 1500
        min_verts_match = re.search(r"kMinCharacterWhiteboxVertices\s*=\s*(\d+)", header_text)
        assert min_verts_match is not None, "kMinCharacterWhiteboxVertices not found"
        assert int(min_verts_match.group(1)) == MIN_CHARACTER_WHITEBOX_VERTICES

    def test_camera_projection_eccentric_principal_point_zero_mirror(self) -> None:
        """Eccentric principal point (cx != W/2) must not introduce horizontal mirror offset."""
        width = 832
        height = 480
        cx = 400.0  # W/2 would be 416.0
        cy = 250.0  # H/2 would be 240.0
        fx = 600.0
        fy = 600.0
        z_near = 0.1
        z_far = 100.0

        cam = CameraParams(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            z_near=z_near,
            z_far=z_far,
        )

        proj = cam.projection_matrix()
        # Verify p02 matches the corrected canonical formula (W - 2*cx) / W
        # Note: p02 uses (W - 2*cx)/W to fix horizontal mirror; p12 retains
        # (2*cy - H)/H because OpenGL Y-axis is upward while image cy is top-down.
        expected_p02 = (float(width) - 2.0 * cx) / float(width)
        expected_p12 = (2.0 * cy - float(height)) / float(height)
        assert abs(proj[0, 2] - expected_p02) < 1e-12
        assert abs(proj[1, 2] - expected_p12) < 1e-12

        # Verify exact intrinsics reconstruction from projection matrix elements
        recovered_cx = (1.0 - proj[0, 2]) * float(width) / 2.0
        recovered_cy = (proj[1, 2] + 1.0) * float(height) / 2.0
        assert abs(recovered_cx - cx) < 1e-12
        assert abs(recovered_cy - cy) < 1e-12

        # Verify right-handed view matrix determinant: R = eye(3), det(R) = +1
        rot = cam.view_matrix[:3, :3]
        det_r = float(np.linalg.det(rot))
        assert abs(det_r - 1.0) < 1e-6, f"Rotation matrix must have det=+1, got {det_r}"
        ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3)))
        assert ortho_err < 1e-6

    def test_fgpp_exr_15ch_bit_level_roundtrip(self) -> None:
        """15-channel OpenEXR write/read preserves bit-exact UINT32 Cryptomatte IDs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            w, h = 64, 48
            # Test values for Cryptomatte ID (including high bit set to verify unsigned 32-bit preservation)
            crypto_ids = np.array(
                [[0xDEADBEEF, 0xCAFEBABE, 0x80000001, 0x00000042]],
                dtype=np.uint32,
            )
            full_crypto = np.tile(crypto_ids, (h, w // 4))

            cam = CameraParams(
                fx=50.0,
                fy=50.0,
                cx=float(w) / 2.0,
                cy=float(h) / 2.0,
                width=w,
                height=h,
            )
            depth = np.full((h, w), 3.14159, dtype=np.float32)
            normal = np.zeros((h, w, 3), dtype=np.float16)
            normal[:, :, 2] = np.float16(1.0)
            radiance = np.full((h, w, 3), 0.75, dtype=np.float16)

            frame = FGPPFrame(
                camera=cam,
                depth=depth,
                world_normal=normal,
                direct_radiance=radiance,
                cryptomatte_id=full_crypto,
                frame_index=0,
            )
            seq = FGPPSequence(frames=[frame], fps=24.0, scene_id="contract_test")

            exr_path = tmp_path / "frame_000000.exr"
            meta_path = tmp_path / "metadata.json"

            # Export and re-import
            export_fgpp(seq, tmp_path)
            assert exr_path.is_file()
            assert meta_path.is_file()

            reloaded = import_fgpp(tmp_path)
            if isinstance(reloaded, FGPPSequence):
                reloaded_frames = reloaded.frames
            else:
                reloaded_frames = list(reloaded)
            assert len(reloaded_frames) == 1
            rf0 = reloaded_frames[0]

            # Verify UINT32 exact bit identity (no interpolation, no float casting)
            assert rf0.cryptomatte_id is not None
            assert rf0.cryptomatte_id.dtype == np.uint32
            assert np.array_equal(rf0.cryptomatte_id, full_crypto)

            # Verify depth float32 precision
            assert rf0.depth is not None
            assert np.allclose(rf0.depth, depth, atol=1e-5)


# ===========================================================================
# Contract 2: @aether//color <-> synapse.style.aces_color
# ===========================================================================


class TestAetherColorContract:
    """Color space math alignment between @aether//color and synapse.style."""

    def test_srgb_to_acescg_matrix_parity_with_aether(self) -> None:
        """Matrix kSrgbToAcesCg from aether/color/src/color_space.cc matches _M_SRGB_TO_ACESCG."""
        aether_root = _resolve_aether_root()
        if aether_root is None or not (aether_root / "color").exists():
            if _is_strict_deps():
                pytest.fail("SYNAPSE_CI_STRICT_DEPS=1: sibling @aether repo not found.")
            pytest.skip("Sibling @aether repo not found; skipping color_space.cc check.")

        color_space_cc = aether_root / "color" / "src" / "color_space.cc"
        assert color_space_cc.is_file(), f"color_space.cc missing at {color_space_cc}"
        cc_text = color_space_cc.read_text(encoding="utf-8")

        pattern = re.compile(
            r"kSrgbToAcesCg\s*=\s*Mat3\(\s*"
            r"Vec3\(([^,]+)f,\s*([^,]+)f,\s*([^)]+)f\),\s*"
            r"Vec3\(([^,]+)f,\s*([^,]+)f,\s*([^)]+)f\),\s*"
            r"Vec3\(([^,]+)f,\s*([^,]+)f,\s*([^)]+)f\)\s*\)",
            re.MULTILINE,
        )
        match = pattern.search(cc_text)
        assert match is not None, "kSrgbToAcesCg matrix definition not found in color_space.cc"

        cols = [float(match.group(i)) for i in range(1, 10)]
        cpp_matrix = np.array(
            [
                [cols[0], cols[3], cols[6]],
                [cols[1], cols[4], cols[7]],
                [cols[2], cols[5], cols[8]],
            ],
            dtype=np.float64,
        )

        # Compare with synapse _M_SRGB_TO_ACESCG
        diff = np.max(np.abs(cpp_matrix - _M_SRGB_TO_ACESCG))
        assert diff < 1e-6, f"kSrgbToAcesCg diff {diff} exceeds 1e-6"

    def test_d65_white_point_chromaticity(self) -> None:
        """D65 standard illuminant chromaticity (0.3127, 0.3290) tristimulus check."""
        d65_xyz = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
        sum_xyz = np.sum(d65_xyz)
        x = d65_xyz[0] / sum_xyz
        y = d65_xyz[1] / sum_xyz
        assert abs(x - 0.3127) < 5e-4
        assert abs(y - 0.3290) < 5e-4

        # D65 white sRGB [1, 1, 1] converts to ACEScg [1, 1, 1]
        white_srgb = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        white_aces = linear_srgb_to_acescg(white_srgb)
        assert np.allclose(white_aces, [1.0, 1.0, 1.0], atol=1e-3)

    def test_negative_gamut_linear_slope_and_symmetry(self) -> None:
        """sRGB <-> Linear handles negative wide-gamut excursions symmetrically without NaN."""
        neg_vals = np.array([-0.5, -0.01, -0.001, 0.0, 0.001, 0.01, 0.5], dtype=np.float64)
        lin = srgb_to_linear(neg_vals)
        assert np.all(np.isfinite(lin))
        # Odd symmetry: f(-x) == -f(x)
        assert np.allclose(lin[:3], -lin[-1:-4:-1], atol=1e-12)

        # Invert back
        re_srgb = linear_to_srgb(lin)
        assert np.allclose(re_srgb, neg_vals, atol=1e-12)

    def test_soft_knee_bounded_tonemap(self) -> None:
        """Forward soft-knee tonemap is strictly monotonic and bounded."""
        y_vals = np.array([0.0, 0.5, 0.8, 1.0, 2.0, 10.0, 100.0, 1000.0], dtype=np.float64)
        t_vals = soft_knee_forward_tonemap(y_vals)

        # Must be strictly finite and non-negative
        assert np.all(np.isfinite(t_vals))
        assert np.all(t_vals >= 0.0)

        # Strictly monotonic: t_vals[i+1] > t_vals[i]
        assert np.all(np.diff(t_vals) > 0.0)

        # Exact round-trip with soft_knee_inverse_tonemap
        recovered = soft_knee_inverse_tonemap(t_vals)
        assert np.allclose(recovered, y_vals, atol=1e-10)


# ===========================================================================
# Contract 3: render-farm / grand-studio RFC 0039 JobBundle
# ===========================================================================


class TestRenderFarmJobBundleContract:
    """RFC 0039 JobBundle deserialization and validation contract."""

    def test_rfc0039_job_bundle_creation_and_execution(self) -> None:
        """Parse upstream RFC 0039 JobBundle structure into BundleManifest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test_asset.txt"
            test_file.write_text("render_farm_payload", encoding="utf-8")
            file_sha256 = _compute_sha256(test_file)

            raw_bundle = {
                "job_id": "job-rf-2026-001",
                "project_name": "cinematic_vfx",
                "closure_type": ClosureType.NEURAL_VIDEO,
                "engine": "synapse.worker",
                "scene_spec": {"fgpp_path": "/data/fgpp", "fps": 24, "frames": 48},
                "asset_manifest": [
                    {
                        "path": str(test_file),
                        "sha256": file_sha256,
                        "byte_size": test_file.stat().st_size,
                    }
                ],
                "output_sink": {"result_video": str(tmp_path / "outputs" / "out.mp4")},
                "payload": {
                    "seed": 42,
                    "style_preset": "cinematic",
                },
            }

            # Deserialise via RFC 0039 adapter
            manifest = BundleManifest.from_rfc0039_job_bundle(raw_bundle)
            assert manifest.job_id == "job-rf-2026-001"
            assert manifest.stage == "neural_video_generate"
            assert manifest.inputs == {"fgpp_path": "/data/fgpp"}
            assert manifest.outputs == {"result_video": str(tmp_path / "outputs" / "out.mp4")}
            assert manifest.params["seed"] == 42
            assert manifest.params["style_preset"] == "cinematic"

    def test_rfc0039_job_bundle_asset_tamper_fails_code_20(self) -> None:
        """Tampering with an asset in asset_manifest must raise AssetNotFoundError (code 20)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "genuine_model.glb"
            test_file.write_bytes(b"\x00" * 128)

            # Provide a deliberately corrupted expected SHA-256
            tampered_bundle = {
                "job_id": "job-tampered-001",
                "closure_type": ClosureType.MESH_RECONSTRUCT,
                "asset_manifest": [
                    {
                        "path": str(test_file),
                        "sha256": "0" * 64,  # Bad hash
                    }
                ],
            }

            with pytest.raises(AssetNotFoundError) as exc_info:
                BundleManifest.from_rfc0039_job_bundle(tampered_bundle)

            assert exc_info.value.exit_code == 20
            assert "SHA-256 mismatch" in str(exc_info.value)

    def test_all_closure_types_mapped(self) -> None:
        """All 6 RFC 0039 ClosureType enum members must map to valid stages."""
        expected_closure_types = {
            ClosureType.NEURAL_VIDEO,
            ClosureType.SURGICAL_INPAINT,
            ClosureType.MESH_RECONSTRUCT,
            ClosureType.OPTICAL_POLISH,
            ClosureType.MOCAP_IK,
            ClosureType.DELIGHTING,
        }
        mapped_keys = set(CLOSURE_TYPE_MAP.keys())
        assert expected_closure_types.issubset(mapped_keys)
        for ctype, stage in CLOSURE_TYPE_MAP.items():
            assert isinstance(stage, str) and len(stage) > 0


# ===========================================================================
# Contract 4: tools/data-inspector/native <-> Electron ShmFrameReader
# ===========================================================================


class TestDataInspectorShmContract:
    """Shared-memory protocol contract between native C++ renderer and Electron."""

    INSPECTOR_SHM_MAGIC = 0x494E5350  # "INSP"
    INSPECTOR_SHM_VERSION = 1
    HEADER_SIZE = 64
    BUFFER_COUNT = 3

    def test_shm_header_seqlock_protocol_pack_and_parse(self) -> None:
        """Validate 64-byte InspectorShmHeader Seqlock packing and tear-free read."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            width = 64
            height = 64
            frame_bytes = width * height * 4
            total_size = self.HEADER_SIZE + self.BUFFER_COUNT * frame_bytes

            shm_file = tmp_path / "test_inspector_shm.bin"
            shm_file.write_bytes(b"\x00" * total_size)

            with open(shm_file, "r+b") as f:
                mm = mmap.mmap(f.fileno(), total_size)

                # 1. Write initial stable header (frameReady = 2, writeIdx = 0)
                magic = self.INSPECTOR_SHM_MAGIC
                version = self.INSPECTOR_SHM_VERSION
                write_idx = 0
                buf_count = self.BUFFER_COUNT
                frame_ready = 2  # Even -> Stable
                fps = 60.0
                delta_time = 0.01666
                debug_view = 2  # kNormal

                header_bin = struct.pack(
                    "<IIIIIIQffB23s",
                    magic,
                    version,
                    width,
                    height,
                    write_idx,
                    buf_count,
                    frame_ready,
                    fps,
                    delta_time,
                    debug_view,
                    b"\x00" * 23,
                )
                assert len(header_bin) == self.HEADER_SIZE
                mm[: self.HEADER_SIZE] = header_bin

                # 2. Write pixel data into buffer 0
                buf0_offset = self.HEADER_SIZE + write_idx * frame_bytes
                pixels0 = np.full((height, width, 4), 128, dtype=np.uint8)
                mm[buf0_offset : buf0_offset + frame_bytes] = pixels0.tobytes()

                # 3. Simulate reader
                h_magic, h_ver, h_w, h_h, h_widx, h_bc, h_fr, h_fps, h_dt, h_dv = (
                    struct.unpack_from("<IIIIIIQffB", mm, 0)
                )
                assert h_magic == self.INSPECTOR_SHM_MAGIC
                assert h_ver == 1
                assert h_w == width
                assert h_h == height
                assert h_fr % 2 == 0  # Stable

                read_offset = self.HEADER_SIZE + h_widx * frame_bytes
                read_pixels = np.frombuffer(
                    mm[read_offset : read_offset + frame_bytes], dtype=np.uint8
                ).reshape(h_h, h_w, 4)
                assert np.array_equal(read_pixels, pixels0)

                # 4. Simulate writer busy (frameReady = 3, odd)
                struct.pack_into("<Q", mm, 24, 3)
                _, _, _, _, _, _, h_fr_busy, _, _, _ = struct.unpack_from("<IIIIIIQffB", mm, 0)
                assert h_fr_busy % 2 != 0, "Odd sequence must indicate writer busy"

                mm.close()
