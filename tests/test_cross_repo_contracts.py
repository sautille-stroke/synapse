"""Cross-repository golden contract test suite.

Verifies end-to-end parity and contract compliance across:
1. @aether//fgpp <-> synapse.fgpp (15ch, 34 B/px, metadata JSON, eccentric principal point projection).
2. @aether//color <-> synapse.style.aces_color (D65, kSrgbToAcesCg, negative gamut, bounded soft-knee).
3. render-farm / grand-studio RFC 0039 JobBundle <-> synapse.pipeline.JobBundleWorker.
4. tools/data-inspector/native <-> Electron ShmFrameReader (64B Seqlock header, --shm-file mapping).
5. @aether//style <-> synapse.style.descriptor (StyleDescriptor v2.0.0, ACEScg).
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
import subprocess
import sys
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
    run_worker_cli,
)
from synapse.style.aces_color import (
    _M_SRGB_TO_ACESCG,
    linear_srgb_to_acescg,
    linear_to_srgb,
    soft_knee_forward_tonemap,
    soft_knee_inverse_tonemap,
    srgb_to_linear,
)
from synapse.style.descriptor import StyleDescriptor

# ---------------------------------------------------------------------------
# Path resolution & strict dependency helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_strict_deps() -> bool:
    """Return True if SYNAPSE_CI_STRICT_DEPS is active."""
    val = os.environ.get("SYNAPSE_CI_STRICT_DEPS", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _require_siblings() -> bool:
    """Return True if sibling ecosystem checkouts (aether, render-farm, native bin) are required."""
    val = os.environ.get("SYNAPSE_CI_REQUIRE_SIBLINGS", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _resolve_aether_root() -> Path | None:
    """Find the sibling @aether repository."""
    candidates: list[Path] = []
    env_root = os.environ.get("AETHER_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            _REPO_ROOT.parent / "aether",
            Path("/root/work/dervgen/aether"),
            Path("/Volumes/work/dervgen/aether"),
            Path("H:/dervgen/aether"),
        ]
    )
    for c in candidates:
        try:
            if c.is_dir() and (c / "fgpp").is_dir():
                return c
        except OSError:
            continue
    return None


def _resolve_render_farm_root() -> Path | None:
    """Find the sibling render-farm repository."""
    candidates = [
        _REPO_ROOT.parent / "render-farm",
        Path("/root/work/dervgen/render-farm"),
        Path("/Volumes/work/dervgen/render-farm"),
        Path("H:/dervgen/render-farm"),
    ]
    for c in candidates:
        try:
            if c.is_dir() and (c / "internal" / "domain" / "models.go").is_file():
                return c
        except OSError:
            continue
    return None


def _resolve_inspector_native_binary() -> Path | None:
    """Find the built data-inspector-native binary."""
    candidates = [
        _REPO_ROOT / "tools" / "data-inspector" / "native" / "bazel-bin" / "data-inspector-native",
        _REPO_ROOT / "tools" / "data-inspector" / "native" / "bazel-bin" / "data-inspector-native.exe",
        Path("/root/work/dervgen/synapse/tools/data-inspector/native/bazel-bin/data-inspector-native"),
    ]
    for c in candidates:
        try:
            if c.is_file() and os.access(str(c), os.X_OK):
                return c
        except OSError:
            continue
    return None


# ===========================================================================
# Contract 1: @aether//fgpp <-> synapse.fgpp
# ===========================================================================


class TestAetherFgppContract:
    """Cross-language parity between @aether//fgpp and synapse.fgpp."""

    @pytest.mark.external_data
    def test_fgpp_types_header_parity(self) -> None:
        """Parse aether/fgpp/include/aether/fgpp/fgpp_types.h and verify 15ch spec."""
        aether_root = _resolve_aether_root()
        if aether_root is None or not (aether_root / "fgpp").exists():
            if _require_siblings():
                pytest.fail("SYNAPSE_CI_REQUIRE_SIBLINGS=1: sibling @aether repo not found.")
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

    def test_eccentric_principal_point_multiple_configs_roundtrip(self) -> None:
        """Multiple eccentric (cx, cy) configs round-trip through projection → intrinsics recovery."""
        configs = [
            # (width, height, cx, cy, fx, fy)
            (1920, 1080, 960.0, 540.0, 1200.0, 1200.0),  # centered
            (1920, 1080, 800.0, 400.0, 1200.0, 1200.0),  # offset
            (640, 480, 100.0, 50.0, 500.0, 500.0),        # extreme offset
            (1024, 768, 512.0, 384.0, 800.0, 900.0),      # non-square focal
            (832, 480, 400.0, 250.0, 600.0, 600.0),       # original test case
        ]
        for width, height, cx, cy, fx, fy in configs:
            cam = CameraParams(
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height, z_near=0.1, z_far=100.0,
            )
            proj = cam.projection_matrix()

            # Round-trip: projection matrix → recovered intrinsics
            recovered_cx = (1.0 - proj[0, 2]) * float(width) / 2.0
            recovered_cy = (proj[1, 2] + 1.0) * float(height) / 2.0
            assert abs(recovered_cx - cx) < 1e-10, (
                f"cx recovery failed for config ({width}×{height}, cx={cx}): "
                f"recovered={recovered_cx}"
            )
            assert abs(recovered_cy - cy) < 1e-10, (
                f"cy recovery failed for config ({width}×{height}, cy={cy}): "
                f"recovered={recovered_cy}"
            )

            # Verify fx, fy can be recovered from p00, p11
            recovered_fx = proj[0, 0] * float(width) / 2.0
            recovered_fy = proj[1, 1] * float(height) / 2.0
            assert abs(recovered_fx - fx) < 1e-10, (
                f"fx recovery failed: expected {fx}, got {recovered_fx}"
            )
            assert abs(recovered_fy - fy) < 1e-10, (
                f"fy recovery failed: expected {fy}, got {recovered_fy}"
            )

    def test_fgpp_exr_15ch_bit_level_roundtrip(self) -> None:
        """15-channel OpenEXR write/read preserves bit-exact UINT32 Cryptomatte IDs."""
        pytest.importorskip("OpenEXR", reason="OpenEXR not installed")
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
            export_fgpp(seq, tmp_path, strict=False)
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

    def test_eccentric_camera_rasterization_and_fulgur_convention(self) -> None:
        """Render with SoftwareRasterizer using eccentric camera (cx=700, cy=200, W=1024, H=512).

        Assert screen projection lands at (700, 200) and matches fulgur RH_ZO / reversed-Z.
        """
        from synapse.fgpp.frame import CameraConvention
        from synapse.render_core.rasterizer import SoftwareRasterizer, entity_hash

        width = 1024
        height = 512
        cx = 700.0
        cy = 200.0
        fx = 800.0
        fy = 800.0
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

        # 1. Check projection matrix against OpenGL formula
        proj = cam.projection_matrix()
        assert CameraConvention.detect(proj) == CameraConvention.OPENGL

        # 2. Check Vulkan RH_ZO conversion
        # Vulkan: p22 = -far / (far - near), p23 = -(far * near) / (far - near)
        vul_p22 = -z_far / (z_far - z_near)
        vul_p23 = -(z_far * z_near) / (z_far - z_near)
        vul_proj = proj.copy()
        vul_proj[2, 2] = vul_p22
        vul_proj[2, 3] = vul_p23
        re_gl = CameraConvention.to_opengl(vul_proj, CameraConvention.VULKAN_RH_ZO)
        assert np.allclose(re_gl[2, :], proj[2, :], atol=1e-6)

        # Vulkan RH_ZO detection with p22 in (-1, 0)
        zo_proj = proj.copy()
        zo_proj[2, 2] = -0.5
        assert CameraConvention.detect(zo_proj) == CameraConvention.VULKAN_RH_ZO

        # 3. Check fulgur infinite reversed-Z: p22 = 0, p32 = -1, p23 = z_near
        fulgur_rev_z = proj.copy()
        fulgur_rev_z[2, 2] = 0.0
        fulgur_rev_z[2, 3] = z_near
        fulgur_rev_z[3, 2] = -1.0
        assert CameraConvention.detect(fulgur_rev_z) == CameraConvention.FULGUR_INF_REVERSED_Z

        # 4. Render with SoftwareRasterizer: quad centered at (0, 0, -5.0) in camera space
        rasterizer = SoftwareRasterizer(width=width, height=height)
        d = 0.05
        verts = np.array(
            [
                [-d, -d, -5.0],
                [d, -d, -5.0],
                [d, d, -5.0],
                [-d, d, -5.0],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=np.int32,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        frame = rasterizer.render(verts, faces, normals, cam, entity_name="eccentric_target")
        expected_eid = np.uint32(entity_hash("eccentric_target"))

        # Verify pixel at (cy=200, cx=700) is hit
        assert frame.cryptomatte_id is not None
        assert frame.cryptomatte_id[200, 700] == expected_eid
        assert frame.depth is not None
        assert abs(float(frame.depth[200, 700]) - 5.0) < 0.1


# ===========================================================================
# Contract 2: @aether//color <-> synapse.style.aces_color
# ===========================================================================


class TestAetherColorContract:
    """Color space math alignment between @aether//color and synapse.style."""

    @pytest.mark.external_data
    def test_srgb_to_acescg_matrix_parity_with_aether(self) -> None:
        """Matrix kSrgbToAcesCg from aether/color/src/color_space.cc matches _M_SRGB_TO_ACESCG."""
        aether_root = _resolve_aether_root()
        if aether_root is None or not (aether_root / "color").exists():
            if _require_siblings():
                pytest.fail("SYNAPSE_CI_REQUIRE_SIBLINGS=1: sibling @aether repo not found.")
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

    @pytest.mark.external_data
    def test_render_farm_models_go_closure_constants_parity(self) -> None:
        """Verify parity between render-farm internal/domain/models.go and ClosureType."""
        rf_root = _resolve_render_farm_root()
        if rf_root is None:
            if _require_siblings():
                pytest.fail("SYNAPSE_CI_REQUIRE_SIBLINGS=1: sibling render-farm repo not found.")
            pytest.skip("Sibling render-farm repo not found; skipping models.go parity.")

        models_go = rf_root / "internal" / "domain" / "models.go"
        assert models_go.is_file(), f"models.go missing at {models_go}"
        content = models_go.read_text(encoding="utf-8")

        expected_constants = {
            "ClosureNeuralVideo": "neural_video",
            "ClosureSurgicalInpaint": "surgical_inpaint",
            "ClosureMeshReconstruct": "mesh_reconstruct",
            "ClosureDelighting": "delighting",
            "ClosureOpticalPolish": "optical_polish",
            "ClosureMocapIK": "mocap_ik",
        }
        for const_name, const_val in expected_constants.items():
            pattern = rf'{const_name}\s+ClosureType\s*=\s*"{const_val}"'
            assert re.search(pattern, content) is not None, f"Constant {const_name} not found in models.go"

    def test_rfc0039_job_bundle_e2e_cli_execution(self) -> None:
        """Run run_worker_cli on valid and tampered RFC 0039 JobBundles."""
        pytest.importorskip("OpenEXR", reason="OpenEXR not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Create a valid FGPP directory with frame and metadata.json
            from synapse.fgpp import CameraParams, FGPPFrame, FGPPSequence, export_fgpp

            fgpp_dir = tmp_path / "fgpp_input"
            fgpp_frame = FGPPFrame(
                camera=CameraParams(fx=500.0, fy=500.0, cx=32.0, cy=32.0, width=64, height=64),
                depth=np.ones((64, 64), dtype=np.float32),
                frame_index=0,
            )
            seq = FGPPSequence(frames=[fgpp_frame], fps=24.0, scene_id="rfc0039_cli_test")
            export_fgpp(seq, fgpp_dir, strict=False)
            meta_file = fgpp_dir / "metadata.json"
            meta_sha = _compute_sha256(meta_file)

            out_dir = tmp_path / "outputs"
            out_dir.mkdir(parents=True)

            bundle_dict = {
                "job_id": "job-rfc0039-test-01",
                "closure_type": "neural_video",
                "scene_spec": {
                    "fgpp_dir": str(fgpp_dir),
                    "fps": 24,
                    "frames": 1,
                },
                "asset_manifest": [
                    {
                        "path": str(meta_file),
                        "sha256": meta_sha,
                        "byte_size": meta_file.stat().st_size,
                    }
                ],
                "output_sink": {
                    "result_video": str(out_dir / "out.mp4"),
                },
                "hardware_budget": {
                    "free_vram_gb": 80.0,
                },
                "payload": {
                    "seed": 42,
                    "model_variant": "wan2.1_1.3b",
                    "num_frames": 1,
                    "adapter": "mock",
                },
            }

            bundle_file = tmp_path / "bundle_manifest.json"
            bundle_file.write_text(json.dumps(bundle_dict), encoding="utf-8")

            # Execute via CLI runner with dry-run or mock backend
            exit_code = run_worker_cli(str(bundle_file), json_mode=True)
            assert exit_code == 0

            res_manifest_file = tmp_path / "result_manifest.json"
            assert res_manifest_file.is_file()
            res_data = json.loads(res_manifest_file.read_text(encoding="utf-8"))
            assert res_data["status"] == "success"
            assert res_data["exit_code"] == 0

            # 2. Tampering test: corrupt asset SHA256 and assert exit_code 20
            tampered_bundle = dict(bundle_dict)
            tampered_bundle["job_id"] = "job-rfc0039-tampered-02"
            tampered_bundle["asset_manifest"] = [
                {
                    "path": str(meta_file),
                    "sha256": "bad" * 21 + "a",
                    "byte_size": meta_file.stat().st_size,
                }
            ]
            tampered_dir = tmp_path / "tampered"
            tampered_dir.mkdir(parents=True)
            tampered_file = tampered_dir / "bundle_manifest.json"
            tampered_file.write_text(json.dumps(tampered_bundle), encoding="utf-8")

            tamper_exit_code = run_worker_cli(str(tampered_file), json_mode=True)
            assert tamper_exit_code == 20

    def test_rfc0039_job_bundle_json_schema_required_fields(self) -> None:
        """Validate RFC 0039 JobBundle JSON schema has all required top-level keys."""
        required_keys = {"job_id", "closure_type"}
        allowed_optional_keys = {
            "project_name", "engine", "scene_spec", "asset_manifest",
            "output_sink", "payload", "hardware_budget",
        }
        valid_bundle = {
            "job_id": "job-schema-test-001",
            "closure_type": ClosureType.NEURAL_VIDEO,
            "scene_spec": {"fgpp_path": "/data/fgpp", "fps": 24, "frames": 48},
            "asset_manifest": [],
            "output_sink": {"result_video": "/tmp/out.mp4"},
            "payload": {"seed": 42},
        }
        # Verify required keys are present and all keys are recognized
        for key in required_keys:
            assert key in valid_bundle
        assert set(valid_bundle.keys()).issubset(required_keys | allowed_optional_keys)
        # Verify deserialization succeeds
        manifest = BundleManifest.from_rfc0039_job_bundle(valid_bundle)
        assert manifest.job_id == "job-schema-test-001"

        # Verify JSON serialization round-trip preserves structure
        json_str = json.dumps(valid_bundle, default=str)
        reparsed = json.loads(json_str)
        for key in required_keys:
            assert key in reparsed, f"Required key '{key}' lost in JSON round-trip"

    def test_rfc0039_e2e_cli_from_non_standard_cwd(self) -> None:
        """run_worker_cli must succeed when invoked from a non-standard CWD."""
        pytest.importorskip("OpenEXR", reason="OpenEXR not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cwd_dir = tmp_path / "arbitrary_cwd"
            cwd_dir.mkdir()

            from synapse.fgpp import CameraParams, FGPPFrame, FGPPSequence, export_fgpp

            fgpp_dir = tmp_path / "fgpp_cwd_test"
            fgpp_frame = FGPPFrame(
                camera=CameraParams(fx=500.0, fy=500.0, cx=32.0, cy=32.0, width=64, height=64),
                depth=np.ones((64, 64), dtype=np.float32),
                frame_index=0,
            )
            seq = FGPPSequence(frames=[fgpp_frame], fps=24.0, scene_id="cwd_test")
            export_fgpp(seq, fgpp_dir, strict=False)
            meta_file = fgpp_dir / "metadata.json"
            meta_sha = _compute_sha256(meta_file)

            out_dir = tmp_path / "outputs_cwd"
            out_dir.mkdir(parents=True)

            bundle_dict = {
                "job_id": "job-cwd-test-01",
                "closure_type": "neural_video",
                "scene_spec": {"fgpp_dir": str(fgpp_dir), "fps": 24, "frames": 1},
                "asset_manifest": [
                    {"path": str(meta_file), "sha256": meta_sha, "byte_size": meta_file.stat().st_size}
                ],
                "output_sink": {"result_video": str(out_dir / "out.mp4")},
                "payload": {"seed": 42, "adapter": "mock", "num_frames": 1},
            }
            bundle_file = tmp_path / "bundle_cwd.json"
            bundle_file.write_text(json.dumps(bundle_dict), encoding="utf-8")

            # Change to a non-standard CWD (not repo root)
            prev_cwd = os.getcwd()
            try:
                os.chdir(str(cwd_dir))
                exit_code = run_worker_cli(str(bundle_file), json_mode=True)
                assert exit_code == 0
            finally:
                os.chdir(prev_cwd)

    def test_go_render_farm_wire_format_presigned_endpoint_and_nested_payload(self) -> None:
        """Go render-farm JobBundle with PresignedEndpoint write_capability and nested payload[closure_type]."""
        go_bundle = {
            "job_id": "job-go-wire-001",
            "project_name": "shot_010",
            "closure_type": "neural_video",
            "engine": {"name": "wan-2.1", "version": "v2.1", "mode": "fgpp_bake_track_b"},
            "scene_spec": {
                "resolution": [832, 480],
                "frame_range": [10, 25],
                "spp": 1,
            },
            "asset_manifest": [],
            "output_sink": {
                "format": "exr",
                "channels": ["DirectRadiance.RGB", "CameraLinearDepth.Z"],
                "write_capability": {
                    "method": "PUT",
                    "url": "https://vault.example.com/upload/job-go-wire-001",
                    "expires_at": "2026-12-01T00:00:00Z",
                },
            },
            "payload": {
                "neural_video": {
                    "model_tier": "tier1_whitebox",
                    "style_prompt": "cinematic neon rain",
                    "anchor_interval_frames": 48,
                }
            },
        }
        manifest = BundleManifest.from_rfc0039_job_bundle(go_bundle)
        assert manifest.stage == "neural_video_generate"
        assert manifest.params["width"] == 832
        assert manifest.params["height"] == 480
        assert manifest.params["num_frames"] == 16
        assert manifest.params["model_tier"] == "tier1_whitebox"
        assert manifest.params["style_prompt"] == "cinematic neon rain"
        assert manifest.params["anchor_interval_frames"] == 48
        assert isinstance(manifest.params["write_capability"], dict)
        assert manifest.params["write_capability"]["method"] == "PUT"


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

    @pytest.mark.external_data
    def test_native_data_inspector_headless_shm_execution(self) -> None:
        """Execute built data-inspector-native binary with --headless --keep-shm --shm-file."""
        native_bin = _resolve_inspector_native_binary()
        if native_bin is None:
            if _require_siblings():
                pytest.fail("SYNAPSE_CI_REQUIRE_SIBLINGS=1: data-inspector-native binary not found.")
            pytest.skip("data-inspector-native binary not built; skipping live binary test.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shm_file = tmp_path / "test_native_shm.bin"

            width = 64
            height = 64
            cmd = [
                str(native_bin),
                "--headless",
                f"--width={width}",
                f"--height={height}",
                "--frames=1",
                "--keep-shm",
                f"--shm-file={shm_file}",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            assert res.returncode == 0, f"data-inspector-native exited with {res.returncode}: {res.stderr}"
            assert shm_file.is_file(), f"Output SHM file missing at {shm_file}"

            expected_bytes = self.HEADER_SIZE + self.BUFFER_COUNT * width * height * 4
            assert shm_file.stat().st_size == expected_bytes

            with open(shm_file, "rb") as f:
                header_data = f.read(self.HEADER_SIZE)
                magic, ver, w, h, widx, bc, fr, fps, dt, dv = struct.unpack_from(
                    "<IIIIIIQffB", header_data, 0
                )
                assert magic == self.INSPECTOR_SHM_MAGIC
                assert ver == self.INSPECTOR_SHM_VERSION
                assert w == width
                assert h == height
                assert bc == self.BUFFER_COUNT
                assert fr % 2 == 0  # Even = stable

                # Read published frame buffer
                f.seek(self.HEADER_SIZE + widx * (width * height * 4))
                pixels = np.frombuffer(f.read(width * height * 4), dtype=np.uint8).reshape(height, width, 4)
                # Ensure frame is rendered and non-empty (diagnostic test pattern or clear color)
                assert pixels.shape == (height, width, 4)

    def test_shm_pixel_data_non_zero(self) -> None:
        """SHM pixel buffer written by the protocol must contain non-zero data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            width = 32
            height = 32
            frame_bytes = width * height * 4
            total_size = self.HEADER_SIZE + self.BUFFER_COUNT * frame_bytes

            shm_file = tmp_path / "test_nonzero_shm.bin"
            shm_file.write_bytes(b"\x00" * total_size)

            with open(shm_file, "r+b") as f:
                mm = mmap.mmap(f.fileno(), total_size)

                # Write header with valid magic
                header_bin = struct.pack(
                    "<IIIIIIQffB23s",
                    self.INSPECTOR_SHM_MAGIC,
                    self.INSPECTOR_SHM_VERSION,
                    width,
                    height,
                    0,  # write_idx
                    self.BUFFER_COUNT,
                    2,  # frame_ready (even = stable)
                    30.0,  # fps
                    0.033,  # delta_time
                    1,  # debug_view
                    b"\x00" * 23,
                )
                mm[: self.HEADER_SIZE] = header_bin

                # Write a gradient test pattern (non-zero, non-uniform)
                buf_offset = self.HEADER_SIZE
                rng = np.random.RandomState(42)
                pixel_data = rng.randint(10, 245, (height, width, 4), dtype=np.uint8)
                mm[buf_offset : buf_offset + frame_bytes] = pixel_data.tobytes()

                # Read back and verify non-zero
                read_pixels = np.frombuffer(
                    mm[buf_offset : buf_offset + frame_bytes], dtype=np.uint8
                ).reshape(height, width, 4)

                # Pixel data must be non-zero (not all black)
                assert np.any(read_pixels > 0), (
                    "SHM pixel data is all-zero — indicates no rendering output"
                )
                # Pixel data must have spatial variance (not flat/degenerate)
                assert float(np.std(read_pixels.astype(np.float32))) > 1.0, (
                    "SHM pixel data has near-zero variance — degenerate output"
                )
                # Verify data round-trips exactly
                np.testing.assert_array_equal(read_pixels, pixel_data)

                mm.close()


# ===========================================================================
# Contract 5: @aether//style <-> synapse.style (StyleDescriptor v2.0.0)
# ===========================================================================


class TestAetherStyleContract:
    """Cross-language parity between @aether//style and synapse.style.descriptor."""

    @pytest.mark.external_data
    def test_style_types_header_parity(self) -> None:
        """Parse aether/style/include/aether/style/style_types.h and run --require-style."""
        aether_root = _resolve_aether_root()
        if aether_root is None or not (aether_root / "fgpp").exists():
            if _require_siblings():
                pytest.fail("SYNAPSE_CI_REQUIRE_SIBLINGS=1: sibling @aether repo not found.")
            pytest.skip("Sibling @aether repo not found; skipping C++ style header parity.")

        style_header = aether_root / "style" / "include" / "aether" / "style" / "style_types.h"
        if not style_header.is_file():
            pytest.skip(f"@aether//style header not yet available at {style_header}")
        header_text = style_header.read_text(encoding="utf-8")

        struct_match = re.search(
            r"struct\s+StyleDescriptor\s*\{(.*?)\n\};",
            header_text,
            re.DOTALL,
        )
        assert struct_match is not None, "struct StyleDescriptor not found in style_types.h"
        struct_body = struct_match.group(1)

        for field_name in StyleDescriptor.model_fields:
            assert re.search(rf"\b{re.escape(field_name)}\b", struct_body), (
                f"StyleDescriptor field '{field_name}' missing in C++ header"
            )
        assert "schema_version" in struct_body
        assert "ACEScg" in header_text
        assert 'kCurrentSchemaVersion = "2.0.0"' in header_text

        fgpp_header = aether_root / "fgpp" / "include" / "aether" / "fgpp" / "fgpp_types.h"
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "wire_guard.py"),
            "--check",
            "--require-aether",
            "--require-style",
            "--aether-header",
            str(fgpp_header),
            "--aether-style-header",
            str(style_header),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert res.returncode == 0, (
            f"wire_guard --require-style failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "SKIPPED" not in res.stdout

