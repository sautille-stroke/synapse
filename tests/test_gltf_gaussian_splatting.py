"""Unit tests for M59 KHR_gaussian_splatting glTF 2.0 (.glb/SPZ) & .ply conversion."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from synapse.cli import main_synapse
from synapse.errors import AssetNotFoundError
from synapse.recon import (
    KHR_GS_EXTENSION,
    KHR_GS_SPZ_EXTENSION,
    GaussianSplatCloud,
    convert_3dgs_asset,
    export_gltf_gaussian_splat,
    load_gltf_gaussian_splat,
    load_ply_3dgs,
    save_ply_3dgs,
)


def _sample_cloud(num_splats: int = 256, *, with_sh_rest: bool = True) -> GaussianSplatCloud:
    rng = np.random.default_rng(59)
    rots = rng.normal(size=(num_splats, 4)).astype(np.float32)
    rots /= np.maximum(np.linalg.norm(rots, axis=-1, keepdims=True), 1e-6)
    sh_rest = (
        rng.normal(0.0, 0.05, size=(num_splats, 15, 3)).astype(np.float32)
        if with_sh_rest
        else None
    )
    return GaussianSplatCloud(
        xyz=rng.uniform(-1.0, 1.0, size=(num_splats, 3)).astype(np.float32),
        sh_dc=rng.uniform(-0.8, 0.8, size=(num_splats, 3)).astype(np.float32),
        sh_rest=sh_rest,
        scales=rng.uniform(0.01, 0.08, size=(num_splats, 3)).astype(np.float32),
        rotations=rots,
        opacities=rng.uniform(0.2, 0.98, size=(num_splats,)).astype(np.float32),
    )


class TestKHRGaussianSplattingGLTF:
    """Export and import tests for KHR_gaussian_splatting .glb and SPZ."""

    def test_uncompressed_glb_structure_and_exact_roundtrip(self, tmp_path: Path) -> None:
        cloud = _sample_cloud(128, with_sh_rest=True)
        glb_path = tmp_path / "splats.glb"
        out, report = export_gltf_gaussian_splat(cloud, glb_path, compress_spz=False)
        assert out.is_file()
        assert report.num_gaussians == 128
        assert report.sh_degree == 3
        assert report.compressed_spz is False
        assert report.roundtrip_psnr_db >= 90.0

        # Validate GLB binary magic, version, chunk headers & KHR extension metadata
        raw = out.read_bytes()
        magic, version, total_len = struct.unpack_from("<III", raw, 0)
        assert magic == 0x46546C67
        assert version == 2
        assert total_len == len(raw)
        json_len, json_type = struct.unpack_from("<II", raw, 12)
        assert json_type == 0x4E4F534A
        gltf_doc = json.loads(raw[20 : 20 + json_len].decode("utf-8"))
        assert KHR_GS_EXTENSION in gltf_doc["extensionsUsed"]
        prim = gltf_doc["meshes"][0]["primitives"][0]
        attrs = prim["attributes"]
        assert "POSITION" in attrs
        assert f"{KHR_GS_EXTENSION}:ROTATION" in attrs
        assert f"{KHR_GS_EXTENSION}:SCALE" in attrs
        assert f"{KHR_GS_EXTENSION}:OPACITY" in attrs
        assert f"{KHR_GS_EXTENSION}:SH_DEGREE_0_COEF_0" in attrs
        assert f"{KHR_GS_EXTENSION}:SH_DEGREE_3_COEF_6" in attrs

        reloaded = load_gltf_gaussian_splat(out)
        assert np.allclose(reloaded.xyz, cloud.xyz, atol=1e-6)
        assert np.allclose(reloaded.sh_dc, cloud.sh_dc, atol=1e-6)
        assert np.allclose(reloaded.scales, cloud.scales, atol=1e-6)
        assert np.allclose(reloaded.opacities, cloud.opacities, atol=1e-6)
        assert reloaded.sh_rest is not None and cloud.sh_rest is not None
        assert np.allclose(reloaded.sh_rest, cloud.sh_rest, atol=1e-6)
        # Unit quaternions q and -q represent identical rotations; abs dot >= 1 - 1e-5
        dots = np.abs(np.sum(reloaded.rotations * cloud.rotations, axis=-1))
        assert np.all(dots >= 1.0 - 1e-5)

    @pytest.mark.parametrize("with_sh_rest", [False, True])
    def test_spz_glb_achieves_3x_compression_and_50db_psnr(
        self,
        tmp_path: Path,
        with_sh_rest: bool,
    ) -> None:
        cloud = _sample_cloud(512, with_sh_rest=with_sh_rest)
        spz_glb = tmp_path / f"compressed_{with_sh_rest}.glb"
        out, report = export_gltf_gaussian_splat(cloud, spz_glb, compress_spz=True)
        assert out.is_file()
        assert report.compressed_spz is True
        assert report.compression_ratio >= 3.0
        assert report.roundtrip_psnr_db >= 50.0

        raw = out.read_bytes()
        json_len, _ = struct.unpack_from("<II", raw, 12)
        gltf_doc = json.loads(raw[20 : 20 + json_len].decode("utf-8"))
        assert KHR_GS_SPZ_EXTENSION in gltf_doc["extensionsUsed"]

        reloaded = load_gltf_gaussian_splat(out)
        assert reloaded.num_gaussians == 512
        assert reloaded.sh_band_count == (15 if with_sh_rest else 0)
        assert np.allclose(reloaded.xyz, cloud.xyz, atol=1e-4)


class TestPlyGltfBidirectionalConverterAndCLI:
    """Tests for .ply <-> .glb / .s3dgs conversion and recon CLI subcommands."""

    def test_ply_save_load_and_bidirectional_conversion(self, tmp_path: Path) -> None:
        cloud = _sample_cloud(256, with_sh_rest=True)
        ply_path = save_ply_3dgs(cloud, tmp_path / "scene.ply")
        assert ply_path.is_file()
        loaded_ply = load_ply_3dgs(ply_path)
        assert np.allclose(loaded_ply.xyz, cloud.xyz, atol=1e-6)
        assert loaded_ply.sh_rest is not None and cloud.sh_rest is not None
        assert np.allclose(loaded_ply.sh_rest, cloud.sh_rest, atol=1e-6)

        glb_path = tmp_path / "scene.glb"
        rep1 = convert_3dgs_asset(ply_path, glb_path, compress_spz=True)
        assert rep1.source_format == "ply"
        assert rep1.target_format == "glb"
        assert rep1.compression_ratio >= 3.0
        assert rep1.roundtrip_psnr_db >= 50.0

        ply_rt = tmp_path / "scene_rt.ply"
        rep2 = convert_3dgs_asset(glb_path, ply_rt)
        assert rep2.source_format == "glb"
        assert rep2.target_format == "ply"
        assert rep2.roundtrip_psnr_db >= 90.0
        assert load_ply_3dgs(ply_rt).num_gaussians == 256

    def test_cli_export_gltf_and_convert_json(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cloud = _sample_cloud(512, with_sh_rest=False)
        ply_in = save_ply_3dgs(cloud, tmp_path / "in.ply")
        glb_out = tmp_path / "out.glb"
        ply_back = tmp_path / "back.ply"

        capsys.readouterr()
        rc = main_synapse(
            [
                "recon",
                "export-gltf-3dgs",
                "--input",
                str(ply_in),
                "--output",
                str(glb_out),
                "--compress-spz",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "ok"
        assert payload["compressed_spz"] is True
        assert payload["compression_ratio"] >= 3.0
        assert payload["roundtrip_psnr_db"] >= 50.0

        rc2 = main_synapse(
            [
                "recon",
                "convert-3dgs",
                "--input",
                str(glb_out),
                "--output",
                str(ply_back),
                "--json",
            ]
        )
        assert rc2 == 0
        payload2 = json.loads(capsys.readouterr().out)
        assert payload2["status"] == "ok"
        assert payload2["source_format"] == "glb"
        assert payload2["target_format"] == "ply"
        assert ply_back.is_file()

        with pytest.raises(AssetNotFoundError) as exc_info:
            load_gltf_gaussian_splat(tmp_path / "missing.glb")
        assert exc_info.value.exit_code == 20
