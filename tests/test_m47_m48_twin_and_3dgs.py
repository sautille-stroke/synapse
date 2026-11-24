"""Tests for M47-M48: C++ IK bridge, Slang AD dispatch, USD/OpenPBR, 3DGS, and E6."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
from synapse.bridge.aether_bridge import (
    has_aether_animation,
    has_aether_compress,
    reset_aether_cache,
)
from synapse.cli import main as cli_main
from synapse.errors import (
    BackendUnavailableError,
    CircuitBreakerError,
    QualityGateError,
)
from synapse.inspect.jev_critic import MockJevTransport, VisualJevCritic
from synapse.recon.delight import blend_multiview_seams
from synapse.recon.gs_quantizer import (
    GaussianSplatCloud,
    dequantize_3dgs,
    quantize_3dgs,
)
from synapse.recon.kinematics import solve_zero_slip_ik, validate_foot_slide_gate
from synapse.recon.loop import (
    ClosedLoopRunner,
    FulgurSlangADRendererAdapter,
    InverseOptimConfig,
    create_inverse_optimizer,
)
from synapse.recon.pipeline import ReconPipeline
from synapse.recon.types import MeshGeometry, MocapKinematics
from synapse.recon.usd_packager import build_rfc0058_usda, validate_usd_package


def test_pr197_ik_bridge_and_neg_gates() -> None:
    """PR-197 + NEG-100-08/09: native IK delegation, fail-closed missing, and slip > 0.5cm gate."""
    reset_aether_cache()

    t_len = 6
    positions = np.zeros((t_len, 9, 3), dtype=np.float32)
    positions[:, 1] = [0.0, 0.9, 0.0]  # left hip
    positions[:, 4] = [0.0, 0.45, 0.05]  # left knee
    positions[:, 7] = [0.0, 0.02, 0.0]  # left foot
    positions[:, 2] = [0.2, 0.9, 0.0]  # right hip
    positions[:, 5] = [0.2, 0.45, 0.05]  # right knee
    positions[:, 8] = [0.2, 0.02, 0.0]  # right foot
    # Inject small sliding within hip reach
    for t in range(1, t_len):
        positions[t, 7, 0] = 0.004 * t
    contacts = np.ones((t_len, 2), dtype=np.float32)

    # NEG-100-08: backend="native" without aether_animation MUST fail closed with exit 30
    with mock.patch("synapse.bridge.aether_bridge.get_aether_animation", return_value=None):
        assert has_aether_animation() is False
        with pytest.raises(BackendUnavailableError) as e30:
            solve_zero_slip_ik({"joint_positions": positions}, contacts, backend="native")
        assert e30.value.exit_code == 30

    # Inject mock aether_animation module and verify parity + ik_backend telemetry
    calls: list[int] = []

    class _MockAnimMod:
        @staticmethod
        def solve_two_bone_ik(
            hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, target: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            calls.append(1)
            from synapse.recon.kinematics import solve_two_bone_ik

            return solve_two_bone_ik(hip, knee, ankle, target)

    with mock.patch.dict(sys.modules, {"aether_animation": _MockAnimMod()}):
        reset_aether_cache()
        assert has_aether_animation() is True
        solved = solve_zero_slip_ik(
            {"joint_positions": positions},
            contacts,
            backend="native",
            strict_slip_gate=True,
        )
        assert len(calls) > 0
        assert solved["telemetry"]["ik_backend"] == "aether_animation"
        assert solved["telemetry"]["max_foot_slide_cm"] <= 0.5
    reset_aether_cache()

    # NEG-100-09: > 0.5 cm sliding fails strict gate with ExitCode 40
    with pytest.raises(QualityGateError) as e40:
        validate_foot_slide_gate(0.75, max_foot_slide_cm=0.5)
    assert e40.value.exit_code == 40

    # Unreachable hip displacement causing post-IK ankle drift > 0.5 cm triggers exit 40
    unreach = np.copy(positions)
    unreach[2:, 1, 0] = 5.0  # hip flies 5 m away so leg cannot reach anchor
    unreach[2:, 4, 0] = 5.0
    unreach[2:, 7, 0] = 5.0
    with pytest.raises(QualityGateError) as e40_ik:
        solve_zero_slip_ik(
            {"joint_positions": unreach},
            contacts,
            backend="python",
            strict_slip_gate=True,
            max_foot_slide_cm=0.5,
        )
    assert e40_ik.value.exit_code == 40


def test_pr198_slang_ad_dispatch_and_fail_closed() -> None:
    """PR-198: FulgurSlangADRendererAdapter fails closed (exit 30) when missing and works when injected."""
    rng = np.random.default_rng(198)
    frames = rng.uniform(0.2, 0.8, (5, 16, 16, 3)).astype(np.float32)
    normals = np.zeros((5, 16, 16, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    uv_maps = np.zeros((5, 16, 16, 2), dtype=np.float32)
    u = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    v = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    uv_maps[..., 0] = u[None, None, :]
    uv_maps[..., 1] = v[None, :, None]

    # Missing Slang backend fails closed with BackendUnavailableError (exit 30)
    with pytest.raises(BackendUnavailableError) as e30:
        create_inverse_optimizer(
            InverseOptimConfig(renderer_backend="fulgur_slang_ad", strict_backend=True)
        )
    assert e30.value.exit_code == 30

    adapter = FulgurSlangADRendererAdapter()
    with pytest.raises(BackendUnavailableError) as e30_run:
        adapter.optimize_material_from_video(frames, normals, uv_maps)
    assert e30_run.value.exit_code == 30

    # Inject Slang AD backend and run ClosedLoopRunner
    class _StubSlangBackend:
        @staticmethod
        def render_and_grad(
            albedo: np.ndarray,
            obs_frames: np.ndarray,
            nrms: np.ndarray,
            uvs: np.ndarray,
            l_dirs: np.ndarray,
        ) -> np.ndarray:
            _ = (nrms, uvs, l_dirs)
            target_mean = np.mean(obs_frames, axis=(0, 1, 2))
            return 2.0 * (albedo - target_mean[None, None, :])

    runner = ClosedLoopRunner(
        optim_config=InverseOptimConfig(
            renderer_backend="fulgur_slang_ad",
            max_iterations=8,
            learning_rate=0.25,
            texture_resolution=16,
            strict_backend=True,
        )
    )
    res = runner.run(frames, normals, uv_maps, slang_backend=_StubSlangBackend())
    assert res.inverse_result.backend_used == "fulgur_slang_ad"
    assert res.to_dict()["inverse_backend"] == "fulgur_slang_ad"


def test_pr199_seam_blending_rfc0058_usd_and_jev_hitl(tmp_path: Path) -> None:
    """PR-199: DP Graph-Cut seam blend <= 2.0 ΔE00, RFC 0058 OpenUSD/OpenPBR/UsdSkel, JEV + HITL."""
    # Aligned views with small lighting offset -> DP seam cut ΔE00 <= 2.0
    v0 = np.full((24, 24, 3), 0.45, dtype=np.float32)
    v1 = np.full((24, 24, 3), 0.46, dtype=np.float32)
    blended, seam_de00 = blend_multiview_seams([v0, v1], feather_width=8)
    assert blended.shape == (24, 24, 3)
    assert seam_de00 <= 2.0

    # RFC 0058 OpenUSD + OpenPBR + UsdSkel export & strict disk-ref validation
    tex_file = tmp_path / "textures" / "albedo.png"
    tex_file.parent.mkdir(parents=True, exist_ok=True)
    tex_file.write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")

    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    mesh = MeshGeometry(vertices=verts, faces=faces, normals=verts.copy(), uvs=uvs)
    kin = MocapKinematics(
        frame_count=2,
        joint_names=["root", "left_foot"],
        joint_positions=np.zeros((2, 2, 3), dtype=np.float32),
    )

    usda_path = build_rfc0058_usda(
        mesh,
        tmp_path,
        kinematics=kin,
        texture_paths={"albedo": tex_file},
    )
    val = validate_usd_package(usda_path, strict=True)
    assert val.valid is True
    assert val.has_usd_geom_mesh is True
    assert val.has_openpbr_material is True
    assert val.has_usd_skel is True
    assert val.missing_assets == []

    # JEV + 15-min HITL circuit breaker via ReconPipeline
    from synapse.render_core.procedural import ProceduralMeshType, generate_procedural_mesh

    sphere = generate_procedural_mesh(ProceduralMeshType.SPHERE)
    high_mesh = MeshGeometry(
        vertices=sphere.vertices.astype(np.float32),
        faces=sphere.faces.astype(np.int32),
        normals=sphere.normals.astype(np.float32) if sphere.normals is not None else None,
        uvs=sphere.uvs.astype(np.float32) if sphere.uvs is not None else None,
    )
    pipe = ReconPipeline(target_quad_faces=120, geometry_backend="python")
    frames = np.full((3, 32, 32, 3), 0.5, dtype=np.float32)

    critic = VisualJevCritic(MockJevTransport())
    ok_res = pipe.run(
        frames,
        initial_mesh=high_mesh,
        has_gen3d_highpoly=True,
        enable_jev_gate=True,
        jev_critic=critic,
        hitl_ledger_path=tmp_path / "hitl.json",
        elapsed_seconds_override=5.0,
    )
    assert "jev_inspection" in ok_res.telemetry
    assert ok_res.telemetry["jev_inspection"]["overall_quality_score"] >= 0.85

    with pytest.raises(CircuitBreakerError) as e40:
        pipe.run(
            frames,
            initial_mesh=high_mesh,
            has_gen3d_highpoly=True,
            enable_jev_gate=True,
            jev_critic=critic,
            hitl_ledger_path=tmp_path / "hitl.json",
            elapsed_seconds_override=950.0,
        )
    assert e40.value.exit_code == 40


def test_pr200_3dgs_quantizer_draco_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """PR-200: 3DGS compression >= 70%, ΔPSNR <= 0.5 dB, Draco exit 30, CLI compress-3dgs."""
    reset_aether_cache()
    n = 512
    rng = np.random.default_rng(200)
    rots = rng.normal(size=(n, 4)).astype(np.float32)
    rots /= np.maximum(np.linalg.norm(rots, axis=-1, keepdims=True), 1e-6)
    cloud = GaussianSplatCloud(
        xyz=rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32),
        sh_dc=rng.uniform(-0.8, 0.8, size=(n, 3)).astype(np.float32),
        sh_rest=rng.normal(0.0, 0.05, size=(n, 15, 3)).astype(np.float32),
        scales=rng.uniform(0.01, 0.08, size=(n, 3)).astype(np.float32),
        rotations=rots,
        opacities=rng.uniform(0.2, 0.98, size=(n,)).astype(np.float32),
    )

    quantized = quantize_3dgs(cloud, require_draco=False)
    assert quantized.metrics.compression_ratio >= 0.70
    assert quantized.metrics.psnr_delta_db <= 0.5
    assert quantized.metrics.recon_vs_orig_psnr_db >= 40.0

    decoded = dequantize_3dgs(quantized.blob)
    assert decoded.num_gaussians == n
    assert decoded.sh_band_count == 15
    assert decoded.to_fulgur_desc_dict()["sh_degree"] == 3

    # require_draco=True must fail closed with exit 30 when aether_compress absent
    with mock.patch("synapse.bridge.aether_bridge.get_aether_compress", return_value=None):
        assert has_aether_compress() is False
        with pytest.raises(BackendUnavailableError) as e30:
            quantize_3dgs(cloud, require_draco=True)
        assert e30.value.exit_code == 30

    # CLI compress-3dgs --json
    npz_path = tmp_path / "splats.npz"
    out_s3dgs = tmp_path / "splats.s3dgs"
    np.savez(
        npz_path,
        xyz=cloud.xyz,
        sh_dc=cloud.sh_dc,
        sh_rest=cloud.sh_rest if cloud.sh_rest is not None else np.zeros((n, 0, 3), dtype=np.float32),
        scales=cloud.scales,
        rotations=cloud.rotations,
        opacities=cloud.opacities,
    )
    capsys.readouterr()
    rc = cli_main(["recon", "compress-3dgs", "--input", str(npz_path), "--output", str(out_s3dgs), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["compression_ratio"] >= 0.70
    assert out_s3dgs.is_file()

    rc_draco = cli_main(
        ["recon", "compress-3dgs", "--input", str(npz_path), "--output", str(out_s3dgs), "--require-draco", "--json"]
    )
    assert rc_draco == 30


def test_pr201_colab_runner_e6_twin_benchmark(tmp_path: Path) -> None:
    """PR-201: colab_runner run_colab_suite(['e6']) emits e6_metrics.json and OpenUSD asset."""
    from scripts.colab_runner import run_colab_suite

    ckpt_dir = tmp_path / "checkpoints"
    state = run_colab_suite(
        dataset_dir=tmp_path / "unused_datasets",
        checkpoint_dir=ckpt_dir,
        experiments=["e6"],
        mode="mock",
    )
    assert "e6" in state["completed"]
    e6 = state["completed"]["e6"]
    assert e6["status"] == "PASS"
    assert e6["usd_valid"] is True
    assert e6["compression_ratio"] >= 0.70
    assert e6["seam_delta_e00"] <= 2.0
    assert (ckpt_dir / "e6_metrics.json").is_file()
    _ = Any, SimpleNamespace
