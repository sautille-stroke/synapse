"""Wave 4 cross-repo integration and native parity tests for RFC 0060 and RFC 0061 Phase 4."""

from __future__ import annotations

import os

import numpy as np
import pytest
from synapse.bridge.aether_bridge import (
    get_aether_animation,
    get_aether_compress,
    get_aether_geo,
    reset_aether_cache,
)
from synapse.bridge.foundary_client import (
    FOUNDARY_SERVICE_NAMES,
    FoundaryClient,
    FoundaryRequest,
    FoundaryServiceName,
    MockFoundaryTransport,
)
from synapse.recon.delight import blend_multiview_seams
from synapse.recon.gs_quantizer import GaussianSplatCloud, dequantize_3dgs, quantize_3dgs
from synapse.recon.kinematics import solve_zero_slip_ik

_REQUIRE_NATIVE = os.environ.get("SYNAPSE_REQUIRE_NATIVE_WAVE4", "0") == "1"


def _require_or_skip(mod: object | None, name: str) -> None:
    if mod is not None:
        return
    msg = f"Native extension '{name}' unavailable (set AETHER_ROOT to built @aether repo)."
    if _REQUIRE_NATIVE:
        pytest.fail(msg)
    pytest.skip(msg)


def test_foundary_rfc0060_services_and_serving_recipes_catalog() -> None:
    """Verify RFC 0060 Foundary service schemas and recipe discovery client."""
    expected: tuple[FoundaryServiceName, ...] = (
        "sam2",
        "dust3r",
        "metric3d",
        "cotracker",
        "wham",
        "trellis",
        "intrinsic_image",
    )
    for svc in expected:
        assert svc in FOUNDARY_SERVICE_NAMES
        req = FoundaryRequest(service=svc, job_id=f"job-{svc}")
        client = FoundaryClient(MockFoundaryTransport())
        resp = client.transport.submit(req)
        assert resp.status == "ok"
        assert resp.service == svc

    client = FoundaryClient(MockFoundaryTransport())
    recipes_0060 = client.list_serving_recipes(rfc="RFC-0060")
    ids = {r["id"] for r in recipes_0060}
    assert ids == {
        "sam2",
        "dust3r",
        "metric3d-v2",
        "cotracker3",
        "wham",
        "trellis",
        "intrinsic-image",
    }
    seg_only = client.list_serving_recipes(rfc="RFC-0060", role="segmentation")
    assert [r["id"] for r in seg_only] == ["sam2"]


@pytest.mark.external_data
def test_native_draco_point_cloud_3dgs_parity() -> None:
    """RFC 0061 Phase 4: real aether_compress Draco point-cloud quantization >= 70%."""
    reset_aether_cache()
    comp_mod = get_aether_compress()
    _require_or_skip(comp_mod, "aether_compress")
    assert hasattr(comp_mod, "compress_point_cloud")
    assert hasattr(comp_mod, "decompress_point_cloud")

    rng = np.random.default_rng(61)
    n = 4096
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

    quantized = quantize_3dgs(cloud, require_draco=True)
    assert quantized.metrics.draco_used is True
    assert quantized.metrics.compression_ratio >= 0.70
    assert quantized.metrics.xyz_rmse < 5e-3

    decoded = dequantize_3dgs(quantized.blob)
    assert decoded.num_gaussians == n
    assert decoded.sh_band_count == 15


@pytest.mark.external_data
def test_native_two_bone_ik_oracle_parity() -> None:
    """RFC 0060: real aether_animation Two-Bone IK matches Python Oracle (<1e-4) and <= 0.2cm slip."""
    reset_aether_cache()
    anim_mod = get_aether_animation()
    _require_or_skip(anim_mod, "aether_animation")
    assert hasattr(anim_mod, "solve_two_bone_ik")

    t_len = 8
    positions = np.zeros((t_len, 9, 3), dtype=np.float32)
    positions[:, 1] = [0.0, 0.9, 0.0]
    positions[:, 4] = [0.0, 0.45, 0.05]
    positions[:, 7] = [0.0, 0.02, 0.0]
    positions[:, 2] = [0.2, 0.9, 0.0]
    positions[:, 5] = [0.2, 0.45, 0.05]
    positions[:, 8] = [0.2, 0.02, 0.0]
    for t in range(1, t_len):
        # Multi-centimeter raw foot slip exercises non-trivial IK rotations above
        # aether::quatFromVectors's 0.08 deg (1 - 1e-6) near-identity cutoff.
        positions[t, 7, 0] = 0.025 * t
        positions[t, 8, 2] = 0.015 * t
    contacts = np.ones((t_len, 2), dtype=np.float32)

    native_res = solve_zero_slip_ik(
        {"joint_positions": positions},
        contacts,
        backend="native",
        strict_slip_gate=True,
    )
    oracle_res = solve_zero_slip_ik(
        {"joint_positions": positions},
        contacts,
        backend="python",
        strict_slip_gate=True,
    )

    assert native_res["telemetry"]["ik_backend"] == "aether_animation"
    assert native_res["telemetry"]["max_foot_slide_cm"] <= 0.2
    max_diff = float(np.max(np.abs(native_res["joint_positions"] - oracle_res["joint_positions"])))
    assert max_diff < 1e-4


@pytest.mark.external_data
def test_native_graph_cut_seam_blend_parity() -> None:
    """RFC 0060: real aether_geo.graph_cut_seam_blend keeps seam discontinuity <= 2.0 ΔE00."""
    reset_aether_cache()
    geo_mod = get_aether_geo()
    _require_or_skip(geo_mod, "aether_geo")
    assert hasattr(geo_mod, "graph_cut_seam_blend")

    v0 = np.full((24, 24, 3), 0.45, dtype=np.float32)
    v1 = np.full((24, 24, 3), 0.46, dtype=np.float32)
    blended, seam_de00 = blend_multiview_seams([v0, v1], feather_width=8)
    assert blended.shape == (24, 24, 3)
    assert seam_de00 <= 2.0
