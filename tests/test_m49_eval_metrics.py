"""Unit tests for M49: PR #189 5-defect hardening and P0/P1/P2 evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from synapse.inspect.eval_report import (
    EvalReportData,
    TopologyResult,
    generate_eval_html_report,
)
from synapse.inspect.model_evaluator import evaluate_profile_gates, evaluate_single_model
from synapse.metrics import (
    TRACK_A_ASSET,
    add_s_norm,
    dino_cosine_sim,
    lpips_vgg,
    patch_lpips,
    reprojection_error_px,
)
from synapse.recon.geometry import (
    compute_euler_characteristic,
    compute_seam_discontinuity_deg,
    count_non_manifold_edges,
    count_non_manifold_vertices,
    discrete_mean_curvature_variance,
    surface_roughness_var,
)


def _cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ],
        dtype=np.int64,
    )
    return verts, faces


def test_euler_characteristic_ignores_orphan_vertices() -> None:
    """PR-203 Defect #1: unreferenced vertices in buffer must not inflate chi."""
    verts, faces = _cube_mesh()
    orphan_verts = np.vstack([verts, np.zeros((25, 3), dtype=np.float64)])
    assert compute_euler_characteristic(orphan_verts, faces) == 2


def test_count_non_manifold_vertices_bowtie() -> None:
    """PR-203 Defect #2: bowtie/pinched vertex joining two umbrellas must be detected."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # shared apex vertex 0 (bowtie)
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, -1.0],
            [0.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 4],
            [4, 6, 5],
        ],
        dtype=np.int64,
    )
    # Edges all have degree 2, yet vertex 0 connects two disconnected cones
    assert count_non_manifold_edges(verts, faces) == 0
    assert count_non_manifold_vertices(verts, faces) == 1


def test_seam_discontinuity_split_uv_and_across_seam_angle() -> None:
    """PR-203 Defect #3: detects seams on split-UV meshes (uv_faces=None) across faces."""
    # Two orthogonal triangles meeting along Y-axis edge [(0,0,0) - (0,1,0)],
    # represented as duplicated UV-split vertices (0,1,2) and (3,4,5) -> 90 deg crease.
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [0.0, 1.0, 0.0],  # 1
            [1.0, 0.0, 0.0],  # 2 (lies in XY plane -> normal +Z)
            [0.0, 1.0, 0.0],  # 3 (duplicate of 1)
            [0.0, 0.0, 0.0],  # 4 (duplicate of 0)
            [0.0, 0.0, 1.0],  # 5 (lies in YZ plane -> normal +X)
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    uvs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.4, 0.0],
            [0.8, 1.0],  # split UV across seam
            [0.8, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    angle = compute_seam_discontinuity_deg(verts, faces, uvs, uv_faces=None)
    assert 89.0 <= angle <= 91.0

    # Coplanar split-UV triangles -> angle across seam must be ~0 deg
    verts_flat = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    flat_angle = compute_seam_discontinuity_deg(verts_flat, faces, uvs, uv_faces=None)
    assert flat_angle == pytest.approx(0.0, abs=1e-5)


def test_model_evaluator_no_fake_uv_defaults_and_html_report(tmp_path: Path) -> None:
    """PR-203 Defects #4 & #5: missing UVs omit fake passing metrics and fail when required."""
    obj_path = tmp_path / "no_uv_tet.obj"
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n",
        encoding="utf-8",
    )
    res = evaluate_single_model(
        model_path=obj_path,
        profile_name="TRACK_A_ASSET",
        resolution=32,
        num_views=4,
        geometry_backend="python",
        require_reference=False,
        require_uv=False,
    )
    assert res.topology.uv_available is False
    assert "uv_area_distortion_ratio" not in res.metrics
    assert "seam_discontinuity_deg" not in res.metrics
    assert "texel_density_variance" not in res.metrics

    # Require UVs explicitly -> fail-closed
    res_req = evaluate_single_model(
        model_path=obj_path,
        profile_name="TRACK_A_ASSET",
        resolution=32,
        num_views=4,
        geometry_backend="python",
        require_reference=False,
        require_uv=True,
    )
    assert res_req.overall_passed is False
    assert res_req.gate_results.get("uv_required_not_evaluated") is False

    # TRACK_A_ASSET registers UV thresholds
    assert TRACK_A_ASSET["uv_area_distortion_ratio"] == pytest.approx(1.5)
    assert TRACK_A_ASSET["seam_discontinuity_deg"] == pytest.approx(5.0)
    assert TRACK_A_ASSET["texel_density_variance"] == pytest.approx(0.3)

    # HTML report renders Euler characteristic, non-manifold counts, and N/A UVs
    html_out = tmp_path / "eval_report.html"
    generate_eval_html_report(
        EvalReportData(timestamp="2026-11-24 UTC", profile_name="TRACK_A_ASSET", models=[res]),
        html_out,
    )
    html_text = html_out.read_text(encoding="utf-8")
    assert "Euler Characteristic" in html_text
    assert "Non-Manifold Edges" in html_text
    assert "Non-Manifold Vertices" in html_text
    assert "N/A (No UVs)" in html_text

    # Gate evaluates non-passing UV metrics cleanly
    bad_uv = evaluate_profile_gates(
        {
            "is_manifold": 1.0,
            "is_watertight": 1.0,
            "num_degenerate_faces": 0.0,
            "self_intersecting_pairs": 0.0,
            "chamfer_mm": 1.0,
            "uv_area_distortion_ratio": 2.5,
            "seam_discontinuity_deg": 12.0,
            "texel_density_variance": 0.8,
        },
        "TRACK_A_ASSET",
    )
    assert bad_uv["uv_area_distortion_ratio"] is False
    assert bad_uv["seam_discontinuity_deg"] is False
    assert bad_uv["texel_density_variance"] is False


def test_perceptual_metrics_zero_torch_and_monotonic() -> None:
    """PR-204/205: dino_cosine_sim, lpips_vgg, patch_lpips stay torch-free in auto mode."""
    had_torch_before = "torch" in sys.modules
    rng = np.random.default_rng(7)
    img = (rng.uniform(0.1, 0.9, (128, 128, 3)) * 255).astype(np.uint8)
    # Local patch corruption vs global similarity
    corrupted = img.copy()
    corrupted[32:96, 32:96, :] = 0

    assert dino_cosine_sim(img, img, backend="auto") == pytest.approx(1.0, abs=1e-6)
    sim_corrupt = dino_cosine_sim(corrupted, img, backend="auto")
    assert sim_corrupt < 0.99

    assert lpips_vgg(img, img, backend="auto") == pytest.approx(0.0, abs=1e-6)
    global_lp = lpips_vgg(corrupted, img, backend="auto")
    local_worst = patch_lpips(
        corrupted, img, patch_size=32, stride=16, aggregation="max_quartile", backend="auto"
    )
    assert local_worst > global_lp > 0.0

    if not had_torch_before:
        assert "torch" not in sys.modules


def test_curvature_and_surface_roughness_monotonic() -> None:
    """PR-205: discrete_mean_curvature_variance & surface_roughness_var separate flat vs bumpy."""
    # Regular 5x5 grid in XY plane (flat)
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6))
    verts_flat = np.column_stack([gx.ravel(), gy.ravel(), np.zeros_like(gx.ravel())])
    faces_list = []
    for iy in range(5):
        for ix in range(5):
            v0 = iy * 6 + ix
            v1 = v0 + 1
            v2 = v0 + 6
            v3 = v2 + 1
            faces_list.append([v0, v1, v2])
            faces_list.append([v1, v3, v2])
    faces = np.asarray(faces_list, dtype=np.int64)

    assert discrete_mean_curvature_variance(verts_flat, faces) == pytest.approx(0.0, abs=1e-8)
    assert surface_roughness_var(verts_flat, faces) == pytest.approx(0.0, abs=1e-8)

    # Displace alternating Z heights to create severe jagged surface ripples
    verts_bumpy = verts_flat.copy()
    rng = np.random.default_rng(123)
    verts_bumpy[:, 2] = rng.uniform(-0.35, 0.35, size=len(verts_flat))
    assert discrete_mean_curvature_variance(verts_bumpy, faces) > 1e-2
    assert surface_roughness_var(verts_bumpy, faces) > 1e-2


def test_pose_and_3d_metrics_add_s_norm_and_reprojection_error() -> None:
    """PR-206: add_s_norm & reprojection_error_px exactness and monotonic scaling."""
    pts = np.array(
        [
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    assert add_s_norm(pts, pts) == pytest.approx(0.0, abs=1e-9)
    shifted = pts + np.array([0.0, 0.5, 0.0])
    score = add_s_norm(shifted, pts, diameter=2.0)
    # Closest point distance may be < shift due to nearest-neighbor matching, always > 0
    assert score > 0.0

    k_mat = np.array(
        [
            [500.0, 0.0, 256.0],
            [0.0, 500.0, 256.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_mat = np.eye(3, dtype=np.float64)
    t_vec = np.zeros(3, dtype=np.float64)
    proj = (pts @ k_mat.T)[:, :2] / pts[:, 2:3]
    assert reprojection_error_px(pts, proj, k_mat, r_mat, t_vec) == pytest.approx(0.0, abs=1e-9)

    obs_offset = proj + np.array([3.0, 4.0])  # Euclidean offset = 5.0 px
    assert reprojection_error_px(pts, obs_offset, k_mat, r_mat, t_vec) == pytest.approx(
        5.0, abs=1e-7
    )


def test_topology_result_dataclass_defaults() -> None:
    """Backwards compatibility for TopologyResult keyword construction."""
    t = TopologyResult(
        is_manifold=True,
        is_watertight=True,
        num_vertices=4,
        num_faces=4,
        num_connected_components=1,
        has_degenerate_faces=False,
        num_boundary_edges=0,
    )
    assert t.num_non_manifold_vertices is None
    assert t.uv_available is None
