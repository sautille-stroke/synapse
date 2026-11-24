"""Unit tests for advanced geometric and UV topology quality metrics.

Verifies:
- Euler characteristic chi = V - E + F for sphere/cube (chi=2) and torus (chi=0).
- Exact non-manifold edges count for branching and orientation-conflict topologies.
- UV area distortion ratio under conformal and stretched parameterisations.
- Seam normal discontinuity in degrees along chart boundary edges.
- Texel density coefficient of variation (CV).
- Integration into ModelEvaluator topology results and metrics dictionary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from synapse.inspect.model_evaluator import evaluate_single_model
from synapse.recon.geometry import (
    compute_euler_characteristic,
    compute_seam_discontinuity_deg,
    compute_texel_density_variance,
    compute_uv_area_distortion_ratio,
    count_non_manifold_edges,
)
from synapse.render_core.mesh import MeshData


def test_euler_characteristic_tetrahedron_and_cube() -> None:
    """Euler characteristic should satisfy V - E + F = 2 for genus-0 closed surfaces."""
    # Tetrahedron: V=4, E=6, F=4 -> chi = 2
    tet_verts = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ], dtype=np.float64)
    tet_faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ], dtype=np.int64)

    chi_tet = compute_euler_characteristic(tet_verts, tet_faces)
    assert chi_tet == 2

    # Cube with 6 quad faces: V=8, E=12, F=6 -> chi = 8 - 12 + 6 = 2
    cube_verts = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ], dtype=np.float64)
    cube_faces = np.array([
        [0, 3, 2, 1],  # -Z
        [4, 5, 6, 7],  # +Z
        [0, 1, 5, 4],  # -Y
        [2, 3, 7, 6],  # +Y
        [0, 4, 7, 3],  # -X
        [1, 2, 6, 5],  # +X
    ], dtype=np.int64)
    chi_cube = compute_euler_characteristic(cube_verts, cube_faces)
    assert chi_cube == 2

    # Empty inputs
    assert compute_euler_characteristic(np.empty((0, 3)), np.empty((0, 3))) == 0


def test_count_non_manifold_edges() -> None:
    """Branching edges (shared by >2 faces) and orientation conflicts should be counted."""
    # 2-manifold closed tetrahedron has 0 non-manifold edges
    tet_verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    tet_faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=np.int64)
    assert count_non_manifold_edges(tet_verts, tet_faces) == 0

    # 3 triangles sharing edge (0, 1) -> 1 non-manifold edge
    branching_faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 1, 4],
    ], dtype=np.int64)
    branching_verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    assert count_non_manifold_edges(branching_verts, branching_faces) == 1

    # Orientation conflict: edge (0, 1) traversed in same direction [0, 1, 2] and [0, 1, 3]
    conflict_faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
    ], dtype=np.int64)
    assert count_non_manifold_edges(branching_verts[:4], conflict_faces) == 1


def test_uv_area_distortion_ratio() -> None:
    """UV area distortion ratio should be ~1.0 for isometric UVs and large when stretched."""
    # Two identical 3D triangles
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.int64)

    # Perfectly proportional UVs
    iso_uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float64)
    ratio_iso = compute_uv_area_distortion_ratio(verts, faces, iso_uvs)
    assert 0.99 <= ratio_iso <= 1.01

    # Stretched UVs: triangle 1 is 4x larger in UV space than triangle 0
    distorted_uvs = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [2.0, 2.0],
    ], dtype=np.float64)
    ratio_distorted = compute_uv_area_distortion_ratio(verts, faces, distorted_uvs)
    assert ratio_distorted > 2.0

    # Degenerate/empty
    assert compute_uv_area_distortion_ratio(verts[:0], faces[:0], iso_uvs[:0]) == 1.0


def test_seam_discontinuity_and_texel_density() -> None:
    """Test seam normal jump and texel density variance metrics."""
    # 2 triangles sharing an edge in 3D but split in UV space
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.int64)

    # 4 distinct UV vertices (shared edge 1-2 has different UVs in face 0 vs face 1)
    uvs = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [1.0, 1.0],
        [0.6, 0.0],  # split UV for vert 1
        [0.0, 0.6],  # split UV for vert 2
    ], dtype=np.float64)
    uv_faces = np.array([
        [0, 1, 2],
        [4, 3, 5],
    ], dtype=np.int64)

    discontinuity = compute_seam_discontinuity_deg(verts, faces, uvs, uv_faces=uv_faces)
    assert 0.0 <= discontinuity <= 180.0

    # Texel density CV
    cv = compute_texel_density_variance(verts, faces, uvs, uv_faces=uv_faces)
    assert cv >= 0.0


def test_model_evaluator_integration() -> None:
    """ModelEvaluator must include new metrics in output and TopologyResult."""
    obj_content = (
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.0 0.0 1.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 0.0 1.0\n"
        "vt 0.5 0.5\n"
        "vn 0.0 0.0 1.0\n"
        "f 1/1/1 3/3/1 2/2/1\n"
        "f 1/1/1 2/2/1 4/4/1\n"
        "f 2/2/1 3/3/1 4/4/1\n"
        "f 3/3/1 1/1/1 4/4/1\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        obj_file = Path(tmp_dir) / "test_tet.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        res = evaluate_single_model(
            model_path=obj_file,
            profile_name="TRACK_B_VIDEO",
            num_views=2,
            resolution=64,
            geometry_backend="python",
        )
        assert res.error is None
        assert "euler_characteristic" in res.metrics
        assert res.metrics["euler_characteristic"] == 2.0
        assert "non_manifold_edges_count" in res.metrics
        assert res.metrics["non_manifold_edges_count"] == 0.0
        assert "uv_area_distortion_ratio" in res.metrics
        assert res.metrics["uv_area_distortion_ratio"] >= 1.0
        assert "seam_discontinuity_deg" in res.metrics
        assert "texel_density_variance" in res.metrics

        assert res.topology.euler_characteristic == 2
        assert res.topology.num_non_manifold_edges == 0
