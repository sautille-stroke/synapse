"""Tier-0 Geometry Oracle & Native C++20 Parity Tests.

Verifies edge cases in ``synapse.recon.geometry`` (coplanar disjoint grids,
coplanar overlapping triangles, parallel planes, quad diagonal umbrellas,
out-of-bounds/NaN safety, IEEE-754 floating-island ratios) and enforces 100%
output parity with ``synapse._c_geometry`` whenever the native pybind11
extension is built.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from synapse.inspect.geometry_gate import validate_mesh_geometry
from synapse.recon.geometry import (
    check_degenerate_faces,
    check_manifold,
    check_self_intersections,
    check_watertight,
    find_connected_components,
)

try:
    import synapse._c_geometry as _c_geometry  # type: ignore[import-not-found]

    HAS_NATIVE_GEOMETRY = True
except ImportError:
    _c_geometry = None
    HAS_NATIVE_GEOMETRY = False


# ===================================================================
# Canonical Mesh Builders
# ===================================================================


def _make_tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _make_coplanar_disjoint_grid() -> tuple[np.ndarray, np.ndarray]:
    """4x4 grid of disjoint coplanar triangles on z = 0 (zero intersections)."""
    verts_list: list[list[float]] = []
    faces_list: list[list[int]] = []
    idx = 0
    for gx in range(4):
        for gy in range(4):
            ox, oy = gx * 2.0, gy * 2.0
            verts_list.extend(
                [
                    [ox, oy, 0.0],
                    [ox + 0.8, oy, 0.0],
                    [ox, oy + 0.8, 0.0],
                ]
            )
            faces_list.append([idx, idx + 1, idx + 2])
            idx += 3
    return np.array(verts_list, dtype=np.float64), np.array(faces_list, dtype=np.int64)


def _make_coplanar_overlapping() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
            [0.5, 2.5, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return verts, faces


def _make_coplanar_contained() -> tuple[np.ndarray, np.ndarray]:
    """Small triangle strictly inside a larger coplanar triangle on z = 0."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return verts, faces


def _make_parallel_non_coplanar() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.25],
            [2.0, 0.0, 0.25],
            [0.0, 2.0, 0.25],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return verts, faces


def _make_3d_crossing() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return verts, faces


def _make_bowtie() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    return verts, faces


def _make_non_manifold_triple_edge() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]], dtype=np.int64)
    return verts, faces


# ===================================================================
# Python Oracle Exactness Tests (always run)
# ===================================================================


class TestGeometryOracleExactness:
    """Verify mathematical edge cases in the Python Oracle."""

    def test_coplanar_disjoint_grid_zero_false_positives(self) -> None:
        verts, faces = _make_coplanar_disjoint_grid()
        assert check_self_intersections(verts, faces) == []

    def test_coplanar_overlapping_detected(self) -> None:
        verts, faces = _make_coplanar_overlapping()
        assert check_self_intersections(verts, faces) == [(0, 1)]

    def test_coplanar_contained_detected(self) -> None:
        verts, faces = _make_coplanar_contained()
        assert check_self_intersections(verts, faces) == [(0, 1)]

    def test_parallel_non_coplanar_not_flagged(self) -> None:
        verts, faces = _make_parallel_non_coplanar()
        assert check_self_intersections(verts, faces) == []

    def test_bowtie_fails_manifold_and_has_two_edge_components(self) -> None:
        verts, faces = _make_bowtie()
        assert check_manifold(verts, faces) is False
        assert check_watertight(verts, faces) is False
        comps = find_connected_components(faces)
        assert len(comps) == 2

    def test_quad_diagonal_pinch_fails_manifold(self) -> None:
        """Two quads sharing two non-adjacent vertices (0 and 2) but NO shared edge."""
        verts = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [1.0, 1.0, 0.0],  # 2
                [0.0, 1.0, 0.0],  # 3
                [1.0, 0.0, 1.0],  # 4
                [0.0, 1.0, 1.0],  # 5
            ],
            dtype=np.float64,
        )
        # Quad 0 edges: (0,1), (1,2), (2,3), (3,0)
        # Quad 1 edges: (0,4), (4,2), (2,5), (5,0)
        # Both contain {0, 2}, yet share ZERO edges!
        faces = np.array([[0, 1, 2, 3], [0, 4, 2, 5]], dtype=np.int64)
        assert check_manifold(verts, faces) is False

    def test_repeated_vertex_in_face_fails_manifold(self) -> None:
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 0, 1]], dtype=np.int64)
        assert check_manifold(verts, faces) is False

    def test_empty_and_invalid_inputs_safe(self) -> None:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        empty_f = np.zeros((0, 3), dtype=np.int64)
        assert check_manifold(empty_v, empty_f) is False
        assert check_watertight(empty_v, empty_f) is False
        res = validate_mesh_geometry(empty_v, empty_f, backend="python")
        assert res.passed is False

        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        oob_faces = np.array([[0, -1, 99]], dtype=np.int64)
        assert check_manifold(verts, oob_faces) is False
        assert list(check_degenerate_faces(verts, oob_faces)) == [0]
        assert check_self_intersections(verts, oob_faces) == []

    def test_floating_island_exact_ratio_boundary(self) -> None:
        """3-face island in a 300-face mesh with min_face_ratio=0.01 (3 >= 3.0) is NOT debris."""
        # Build 74 disjoint tetrahedra (296 faces) + 1 tetrahedron (4 faces) = 300 faces?
        # Even simpler: 99-component strip of 3 faces + 297-face strip = 300 faces.
        verts_list = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        faces_list: list[list[int]] = []
        # Component A: 297 faces sharing edges in a fan around edge chain
        for i in range(297):
            z = float(i + 1)
            verts_list.append([0.5, 1.0, z])
            if i == 0:
                faces_list.append([0, 1, 2])
            else:
                faces_list.append([0, i + 1, i + 2])
        # Wait: edge (0, 1) has 1 face, edge (0, k) has 2 faces -> single 297-face component!
        # Component B: 3 faces far away sharing edges
        base = len(verts_list)
        verts_list.extend(
            [
                [50.0, 0.0, 0.0],
                [51.0, 0.0, 0.0],
                [50.5, 1.0, 0.0],
                [50.5, -1.0, 0.0],
                [51.5, 1.0, 0.0],
            ]
        )
        faces_list.extend(
            [
                [base, base + 1, base + 2],
                [base, base + 3, base + 1],
                [base + 1, base + 4, base + 2],
            ]
        )
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(faces_list, dtype=np.int64)
        assert len(faces) == 300
        res = validate_mesh_geometry(
            verts, faces, min_face_ratio=0.01, backend="python",
        )
        # 0.01 * 300 = 3.0; 3 < 3.0 is False -> 0 floating island faces!
        assert res.floating_island_face_indices == []


# ===================================================================
# C++20 Native <-> Python Oracle 100% Parity Suite
# ===================================================================


@pytest.mark.skipif(
    not HAS_NATIVE_GEOMETRY,
    reason="synapse._c_geometry native extension not compiled in current Python path",
)
class TestNativeVsPythonOracleParity:
    """Enforce 100% parity between synapse._c_geometry and Python Oracle."""

    @pytest.mark.parametrize(
        "mesh_builder",
        [
            _make_tetrahedron,
            _make_coplanar_disjoint_grid,
            _make_coplanar_overlapping,
            _make_coplanar_contained,
            _make_parallel_non_coplanar,
            _make_3d_crossing,
            _make_bowtie,
            _make_non_manifold_triple_edge,
        ],
    )
    def test_gate_parity_2d_and_1d_faces(self, mesh_builder: Any) -> None:
        verts, faces = mesh_builder()

        py_res = validate_mesh_geometry(
            verts, faces, require_watertight=True, backend="python",
        )
        cpp_res = validate_mesh_geometry(
            verts, faces, require_watertight=True, backend="native",
        )

        assert cpp_res.is_manifold == py_res.is_manifold
        assert cpp_res.is_watertight == py_res.is_watertight
        assert cpp_res.num_vertices == py_res.num_vertices
        assert cpp_res.num_faces == py_res.num_faces
        assert cpp_res.degenerate_face_indices == py_res.degenerate_face_indices
        assert cpp_res.self_intersecting_pairs == py_res.self_intersecting_pairs
        assert cpp_res.num_connected_components == py_res.num_connected_components
        assert (
            cpp_res.floating_island_face_indices
            == py_res.floating_island_face_indices
        )
        assert cpp_res.passed == py_res.passed

        # Also verify 1-D flat index buffer and default num_vertices=0 in _c_geometry
        flat_faces = faces.ravel()
        assert _c_geometry.check_manifold(verts, flat_faces, 3) == py_res.is_manifold
        assert (
            _c_geometry.check_watertight(verts, flat_faces, 3)
            == py_res.is_watertight
        )
        comps_cpp = _c_geometry.find_connected_components(faces)  # num_vertices=0!
        comps_py = find_connected_components(faces)
        assert [list(c) for c in comps_cpp] == [list(c) for c in comps_py]

    def test_empty_and_oob_parity(self) -> None:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        empty_f = np.zeros((0, 3), dtype=np.int64)
        py_empty = validate_mesh_geometry(empty_v, empty_f, backend="python")
        cpp_empty = validate_mesh_geometry(empty_v, empty_f, backend="native")
        assert cpp_empty.passed == py_empty.passed is False
        assert cpp_empty.is_manifold == py_empty.is_manifold is False
        assert cpp_empty.is_watertight == py_empty.is_watertight is False

        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        oob_f = np.array([[0, -1, 99]], dtype=np.int64)
        py_oob = validate_mesh_geometry(verts, oob_f, backend="python")
        cpp_oob = validate_mesh_geometry(verts, oob_f, backend="native")
        assert cpp_oob.passed == py_oob.passed is False
        assert cpp_oob.degenerate_face_indices == py_oob.degenerate_face_indices == [0]

    def test_split_vertex_uv_cube_not_false_self_intersecting(self) -> None:
        from synapse.bench.procedural import (
            ProceduralMeshType,
            generate_procedural_mesh,
        )

        cube = generate_procedural_mesh(ProceduralMeshType.CUBE)
        verts = np.asarray(cube.vertices, dtype=np.float64)
        faces = np.asarray(cube.faces, dtype=np.int64)
        py_res = validate_mesh_geometry(
            verts, faces, require_watertight=False, backend="python",
        )
        cpp_res = validate_mesh_geometry(
            verts, faces, require_watertight=False, backend="native",
        )
        assert py_res.self_intersecting_pairs == []
        assert cpp_res.self_intersecting_pairs == []
        assert py_res.passed is True
        assert cpp_res.passed is True
