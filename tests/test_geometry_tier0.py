"""Tier 0 Geometric Topology Gate — comprehensive unit tests.

Tests all Python Oracle functions in ``synapse.recon.geometry`` and the
unified ``synapse.inspect.geometry_gate.validate_mesh_geometry`` against
a battery of known-good and known-bad mesh topologies.

Test fixtures include:
  - Valid tetrahedron (closed, watertight, manifold)
  - Valid open plane (manifold but NOT watertight)
  - Degenerate needle triangles (area → 0)
  - Non-manifold "bowtie" pinch vertex
  - Self-intersecting crossed quads
  - Floating debris islands
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.inspect.geometry_gate import validate_mesh_geometry
from synapse.inspect.schemas import GeometricValidationResult
from synapse.recon.geometry import (
    check_degenerate_faces,
    check_manifold,
    check_self_intersections,
    check_watertight,
    find_connected_components,
    prune_floating_islands,
)

# ===================================================================
# Fixtures — canonical test meshes
# ===================================================================


@pytest.fixture()
def tetrahedron():
    """Regular tetrahedron — closed, watertight, 2-manifold, zero defects."""
    verts = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ])
    return verts, faces


@pytest.fixture()
def open_plane():
    """Single triangle — manifold but not watertight (boundary edges)."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2]])
    return verts, faces


@pytest.fixture()
def needle_triangle():
    """Extremely thin needle triangle — should be flagged as degenerate."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1e-12, 0.0],  # Near-zero area
    ])
    faces = np.array([[0, 1, 2]])
    return verts, faces


@pytest.fixture()
def bowtie_mesh():
    """Bowtie / hourglass — two triangles sharing only a single vertex.

    This is a non-manifold pinch point at vertex 2.
    """
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],  # shared pinch vertex
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [2, 3, 4],
    ])
    return verts, faces


@pytest.fixture()
def self_intersecting_mesh():
    """Two triangles that cross through each other in 3D space."""
    verts = np.array([
        # Triangle 0: lies in the XY plane
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        # Triangle 1: lies in the XZ plane, crossing through tri 0
        [-1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ])
    return verts, faces


@pytest.fixture()
def mesh_with_debris():
    """Tetrahedron + tiny floating debris cube far away."""
    # Main body: tetrahedron
    main_verts = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ])
    main_faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ])
    # Debris: single isolated triangle far away
    debris_verts = np.array([
        [100.0, 100.0, 100.0],
        [100.1, 100.0, 100.0],
        [100.0, 100.1, 100.0],
    ])
    debris_faces = np.array([[4, 5, 6]])

    verts = np.vstack([main_verts, debris_verts])
    faces = np.vstack([main_faces, debris_faces])
    return verts, faces


# ===================================================================
# Tests — check_degenerate_faces
# ===================================================================


class TestCheckDegenerateFaces:
    """Tests for degenerate / needle triangle detection."""

    def test_healthy_tetrahedron(self, tetrahedron):
        verts, faces = tetrahedron
        result = check_degenerate_faces(verts, faces)
        assert len(result) == 0

    def test_needle_detected(self, needle_triangle):
        verts, faces = needle_triangle
        result = check_degenerate_faces(verts, faces, min_area=1e-8)
        assert len(result) == 1
        assert result[0] == 0

    def test_zero_area_face(self):
        """Three collinear points — zero area face."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        result = check_degenerate_faces(verts, faces)
        assert len(result) == 1

    def test_high_aspect_ratio(self):
        """Very elongated but nonzero-area triangle."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [500.0, 0.01, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        result = check_degenerate_faces(
            verts, faces, min_area=1e-10, max_aspect_ratio=50.0,
        )
        assert len(result) == 1

    def test_empty_mesh(self):
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        result = check_degenerate_faces(verts, faces)
        assert len(result) == 0

    def test_quad_degenerate_second_triangle(self):
        """Quad where v0,v1,v2 form a healthy sub-tri but v0,v2,v3 is a degenerate needle."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ])
        # Make v3 nearly collinear with v0→v2 → degenerate needle
        verts[3] = verts[0] + (verts[2] - verts[0]) * 2.0 + np.array([0, 1e-14, 0])
        faces = np.array([[0, 1, 2, 3]])
        result = check_degenerate_faces(verts, faces, min_area=1e-8)
        assert len(result) >= 1
        assert 0 in result


# ===================================================================
# Tests — check_self_intersections
# ===================================================================


class TestCheckSelfIntersections:
    """Tests for triangle-triangle intersection detection."""

    def test_no_intersection_tetrahedron(self, tetrahedron):
        verts, faces = tetrahedron
        pairs = check_self_intersections(verts, faces)
        assert len(pairs) == 0

    def test_crossing_triangles_detected(self, self_intersecting_mesh):
        verts, faces = self_intersecting_mesh
        pairs = check_self_intersections(verts, faces)
        assert len(pairs) >= 1
        assert (0, 1) in pairs or (1, 0) in pairs

    def test_adjacent_faces_not_flagged(self):
        """Two triangles sharing an edge should NOT be flagged."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 3, 1],
        ])
        pairs = check_self_intersections(verts, faces)
        assert len(pairs) == 0

    def test_empty_mesh(self):
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        pairs = check_self_intersections(verts, faces)
        assert len(pairs) == 0

    def test_quad_self_intersection_second_triangle(self):
        """Two quads: first sub-triangles don't intersect but second sub-triangles do."""
        verts = np.array([
            # Quad A: XY plane
            [-2.0, -1.0, 0.0],   # 0
            [2.0, -1.0, 0.0],    # 1
            [2.0, 1.0, 0.0],     # 2
            [-2.0, 1.0, 0.0],    # 3
            # Quad B: crosses quad A in XZ plane
            [0.0, 0.0, -2.0],    # 4
            [10.0, 0.0, -2.0],   # 5 — far right
            [10.0, 0.0, 2.0],    # 6 — far right
            [0.0, 0.0, 2.0],     # 7
        ])
        faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        pairs = check_self_intersections(verts, faces)
        assert len(pairs) >= 1
        assert (0, 1) in pairs


# ===================================================================
# Tests — find_connected_components
# ===================================================================


class TestFindConnectedComponents:
    """Tests for mesh connected component decomposition."""

    def test_single_component(self, tetrahedron):
        _, faces = tetrahedron
        components = find_connected_components(faces)
        assert len(components) == 1
        assert len(components[0]) == 4

    def test_two_components(self, mesh_with_debris):
        _, faces = mesh_with_debris
        components = find_connected_components(faces)
        assert len(components) == 2
        # Largest component first
        assert len(components[0]) == 4
        assert len(components[1]) == 1

    def test_isolated_triangles(self):
        """Three completely disconnected triangles."""
        faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        components = find_connected_components(faces)
        assert len(components) == 3

    def test_empty_mesh(self):
        faces = np.zeros((0, 3), dtype=np.int64)
        components = find_connected_components(faces)
        assert len(components) == 0


# ===================================================================
# Tests — prune_floating_islands
# ===================================================================


class TestPruneFloatingIslands:
    """Tests for floating debris removal."""

    def test_no_pruning_needed(self, tetrahedron):
        verts, faces = tetrahedron
        clean_v, clean_f, mask = prune_floating_islands(verts, faces)
        assert len(clean_f) == len(faces)
        assert np.all(mask)

    def test_debris_removed(self, mesh_with_debris):
        verts, faces = mesh_with_debris
        clean_v, clean_f, mask = prune_floating_islands(
            verts, faces, min_face_ratio=0.25,
        )
        # Debris (1 face out of 5 = 20%) < 25% threshold → pruned
        assert len(clean_f) == 4
        assert len(clean_v) == 4

    def test_vertex_remapping(self, mesh_with_debris):
        verts, faces = mesh_with_debris
        clean_v, clean_f, _ = prune_floating_islands(
            verts, faces, min_face_ratio=0.25,
        )
        # All face indices should be valid for cleaned vertices
        assert np.all(clean_f >= 0)
        assert np.all(clean_f < len(clean_v))


# ===================================================================
# Tests — check_manifold / check_watertight (existing, but verify)
# ===================================================================


class TestManifoldWatertight:
    """Verify existing manifold/watertight checks with new test meshes."""

    def test_tetrahedron_manifold_watertight(self, tetrahedron):
        verts, faces = tetrahedron
        assert check_manifold(verts, faces)
        assert check_watertight(verts, faces)

    def test_open_plane_manifold_not_watertight(self, open_plane):
        verts, faces = open_plane
        assert check_manifold(verts, faces)
        assert not check_watertight(verts, faces)

    def test_bowtie_not_manifold(self, bowtie_mesh):
        verts, faces = bowtie_mesh
        # Bowtie has a pinch vertex at vertex 2 — must fail both manifold & watertight
        assert not check_manifold(verts, faces)
        assert not check_watertight(verts, faces)


# ===================================================================
# Tests — validate_mesh_geometry (unified gate)
# ===================================================================


class TestValidateMeshGeometry:
    """Tests for the unified Tier 0 gate function."""

    def test_healthy_tetrahedron_passes(self, tetrahedron):
        verts, faces = tetrahedron
        result = validate_mesh_geometry(verts, faces, backend="python")
        assert isinstance(result, GeometricValidationResult)
        assert result.passed is True
        assert result.is_manifold is True
        assert result.is_watertight is True
        assert len(result.degenerate_face_indices) == 0
        assert len(result.self_intersecting_pairs) == 0
        assert result.num_connected_components == 1
        assert len(result.failure_reasons) == 0
        assert result.num_vertices == 4
        assert result.num_faces == 4

    def test_needle_fails(self, needle_triangle):
        verts, faces = needle_triangle
        result = validate_mesh_geometry(verts, faces, backend="python")
        assert result.passed is False
        assert len(result.degenerate_face_indices) > 0
        assert any("degenerate" in r.lower() for r in result.failure_reasons)

    def test_self_intersection_fails(self, self_intersecting_mesh):
        verts, faces = self_intersecting_mesh
        result = validate_mesh_geometry(verts, faces, backend="python")
        assert result.passed is False
        assert len(result.self_intersecting_pairs) > 0
        assert any("intersect" in r.lower() for r in result.failure_reasons)

    def test_debris_detected(self, mesh_with_debris):
        verts, faces = mesh_with_debris
        result = validate_mesh_geometry(
            verts, faces, min_face_ratio=0.25, backend="python",
        )
        assert result.num_connected_components == 2
        assert len(result.floating_island_face_indices) > 0

    def test_require_watertight_open_fails(self, open_plane):
        verts, faces = open_plane
        result = validate_mesh_geometry(
            verts, faces, require_watertight=True, backend="python",
        )
        assert result.passed is False
        assert result.is_watertight is False
        assert any("watertight" in r.lower() for r in result.failure_reasons)

    def test_result_serializable(self, tetrahedron):
        verts, faces = tetrahedron
        result = validate_mesh_geometry(verts, faces, backend="python")
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "passed" in d
        assert "is_manifold" in d

    def test_auto_backend_removed(self, tetrahedron):
        """'auto' backend no longer exists — should raise ValueError."""
        verts, faces = tetrahedron
        with pytest.raises(ValueError, match="Invalid backend"):
            validate_mesh_geometry(verts, faces, backend="auto")  # type: ignore[arg-type]

    def test_native_backend_unavailable_raises(self, tetrahedron):
        """Native backend raises BackendUnavailableError(exit_code=30) when _c_geometry missing (P1-ARCH-03)."""
        import sys
        from unittest.mock import patch

        from synapse.errors import BackendUnavailableError

        verts, faces = tetrahedron
        with patch.dict(sys.modules, {"synapse._c_geometry": None}):
            with pytest.raises(BackendUnavailableError) as exc_info:
                validate_mesh_geometry(verts, faces, backend="native")
            assert exc_info.value.exit_code == 30
            assert "python scripts/build_native.py" in exc_info.value.agent_fix_hint
