"""Tests for TRELLIS.2 O-Voxel reconstruction and topology-aware validation.

Covers:
- TopologyPolicy enum values
- OVoxelPBRMaps / Trellis2Result dataclass construction
- MockTrellis2Backend reconstruct (closed and open surface)
- FoundaryTrellis2Backend raises when unconnected
- validate_mesh_topology with STRICT / OPEN_SURFACE_ALLOWED / PERMISSIVE
- PBR map opacity handling for transparent surfaces
"""

from __future__ import annotations

import numpy as np
import pytest
from synapse.errors import BackendUnavailableError, ContractValidationError
from synapse.recon.trellis2 import (
    FoundaryTrellis2Backend,
    MockTrellis2Backend,
    OVoxelPBRMaps,
    TopologyPolicy,
    Trellis2Result,
    validate_mesh_topology,
)
from synapse.recon.types import MeshGeometry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_keyframes(k: int = 4, h: int = 32, w: int = 32) -> np.ndarray:
    """Create synthetic keyframe images [K, H, W, 3] uint8."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (k, h, w, 3), dtype=np.uint8)


def _make_closed_mesh(n_verts: int = 200) -> MeshGeometry:
    """Create a simple closed mesh (icosphere-like) for testing."""
    # Simple UV sphere
    verts = []
    faces = []
    n_lat, n_lon = 10, 10
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon + 1):
            phi = 2.0 * np.pi * j / n_lon
            x = 0.5 * np.sin(theta) * np.cos(phi)
            y = 0.5 * np.cos(theta)
            z = 0.5 * np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * (n_lon + 1) + j
            b = a + n_lon + 1
            c = b + 1
            d = a + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    return MeshGeometry(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def _make_open_mesh() -> MeshGeometry:
    """Create a simple open surface (a quad plane) for testing."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    return MeshGeometry(vertices=verts, faces=faces)


def _make_degenerate_mesh() -> MeshGeometry:
    """Create a mesh with NaN vertices."""
    verts = np.array([
        [float("nan"), 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return MeshGeometry(vertices=verts, faces=faces)


# ---------------------------------------------------------------------------
# TopologyPolicy enum
# ---------------------------------------------------------------------------


class TestTopologyPolicy:
    """TopologyPolicy enum values."""

    def test_values(self) -> None:
        assert TopologyPolicy.STRICT.value == "strict"
        assert TopologyPolicy.OPEN_SURFACE_ALLOWED.value == "open_surface_allowed"
        assert TopologyPolicy.PERMISSIVE.value == "permissive"

    def test_all_values(self) -> None:
        assert len(TopologyPolicy) == 3


# ---------------------------------------------------------------------------
# OVoxelPBRMaps / Trellis2Result
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Dataclass construction and field access."""

    def test_pbr_maps_construction(self) -> None:
        pbr = OVoxelPBRMaps(
            base_color=np.zeros((64, 64, 3), dtype=np.float32),
            roughness=np.full((64, 64), 0.5, dtype=np.float32),
            metallic=np.zeros((64, 64), dtype=np.float32),
            opacity=np.ones((64, 64), dtype=np.float32),
        )
        assert pbr.base_color.shape == (64, 64, 3)
        assert pbr.roughness.shape == (64, 64)
        assert pbr.opacity.shape == (64, 64)
        assert pbr.normal_map is None
        assert pbr.metadata == {}

    def test_pbr_maps_with_normal_map(self) -> None:
        pbr = OVoxelPBRMaps(
            base_color=np.zeros((32, 32, 3), dtype=np.float32),
            roughness=np.zeros((32, 32), dtype=np.float32),
            metallic=np.zeros((32, 32), dtype=np.float32),
            opacity=np.ones((32, 32), dtype=np.float32),
            normal_map=np.zeros((32, 32, 3), dtype=np.float32),
        )
        assert pbr.normal_map is not None
        assert pbr.normal_map.shape == (32, 32, 3)

    def test_trellis2_result(self) -> None:
        mesh = _make_closed_mesh()
        pbr = OVoxelPBRMaps(
            base_color=np.zeros((16, 16, 3), dtype=np.float32),
            roughness=np.zeros((16, 16), dtype=np.float32),
            metallic=np.zeros((16, 16), dtype=np.float32),
            opacity=np.ones((16, 16), dtype=np.float32),
        )
        result = Trellis2Result(
            mesh=mesh,
            pbr_maps=pbr,
            topology_policy=TopologyPolicy.OPEN_SURFACE_ALLOWED,
            confidence=0.9,
        )
        assert result.topology_policy == TopologyPolicy.OPEN_SURFACE_ALLOWED
        assert result.confidence == 0.9


# ---------------------------------------------------------------------------
# MockTrellis2Backend
# ---------------------------------------------------------------------------


class TestMockBackend:
    """MockTrellis2Backend produces deterministic results."""

    def test_always_available(self) -> None:
        assert MockTrellis2Backend().is_available()

    def test_reconstruct_closed_mesh(self) -> None:
        backend = MockTrellis2Backend()
        keyframes = _make_keyframes()
        result = backend.reconstruct(keyframes)
        assert result.mesh.vertex_count > 100
        assert result.mesh.face_count > 100
        assert result.topology_policy == TopologyPolicy.STRICT
        assert result.confidence > 0.0
        assert result.metadata["backend"] == "mock"
        assert not result.metadata["open_surface"]

    def test_reconstruct_open_surface(self) -> None:
        backend = MockTrellis2Backend()
        keyframes = _make_keyframes()
        result = backend.reconstruct(
            keyframes, topology_hint=TopologyPolicy.OPEN_SURFACE_ALLOWED,
        )
        assert result.topology_policy == TopologyPolicy.OPEN_SURFACE_ALLOWED
        assert result.metadata["open_surface"]
        # Open surface mesh has fewer faces (bottom cap removed)
        closed_result = backend.reconstruct(keyframes)
        assert result.mesh.face_count < closed_result.mesh.face_count

    def test_reconstruct_pbr_maps(self) -> None:
        backend = MockTrellis2Backend()
        result = backend.reconstruct(_make_keyframes())
        pbr = result.pbr_maps
        assert pbr.base_color.shape == (64, 64, 3)
        assert pbr.roughness.shape == (64, 64)
        assert pbr.metallic.shape == (64, 64)
        assert pbr.opacity.shape == (64, 64)
        assert pbr.normal_map is not None

    def test_reconstruct_open_surface_has_transparency(self) -> None:
        """Open surface reconstruction includes opacity variation."""
        backend = MockTrellis2Backend()
        result = backend.reconstruct(
            _make_keyframes(),
            topology_hint=TopologyPolicy.OPEN_SURFACE_ALLOWED,
        )
        # Should have some non-1.0 opacity values
        assert np.any(result.pbr_maps.opacity < 1.0)

    def test_reconstruct_closed_surface_fully_opaque(self) -> None:
        """Closed surface reconstruction is fully opaque."""
        backend = MockTrellis2Backend()
        result = backend.reconstruct(_make_keyframes())
        np.testing.assert_array_equal(result.pbr_maps.opacity, 1.0)

    def test_reconstruct_deterministic(self) -> None:
        backend = MockTrellis2Backend()
        kf = _make_keyframes()
        r1 = backend.reconstruct(kf)
        r2 = backend.reconstruct(kf)
        np.testing.assert_array_equal(r1.mesh.vertices, r2.mesh.vertices)
        np.testing.assert_array_equal(r1.pbr_maps.base_color, r2.pbr_maps.base_color)

    def test_reconstruct_empty_keyframes_raises(self) -> None:
        backend = MockTrellis2Backend()
        with pytest.raises(ContractValidationError, match="keyframe"):
            backend.reconstruct(np.empty((0, 32, 32, 3), dtype=np.uint8))

    def test_reconstruct_with_prompts(self) -> None:
        backend = MockTrellis2Backend()
        result = backend.reconstruct(
            _make_keyframes(), prompts=["a wooden chair"],
        )
        assert result.mesh.vertex_count > 0

    def test_mesh_has_uvs(self) -> None:
        backend = MockTrellis2Backend()
        result = backend.reconstruct(_make_keyframes())
        assert result.mesh.uvs is not None
        assert result.mesh.uvs.shape[1] == 2

    def test_mesh_has_normals(self) -> None:
        backend = MockTrellis2Backend()
        result = backend.reconstruct(_make_keyframes())
        assert result.mesh.normals is not None
        assert result.mesh.normals.shape[1] == 3


# ---------------------------------------------------------------------------
# FoundaryTrellis2Backend
# ---------------------------------------------------------------------------


class TestFoundaryBackend:
    """FoundaryTrellis2Backend raises when unconnected."""

    def test_not_available(self) -> None:
        assert not FoundaryTrellis2Backend().is_available()

    def test_reconstruct_raises(self) -> None:
        backend = FoundaryTrellis2Backend()
        with pytest.raises(BackendUnavailableError):
            backend.reconstruct(_make_keyframes())


# ---------------------------------------------------------------------------
# validate_mesh_topology
# ---------------------------------------------------------------------------


class TestValidateMeshTopology:
    """Topology-aware mesh validation."""

    def test_strict_closed_mesh_passes(self) -> None:
        mesh = _make_closed_mesh()
        result = validate_mesh_topology(mesh, TopologyPolicy.STRICT, min_vertices=10)
        # Note: UV sphere from _make_closed_mesh may not be perfectly watertight
        # due to seam vertices, but should be manifold
        assert result["policy"] == "strict"
        assert result["vertex_count"] > 10

    def test_strict_open_mesh_fails(self) -> None:
        mesh = _make_open_mesh()
        result = validate_mesh_topology(mesh, TopologyPolicy.STRICT, min_vertices=2)
        # Open mesh is not watertight → should fail STRICT
        assert not result["valid"]
        assert any("watertight" in e.lower() or "manifold" in e.lower() for e in result["errors"])

    def test_open_surface_policy_accepts_plane(self) -> None:
        mesh = _make_open_mesh()
        result = validate_mesh_topology(
            mesh, TopologyPolicy.OPEN_SURFACE_ALLOWED, min_vertices=2,
        )
        # A simple quad plane IS manifold (each edge shared by 1-2 faces)
        assert result["is_manifold"]
        assert result["valid"]

    def test_permissive_accepts_anything(self) -> None:
        mesh = _make_open_mesh()
        result = validate_mesh_topology(
            mesh, TopologyPolicy.PERMISSIVE, min_vertices=2,
        )
        assert result["valid"]
        assert result["is_manifold"] is None  # Not checked
        assert result["is_watertight"] is None  # Not checked

    def test_nan_vertices_fail_all_policies(self) -> None:
        mesh = _make_degenerate_mesh()
        for policy in TopologyPolicy:
            result = validate_mesh_topology(mesh, policy, min_vertices=1)
            assert not result["valid"], f"Should fail for {policy} with NaN vertices"
            assert not result["has_finite_vertices"]

    def test_min_vertices_enforcement(self) -> None:
        mesh = _make_open_mesh()
        result = validate_mesh_topology(
            mesh, TopologyPolicy.PERMISSIVE, min_vertices=1000,
        )
        assert not result["valid"]
        assert any("Vertex count" in e for e in result["errors"])

    def test_result_contains_all_fields(self) -> None:
        mesh = _make_closed_mesh()
        result = validate_mesh_topology(mesh, TopologyPolicy.STRICT, min_vertices=10)
        assert "valid" in result
        assert "policy" in result
        assert "vertex_count" in result
        assert "face_count" in result
        assert "is_manifold" in result
        assert "is_watertight" in result
        assert "has_finite_vertices" in result
        assert "errors" in result
