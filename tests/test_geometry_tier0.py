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

import os
import sys
from pathlib import Path

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
    quad_remesh,
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


class TestBuildNativeScript:
    """PR-62 (AUD-P0-02): Verify build_native.py dry-run and direct backend support."""

    def test_build_native_dry_run_exits_zero(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "build_native.py"
        res = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"build_native.py --dry-run failed: {res.stderr}"

    def test_build_native_direct_dry_run(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "build_native.py"
        res = subprocess.run(
            [sys.executable, str(script), "--dry-run", "--backend=direct"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"build_native.py --dry-run --backend=direct failed: {res.stderr}"

    def test_build_native_pybind11_missing_exit_30(self) -> None:
        """When pybind11 is absent, build_native.py must exit with code 30 (BACKEND_UNAVAILABLE).

        Tests the _emit_error path by directly invoking _find_pybind11_include
        and _build_direct_fallback with a rigged environment where pybind11
        is genuinely unreachable.
        """
        import importlib
        import importlib.util
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "build_native.py"

        # Create a wrapper that blocks pybind11 AND clears bazel cache path
        wrapper = (
            "import sys, os, importlib, importlib.util\n"
            "# Block pybind11 from import\n"
            "class _Block:\n"
            "    def find_module(self, n, p=None):\n"
            "        return self if n == 'pybind11' else None\n"
            "    def load_module(self, n):\n"
            "        raise ImportError('blocked')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "sys.modules.pop('pybind11', None)\n"
            "# Redirect HOME to a temp dir so bazel cache is not found\n"
            "import tempfile\n"
            "os.environ['HOME'] = tempfile.mkdtemp()\n"
            f"sys.argv = ['{script}', '--backend=direct']\n"
            f"spec = importlib.util.spec_from_file_location('build_native', r'{script}')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "sys.exit(mod.main())\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        # Should exit 30 when pybind11 is missing
        assert res.returncode == 30, (
            f"Expected exit code 30, got {res.returncode}.\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
        assert "agent_fix_hint" in res.stderr

    def test_build_native_includes_aether_core_spatial(self) -> None:
        """Dry-run must include aether/core/include and aether/spatial/include in compiler args."""
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "build_native.py"
        aether_root = repo_root.parent / "aether"
        if not aether_root.is_dir():
            pytest.skip("aether sibling directory not found")

        env = os.environ.copy()
        env.pop("AETHER_ROOT", None)
        res = subprocess.run(
            [sys.executable, str(script), "--dry-run", "--backend=direct"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0, f"dry-run failed: {res.stderr}"
        # Check that aether core and spatial includes appear in the compiler command
        output = res.stdout + res.stderr
        assert "core/include" in output, "aether/core/include not found in compiler args"
        assert "spatial/include" in output, "aether/spatial/include not found in compiler args"


@pytest.mark.skipif(
    sys.platform == "win32" or sys.platform == "darwin",
    reason="Real native compile requires Linux (ubuntu24)",
)
class TestBuildNativeRealCompile:
    """PR-84: Real compile test — build_native.py produces a loadable .so."""

    def test_real_direct_compile_and_load(self, tmp_path: Path) -> None:
        """build_native.py --backend=direct --output-dir=<tmp> produces a
        loadable _c_geometry .so with working validate_mesh exported."""
        import importlib
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "build_native.py"

        # Build into tmp_path
        res = subprocess.run(
            [
                sys.executable,
                str(script),
                "--backend=direct",
                f"--output-dir={tmp_path}",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=120,
        )
        assert res.returncode == 0, (
            f"build_native.py real compile failed (rc={res.returncode}):\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )

        # Locate the built .so
        so_files = list(tmp_path.glob("_c_geometry*"))
        assert len(so_files) >= 1, f"No _c_geometry .so found in {tmp_path}: {list(tmp_path.iterdir())}"

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("_c_geometry", str(so_files[0]))
        assert spec is not None, f"Failed to create module spec from {so_files[0]}"
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Verify validate_mesh_geometry is exported and callable
        assert hasattr(mod, "validate_mesh_geometry"), (
            f"Module does not export validate_mesh_geometry. Dir: {dir(mod)}"
        )

        # Smoke test: validate a simple tetrahedron
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
            dtype=np.int64,
        )
        result = mod.validate_mesh_geometry(
            vertices=verts,
            faces=faces,
            verts_per_face=3,
            require_watertight=True,
            min_area=1e-8,
            max_aspect_ratio=100.0,
            min_face_ratio=0.01,
        )
        assert isinstance(result, dict)
        assert result.get("is_manifold") is True
        assert result.get("passed") is True


# ===================================================================
# PR-80: New tests for tracking, geometry, and stages
# ===================================================================


class TestGeometryGateNativeBackend:
    """PR-80: backend="native" validate_recon_mesh_manifold hits C++ path
    without fallback exception (test-a)."""

    def test_native_backend_validate_recon_mesh_manifold_dict_access(self) -> None:
        """When _c_geometry returns a dict, validate_recon_mesh_manifold
        reads result["is_manifold"] without AttributeError."""
        import sys
        from unittest.mock import MagicMock

        import synapse
        from synapse.recon.geometry import validate_recon_mesh_manifold

        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int64)

        # Mock _c_geometry returning a dict (not an object with attributes)
        mock_cg = MagicMock()
        mock_cg.validate_mesh_geometry.return_value = {
            "is_manifold": True,
            "is_watertight": False,
            "degenerate_face_indices": [],
            "self_intersecting_pairs": [],
            "num_connected_components": 1,
            "floating_island_face_indices": [],
            "num_faces": 1,
        }

        old_mod = sys.modules.get("synapse._c_geometry")
        old_attr = getattr(synapse, "_c_geometry", None)
        sys.modules["synapse._c_geometry"] = mock_cg
        synapse._c_geometry = mock_cg  # type: ignore[attr-defined]
        try:
            result = validate_recon_mesh_manifold(v, f, backend="native")
            assert result is True
            mock_cg.validate_mesh_geometry.assert_called_once()
        finally:
            if old_mod is None:
                sys.modules.pop("synapse._c_geometry", None)
            else:
                sys.modules["synapse._c_geometry"] = old_mod
            if old_attr is None:
                if hasattr(synapse, "_c_geometry"):
                    delattr(synapse, "_c_geometry")
            else:
                synapse._c_geometry = old_attr  # type: ignore[attr-defined]

    def test_native_backend_validate_mesh_geometry_dict_access(self) -> None:
        """validate_mesh_geometry with backend='native' reads dict keys
        without AttributeError."""
        import sys
        from unittest.mock import MagicMock

        import synapse

        mock_cg = MagicMock()
        mock_cg.validate_mesh_geometry.return_value = {
            "is_manifold": True,
            "is_watertight": True,
            "degenerate_face_indices": np.array([], dtype=np.int64),
            "self_intersecting_pairs": [],
            "num_connected_components": 1,
            "floating_island_face_indices": np.array([], dtype=np.int64),
            "num_faces": 4,
        }

        v = np.array([
            [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
        ], dtype=np.float64)
        f = np.array([
            [0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2],
        ], dtype=np.int64)

        old_mod = sys.modules.get("synapse._c_geometry")
        old_attr = getattr(synapse, "_c_geometry", None)
        sys.modules["synapse._c_geometry"] = mock_cg
        synapse._c_geometry = mock_cg  # type: ignore[attr-defined]
        try:
            result = validate_mesh_geometry(v, f, backend="native")
            assert result.passed is True
            assert result.is_manifold is True
            mock_cg.validate_mesh_geometry.assert_called_once()
        finally:
            if old_mod is None:
                sys.modules.pop("synapse._c_geometry", None)
            else:
                sys.modules["synapse._c_geometry"] = old_mod
            if old_attr is None:
                if hasattr(synapse, "_c_geometry"):
                    delattr(synapse, "_c_geometry")
            else:
                synapse._c_geometry = old_attr  # type: ignore[attr-defined]


# ===================================================================
# PR-80: quad_remesh no duplicate-vertex faces (test-b)
# ===================================================================


class TestQuadRemeshNoDuplicateVertex:
    """PR-80: quad_remesh output has no duplicate-vertex faces."""

    def test_odd_triangle_count_no_duplicate_vertex_quads(self) -> None:
        """When triangle count is odd, quad_remesh must not generate
        [a,b,c,c] duplicate-vertex quad faces. Uses Catmull-Clark
        midpoint subdivision instead."""
        # 3 triangles forming a fan — one will remain unpaired
        v = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [-0.5, 1.0, 0.0],
        ], dtype=np.float32)
        f = np.array([
            [0, 1, 2],  # can pair with tri 1
            [0, 2, 3],  # can pair with tri 0
            [0, 3, 1],  # unpaired (odd one out)
        ], dtype=np.int32)

        new_v, new_f = quad_remesh(v, f)

        # All output faces must be quads (4 vertices each)
        assert new_f.shape[1] == 4, f"Expected quad faces, got {new_f.shape[1]}-gons"

        # No face should have duplicate vertices
        for face_idx in range(len(new_f)):
            face = new_f[face_idx]
            unique_verts = set(int(x) for x in face)
            assert len(unique_verts) == 4, (
                f"Face {face_idx} has duplicate vertices: {face.tolist()}"
            )

    def test_even_triangle_count_all_quads(self) -> None:
        """Cube (12 triangles) should remesh to 6 quads with no duplicates."""
        v = np.array([
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ], dtype=np.float32)
        f = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        new_v, new_f = quad_remesh(v, f)
        assert new_f.shape[1] == 4
        assert len(new_f) == 6

        for face_idx in range(len(new_f)):
            face = new_f[face_idx]
            unique_verts = set(int(x) for x in face)
            assert len(unique_verts) == 4, (
                f"Face {face_idx} has duplicate vertices: {face.tolist()}"
            )

    def test_single_triangle_subdivision(self) -> None:
        """A single triangle should be subdivided into 3 quads."""
        v = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int32)

        new_v, new_f = quad_remesh(v, f)

        # No pairing possible → all tris subdivided
        # Single triangle → 3 quads, 4 new verts
        assert new_f.shape[1] == 4
        for face_idx in range(len(new_f)):
            face = new_f[face_idx]
            unique_verts = set(int(x) for x in face)
            assert len(unique_verts) == 4, (
                f"Face {face_idx} has duplicate vertices: {face.tolist()}"
            )
