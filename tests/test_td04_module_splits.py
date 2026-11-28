"""Structural and backward-compatibility tests for PR-TD04 module splits."""

from __future__ import annotations

from pathlib import Path

from synapse.pipeline import bundle_worker, manifest, sandbox
from synapse.recon import geometry
from synapse.recon.geometry import (
    decimate,
    icp,
    intersections,
    metrics,
    remesh,
    topology,
    uv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_geometry_package_reexports_and_identity() -> None:
    """``synapse.recon.geometry`` must re-export every split submodule symbol cleanly."""
    expected = {
        "compute_normals": topology.compute_normals,
        "check_manifold": topology.check_manifold,
        "check_watertight": topology.check_watertight,
        "check_degenerate_faces": topology.check_degenerate_faces,
        "find_connected_components": topology.find_connected_components,
        "prune_floating_islands": topology.prune_floating_islands,
        "validate_recon_mesh_manifold": topology.validate_recon_mesh_manifold,
        "check_self_intersections": intersections.check_self_intersections,
        "_qem_decimate": decimate._qem_decimate,
        "quad_remesh": remesh.quad_remesh,
        "_unwrap_uv_atlas_with_seams": uv._unwrap_uv_atlas_with_seams,
        "_generate_uvs": uv._generate_uvs,
        "compute_uv_area_distortion_ratio": uv.compute_uv_area_distortion_ratio,
        "_weld_vertex_positions": uv._weld_vertex_positions,
        "_compute_face_normals": uv._compute_face_normals,
        "compute_seam_discontinuity_deg": uv.compute_seam_discontinuity_deg,
        "compute_texel_density_variance": uv.compute_texel_density_variance,
        "refine_pose_6dof": icp.refine_pose_6dof,
        "compute_euler_characteristic": metrics.compute_euler_characteristic,
        "count_non_manifold_edges": metrics.count_non_manifold_edges,
        "count_non_manifold_vertices": metrics.count_non_manifold_vertices,
        "_triangulate_faces": metrics._triangulate_faces,
        "discrete_mean_curvature_variance": metrics.discrete_mean_curvature_variance,
        "surface_roughness_var": metrics.surface_roughness_var,
    }
    for name, fn in expected.items():
        assert getattr(geometry, name) is fn, f"Missing or mismatched re-export: {name}"
        assert name in geometry.__all__


def test_geometry_submodules_within_line_budget() -> None:
    """Every module in ``recon/geometry/`` must stay <= 600 lines."""
    geo_dir = REPO_ROOT / "python" / "synapse" / "recon" / "geometry"
    assert not (REPO_ROOT / "python" / "synapse" / "recon" / "geometry.py").exists()
    py_files = sorted(geo_dir.glob("*.py"))
    assert len(py_files) >= 6
    for py_file in py_files:
        line_count = len(py_file.read_text(encoding="utf-8").splitlines())
        assert line_count <= 600, f"{py_file.name} exceeds 600 lines ({line_count})"


def test_bundle_worker_split_reexports_and_line_budget() -> None:
    """``bundle_worker`` must re-export manifest/sandbox APIs and stay <= 800 lines."""
    assert bundle_worker.BundleManifest is manifest.BundleManifest
    assert bundle_worker.HardwareBudget is manifest.HardwareBudget
    assert bundle_worker.EngineSpec is manifest.EngineSpec
    assert bundle_worker.OutputSink is manifest.OutputSink
    assert bundle_worker.SceneSpec is manifest.SceneSpec
    assert bundle_worker.ResultManifest is manifest.ResultManifest
    assert bundle_worker.CLOSURE_TYPE_MAP is manifest.CLOSURE_TYPE_MAP
    assert bundle_worker.ClosureType is manifest.ClosureType
    assert bundle_worker._compute_sha256 is sandbox._compute_sha256
    assert bundle_worker._check_ancestor_symlinks is sandbox._check_ancestor_symlinks
    assert bundle_worker._extract_manifest_traceboard_param is sandbox._extract_manifest_traceboard_param
    assert bundle_worker._validate_manifest_traceboard_dir is sandbox._validate_manifest_traceboard_dir

    pipeline_dir = REPO_ROOT / "python" / "synapse" / "pipeline"
    for rel, max_lines in (
        ("bundle_worker.py", 800),
        ("manifest.py", 600),
        ("sandbox.py", 600),
        ("stages/core.py", 600),
    ):
        target = pipeline_dir / rel
        assert target.is_file(), f"Missing split module: {target}"
        count = len(target.read_text(encoding="utf-8").splitlines())
        assert count <= max_lines, f"{rel} has {count} lines (limit {max_lines})"
