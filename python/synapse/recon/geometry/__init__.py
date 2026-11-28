"""Ground Truth Oracle for geometric algorithm correctness verification.

This package provides pure-Python (NumPy) reference implementations of mesh
geometry checks (manifold, watertight, degenerate faces, self-intersections,
connected components, etc.).

**This is NOT production code.**  Normal workflows use the C++20 native kernel
(``aether_geo`` via ``@aether//geo``) for actual geometry validation.
If the C++ extension is unavailable, the geometry gate raises ``ImportError``
rather than falling back to this module.

This Python Oracle exists solely to:

1. Serve as the algorithm correctness reference (ground truth).
2. Enable parity testing against the C++ implementation
   (see ``tests/test_geometry_oracle_parity.py``).

Additionally includes: area-weighted normal estimation, Kabsch/ICP 6-DoF rigid
alignment, and quad-dominant remeshing — utilities used by the recon pipeline.
"""

from __future__ import annotations

from synapse.recon.geometry.decimate import _qem_decimate
from synapse.recon.geometry.icp import refine_pose_6dof
from synapse.recon.geometry.intersections import check_self_intersections
from synapse.recon.geometry.metrics import (
    _triangulate_faces,
    compute_euler_characteristic,
    count_non_manifold_edges,
    count_non_manifold_vertices,
    discrete_mean_curvature_variance,
    surface_roughness_var,
)
from synapse.recon.geometry.remesh import quad_remesh
from synapse.recon.geometry.topology import (
    check_degenerate_faces,
    check_manifold,
    check_watertight,
    compute_normals,
    find_connected_components,
    prune_floating_islands,
    validate_recon_mesh_manifold,
)
from synapse.recon.geometry.uv import (
    _compute_face_normals,
    _generate_uvs,
    _unwrap_uv_atlas_with_seams,
    _weld_vertex_positions,
    compute_seam_discontinuity_deg,
    compute_texel_density_variance,
    compute_uv_area_distortion_ratio,
)

__all__ = [
    "_compute_face_normals",
    "_generate_uvs",
    "_qem_decimate",
    "_triangulate_faces",
    "_unwrap_uv_atlas_with_seams",
    "_weld_vertex_positions",
    "check_degenerate_faces",
    "check_manifold",
    "check_self_intersections",
    "check_watertight",
    "compute_euler_characteristic",
    "compute_normals",
    "compute_seam_discontinuity_deg",
    "compute_texel_density_variance",
    "compute_uv_area_distortion_ratio",
    "count_non_manifold_edges",
    "count_non_manifold_vertices",
    "discrete_mean_curvature_variance",
    "find_connected_components",
    "prune_floating_islands",
    "quad_remesh",
    "refine_pose_6dof",
    "surface_roughness_var",
    "validate_recon_mesh_manifold",
]
