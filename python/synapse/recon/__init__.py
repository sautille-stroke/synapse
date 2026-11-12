"""synapse.recon — Video-to-3D Inverse Reconstruction & Asset Distillation.

Provides native implementations for 4D point tracking, 6-DoF camera trajectory
solving, watertight manifold mesh verification, quad-dominant remeshing,
zero-slip foot kinematics IK, and intrinsic material de-lighting.
"""

from __future__ import annotations

from synapse.recon.delight import (
    decompose_intrinsic,
    delight,
    evaluate_spherical_harmonics_basis,
)
from synapse.recon.geometry import (
    check_degenerate_faces,
    check_manifold,
    check_self_intersections,
    check_watertight,
    compute_normals,
    find_connected_components,
    prune_floating_islands,
    quad_remesh,
    refine_pose_6dof,
)
from synapse.recon.kinematics import (
    solve_two_bone_ik,
    solve_zero_slip_ik,
)
from synapse.recon.pipeline import ReconPipeline
from synapse.recon.tracking import (
    solve_camera_trajectory,
    solve_point_tracks,
)
from synapse.recon.types import (
    CameraExtrinsics,
    CameraTrajectory,
    DelightResult,
    FootContactInterval,
    MeshGeometry,
    MocapKinematics,
    ReconResult,
)

__all__ = [
    "CameraExtrinsics",
    "CameraTrajectory",
    "DelightResult",
    "FootContactInterval",
    "MeshGeometry",
    "MocapKinematics",
    "ReconPipeline",
    "ReconResult",
    "check_degenerate_faces",
    "check_manifold",
    "check_self_intersections",
    "check_watertight",
    "compute_normals",
    "decompose_intrinsic",
    "delight",
    "evaluate_spherical_harmonics_basis",
    "find_connected_components",
    "prune_floating_islands",
    "quad_remesh",
    "refine_pose_6dof",
    "solve_camera_trajectory",
    "solve_point_tracks",
    "solve_two_bone_ik",
    "solve_zero_slip_ik",
]
