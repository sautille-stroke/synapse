"""synapse.recon — Video-to-3D Inverse Reconstruction & Asset Distillation.

Provides native implementations for 4D point tracking, 6-DoF camera trajectory
solving, watertight manifold mesh verification, quad-dominant remeshing,
zero-slip foot kinematics IK, and intrinsic material de-lighting.
"""

from __future__ import annotations

from synapse.recon.backend import (
    SynapseReconBackend,
    get_recon_backend,
)
from synapse.recon.delight import (
    blend_multiview_seams,
    decompose_intrinsic,
    delight,
    delight_multiview_albedo,
    evaluate_spherical_harmonics_basis,
)
from synapse.recon.geometry import (
    check_degenerate_faces,
    check_manifold,
    check_self_intersections,
    check_watertight,
    compute_euler_characteristic,
    compute_normals,
    count_non_manifold_edges,
    count_non_manifold_vertices,
    discrete_mean_curvature_variance,
    find_connected_components,
    prune_floating_islands,
    quad_remesh,
    refine_pose_6dof,
    surface_roughness_var,
    validate_recon_mesh_manifold,
)
from synapse.recon.gs_quantizer import (
    GaussianSplatCloud,
    QuantizationMetrics,
    Quantized3DGS,
    dequantize_3dgs,
    quantize_3dgs,
)
from synapse.recon.io import (
    VideoMetadata,
    export_recon_fgpp_sequence,
    load_video_frames,
)
from synapse.recon.kinematics import (
    solve_two_bone_ik,
    solve_zero_slip_ik,
    validate_foot_slide_gate,
)
from synapse.recon.multiview import (
    Hunyuan3DReconstructionAdapter,
    MultiViewInitializer,
    TrellisReconstructionAdapter,
    select_adaptive_keyframes,
)
from synapse.recon.pipeline import ReconPipeline
from synapse.recon.tracking import (
    anchor_metric_scale,
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
from synapse.recon.usd_packager import (
    UsdPackageSpec,
    UsdPackageValidationResult,
    build_rfc0058_usda,
    validate_usd_package,
)

__all__ = [
    "CameraExtrinsics",
    "CameraTrajectory",
    "DelightResult",
    "FootContactInterval",
    "GaussianSplatCloud",
    "Hunyuan3DReconstructionAdapter",
    "MeshGeometry",
    "MocapKinematics",
    "MultiViewInitializer",
    "QuantizationMetrics",
    "Quantized3DGS",
    "ReconPipeline",
    "ReconResult",
    "SynapseReconBackend",
    "TrellisReconstructionAdapter",
    "UsdPackageSpec",
    "UsdPackageValidationResult",
    "VideoMetadata",
    "anchor_metric_scale",
    "blend_multiview_seams",
    "build_rfc0058_usda",
    "get_recon_backend",
    "check_degenerate_faces",
    "check_manifold",
    "check_self_intersections",
    "check_watertight",
    "compute_euler_characteristic",
    "compute_normals",
    "count_non_manifold_edges",
    "count_non_manifold_vertices",
    "dequantize_3dgs",
    "discrete_mean_curvature_variance",
    "decompose_intrinsic",
    "delight",
    "delight_multiview_albedo",
    "evaluate_spherical_harmonics_basis",
    "export_recon_fgpp_sequence",
    "find_connected_components",
    "load_video_frames",
    "prune_floating_islands",
    "quad_remesh",
    "quantize_3dgs",
    "refine_pose_6dof",
    "select_adaptive_keyframes",
    "surface_roughness_var",
    "solve_camera_trajectory",
    "solve_point_tracks",
    "solve_two_bone_ik",
    "solve_zero_slip_ik",
    "validate_foot_slide_gate",
    "validate_recon_mesh_manifold",
    "validate_usd_package",
]
