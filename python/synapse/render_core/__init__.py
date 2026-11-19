"""Tier-0 shared rendering primitives for Synapse.

Extracted from ``synapse.bench`` to eliminate reverse dependencies from
non-bench packages (``models``, ``rsi``, ``inspect``, ``generators``)
back into the top-level evaluation suite.

This package may only import from: ``synapse.errors``, ``synapse.telemetry``,
``synapse.fgpp``, standard library, and ``numpy``.
"""

from synapse.render_core.backend import MockVgBackend, VgBackend
from synapse.render_core.camera import CameraTrajectoryType, generate_cameras

# Camera mathematics SSOT (look-at, projection, pose conversion).
from synapse.render_core.camera_math import (
    look_at,
    look_at_rotation,
    perspective_projection,
    pose_from_view_matrix,
    quat_to_matrix,
    view_matrix_from_pose,
)
from synapse.render_core.conditions import (
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    ConditionLevel,
    Conditions,
)
from synapse.render_core.mesh import MeshData, load_obj
from synapse.render_core.morphology import (
    binary_dilation,
    binary_erosion,
    circular_structuring_element,
)
from synapse.render_core.procedural import (
    ProceduralMesh,
    ProceduralMeshType,
    generate_humanoid_proxy,
    generate_procedural_mesh,
)

# Re-export pyramid primitives used by composite/blend and refine/fusion.
from synapse.render_core.pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    downsample2x,
    gaussian_kernel_1d,
    reconstruct_laplacian_pyramid,
    separable_gaussian,
    upsample2x,
)
from synapse.render_core.rasterizer import SoftwareRasterizer, entity_hash

__all__ = [
    "CROSS_CONDITIONS",
    "RENDERING_CONDITION_FACTORS",
    "SINGLE_FACTOR_CONDITIONS",
    "CameraTrajectoryType",
    "ConditionLevel",
    "Conditions",
    "ProceduralMesh",
    "ProceduralMeshType",
    "MeshData",
    "load_obj",
    "SoftwareRasterizer",
    "entity_hash",
    "generate_cameras",
    "generate_humanoid_proxy",
    "generate_procedural_mesh",
    "MockVgBackend",
    "VgBackend",
    "binary_dilation",
    "binary_erosion",
    "circular_structuring_element",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "downsample2x",
    "gaussian_kernel_1d",
    "reconstruct_laplacian_pyramid",
    "separable_gaussian",
    "upsample2x",
    "look_at",
    "look_at_rotation",
    "perspective_projection",
    "pose_from_view_matrix",
    "quat_to_matrix",
    "view_matrix_from_pose",
]
