"""Tier-0 shared rendering primitives for Synapse.

Extracted from ``synapse.bench`` to eliminate reverse dependencies from
non-bench packages (``models``, ``rsi``, ``inspect``, ``generators``)
back into the top-level evaluation suite.

This package may only import from: ``synapse.errors``, ``synapse.telemetry``,
``synapse.fgpp``, standard library, and ``numpy``.
"""

from synapse.render_core.camera import CameraTrajectoryType, generate_cameras
from synapse.render_core.conditions import (
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    ConditionLevel,
    Conditions,
)
from synapse.render_core.procedural import (
    ProceduralMesh,
    ProceduralMeshType,
    generate_humanoid_proxy,
    generate_procedural_mesh,
)
from synapse.render_core.rasterizer import SoftwareRasterizer, entity_hash

# Re-export pyramid primitives used by composite/blend and refine/fusion.
from synapse.render_core.pyramid import (  # noqa: F401
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    downsample2x,
    gaussian_kernel_1d,
    reconstruct_laplacian_pyramid,
    separable_gaussian,
    upsample2x,
)

__all__ = [
    "CROSS_CONDITIONS",
    "RENDERING_CONDITION_FACTORS",
    "SINGLE_FACTOR_CONDITIONS",
    "CameraTrajectoryType",
    "ConditionLevel",
    "Conditions",
    "ProceduralMesh",
    "ProceduralMeshType",
    "SoftwareRasterizer",
    "entity_hash",
    "generate_cameras",
    "generate_humanoid_proxy",
    "generate_procedural_mesh",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "downsample2x",
    "gaussian_kernel_1d",
    "reconstruct_laplacian_pyramid",
    "separable_gaussian",
    "upsample2x",
]
