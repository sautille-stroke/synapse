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
]
