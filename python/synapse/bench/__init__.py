"""Synapse-Bench evaluation dataset — public API.

Provides programmatic access to the benchmark scenario catalogue,
rendering condition factors, camera trajectory generators, and
procedural GT geometry.
"""

from synapse.bench.camera import (
    CameraTrajectoryType,
    generate_cameras,
)
from synapse.bench.conditions import (
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    Conditions,
    ConditionLevel,
)
from synapse.bench.procedural import (
    ProceduralMesh,
    ProceduralMeshType,
    generate_procedural_mesh,
)
from synapse.bench.scenario import (
    Asset,
    AssetSource,
    ExpectedMetrics,
    Scenario,
    ScenarioTier,
    list_scenarios,
    load_scenario,
)

__all__ = [
    "Scenario",
    "ScenarioTier",
    "Asset",
    "AssetSource",
    "ExpectedMetrics",
    "list_scenarios",
    "load_scenario",
    "Conditions",
    "ConditionLevel",
    "RENDERING_CONDITION_FACTORS",
    "SINGLE_FACTOR_CONDITIONS",
    "CROSS_CONDITIONS",
    "CameraTrajectoryType",
    "generate_cameras",
    "ProceduralMesh",
    "ProceduralMeshType",
    "generate_procedural_mesh",
]
