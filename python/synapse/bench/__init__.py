"""Synapse-Bench evaluation dataset — public API.

Provides programmatic access to the benchmark scenario catalogue,
rendering condition factors, camera trajectory generators, procedural
GT geometry, and the pure-NumPy software rasterizer.
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
from synapse.bench.model_loader import (
    MeshData,
    load_mesh,
    load_obj,
    load_procedural,
)
from synapse.bench.procedural import (
    ProceduralMesh,
    ProceduralMeshType,
    generate_procedural_mesh,
)
from synapse.bench.rasterizer import (
    SoftwareRasterizer,
    entity_hash,
)
from synapse.bench.procedural import (
    generate_humanoid_proxy,
)
from synapse.bench.registry import (
    AssetEntry,
    AssetRegistry,
    ScenarioRegistry,
)
from synapse.bench.bundle_exporter import (
    InspectionBundleManifest,
    SourceEntry,
    export_inspection_bundle,
    validate_inspection_bundle,
)
from synapse.bench.report import (
    generate_html_report,
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
    "AssetEntry",
    "AssetRegistry",
    "ScenarioRegistry",
    "generate_html_report",
    "generate_humanoid_proxy",
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
    "SoftwareRasterizer",
    "entity_hash",
    "MeshData",
    "load_mesh",
    "load_obj",
    "load_procedural",
    "InspectionBundleManifest",
    "SourceEntry",
    "validate_inspection_bundle",
    "export_inspection_bundle",
]
