"""Synapse-Bench evaluation dataset — public API.

Provides programmatic access to the benchmark scenario catalogue,
rendering condition factors, camera trajectory generators, procedural
GT geometry, and the pure-NumPy software rasterizer.
"""

from synapse.bench.bundle_exporter import (
    InspectionBundleManifest,
    SourceEntry,
    export_inspection_bundle,
    validate_inspection_bundle,
)
from synapse.bench.camera import (
    CameraTrajectoryType,
    generate_cameras,
)
from synapse.bench.conditions import (
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    ConditionLevel,
    Conditions,
)
from synapse.bench.domain_randomizer import (
    DomainRandomizationConfig,
    DomainRandomizer,
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
    generate_humanoid_proxy,
    generate_procedural_mesh,
)
from synapse.bench.rasterizer import (
    SoftwareRasterizer,
    entity_hash,
)
from synapse.bench.registry import (
    AssetEntry,
    AssetRegistry,
    ScenarioRegistry,
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
    "CROSS_CONDITIONS",
    "RENDERING_CONDITION_FACTORS",
    "SINGLE_FACTOR_CONDITIONS",
    "Asset",
    "AssetEntry",
    "AssetRegistry",
    "AssetSource",
    "CameraTrajectoryType",
    "ConditionLevel",
    "Conditions",
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "ExpectedMetrics",
    "InspectionBundleManifest",
    "MeshData",
    "ProceduralMesh",
    "ProceduralMeshType",
    "Scenario",
    "ScenarioRegistry",
    "ScenarioTier",
    "SoftwareRasterizer",
    "SourceEntry",
    "entity_hash",
    "export_inspection_bundle",
    "generate_cameras",
    "generate_html_report",
    "generate_humanoid_proxy",
    "generate_procedural_mesh",
    "list_scenarios",
    "load_mesh",
    "load_obj",
    "load_procedural",
    "load_scenario",
    "validate_inspection_bundle",
]
