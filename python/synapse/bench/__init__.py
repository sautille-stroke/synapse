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
from synapse.bench.compare import (
    TRACK_DEFINITIONS,
    FourTrackComparisonReport,
    ModelComparisonRow,
    TrackScore,
    aggregate_four_track_scores,
    compare_runs,
)
from synapse.bench.dataset_gen import (
    DatasetConfig,
    DatasetGenerator,
)
from synapse.bench.dataset_visualizer import (
    colorize_depth,
    colorize_flow,
    colorize_mask,
    colorize_normal,
    make_contact_sheet,
)
from synapse.bench.domain_randomizer import (
    DomainRandomizationConfig,
    DomainRandomizer,
)
from synapse.bench.model_loader import (
    MeshData,
    ModelAsset,
    list_builtin_models,
    load_mesh,
    load_model,
    load_obj,
    load_procedural,
)
from synapse.bench.registry import (
    AssetEntry,
    AssetRegistry,
    ExternalDatasetSpec,
    ScenarioRegistry,
    filter_stage0_quality,
    normalize_external_mesh,
)
from synapse.bench.report import (
    generate_html_report,
    render_comparison_html,
)
from synapse.bench.scenario import (
    ALLOWED_EXTERNAL_LICENSES,
    Asset,
    AssetSource,
    ExpectedMetrics,
    Scenario,
    ScenarioTier,
    list_scenarios,
    load_scenario,
    validate_external_license,
)
from synapse.render_core.camera import (
    CameraTrajectoryType,
    generate_cameras,
)
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
from synapse.render_core.rasterizer import (
    SoftwareRasterizer,
    entity_hash,
)

__all__ = [
    "CROSS_CONDITIONS",
    "RENDERING_CONDITION_FACTORS",
    "SINGLE_FACTOR_CONDITIONS",
    "ALLOWED_EXTERNAL_LICENSES",
    "Asset",
    "AssetEntry",
    "AssetRegistry",
    "AssetSource",
    "ExternalDatasetSpec",
    "CameraTrajectoryType",
    "ConditionLevel",
    "Conditions",
    "DatasetConfig",
    "DatasetGenerator",
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "ExpectedMetrics",
    "FourTrackComparisonReport",
    "InspectionBundleManifest",
    "MeshData",
    "ModelAsset",
    "ModelComparisonRow",
    "ProceduralMesh",
    "ProceduralMeshType",
    "Scenario",
    "ScenarioRegistry",
    "ScenarioTier",
    "SoftwareRasterizer",
    "SourceEntry",
    "TRACK_DEFINITIONS",
    "TrackScore",
    "aggregate_four_track_scores",
    "colorize_depth",
    "colorize_flow",
    "colorize_mask",
    "colorize_normal",
    "compare_runs",
    "entity_hash",
    "export_inspection_bundle",
    "filter_stage0_quality",
    "generate_cameras",
    "generate_html_report",
    "generate_humanoid_proxy",
    "generate_procedural_mesh",
    "list_builtin_models",
    "list_scenarios",
    "load_mesh",
    "load_model",
    "load_obj",
    "load_procedural",
    "load_scenario",
    "make_contact_sheet",
    "normalize_external_mesh",
    "render_comparison_html",
    "validate_external_license",
    "validate_inspection_bundle",
]
