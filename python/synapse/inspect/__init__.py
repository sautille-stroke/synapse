"""3D asset inspection --- schemas, alignment, defect reports, and flywheel.

Submodules:
    schemas           InspectionReport v2.0, DefectAnnotation3D, CameraPose,
                      GeometricValidationResult
    align             Fibonacci-sphere camera generation and silhouette-IoU
    macro             Multi-channel residual heatmap and macro camera planning
    jev_critic        Visual Jev calibrated logits critic for defect detection
    flywheel_exporter Refinement roundtrip orchestrator and RL data exporter
    eval_report       3D model topology/render evaluation and HTML report with SVG radar charts
    geometry_gate     Tier 0 deterministic geometry validation gate
"""

from __future__ import annotations

from synapse.inspect.align import (
    align_reference_image_to_mesh,
    compute_silhouette,
    compute_silhouette_iou,
    generate_fibonacci_sphere_cameras,
)
from synapse.inspect.eval_report import (
    EvalReportData,
    ModelEvalResult,
    TopologyResult,
    generate_eval_html_report,
)
from synapse.inspect.flywheel_exporter import (
    DPORecord,
    IntentGroup,
    PRMRecord,
    RoundtripGate,
    StepAnnotation,
    export_rl_flywheel_dataset,
    run_targeted_refinement,
)
from synapse.inspect.geometry_gate import (
    validate_mesh_geometry,
)
from synapse.inspect.jev_critic import (
    MockJevTransport,
    VisualJevCritic,
)
from synapse.inspect.macro import (
    compute_isotopic_residual_heatmap,
    plan_macro_camera_for_roi,
)
from synapse.inspect.model_evaluator import (
    GATE_PROFILES,
    evaluate_3d_models_directory,
    evaluate_single_model,
    normalize_eval_profile,
)
from synapse.inspect.schemas import (
    CameraPose,
    DefectAnnotation3D,
    GeometricValidationResult,
    InspectionReport,
)

__all__ = [
    "CameraPose",
    "DefectAnnotation3D",
    "GeometricValidationResult",
    "InspectionReport",
    "align_reference_image_to_mesh",
    "compute_isotopic_residual_heatmap",
    "compute_silhouette",
    "compute_silhouette_iou",
    "generate_fibonacci_sphere_cameras",
    "plan_macro_camera_for_roi",
    "validate_mesh_geometry",
    "MockJevTransport",
    "VisualJevCritic",
    "DPORecord",
    "IntentGroup",
    "PRMRecord",
    "RoundtripGate",
    "StepAnnotation",
    "export_rl_flywheel_dataset",
    "run_targeted_refinement",
    "EvalReportData",
    "ModelEvalResult",
    "TopologyResult",
    "generate_eval_html_report",
    "GATE_PROFILES",
    "evaluate_3d_models_directory",
    "evaluate_single_model",
    "normalize_eval_profile",
]
