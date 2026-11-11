"""3D asset inspection — schemas, Fibonacci pose alignment, and defect reports.

Submodules:
    schemas     InspectionReport v2.0, DefectAnnotation3D, CameraPose
    align       Fibonacci-sphere camera generation and silhouette-IoU alignment
    macro       Multi-channel residual heatmap and macro camera planning
"""

from __future__ import annotations

from synapse.inspect.align import (
    align_reference_image_to_mesh,
    compute_silhouette,
    compute_silhouette_iou,
    generate_fibonacci_sphere_cameras,
)
from synapse.inspect.macro import (
    compute_isotopic_residual_heatmap,
    plan_macro_camera_for_roi,
)
from synapse.inspect.schemas import (
    CameraPose,
    DefectAnnotation3D,
    InspectionReport,
)

__all__ = [
    "CameraPose",
    "DefectAnnotation3D",
    "InspectionReport",
    "generate_fibonacci_sphere_cameras",
    "compute_silhouette",
    "compute_silhouette_iou",
    "align_reference_image_to_mesh",
    "compute_isotopic_residual_heatmap",
    "plan_macro_camera_for_roi",
]
