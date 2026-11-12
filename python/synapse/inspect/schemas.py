"""InspectionReport v2.0 Pydantic schemas.

Defines the wire-format for 3D asset quality inspection reports,
per-defect annotations with macro-camera metadata, and camera poses
used during automated inspection sweeps.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CameraPose(BaseModel):
    """Camera pose used during inspection evaluation.

    Attributes
    ----------
    position:
        3D world-space camera position ``[x, y, z]``.
    rotation:
        Rotation encoded as a 9-element column-major 3x3 matrix
        **or** a 4-element quaternion ``[x, y, z, w]``.
    fov_deg:
        Vertical field of view in degrees.
    resolution:
        Output image resolution ``(width, height)``.
    """

    position: list[float] = Field(
        ..., min_length=3, max_length=3,
    )
    rotation: list[float] = Field(
        ..., min_length=4, max_length=9,
    )
    fov_deg: float = Field(..., gt=0.0, le=180.0)
    resolution: tuple[int, int] = (512, 512)


class DefectAnnotation3D(BaseModel):
    """Single defect annotation on a 3D asset.

    Attributes
    ----------
    defect_id:
        Unique identifier for this defect instance.
    category:
        Defect taxonomy category.
    severity:
        Severity score in ``[0, 1]``.
    confidence:
        Detection confidence in ``[0, 1]``.
    center_3d:
        3D world position of the defect centre ``[x, y, z]``.
    bounding_radius_m:
        Approximate bounding sphere radius in metres.
    affected_vertices_count:
        Number of mesh vertices affected by this defect.
    macro_camera:
        Optional close-up camera pose for macro inspection.
    macro_crop_uri:
        Optional URI to a cropped macro image of the defect.
    suggested_action:
        Recommended remediation action.
    """

    defect_id: str
    category: Literal[
        "mesh_crumple",
        "texture_stretch",
        "seam_tearing",
        "floating_debris",
        "anatomical_distortion",
        "plastic_shading",
    ]
    severity: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    center_3d: list[float] = Field(
        ..., min_length=3, max_length=3,
    )
    bounding_radius_m: float = Field(..., ge=0.0)
    affected_vertices_count: int = Field(..., ge=0)
    macro_camera: CameraPose | None = None
    macro_crop_uri: str | None = None
    suggested_action: Literal[
        "auto_remesh",
        "auto_inpaint_texture",
        "manual_hitl",
        "accept",
    ] = "accept"


class InspectionReport(BaseModel):
    """InspectionReport v2.0 — top-level quality inspection output.

    Attributes
    ----------
    schema_version:
        Report schema version string.
    asset_id:
        Unique identifier for the inspected asset.
    asset_name:
        Human-readable asset name.
    inspection_timestamp:
        ISO-8601 timestamp of the inspection run.
    overall_quality_score:
        Aggregate quality score in ``[0, 1]``.
    defects:
        List of detected defect annotations.
    requires_hitl:
        Whether the asset requires human-in-the-loop review.
    camera_poses_evaluated:
        Camera poses used during the inspection sweep.
    metrics_summary:
        Key-value summary of computed quality metrics.
    """

    schema_version: str = "2.0.0"
    asset_id: str
    asset_name: str
    inspection_timestamp: str
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    defects: list[DefectAnnotation3D] = Field(default_factory=list)
    requires_hitl: bool = False
    camera_poses_evaluated: list[CameraPose] = Field(
        default_factory=list,
    )
    metrics_summary: dict[str, float] = Field(default_factory=dict)


class GeometricValidationResult(BaseModel):
    """Tier 0 geometric topology validation result.

    Produced by ``validate_mesh_geometry()`` as the first-pass
    deterministic gate before any neural or rendering evaluation.

    Attributes
    ----------
    is_manifold:
        Whether mesh is a valid 2-manifold.
    is_watertight:
        Whether mesh is strictly watertight (closed).
    num_vertices:
        Total vertex count.
    num_faces:
        Total face count.
    degenerate_face_indices:
        Indices of faces flagged as degenerate.
    self_intersecting_pairs:
        Pairs of non-adjacent face indices that geometrically intersect.
    num_connected_components:
        Number of connected components in the mesh.
    floating_island_face_indices:
        Face indices belonging to floating islands (pruned components).
    passed:
        Whether all Tier 0 gates passed.
    failure_reasons:
        Human-readable descriptions of each failed gate.
    """

    is_manifold: bool = Field(..., description="Whether mesh is a valid 2-manifold")
    is_watertight: bool = Field(
        ..., description="Whether mesh is strictly watertight (closed)",
    )
    num_vertices: int = Field(..., ge=0)
    num_faces: int = Field(..., ge=0)
    degenerate_face_indices: list[int] = Field(default_factory=list)
    self_intersecting_pairs: list[tuple[int, int]] = Field(default_factory=list)
    num_connected_components: int = Field(default=1, ge=0)
    floating_island_face_indices: list[int] = Field(default_factory=list)
    passed: bool = Field(..., description="Whether all Tier 0 gates passed")
    failure_reasons: list[str] = Field(default_factory=list)
