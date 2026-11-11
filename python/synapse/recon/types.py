"""Data models and schemas for Video-to-3D Inverse Reconstruction & Asset Distillation.

Covers camera trajectories, mesh geometry, kinematic mocap with foot-planting IK,
and intrinsic material de-lighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CameraExtrinsics:
    """6-DoF camera extrinsics for a single frame.

    Attributes:
        frame: 0-indexed frame index.
        transform: 4x4 homogeneous transformation matrix [R | t].
        position: 3D camera position in world space (m).
        rotation_matrix: 3x3 rotation matrix in SO(3).
        yaw_rad: Camera yaw orientation in radians.
    """

    frame: int
    transform: np.ndarray
    position: np.ndarray
    rotation_matrix: np.ndarray
    yaw_rad: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize extrinsics to JSON-serializable dict."""
        return {
            "frame": self.frame,
            "transform": self.transform.tolist(),
            "cam_pos_m": [round(float(x), 4) for x in self.position],
            "yaw_rad": round(float(self.yaw_rad), 4),
        }


@dataclass
class CameraTrajectory:
    """Solved 6-DoF camera trajectory across a video sequence.

    Attributes:
        frame_count: Number of frames in the sequence.
        intrinsics: Pinhole camera intrinsics (fx, fy, cx, cy).
        extrinsics: List of per-frame CameraExtrinsics.
        metric_scale: Estimated scale alignment factor (to 1:1 metric meters).
        ground_plane_normal: Estimated ground plane normal vector (3,).
        ground_height_m: Estimated ground plane elevation (m).
        reprojection_error_px: Average reprojection error in pixels.
    """

    frame_count: int
    intrinsics: dict[str, float]
    extrinsics: list[CameraExtrinsics] = field(default_factory=list)
    metric_scale: float = 1.0
    ground_plane_normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    ground_height_m: float = 0.0
    reprojection_error_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize trajectory to JSON-compatible dictionary."""
        return {
            "frame_count": self.frame_count,
            "intrinsics": self.intrinsics,
            "metric_scale_alignment": {
                "metric_scale": round(float(self.metric_scale), 4),
                "ground_plane_normal": [round(float(x), 4) for x in self.ground_plane_normal],
                "ground_height_m": round(float(self.ground_height_m), 4),
                "average_reprojection_error_px": round(float(self.reprojection_error_px), 4),
            },
            "extrinsics": [ext.to_dict() for ext in self.extrinsics],
        }


@dataclass
class MeshGeometry:
    """3D polygon mesh representation.

    Attributes:
        vertices: Vertex positions, shape (V, 3), float32/float64.
        faces: Polygon vertex index array, shape (F, 3) or (F, 4), int32.
        normals: Vertex normal vectors, shape (V, 3).
        uvs: Texture coordinates, shape (V, 2) in [0, 1].
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None

    @property
    def vertex_count(self) -> int:
        """Total number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Total number of faces."""
        return int(self.faces.shape[0])

    @property
    def is_quad_dominant(self) -> bool:
        """Whether the majority of faces are quadrilaterals."""
        if self.faces.ndim < 2:
            return False
        return self.faces.shape[1] == 4


@dataclass
class FootContactInterval:
    """Continuous ground-contact interval for a foot effector.

    Attributes:
        foot_index: 0 for left foot, 1 for right foot.
        start_frame: Start frame of stance phase.
        end_frame: End frame of stance phase.
        anchored_pos: Locked ground contact position (x, y, z).
    """

    foot_index: int
    start_frame: int
    end_frame: int
    anchored_pos: np.ndarray


@dataclass
class MocapKinematics:
    """Kinematic motion capture data with foot slip telemetry.

    Attributes:
        frame_count: Number of frames.
        joint_names: List of skeletal joint names.
        joint_positions: Joint 3D coordinates, shape (T, J, 3).
        foot_contacts: Per-frame foot contact mask, shape (T, 2).
        raw_foot_slip_cm_s: Mean horizontal foot slip speed before IK (cm/s).
        clamped_foot_slip_cm_s: Mean horizontal foot slip speed after IK (cm/s).
        max_foot_slip_cm_s: Peak foot slip speed during stance after IK (cm/s).
        raw_sdf_penetration_mm: Max hand-prop/cloth SDF interpenetration (mm).
        resolved_sdf_penetration_mm: Resolved SDF interpenetration (mm).
    """

    frame_count: int
    joint_names: list[str] = field(default_factory=list)
    joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float32)
    )
    foot_contacts: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    raw_foot_slip_cm_s: float = 0.0
    clamped_foot_slip_cm_s: float = 0.0
    max_foot_slip_cm_s: float = 0.0
    raw_sdf_penetration_mm: float = 0.0
    resolved_sdf_penetration_mm: float = 0.0

    def to_telemetry(self) -> dict[str, Any]:
        """Convert kinematics metrics into evaluation telemetry dict."""
        return {
            "frame_count": self.frame_count,
            "raw_foot_sliding_cm_per_sec": round(self.raw_foot_slip_cm_s, 3),
            "clamped_foot_sliding_cm_per_sec": round(self.clamped_foot_slip_cm_s, 3),
            "max_foot_slip_cm_per_sec": round(self.max_foot_slip_cm_s, 3),
            "raw_sdf_penetration_mm": round(self.raw_sdf_penetration_mm, 2),
            "resolved_sdf_penetration_mm": round(self.resolved_sdf_penetration_mm, 2),
            "zero_slip_verified": self.clamped_foot_slip_cm_s < 0.3,
        }


@dataclass
class DelightResult:
    """Intrinsic material decomposition result.

    Attributes:
        albedo: Diffuse reflectance (base color), shape (H, W, 3).
        shading: Estimated irradiance / ambient shadow map, shape (H, W, 3) or (H, W, 1).
        specular: Estimated view-dependent specular reflection, shape (H, W, 3).
        ambient_sh: 2nd-order Spherical Harmonics lighting coefficients, shape (9, 3).
        delta_e_residual: Estimated CIELAB Delta-E shadow residual.
    """

    albedo: np.ndarray
    shading: np.ndarray
    specular: np.ndarray
    ambient_sh: np.ndarray
    delta_e_residual: float = 0.0


@dataclass
class ReconResult:
    """Consolidated result from the Stage 3 Inverse Reconstruction pipeline.

    Attributes:
        mesh: Reconstructed watertight/quad-dominant surface geometry.
        camera_trajectory: Solved 6-DoF camera extrinsics & scale.
        kinematics: Distilled kinematic mocap with zero-slip IK.
        delight: Intrinsic albedo decomposition.
        telemetry: Detailed metric telemetry dictionary.
    """

    mesh: MeshGeometry
    camera_trajectory: CameraTrajectory | None = None
    kinematics: MocapKinematics | None = None
    delight: DelightResult | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
