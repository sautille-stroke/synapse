"""FGPP frame and sequence data classes.

Implements the FGPP channel layout defined in RFC 0044 §4.1.
All channel names, data types, and coordinate-space conventions follow
the authoritative FGPP specification.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synapse.errors import ContractValidationError
from synapse.fgpp.schema import (
    DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
    FGPP_SCHEMA_VERSION,
    is_physical_anchor_frame,
)


@dataclass
class CameraParams:
    """Camera intrinsics and extrinsics for a single frame.

    Coordinate convention: right-handed, Y-up, camera looks along -Z.
    """

    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fov_degrees: float = 60.0

    # Physical lens parameters
    focal_length_mm: float = 35.0
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 20.25

    # Extrinsics — 4×4 world-to-camera matrix (row-major)
    view_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    # 4×4 camera-to-world matrix (row-major)
    world_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    # Depth range (metres)
    z_near: float = 0.1
    z_far: float = 1000.0

    # Lens / shutter parameters for DoF and motion blur conditions
    aperture_fstop: float = 0.0
    shutter_angle_deg: float = 180.0

    def position(self) -> np.ndarray:
        """Camera position in world space."""
        return np.asarray(self.world_matrix[:3, 3].copy())

    def forward(self) -> np.ndarray:
        """Camera forward direction (negative Z axis in camera space)."""
        return np.asarray(-self.world_matrix[:3, 2].copy())

    def projection_matrix(self) -> np.ndarray:
        """Compute 4×4 perspective projection matrix.

        Builds a standard OpenGL-style perspective projection matrix from full
        intrinsics ``(fx, fy, cx, cy, width, height, z_near, z_far)``, preserving
        principal point offsets and independent focal lengths.
        """
        w, h = float(self.width), float(max(self.height, 1))
        near, far = float(self.z_near), float(self.z_far)

        if self.fx > 0 and self.fy > 0:
            p00 = 2.0 * float(self.fx) / w
            p11 = 2.0 * float(self.fy) / h
            p02 = (w - 2.0 * float(self.cx)) / w
            p12 = (2.0 * float(self.cy) - h) / h
        else:
            fov_rad = math.radians(self.fov_degrees or 60.0)
            f = 1.0 / math.tan(fov_rad / 2.0)
            aspect = w / h
            p00 = f / aspect
            p11 = f
            p02 = 0.0
            p12 = 0.0

        p22 = -(far + near) / (far - near)
        p23 = -2.0 * far * near / (far - near)

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = p00
        proj[0, 2] = p02
        proj[1, 1] = p11
        proj[1, 2] = p12
        proj[2, 2] = p22
        proj[2, 3] = p23
        proj[3, 2] = -1.0
        return proj

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-friendly dict."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "fov_degrees": self.fov_degrees,
            "focal_length_mm": self.focal_length_mm,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "view_matrix": self.view_matrix.tolist(),
            "world_matrix": self.world_matrix.tolist(),
            "z_near": self.z_near,
            "z_far": self.z_far,
            "aperture_fstop": self.aperture_fstop,
            "shutter_angle_deg": self.shutter_angle_deg,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraParams:
        """Deserialise from dict[str, Any]."""
        return cls(
            fx=d["fx"],
            fy=d["fy"],
            cx=d["cx"],
            cy=d["cy"],
            width=d["width"],
            height=d["height"],
            fov_degrees=d.get("fov_degrees", 60.0),
            focal_length_mm=d.get("focal_length_mm", 35.0),
            sensor_width_mm=d.get("sensor_width_mm", 36.0),
            sensor_height_mm=d.get("sensor_height_mm", 20.25),
            view_matrix=np.array(d["view_matrix"], dtype=np.float64),
            world_matrix=np.array(d["world_matrix"], dtype=np.float64),
            z_near=d.get("z_near", 0.1),
            z_far=d.get("z_far", 1000.0),
            aperture_fstop=d.get("aperture_fstop", 0.0),
            shutter_angle_deg=d.get("shutter_angle_deg", 180.0),
        )


class CameraConvention:
    """Identifies and converts between camera projection conventions.

    Supported conventions:
    - OPENGL: NDC z in [-1, 1], right-handed
    - VULKAN_RH_ZO: NDC z in [0, 1], right-handed (Vulkan default)
    - FULGUR_INF_REVERSED_Z: Infinite reversed-Z (p22==0, p32==-1, p23==z_near)
    - ORTHOGRAPHIC: Orthographic projection (p32==0)
    """

    OPENGL = "opengl"
    VULKAN_RH_ZO = "vulkan_rh_zo"
    FULGUR_INF_REVERSED_Z = "fulgur_inf_reversed_z"
    ORTHOGRAPHIC = "orthographic"
    UNKNOWN = "unknown"

    @staticmethod
    def detect(proj: np.ndarray) -> str:
        """Detect the camera convention from a 4x4 projection matrix (row-major).

        Returns one of the CameraConvention string constants.
        """
        p22 = float(proj[2, 2])
        p23 = float(proj[2, 3])
        p32 = float(proj[3, 2])

        # Orthographic: p32 == 0
        if abs(p32) < 1e-8:
            return CameraConvention.ORTHOGRAPHIC

        # Fulgur infinite reversed-Z: p22 == 0, p32 == -1, p23 == z_near
        if abs(p22) < 1e-8 and abs(p32 + 1.0) < 1e-6 and p23 > 0:
            return CameraConvention.FULGUR_INF_REVERSED_Z

        # Vulkan RH_ZO [0,1]: p22 = -far/(far-near), p23 = -(far*near)/(far-near)
        # For standard perspective: p32 == -1, and 0 < -p22 < 1
        if abs(p32 + 1.0) < 1e-6 and p22 < 0.0 and p22 > -1.0 - 1e-6:
            return CameraConvention.VULKAN_RH_ZO

        # OpenGL [-1,1]: p22 = -(far+near)/(far-near), p32 == -1
        if abs(p32 + 1.0) < 1e-6:
            return CameraConvention.OPENGL

        return CameraConvention.UNKNOWN

    @staticmethod
    def to_opengl(proj: np.ndarray, convention: str) -> np.ndarray:
        """Convert a projection matrix to OpenGL convention [-1, 1] z range.

        Parameters
        ----------
        proj:
            4x4 row-major projection matrix.
        convention:
            Source convention string.

        Returns
        -------
        np.ndarray
            4x4 projection matrix in OpenGL convention.
        """
        out = proj.copy()
        if convention == CameraConvention.OPENGL:
            return out
        elif convention == CameraConvention.VULKAN_RH_ZO:
            # Convert from [0,1] to [-1,1]: p22' = 2*p22 + 1, p23' = 2*p23
            out[2, 2] = 2.0 * proj[2, 2] + 1.0
            out[2, 3] = 2.0 * proj[2, 3]
            return out
        elif convention == CameraConvention.FULGUR_INF_REVERSED_Z:
            # Infinite reversed-Z: p22=0, p32=-1, p23=z_near
            # Reconstruct a standard OpenGL matrix with z_far = 10000 * z_near
            z_near = float(proj[2, 3])
            z_far = 10000.0 * max(z_near, 1e-7)
            out[2, 2] = -(z_far + z_near) / (z_far - z_near)
            out[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
            return out
        # ORTHOGRAPHIC or UNKNOWN: return as-is
        return out


def _validate_view_matrix_so3(view_row: np.ndarray, frame_id: int | str) -> None:
    """Validate that a 4x4 view matrix has valid SO(3) rotation and [0,0,0,1] bottom row."""
    if view_row.shape != (4, 4):
        return  # Skip validation for non-4x4 matrices

    # Validate bottom row [0, 0, 0, 1]
    if not np.allclose(view_row[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ContractValidationError(
            f"Frame {frame_id}: bottom row of view_matrix {view_row[3]} deviates from [0, 0, 0, 1].",
            agent_fix_hint="Ensure view_matrix has homogeneous last row [0, 0, 0, 1].",
        )

    # Validate rotation determinant and orthogonality
    R = view_row[:3, :3]
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-4:
        raise ContractValidationError(
            f"Frame {frame_id}: rotation determinant {det:.6f} deviates from 1.0 by more than 1e-4.",
            agent_fix_hint="Ensure camera rotation is a proper orthogonal matrix with det(R) = +1.0.",
        )

    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    if ortho_err > 1e-4:
        raise ContractValidationError(
            f"Frame {frame_id}: rotation matrix orthogonality error ||R^T R - I||_F = {ortho_err:.6e} > 1e-4.",
            agent_fix_hint="Ensure camera rotation matrix R satisfies R^T R = I.",
        )


@dataclass
class FGPPFrame:
    """A single FGPP (Feature Guidance Package Protocol) frame.

    Channel semantics follow the FGPP specification (RFC 0044 §4.1):

    +-----------------------+----------+-------------------------------------+
    | Channel               | dtype    | Description                         |
    +=======================+==========+=====================================+
    | direct_radiance       | float16³ | Linear HDR radiance [0, ∞)          |
    | depth                 | float32  | Camera-space linear depth (metres)  |
    | world_normal          | float16³ | World-space normal [-1,1]³          |
    | motion_vector         | float16² | Screen-space subpixel displacement  |
    | base_color            | float16³ | Linear albedo [0,1]                 |
    | roughness_metallic    | float16² | [roughness, metallic] ∈ [0,1]²     |
    | cryptomatte_id        | uint32   | MurmurHash3 entity ID               |
    +-----------------------+----------+-------------------------------------+
    """

    camera: CameraParams

    # Required channels
    depth: np.ndarray  # [H, W] float32

    # Optional channels (None = not available)
    direct_radiance: np.ndarray | None = None  # [H, W, 3] float16
    world_normal: np.ndarray | None = None  # [H, W, 3] float16
    motion_vector: np.ndarray | None = None  # [H, W, 2] float16
    base_color: np.ndarray | None = None  # [H, W, 3] float16
    roughness_metallic: np.ndarray | None = None  # [H, W, 2] float16
    cryptomatte_id: np.ndarray | None = None  # [H, W] uint32

    # Frame index within the sequence
    frame_index: int = 0

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    def available_channels(self) -> list[str]:
        """Return names of non-None channels."""
        channels = ["depth"]
        for name in (
            "direct_radiance",
            "world_normal",
            "motion_vector",
            "base_color",
            "roughness_metallic",
            "cryptomatte_id",
        ):
            if getattr(self, name) is not None:
                channels.append(name)
        return channels


@dataclass
class FGPPSequence:
    """An ordered sequence of FGPP frames with shared metadata."""

    frames: list[FGPPFrame] = field(default_factory=list)
    fps: float = 24.0
    scene_id: str = ""

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.num_frames / self.fps

    def save_metadata(self, output_dir: str | Path) -> Path:
        """Write ``metadata.json`` in FGPP v1.0 format.

        Camera matrices are stored as **column-major** 16-element flat
        arrays (``_col_major``) to match the C++/OpenGL memory layout
        used by ``@aether//fgpp`` and ``fulgur``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        z_nears = [f.camera.z_near for f in self.frames]
        z_fars = [f.camera.z_far for f in self.frames]

        frame_records = []
        for f in self.frames:
            # Column-major: transpose the row-major 4×4 then flatten
            view_col = np.asarray(f.camera.view_matrix, dtype=np.float64).T.reshape(-1).tolist()
            proj_col = (
                np.asarray(f.camera.projection_matrix(), dtype=np.float64).T.reshape(-1).tolist()
            )
            frame_records.append(
                {
                    "frame_index": f.frame_index,
                    "timestamp_sec": f.frame_index / max(self.fps, 1e-6),
                    "is_physical_anchor": is_physical_anchor_frame(f.frame_index),
                    "fov_y_deg": f.camera.fov_degrees,
                    "view_matrix_col_major": view_col,
                    "proj_matrix_col_major": proj_col,
                    "camera": f.camera.to_dict(),
                    "channels": f.available_channels(),
                }
            )

        metadata = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "fps": self.fps,
            "total_frames": self.num_frames,
            "num_frames": self.num_frames,
            "resolution": [
                self.frames[0].camera.width if self.frames else 0,
                self.frames[0].camera.height if self.frames else 0,
            ],
            "z_near_meters": min(z_nears) if z_nears else 0.1,
            "z_far_meters": max(z_fars) if z_fars else 1000.0,
            "anchor_interval_frames": DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
            "frames": frame_records,
        }

        path = output_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_metadata(cls, input_dir: str | Path) -> FGPPSequence:
        """Load an ``FGPPSequence`` from a ``metadata.json`` file.

        Converts ``_col_major`` 16-element arrays back to row-major
        NumPy 4×4 matrices (``reshape((4,4)).T``).  Validates that the
        rotation sub-matrix has ``|det(R) - 1| < 1e-4``, orthogonality error
        ``||R^T R - I||_F <= 1e-4``, and homogeneous last row ``[0, 0, 0, 1]``.
        """
        input_dir = Path(input_dir)
        meta_path = input_dir / "metadata.json"
        if not meta_path.exists():
            raise ContractValidationError(
                f"Missing metadata.json at {meta_path}",
                agent_fix_hint="Ensure metadata.json exists in the sequence directory.",
            )

        raw = json.loads(meta_path.read_text(encoding="utf-8"))

        schema_version = raw.get("schema_version")
        if schema_version != FGPP_SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported FGPP schema_version: {schema_version!r}. Expected {FGPP_SCHEMA_VERSION!r}.",
                agent_fix_hint=f"Ensure metadata.json specifies schema_version='{FGPP_SCHEMA_VERSION}'.",
            )

        fps = raw.get("fps", 24.0)
        scene_id = raw.get("scene_id", "")
        res = raw.get("resolution", [0, 0])

        frames: list[FGPPFrame] = []
        for fr in raw.get("frames", []):
            width = fr.get("width", res[0] if len(res) > 0 else 0)
            height = fr.get("height", res[1] if len(res) > 1 else 0)

            # Reconstruct camera
            if "camera" in fr:
                cam = CameraParams.from_dict(fr["camera"])
                # Validate view_matrix SO(3) even when loaded from camera dict
                _validate_view_matrix_so3(cam.view_matrix, fr.get("frame_index", "?"))
            else:
                # Flat camera reconstruction
                fx, fy = 0.0, 0.0
                cx = float(width) / 2.0 if width > 0 else 0.0
                cy = float(height) / 2.0 if height > 0 else 0.0
                z_near, z_far = 0.1, 1000.0
                fov_deg = float(fr.get("fov_y_deg", fr.get("fov_degrees", 60.0)))

                if "proj_matrix_col_major" in fr and len(fr["proj_matrix_col_major"]) == 16:
                    col_proj = np.array(fr["proj_matrix_col_major"], dtype=np.float64)
                    proj_row = col_proj.reshape((4, 4)).T
                    p00, p11 = proj_row[0, 0], proj_row[1, 1]
                    p02, p12 = proj_row[0, 2], proj_row[1, 2]
                    p22, p23 = proj_row[2, 2], proj_row[2, 3]
                    convention = CameraConvention.detect(proj_row)

                    if abs(p00) > 1e-6 and width > 0:
                        fx = abs(p00) * float(width) / 2.0
                    if abs(p11) > 1e-6 and height > 0:
                        fy = abs(p11) * float(height) / 2.0
                        fov_deg = math.degrees(2.0 * math.atan(float(height) / (2.0 * fy)))

                    if width > 0:
                        cx = (1.0 - p02) * float(width) / 2.0
                    if height > 0:
                        cy = (1.0 - p12) * float(height) / 2.0

                    # Near and far reconstruction based on convention
                    if convention == CameraConvention.FULGUR_INF_REVERSED_Z:
                        # p22==0, p32==-1, p23==z_near
                        if p23 > 0:
                            z_near = float(p23)
                            z_far = 100000.0  # infinite far
                    elif convention == CameraConvention.VULKAN_RH_ZO:
                        # p22 = -far/(far-near), p23 = -(far*near)/(far-near)
                        if abs(p22) > 1e-8 and abs(p23) > 1e-8:
                            z_near = float(p23 / p22)
                            if z_near > 0 and abs(p22 + 1.0) > 1e-8:
                                z_far = float(p23 / (p22 + 1.0))
                                if z_far <= z_near:
                                    z_near, z_far = 0.1, 1000.0
                            else:
                                z_near, z_far = 0.1, 1000.0
                    elif convention == CameraConvention.ORTHOGRAPHIC:
                        # Orthographic: use sensible defaults
                        z_near, z_far = 0.1, 1000.0
                    else:
                        # OpenGL or unknown: standard recovery
                        denom_near = p22 - 1.0
                        denom_far = p22 + 1.0
                        if abs(denom_near) > 1e-8 and abs(denom_far) > 1e-8 and abs(p23) > 1e-8:
                            recovered_near = p23 / denom_near
                            recovered_far = p23 / denom_far
                            if 0 < recovered_near < recovered_far:
                                z_near = float(recovered_near)
                                z_far = float(recovered_far)

                if fx <= 0.0 and fov_deg > 0.0 and height > 0:
                    fy = float(height) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
                    fx = fy

                cam = CameraParams(
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=int(width),
                    height=int(height),
                    fov_degrees=fov_deg,
                    z_near=z_near,
                    z_far=z_far,
                )

            # Recover row-major view matrix from column-major storage
            if "view_matrix_col_major" in fr:
                col_flat = np.array(fr["view_matrix_col_major"], dtype=np.float64)
                if col_flat.shape != (16,):
                    raise ContractValidationError(
                        f"Frame {fr.get('frame_index', '?')}: view_matrix_col_major must contain 16 elements, got {col_flat.shape}.",
                        agent_fix_hint="Ensure view_matrix_col_major is a 16-element array.",
                    )
                view_row = col_flat.reshape((4, 4)).T

                _validate_view_matrix_so3(view_row, fr.get('frame_index', '?'))

                cam.view_matrix = view_row
                cam.world_matrix = np.linalg.inv(view_row)

            frame = FGPPFrame(
                camera=cam,
                depth=np.zeros((cam.height or 1, cam.width or 1), dtype=np.float32),
                frame_index=fr.get("frame_index", 0),
            )
            frames.append(frame)

        return cls(frames=frames, fps=fps, scene_id=scene_id)
