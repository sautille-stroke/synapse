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

import numpy as np

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
    view_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    # 4×4 camera-to-world matrix (row-major)
    world_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    # Depth range (metres)
    z_near: float = 0.1
    z_far: float = 1000.0

    # Lens / shutter parameters for DoF and motion blur conditions
    aperture_fstop: float = 0.0
    shutter_angle_deg: float = 180.0

    def position(self) -> np.ndarray:
        """Camera position in world space."""
        return self.world_matrix[:3, 3].copy()

    def forward(self) -> np.ndarray:
        """Camera forward direction (negative Z axis in camera space)."""
        return -self.world_matrix[:3, 2].copy()

    def projection_matrix(self) -> np.ndarray:
        """Compute 4×4 perspective projection matrix.

        Uses ``fov_degrees``, aspect ratio, ``z_near``, and ``z_far``
        to build a standard OpenGL-style projection matrix.
        """
        fov_rad = math.radians(self.fov_degrees)
        aspect = self.width / max(self.height, 1)
        f = 1.0 / math.tan(fov_rad / 2.0)
        near, far = self.z_near, self.z_far

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0
        return proj

    def to_dict(self) -> dict:
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
    def from_dict(cls, d: dict) -> CameraParams:
        """Deserialise from dict."""
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
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

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
            view_col = (
                np.asarray(f.camera.view_matrix, dtype=np.float64)
                .T.reshape(-1)
                .tolist()
            )
            proj_col = (
                np.asarray(f.camera.projection_matrix(), dtype=np.float64)
                .T.reshape(-1)
                .tolist()
            )
            frame_records.append({
                "frame_index": f.frame_index,
                "timestamp_sec": f.frame_index / max(self.fps, 1e-6),
                "is_physical_anchor": is_physical_anchor_frame(f.frame_index),
                "fov_y_deg": f.camera.fov_degrees,
                "view_matrix_col_major": view_col,
                "proj_matrix_col_major": proj_col,
                "camera": f.camera.to_dict(),
                "channels": f.available_channels(),
            })

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
        rotation sub-matrix has ``|det(R) - 1| < 1e-5``.
        """
        input_dir = Path(input_dir)
        meta_path = input_dir / "metadata.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))

        fps = raw.get("fps", 24.0)
        scene_id = raw.get("scene_id", "")

        frames: list[FGPPFrame] = []
        for fr in raw.get("frames", []):
            # Reconstruct camera
            if "camera" in fr:
                cam = CameraParams.from_dict(fr["camera"])
            else:
                cam = CameraParams(
                    fx=0, fy=0, cx=0, cy=0,
                    width=raw.get("resolution", [0, 0])[0],
                    height=raw.get("resolution", [0, 0])[1],
                )

            # Recover row-major view matrix from column-major storage
            if "view_matrix_col_major" in fr:
                col_flat = np.array(fr["view_matrix_col_major"], dtype=np.float64)
                view_row = col_flat.reshape((4, 4)).T
                cam.view_matrix = view_row
                cam.world_matrix = np.linalg.inv(view_row)

                # Validate rotation determinant
                R = view_row[:3, :3]
                det = np.linalg.det(R)
                if abs(det - 1.0) > 1e-5:
                    raise ValueError(
                        f"Frame {fr.get('frame_index', '?')}: rotation "
                        f"determinant {det:.6f} deviates from 1.0 by "
                        f"more than 1e-5 — likely corrupted camera data."
                    )

            frame = FGPPFrame(
                camera=cam,
                depth=np.zeros(
                    (cam.height or 1, cam.width or 1), dtype=np.float32
                ),
                frame_index=fr.get("frame_index", 0),
            )
            frames.append(frame)

        return cls(frames=frames, fps=fps, scene_id=scene_id)
