"""FGPP frame and sequence data classes.

Mirrors the ``nexus//fgpp`` SSOT channel layout defined in RFC 0044 §4.1.
All channel names, data types, and coordinate-space conventions follow
the authoritative nexus specification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


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

    def position(self) -> np.ndarray:
        """Camera position in world space."""
        return self.world_matrix[:3, 3].copy()

    def forward(self) -> np.ndarray:
        """Camera forward direction (negative Z axis in camera space)."""
        return -self.world_matrix[:3, 2].copy()

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
            "view_matrix": self.view_matrix.tolist(),
            "world_matrix": self.world_matrix.tolist(),
            "z_near": self.z_near,
            "z_far": self.z_far,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraParams":
        """Deserialise from dict."""
        return cls(
            fx=d["fx"],
            fy=d["fy"],
            cx=d["cx"],
            cy=d["cy"],
            width=d["width"],
            height=d["height"],
            fov_degrees=d.get("fov_degrees", 60.0),
            view_matrix=np.array(d["view_matrix"], dtype=np.float64),
            world_matrix=np.array(d["world_matrix"], dtype=np.float64),
            z_near=d.get("z_near", 0.1),
            z_far=d.get("z_far", 1000.0),
        )


@dataclass
class FGPPFrame:
    """A single FGPP (Feature Guidance Package Protocol) frame.

    Channel semantics follow ``nexus//fgpp`` SSOT (RFC 0044 §4.1):

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
    direct_radiance: Optional[np.ndarray] = None  # [H, W, 3] float16
    world_normal: Optional[np.ndarray] = None  # [H, W, 3] float16
    motion_vector: Optional[np.ndarray] = None  # [H, W, 2] float16
    base_color: Optional[np.ndarray] = None  # [H, W, 3] float16
    roughness_metallic: Optional[np.ndarray] = None  # [H, W, 2] float16
    cryptomatte_id: Optional[np.ndarray] = None  # [H, W] uint32

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
        """Write ``metadata.json`` camera trajectory manifest.

        This is the sidecar file specified by the FGPP protocol alongside
        the multi-channel EXR frame files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "version": "1.0",
            "scene_id": self.scene_id,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "camera": f.camera.to_dict(),
                    "channels": f.available_channels(),
                }
                for f in self.frames
            ],
        }

        path = output_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path
