"""3D Model asset representation and utilities for Synapse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelAsset:
    """A high-fidelity 3D mesh asset with geometry, normals, UVs and textures."""

    name: str
    vertices: np.ndarray  # [N, 3] float32
    faces: np.ndarray  # [M, 3] int32
    normals: np.ndarray  # [N, 3] float32
    uvs: np.ndarray | None = None  # [N, 2] float32
    vertex_colors: np.ndarray | None = None  # [N, 3] or [N, 4] float32

    # PBR Textures (optional uint8 RGB/RGBA arrays)
    base_color_texture: np.ndarray | None = None  # [H, W, 3] uint8
    normal_texture: np.ndarray | None = None  # [H, W, 3] uint8
    metallic_roughness_texture: np.ndarray | None = None  # [H, W, 3] uint8
    emissive_texture: np.ndarray | None = None  # [H, W, 3] uint8
    occlusion_texture: np.ndarray | None = None  # [H, W, 3] uint8

    # Base PBR material factors
    base_color_factor: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=np.float32))
    metallic_factor: float = 0.0
    roughness_factor: float = 0.5

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box (min_corner, max_corner)."""
        if self.num_vertices == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def centroid(self) -> np.ndarray:
        """Compute center of axis-aligned bounding box."""
        min_c, max_c = self.bounding_box()
        return (min_c + max_c) * 0.5

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Compute center and radius of bounding sphere."""
        center = self.centroid()
        if self.num_vertices == 0:
            return center, 0.0
        dists = np.linalg.norm(self.vertices - center, axis=1)
        return center, float(np.max(dists))

    def normalize(self, target_radius: float = 1.0) -> ModelAsset:
        """Center the mesh at origin and scale so its bounding sphere radius equals target_radius."""
        center, radius = self.bounding_sphere()
        if radius <= 1e-8:
            return self

        scale = target_radius / radius
        norm_vertices = ((self.vertices - center) * scale).astype(np.float32)

        return ModelAsset(
            name=self.name,
            vertices=norm_vertices,
            faces=self.faces.copy(),
            normals=self.normals.copy(),
            uvs=self.uvs.copy() if self.uvs is not None else None,
            vertex_colors=self.vertex_colors.copy() if self.vertex_colors is not None else None,
            base_color_texture=self.base_color_texture,
            normal_texture=self.normal_texture,
            metallic_roughness_texture=self.metallic_roughness_texture,
            emissive_texture=self.emissive_texture,
            occlusion_texture=self.occlusion_texture,
            base_color_factor=self.base_color_factor.copy(),
            metallic_factor=self.metallic_factor,
            roughness_factor=self.roughness_factor,
        )

    def compute_normals(self) -> ModelAsset:
        """Compute vertex normals via area-weighted face normals."""
        v = self.vertices
        f = self.faces
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        vert_normals = np.zeros_like(v, dtype=np.float32)

        for i in range(3):
            np.add.at(vert_normals, f[:, i], face_normals)

        norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self.normals = (vert_normals / norms).astype(np.float32)
        return self

    def export_obj(self, path: str | Path) -> Path:
        """Export geometry to Wavefront OBJ format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [f"# Synapse ModelAsset: {self.name}"]
        lines.append(f"# Vertices: {self.num_vertices}, Faces: {self.num_faces}")

        for vert in self.vertices:
            lines.append(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}")

        for norm in self.normals:
            lines.append(f"vn {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}")

        if self.uvs is not None:
            for uv in self.uvs:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

        has_uv = self.uvs is not None and len(self.uvs) == len(self.vertices)
        for face in self.faces:
            # 1-based indexing for OBJ
            f1, f2, f3 = face[0] + 1, face[1] + 1, face[2] + 1
            if has_uv:
                lines.append(f"f {f1}/{f1}/{f1} {f2}/{f2}/{f2} {f3}/{f3}/{f3}")
            else:
                lines.append(f"f {f1}//{f1} {f2}//{f2} {f3}//{f3}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def get_summary(self) -> dict[str, Any]:
        """Return metadata summary for logging and manifest serialization."""
        min_c, max_c = self.bounding_box()
        center, radius = self.bounding_sphere()
        return {
            "name": self.name,
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "has_normals": self.normals is not None and len(self.normals) > 0,
            "has_uvs": self.uvs is not None and len(self.uvs) > 0,
            "has_base_color_texture": self.base_color_texture is not None,
            "has_normal_texture": self.normal_texture is not None,
            "bounding_box_min": min_c.tolist(),
            "bounding_box_max": max_c.tolist(),
            "bounding_sphere_center": center.tolist(),
            "bounding_sphere_radius": radius,
        }
