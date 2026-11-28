"""3D Model asset representation and utilities for Synapse.

Extends :class:`synapse.render_core.mesh.MeshData` with PBR textures
and material factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.render_core.mesh import MeshData


@dataclass
class ModelAsset(MeshData):
    """A high-fidelity 3D mesh asset with geometry, normals, UVs and textures.

    Inherits geometry fields (name, vertices, faces, normals, uvs, base_color)
    from :class:`MeshData` and adds PBR texture/material extensions.
    """

    vertex_colors: np.ndarray | None = None  # [N, 3] or [N, 4] float32

    # PBR Textures (optional uint8 RGB/RGBA arrays)
    base_color_texture: np.ndarray | None = None  # [H, W, 3] uint8
    normal_texture: np.ndarray | None = None  # [H, W, 3] uint8
    metallic_roughness_texture: np.ndarray | None = None  # [H, W, 3] uint8
    emissive_texture: np.ndarray | None = None  # [H, W, 3] uint8
    occlusion_texture: np.ndarray | None = None  # [H, W, 3] uint8

    # Base PBR material factors
    base_color_factor: np.ndarray = field(
        default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=np.float32)
    )
    metallic_factor: float = 0.0
    roughness_factor: float = 0.5

    def normalize(self, target_radius: float = 1.0) -> ModelAsset:
        """Center the mesh at origin and scale so its bounding sphere radius equals target_radius."""
        import copy

        center, radius = self.bounding_sphere()
        if radius <= 1e-8:
            return copy.deepcopy(self)

        scale = target_radius / radius
        norm_vertices = ((self.vertices - center) * scale).astype(np.float32)
        topo_vertices = (
            ((self.topology_vertices - center) * scale).astype(np.float32)
            if self.topology_vertices is not None
            else None
        )

        return ModelAsset(
            name=self.name,
            vertices=norm_vertices,
            faces=self.faces.copy(),
            normals=self.normals.copy(),
            uvs=self.uvs.copy() if self.uvs is not None else None,
            base_color=self.base_color.copy() if self.base_color is not None else None,
            topology_vertices=topo_vertices,
            topology_faces=self.topology_faces.copy() if self.topology_faces is not None else None,
            is_normalized=True,
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
        # Delegate to parent MeshData.compute_normals() which mutates self.normals
        super().compute_normals()
        return self

    def to_mesh_data(self) -> MeshData:
        """Convert to canonical :class:`synapse.render_core.mesh.MeshData`."""
        return MeshData(
            name=self.name,
            vertices=self.vertices,
            faces=self.faces,
            normals=self.normals,
            uvs=self.uvs,
            base_color=self.vertex_colors[:, :3] if self.vertex_colors is not None else self.base_color,
            topology_vertices=self.topology_vertices,
            topology_faces=self.topology_faces,
            is_normalized=self.is_normalized,
        )

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
