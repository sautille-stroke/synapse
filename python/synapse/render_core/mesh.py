"""Lightweight mesh data container and pure-Python OBJ loader.

Defined in Tier 0 (render_core) so both Tier 2 (inspect) and Tier 3 (bench)
can share standard mesh representation without upward architectural violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshData:
    """Lightweight mesh container for the rasterizer and quality inspection."""

    name: str
    vertices: np.ndarray  # [N, 3] float32
    faces: np.ndarray  # [M, 3] int32
    normals: np.ndarray  # [N, 3] float32
    uvs: np.ndarray | None = None  # [N, 2] float32
    base_color: np.ndarray | None = None  # [N, 3] float32

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        """Number of triangular faces."""
        return int(self.faces.shape[0])

    def normalize(self, target_radius: float = 1.0) -> MeshData:
        """Center at origin and scale to *target_radius*."""
        if self.num_vertices == 0:
            return self
        center = (self.vertices.min(axis=0) + self.vertices.max(axis=0)) * 0.5
        dists = np.linalg.norm(self.vertices - center, axis=1)
        radius = float(np.max(dists))
        if radius < 1e-8:
            return self
        scale = target_radius / radius
        return MeshData(
            name=self.name,
            vertices=((self.vertices - center) * scale).astype(np.float32),
            faces=self.faces.copy(),
            normals=self.normals.copy(),
            uvs=self.uvs.copy() if self.uvs is not None else None,
            base_color=self.base_color.copy() if self.base_color is not None else None,
        )

    def compute_normals(self) -> MeshData:
        """Recompute vertex normals from face geometry."""
        v = self.vertices
        f = self.faces
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)

        vert_normals = np.zeros_like(v, dtype=np.float32)
        for i in range(3):
            np.add.at(vert_normals, f[:, i], face_normals)

        norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self.normals = (vert_normals / norms).astype(np.float32)
        return self


def load_obj(path: str | Path) -> MeshData:
    """Parse a Wavefront OBJ file into a :class:`MeshData` (pure Python)."""
    p = Path(path)
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []
    face_normals: list[list[int]] = []
    face_uvs: list[list[int]] = []

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]

            if key == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vn" and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vt" and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif key == "f":
                face_v: list[int] = []
                face_n: list[int] = []
                face_t: list[int] = []
                for vert in parts[1:]:
                    indices = vert.split("/")
                    face_v.append(int(indices[0]) - 1)  # OBJ is 1-based
                    if len(indices) > 1 and indices[1]:
                        face_t.append(int(indices[1]) - 1)
                    if len(indices) > 2 and indices[2]:
                        face_n.append(int(indices[2]) - 1)

                # Fan triangulation for polygons
                for i in range(1, len(face_v) - 1):
                    faces.append([face_v[0], face_v[i], face_v[i + 1]])
                    if face_n:
                        face_normals.append([face_n[0], face_n[i], face_n[i + 1]])
                    if face_t:
                        face_uvs.append([face_t[0], face_t[i], face_t[i + 1]])

    verts_arr = np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32)
    faces_arr = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

    # Build per-vertex normals
    if normals and face_normals:
        normals_arr = np.array(normals, dtype=np.float32)
        vert_normals = np.zeros_like(verts_arr)
        counts = np.zeros(len(vertices), dtype=np.float32)
        for fi, fn in enumerate(face_normals):
            for vi, ni in zip(faces[fi], fn, strict=False):
                vert_normals[vi] += normals_arr[ni]
                counts[vi] += 1.0
        counts = np.maximum(counts, 1.0)
        vert_normals /= counts[:, np.newaxis]
        norms_mag = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        vert_normals = (vert_normals / np.maximum(norms_mag, 1e-12)).astype(np.float32)
    else:
        vert_normals = np.zeros_like(verts_arr)

    # UVs
    uvs_arr = None
    if uvs and face_uvs:
        all_uvs = np.array(uvs, dtype=np.float32)
        vert_uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        for fi, ft in enumerate(face_uvs):
            for vi, ti in zip(faces[fi], ft, strict=False):
                vert_uvs[vi] = all_uvs[ti]
        uvs_arr = vert_uvs

    mesh = MeshData(
        name=p.stem,
        vertices=verts_arr,
        faces=faces_arr,
        normals=vert_normals,
        uvs=uvs_arr,
    )
    if not normals:
        mesh.compute_normals()
    return mesh
