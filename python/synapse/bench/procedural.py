"""Procedural GT geometry generators for Synapse-Bench.

These provide ground-truth 3D meshes with known geometry, allowing
pixel-perfect evaluation without external asset dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class ProceduralMeshType(str, Enum):
    """Available procedural mesh types."""

    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    TORUS = "torus"
    PLANE = "plane"


@dataclass
class ProceduralMesh:
    """A programmatically generated triangle mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: Optional[np.ndarray] = None
    mesh_type: str = "unknown"

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the axis-aligned bbox."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def centroid(self) -> np.ndarray:
        """Compute the mesh centroid."""
        return self.vertices.mean(axis=0)

    def save_obj(self, path: str | Path) -> Path:
        """Export to Wavefront OBJ format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"# Synapse-Bench procedural mesh: {self.mesh_type}"]
        lines.append(f"# Vertices: {self.num_vertices}, Faces: {self.num_faces}")

        for v in self.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        for n in self.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

        if self.uvs is not None:
            for uv in self.uvs:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

        for f in self.faces:
            if self.uvs is not None:
                lines.append(
                    f"f {f[0]+1}/{f[0]+1}/{f[0]+1} "
                    f"{f[1]+1}/{f[1]+1}/{f[1]+1} "
                    f"{f[2]+1}/{f[2]+1}/{f[2]+1}"
                )
            else:
                lines.append(
                    f"f {f[0]+1}//{f[0]+1} "
                    f"{f[1]+1}//{f[1]+1} "
                    f"{f[2]+1}//{f[2]+1}"
                )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _generate_cube(size: float = 1.0, **kwargs) -> ProceduralMesh:
    """Unit cube centered at origin."""
    s = size / 2.0
    vertices = []
    normals = []
    uvs = []
    faces = []

    face_defs = [
        ([0, 0, 1], [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ([0, 0, -1], [(-s, s, -s), (s, s, -s), (s, -s, -s), (-s, -s, -s)]),
        ([0, 1, 0], [(-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s)]),
        ([0, -1, 0], [(-s, -s, s), (-s, -s, -s), (s, -s, -s), (s, -s, s)]),
        ([1, 0, 0], [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
        ([-1, 0, 0], [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
    ]

    uv_quad = [(0, 0), (1, 0), (1, 1), (0, 1)]

    for normal, corners in face_defs:
        base = len(vertices)
        for j, c in enumerate(corners):
            vertices.append(c)
            normals.append(normal)
            uvs.append(uv_quad[j])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="cube",
    )


def _generate_sphere(
    radius: float = 1.0,
    segments: int = 32,
    rings: int = 16,
    **kwargs,
) -> ProceduralMesh:
    """UV sphere centered at origin."""
    vertices = []
    normals = []
    uvs = []
    faces = []

    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments

            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)

            nx, ny, nz = x / radius, y / radius, z / radius
            u = j / segments
            v = i / rings

            vertices.append([x, y, z])
            normals.append([nx, ny, nz])
            uvs.append([u, v])

    stride = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1

            if i > 0:
                faces.append([a, c, b])
            if i < rings - 1:
                faces.append([b, c, d])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="sphere",
    )


def _generate_cylinder(
    radius: float = 0.5,
    height: float = 2.0,
    segments: int = 32,
    **kwargs,
) -> ProceduralMesh:
    """Cylinder centered at origin, axis along Y."""
    vertices = []
    normals = []
    uvs = []
    faces = []
    half_h = height / 2.0

    for i in range(2):
        y = -half_h + i * height
        for j in range(segments + 1):
            theta = 2.0 * math.pi * j / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            nx, nz = math.cos(theta), math.sin(theta)

            vertices.append([x, y, z])
            normals.append([nx, 0.0, nz])
            uvs.append([j / segments, i])

    stride = segments + 1
    for j in range(segments):
        a = j
        b = j + 1
        c = j + stride
        d = j + stride + 1
        faces.append([a, c, b])
        faces.append([b, c, d])

    top_center_idx = len(vertices)
    vertices.append([0, half_h, 0])
    normals.append([0, 1, 0])
    uvs.append([0.5, 0.5])
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        idx = len(vertices)
        vertices.append([x, half_h, z])
        normals.append([0, 1, 0])
        uvs.append([0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)])
        if j > 0:
            faces.append([top_center_idx, idx - 1, idx])
    faces.append([top_center_idx, len(vertices) - 1, top_center_idx + 1])

    bot_center_idx = len(vertices)
    vertices.append([0, -half_h, 0])
    normals.append([0, -1, 0])
    uvs.append([0.5, 0.5])
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        idx = len(vertices)
        vertices.append([x, -half_h, z])
        normals.append([0, -1, 0])
        uvs.append([0.5 + 0.5 * math.cos(theta), 0.5 + 0.5 * math.sin(theta)])
        if j > 0:
            faces.append([bot_center_idx, idx, idx - 1])
    faces.append([bot_center_idx, bot_center_idx + 1, len(vertices) - 1])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="cylinder",
    )


def _generate_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_segments: int = 32,
    minor_segments: int = 16,
    **kwargs,
) -> ProceduralMesh:
    """Torus centered at origin, lying in XZ plane."""
    vertices = []
    normals = []
    uvs = []
    faces = []

    for i in range(major_segments + 1):
        theta = 2.0 * math.pi * i / major_segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        for j in range(minor_segments + 1):
            phi = 2.0 * math.pi * j / minor_segments
            cos_p, sin_p = math.cos(phi), math.sin(phi)

            r = major_radius + minor_radius * cos_p
            x = r * cos_t
            y = minor_radius * sin_p
            z = r * sin_t

            nx = cos_p * cos_t
            ny = sin_p
            nz = cos_p * sin_t

            vertices.append([x, y, z])
            normals.append([nx, ny, nz])
            uvs.append([i / major_segments, j / minor_segments])

    stride = minor_segments + 1
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="torus",
    )


def _generate_plane(
    size: float = 2.0,
    subdivisions: int = 1,
    **kwargs,
) -> ProceduralMesh:
    """Flat XZ plane centered at origin (Y=0)."""
    n = subdivisions + 1
    half = size / 2.0
    vertices = []
    normals = []
    uvs = []
    faces = []

    for i in range(n + 1):
        for j in range(n + 1):
            x = -half + size * j / n
            z = -half + size * i / n
            vertices.append([x, 0.0, z])
            normals.append([0.0, 1.0, 0.0])
            uvs.append([j / n, i / n])

    stride = n + 1
    for i in range(n):
        for j in range(n):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return ProceduralMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
        uvs=np.array(uvs, dtype=np.float64),
        mesh_type="plane",
    )


_GENERATORS = {
    ProceduralMeshType.CUBE: _generate_cube,
    ProceduralMeshType.SPHERE: _generate_sphere,
    ProceduralMeshType.CYLINDER: _generate_cylinder,
    ProceduralMeshType.TORUS: _generate_torus,
    ProceduralMeshType.PLANE: _generate_plane,
}


def generate_procedural_mesh(
    mesh_type: ProceduralMeshType | str,
    **params,
) -> ProceduralMesh:
    """Generate a procedural mesh of the given type."""
    if isinstance(mesh_type, str):
        mesh_type = ProceduralMeshType(mesh_type)

    generator = _GENERATORS.get(mesh_type)
    if generator is None:
        raise ValueError(
            f"Unsupported mesh type: {mesh_type}. "
            f"Available: {[t.value for t in ProceduralMeshType]}"
        )

    return generator(**params)
