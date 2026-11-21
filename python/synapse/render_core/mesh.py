"""Lightweight mesh data container and pure-Python OBJ loader.

Defined in Tier 0 (render_core) so both Tier 2 (inspect) and Tier 3 (bench)
can share standard mesh representation without upward architectural violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synapse.errors import AssetNotFoundError, ContractValidationError


@dataclass
class MeshData:
    """Lightweight mesh container for the rasterizer and quality inspection."""

    name: str
    vertices: np.ndarray  # [N, 3] float32
    faces: np.ndarray  # [M, 3] int32
    normals: np.ndarray  # [N, 3] float32
    uvs: np.ndarray | None = None  # [N, 2] float32
    base_color: np.ndarray | None = None  # [N, 3] float32
    topology_vertices: np.ndarray | None = None  # [V_topo, 3] unsplit positional vertices
    topology_faces: np.ndarray | None = None  # [M, 3] unsplit topological face indices
    is_normalized: bool = False

    def get_topology_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (vertices, faces) preserving original topology before UV/normal splits."""
        if self.topology_vertices is not None and self.topology_faces is not None:
            return self.topology_vertices, self.topology_faces
        return self.vertices, self.faces

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        """Number of triangular faces."""
        return int(self.faces.shape[0])

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box (min_corner, max_corner)."""
        if self.num_vertices == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def centroid(self) -> np.ndarray:
        """Compute center of axis-aligned bounding box."""
        min_c, max_c = self.bounding_box()
        return np.asarray((min_c + max_c) * 0.5, dtype=np.float32)

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Compute center and radius of bounding sphere."""
        center = self.centroid()
        if self.num_vertices == 0:
            return center, 0.0
        dists = np.linalg.norm(self.vertices - center, axis=1)
        return center, float(np.max(dists))

    def normalize(self, target_radius: float = 1.0) -> MeshData:
        """Center at origin and scale to *target_radius*."""
        if self.num_vertices == 0:
            return self
        center, radius = self.bounding_sphere()
        if radius < 1e-8:
            return self
        scale = target_radius / radius
        topo_v = (
            ((self.topology_vertices - center) * scale).astype(np.float32)
            if self.topology_vertices is not None
            else None
        )
        return MeshData(
            name=self.name,
            vertices=((self.vertices - center) * scale).astype(np.float32),
            faces=self.faces.copy(),
            normals=self.normals.copy(),
            uvs=self.uvs.copy() if self.uvs is not None else None,
            base_color=self.base_color.copy() if self.base_color is not None else None,
            topology_vertices=topo_v,
            topology_faces=self.topology_faces.copy() if self.topology_faces is not None else None,
            is_normalized=True,
        )

    def compute_normals(self) -> MeshData:
        """Recompute vertex normals from face geometry."""
        v = self.vertices
        f = self.faces
        if len(v) == 0 or len(f) == 0:
            self.normals = np.zeros_like(v, dtype=np.float32)
            return self
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [f"# Synapse Mesh: {self.name}"]
        lines.append(f"# Vertices: {self.num_vertices}, Faces: {self.num_faces}")

        for vert in self.vertices:
            lines.append(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}")

        has_normals = self.normals is not None and len(self.normals) == len(self.vertices)
        if has_normals:
            for norm in self.normals:
                lines.append(f"vn {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}")

        has_uv = self.uvs is not None and len(self.uvs) == len(self.vertices)
        if has_uv and self.uvs is not None:
            for uv in self.uvs:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

        for face in self.faces:
            f1, f2, f3 = face[0] + 1, face[1] + 1, face[2] + 1
            if has_uv and has_normals:
                lines.append(f"f {f1}/{f1}/{f1} {f2}/{f2}/{f2} {f3}/{f3}/{f3}")
            elif has_uv:
                lines.append(f"f {f1}/{f1} {f2}/{f2} {f3}/{f3}")
            elif has_normals:
                lines.append(f"f {f1}//{f1} {f2}//{f2} {f3}//{f3}")
            else:
                lines.append(f"f {f1} {f2} {f3}")

        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def save_obj(self, path: str | Path) -> Path:
        """Alias for export_obj."""
        return self.export_obj(path)


def load_obj(path: str | Path) -> MeshData:
    """Parse a Wavefront OBJ file into a :class:`MeshData` (pure Python)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise AssetNotFoundError(
            f"OBJ file not found: {path}",
            agent_fix_hint=f"Verify that '{path}' exists and is a readable Wavefront .obj file.",
        )

    raw_vertices: list[list[float]] = []
    raw_normals: list[list[float]] = []
    raw_uvs: list[list[float]] = []
    polygon_faces: list[list[tuple[int, int, int]]] = []

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]

            if key == "v" and len(parts) >= 4:
                raw_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vn" and len(parts) >= 4:
                raw_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vt" and len(parts) >= 3:
                raw_uvs.append([float(parts[1]), float(parts[2])])
            elif key == "f":
                poly: list[tuple[int, int, int]] = []
                for vert in parts[1:]:
                    indices = vert.split("/")
                    vi = int(indices[0])
                    if vi == 0:
                        raise ContractValidationError(
                            f"OBJ file '{path}' contains invalid 0 vertex index (OBJ is 1-indexed).",
                            agent_fix_hint="Fix vertex indices in the OBJ file (must be >= 1 or negative relative).",
                        )
                    vi_idx = vi - 1 if vi > 0 else len(raw_vertices) + vi
                    if vi_idx < 0 or vi_idx >= len(raw_vertices):
                        raise ContractValidationError(
                            f"OBJ file '{path}' contains out-of-range vertex index {vi} "
                            f"(vertex count={len(raw_vertices)}).",
                            agent_fix_hint="Verify all face indices in the OBJ file reference valid vertices.",
                        )
                    ti_idx = -1
                    if len(indices) > 1 and indices[1]:
                        ti = int(indices[1])
                        if ti != 0:
                            ti_idx = ti - 1 if ti > 0 else len(raw_uvs) + ti
                    ni_idx = -1
                    if len(indices) > 2 and indices[2]:
                        ni = int(indices[2])
                        if ni != 0:
                            ni_idx = ni - 1 if ni > 0 else len(raw_normals) + ni
                    poly.append((vi_idx, ti_idx, ni_idx))
                if len(poly) >= 3:
                    polygon_faces.append(poly)

    # Fan triangulation with unique (vi, ti, ni) mapping to preserve UV seams and normals,
    # while preserving original positional topology (raw_vertices + topo_faces) for watertight checks.
    vertex_map: dict[tuple[int, int, int], int] = {}
    new_vertices: list[list[float]] = []
    new_normals: list[list[float]] = []
    new_uvs: list[list[float]] = []
    faces: list[list[int]] = []
    topo_faces: list[list[int]] = []

    has_any_uv = False
    has_any_normal = False

    for poly in polygon_faces:
        # Fan triangulation
        for k in range(1, len(poly) - 1):
            tri = [poly[0], poly[k], poly[k + 1]]
            tri_indices: list[int] = []
            topo_faces.append([tri[0][0], tri[1][0], tri[2][0]])
            for corner in tri:
                vi, ti, ni = corner
                if corner not in vertex_map:
                    new_idx = len(new_vertices)
                    vertex_map[corner] = new_idx
                    new_vertices.append(raw_vertices[vi])
                    if 0 <= ti < len(raw_uvs):
                        new_uvs.append(raw_uvs[ti])
                        has_any_uv = True
                    else:
                        new_uvs.append([0.0, 0.0])
                    if 0 <= ni < len(raw_normals):
                        new_normals.append(raw_normals[ni])
                        has_any_normal = True
                    else:
                        new_normals.append([0.0, 0.0, 0.0])
                tri_indices.append(vertex_map[corner])
            faces.append(tri_indices)

    raw_verts_arr = (
        np.array(raw_vertices, dtype=np.float32) if raw_vertices else np.zeros((0, 3), dtype=np.float32)
    )
    if new_vertices:
        verts_arr = np.array(new_vertices, dtype=np.float32)
        faces_arr = np.array(faces, dtype=np.int32)
        topo_faces_arr: np.ndarray | None = np.array(topo_faces, dtype=np.int32)
        topo_verts_arr: np.ndarray | None = raw_verts_arr
        uvs_arr = np.array(new_uvs, dtype=np.float32) if has_any_uv else None
        norms_arr = np.array(new_normals, dtype=np.float32) if has_any_normal else np.zeros_like(verts_arr)
    else:
        verts_arr = raw_verts_arr
        faces_arr = np.zeros((0, 3), dtype=np.int32)
        topo_faces_arr = None
        topo_verts_arr = None
        uvs_arr = np.array(raw_uvs, dtype=np.float32) if raw_uvs else None
        norms_arr = np.array(raw_normals, dtype=np.float32) if raw_normals else np.zeros_like(verts_arr)

    mesh = MeshData(
        name=p.stem,
        vertices=verts_arr,
        faces=faces_arr,
        normals=norms_arr,
        uvs=uvs_arr,
        topology_vertices=topo_verts_arr,
        topology_faces=topo_faces_arr,
    )
    if not has_any_normal or len(norms_arr) == 0 or (len(norms_arr) > 0 and np.all(norms_arr == 0)):
        mesh.compute_normals()
    return mesh
