"""Zero-dependency 3D model loader for Synapse-Bench.

Provides a built-in pure Python/NumPy ``.obj`` parser and bridges to
procedural mesh generators.  ``trimesh`` and ``PIL`` are lazy-imported
inside functions only when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from synapse.bench.procedural import ProceduralMeshType, generate_procedural_mesh


@dataclass
class MeshData:
    """Lightweight mesh container for the rasterizer.

    Unlike :class:`~synapse.models.asset.ModelAsset`, this focuses purely
    on geometry data needed for rasterization.
    """

    name: str
    vertices: np.ndarray  # [N, 3] float32
    faces: np.ndarray  # [M, 3] int32
    normals: np.ndarray  # [N, 3] float32
    uvs: Optional[np.ndarray] = None  # [N, 2] float32
    base_color: Optional[np.ndarray] = None  # [N, 3] float32

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        """Number of triangular faces."""
        return self.faces.shape[0]

    def normalize(self, target_radius: float = 1.0) -> "MeshData":
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

    def compute_normals(self) -> "MeshData":
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


# ---------------------------------------------------------------------------
# Pure Python OBJ parser (zero external dependencies)
# ---------------------------------------------------------------------------


def load_obj(path: str | Path) -> MeshData:
    """Parse a Wavefront OBJ file into a :class:`MeshData`.

    Supports ``v``, ``vn``, ``vt``, and ``f`` directives.
    Face entries may be ``v``, ``v/vt``, ``v//vn``, or ``v/vt/vn``.
    Polygons with more than 3 vertices are naively fan-triangulated.
    """
    path = Path(path)
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []
    face_normals: list[list[int]] = []
    face_uvs: list[list[int]] = []

    with open(path, "r", encoding="utf-8") as f:
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
        # Map face-corner normals to per-vertex by averaging
        vert_normals = np.zeros_like(verts_arr)
        counts = np.zeros(len(vertices), dtype=np.float32)
        for fi, fn in enumerate(face_normals):
            for vi, ni in zip(faces[fi], fn):
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
            for vi, ti in zip(faces[fi], ft):
                vert_uvs[vi] = all_uvs[ti]
        uvs_arr = vert_uvs

    mesh = MeshData(
        name=path.stem,
        vertices=verts_arr,
        faces=faces_arr,
        normals=vert_normals,
        uvs=uvs_arr,
    )

    # Recompute normals if none were provided
    if not normals and mesh.num_faces > 0:
        mesh.compute_normals()

    return mesh


# ---------------------------------------------------------------------------
# Procedural mesh bridge
# ---------------------------------------------------------------------------


def load_procedural(name: str, **kwargs: object) -> MeshData:
    """Load a procedural mesh by name.

    Supported: ``cube``, ``sphere``, ``cylinder``, ``torus``, ``plane``.
    """
    try:
        mesh_type = ProceduralMeshType(name.lower())
    except ValueError as exc:
        raise ValueError(
            f"Unknown procedural mesh '{name}'. "
            f"Available: {[m.value for m in ProceduralMeshType]}"
        ) from exc

    pmesh = generate_procedural_mesh(mesh_type, **kwargs)
    return MeshData(
        name=name,
        vertices=pmesh.vertices.astype(np.float32),
        faces=pmesh.faces.astype(np.int32),
        normals=pmesh.normals.astype(np.float32),
        uvs=pmesh.uvs.astype(np.float32) if pmesh.uvs is not None else None,
    )


def load_mesh(name_or_path: str, normalize: bool = True) -> MeshData:
    """Load a mesh by name (procedural) or file path.

    If *name_or_path* matches a procedural mesh type, generates it.
    If it ends with ``.obj``, parses the OBJ file.
    Otherwise, attempts ``trimesh`` as an optional fallback.
    """
    # Check procedural first
    try:
        mesh = load_procedural(name_or_path)
        if normalize:
            mesh = mesh.normalize()
        return mesh
    except ValueError:
        pass

    # File path
    path = Path(name_or_path)
    if path.suffix.lower() == ".obj" and path.exists():
        mesh = load_obj(path)
        if normalize:
            mesh = mesh.normalize()
        return mesh

    # Try trimesh as optional fallback
    try:
        import trimesh  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            f"Cannot load '{name_or_path}': not a procedural mesh or .obj file, "
            f"and 'trimesh' is not installed. "
            f"Install with: pip install synapse[mesh]"
        ) from exc

    scene_or_mesh: Any = trimesh.load(str(path), force="mesh")
    if hasattr(scene_or_mesh, "vertices"):
        geom: Any = scene_or_mesh
    else:
        geom = list(scene_or_mesh.geometry.values())[0]

    mesh = MeshData(
        name=path.stem,
        vertices=np.asarray(geom.vertices, dtype=np.float32),
        faces=np.asarray(geom.faces, dtype=np.int32),
        normals=np.asarray(geom.vertex_normals, dtype=np.float32),
    )
    if normalize:
        mesh = mesh.normalize()
    return mesh
