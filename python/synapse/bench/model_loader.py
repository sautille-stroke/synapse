"""Zero-dependency 3D model loader for Synapse-Bench.

Provides a built-in pure Python/NumPy ``.obj`` parser and bridges to
procedural mesh generators.  ``trimesh`` and ``PIL`` are lazy-imported
inside functions only when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from synapse.bench.procedural import ProceduralMeshType, generate_procedural_mesh
from synapse.render_core.mesh import MeshData, load_obj

__all__ = [
    "MeshData",
    "load_mesh",
    "load_obj",
    "load_procedural",
]

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

    if name_or_path.lower() == "humanoid_53":
        from synapse.bench.procedural import generate_humanoid_proxy

        verts, faces, _ = generate_humanoid_proxy()
        mesh = MeshData(
            name="humanoid_53",
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=np.zeros_like(verts, dtype=np.float32),
        ).compute_normals()
        if normalize:
            mesh = mesh.normalize()
        return mesh

    # File path
    path = Path(name_or_path)
    if path.suffix.lower() == ".obj" and path.exists():
        mesh = load_obj(path)
        if normalize:
            mesh = mesh.normalize()
        return mesh

    # Try trimesh as optional fallback
    try:
        import trimesh
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


# Register load_mesh into Tier-2 inspect default mesh loader registry
try:
    from synapse.inspect.model_evaluator import register_default_mesh_loader
    register_default_mesh_loader(lambda p: load_mesh(p, normalize=True))
except ImportError:
    pass
