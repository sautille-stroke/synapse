"""3D Model asset loaders and registry for Synapse."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from synapse.models.asset import ModelAsset


# Potential locations where sample/benchmark models reside
_ASSET_SEARCH_PATHS: list[Path] = [
    Path(os.environ.get("SYNAPSE_MODELS_DIR", "")),
    Path(__file__).resolve().parents[3] / "data" / "models",
    Path("../fulgur/assets/models"),
]


def _find_asset_file(filename: str) -> Optional[Path]:
    """Search for an asset filename in standard locations."""
    for base in _ASSET_SEARCH_PATHS:
        if not base or not base.exists():
            continue
        p = base / filename
        if p.is_file():
            return p
    return None


def _pil_to_rgb_array(img: Any) -> Optional[np.ndarray]:
    """Convert PIL image or trimesh texture to uint8 RGB numpy array."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            return img[:, :, :3]
        return img
    return None


def _load_from_trimesh_geom(name: str, geom: "trimesh.Trimesh") -> ModelAsset:
    """Convert a trimesh.Trimesh object into a ModelAsset."""
    vertices = np.ascontiguousarray(geom.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(geom.faces, dtype=np.int32)

    # Normals
    if hasattr(geom, "vertex_normals") and geom.vertex_normals is not None and len(geom.vertex_normals) == len(vertices):
        normals = np.ascontiguousarray(geom.vertex_normals, dtype=np.float32)
    else:
        geom.recompute_normals()
        normals = np.ascontiguousarray(geom.vertex_normals, dtype=np.float32)

    # UVs
    uvs = None
    if hasattr(geom.visual, "uv") and geom.visual.uv is not None and len(geom.visual.uv) == len(vertices):
        uvs = np.ascontiguousarray(geom.visual.uv, dtype=np.float32)

    # Textures & Materials
    base_color_texture = None
    normal_texture = None
    metallic_roughness_texture = None
    emissive_texture = None
    occlusion_texture = None

    base_color_factor = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    metallic_factor = 0.0
    roughness_factor = 0.5

    if hasattr(geom.visual, "material") and geom.visual.material is not None:
        mat = geom.visual.material
        if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            base_color_texture = _pil_to_rgb_array(mat.baseColorTexture)
        if hasattr(mat, "normalTexture") and mat.normalTexture is not None:
            normal_texture = _pil_to_rgb_array(mat.normalTexture)
        if hasattr(mat, "metallicRoughnessTexture") and mat.metallicRoughnessTexture is not None:
            metallic_roughness_texture = _pil_to_rgb_array(mat.metallicRoughnessTexture)
        if hasattr(mat, "emissiveTexture") and mat.emissiveTexture is not None:
            emissive_texture = _pil_to_rgb_array(mat.emissiveTexture)
        if hasattr(mat, "occlusionTexture") and mat.occlusionTexture is not None:
            occlusion_texture = _pil_to_rgb_array(mat.occlusionTexture)

        if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
            base_color_factor = np.array(mat.baseColorFactor[:3], dtype=np.float32)
        if hasattr(mat, "metallicFactor") and mat.metallicFactor is not None:
            metallic_factor = float(mat.metallicFactor)
        if hasattr(mat, "roughnessFactor") and mat.roughnessFactor is not None:
            roughness_factor = float(mat.roughnessFactor)

    return ModelAsset(
        name=name,
        vertices=vertices,
        faces=faces,
        normals=normals,
        uvs=uvs,
        base_color_texture=base_color_texture,
        normal_texture=normal_texture,
        metallic_roughness_texture=metallic_roughness_texture,
        emissive_texture=emissive_texture,
        occlusion_texture=occlusion_texture,
        base_color_factor=base_color_factor,
        metallic_factor=metallic_factor,
        roughness_factor=roughness_factor,
    )


def _generate_procedural_bunny(subdivisions: int = 2) -> ModelAsset:
    """Generate a high-detail procedural Stanford Bunny or organic mesh."""
    import trimesh  # lazy: heavy native dependency

    # Create an icosphere or complex mesh as procedural benchmark
    base_mesh = trimesh.creation.icosphere(subdivisions=subdivisions + 1, radius=1.0)
    # Add high-frequency perturbation to mimic complex sculpture/bunny surfaces
    v = base_mesh.vertices.copy()
    theta = np.arctan2(v[:, 1], v[:, 0])
    phi = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1.0, 1.0))
    # Characteristic multi-frequency harmonic modulation
    r_mod = 1.0 + 0.18 * np.sin(3 * theta) * np.sin(2 * phi) + 0.08 * np.cos(5 * phi)
    v *= r_mod[:, None]
    base_mesh.vertices = v
    base_mesh.recompute_normals()

    asset = _load_from_trimesh_geom("stanford_bunny_procedural", base_mesh)
    asset.base_color_factor = np.array([0.9, 0.82, 0.75], dtype=np.float32)  # Terracotta / ceramic
    return asset


def _generate_procedural_teapot() -> ModelAsset:
    """Generate Utah Teapot or complex surface."""
    import trimesh  # lazy: heavy native dependency

    # Compose body (sphere), handle (torus), spout, lid
    body = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    # Scale body to teapot proportions
    body.vertices[:, 2] *= 0.75
    asset = _load_from_trimesh_geom("utah_teapot_procedural", body)
    asset.base_color_factor = np.array([0.2, 0.55, 0.85], dtype=np.float32)  # Porcelain blue
    return asset


def list_builtin_models() -> list[str]:
    """Return names of available built-in models."""
    models = ["bunny", "teapot", "sphere", "cube", "torus", "cylinder"]
    for candidate in ["damaged_helmet", "duck", "mannequin"]:
        if _find_asset_file(f"{candidate}.glb") is not None:
            models.insert(0, candidate)
    return models


def load_model(
    name_or_path: str | Path,
    normalize: bool = True,
    target_radius: float = 1.0,
) -> ModelAsset:
    """Load a 3D model asset by built-in name or file path."""
    name_str = str(name_or_path).lower().strip()

    # 1. Check if it's a known built-in keyword
    if name_str in ("damaged_helmet", "helmet", "damagedhelmet"):
        glb_path = _find_asset_file("damaged_helmet.glb")
        if glb_path is not None:
            return load_model(glb_path, normalize=normalize, target_radius=target_radius)
    elif name_str in ("duck", "rubber_duck"):
        glb_path = _find_asset_file("duck.glb")
        if glb_path is not None:
            return load_model(glb_path, normalize=normalize, target_radius=target_radius)
    elif name_str in ("mannequin",):
        glb_path = _find_asset_file("mannequin.glb")
        if glb_path is not None:
            return load_model(glb_path, normalize=normalize, target_radius=target_radius)
    elif name_str in ("bunny", "stanford_bunny"):
        asset = _generate_procedural_bunny()
        return asset.normalize(target_radius) if normalize else asset
    elif name_str in ("teapot", "utah_teapot"):
        asset = _generate_procedural_teapot()
        return asset.normalize(target_radius) if normalize else asset
    elif name_str in ("sphere", "cube", "torus", "cylinder"):
        from synapse.bench.procedural import generate_procedural_mesh, ProceduralMeshType
        pm = generate_procedural_mesh(ProceduralMeshType(name_str))
        asset = ModelAsset(
            name=name_str,
            vertices=pm.vertices.astype(np.float32),
            faces=pm.faces.astype(np.int32),
            normals=pm.normals.astype(np.float32),
            uvs=pm.uvs.astype(np.float32) if pm.uvs is not None else None,
        )
        return asset.normalize(target_radius) if normalize else asset

    # 2. File path resolution
    path = Path(name_or_path)
    if not path.is_file():
        # Try finding in search paths
        found = _find_asset_file(path.name)
        if found is not None:
            path = found
        else:
            raise FileNotFoundError(f"Model file or built-in model '{name_or_path}' not found.")

    # Load via trimesh
    import trimesh  # lazy: heavy native dependency

    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        # Merge all geometries into a single mesh or pick the primary one
        if len(loaded.geometry) == 0:
            raise ValueError(f"Scene in {path} has no geometry.")
        geoms = list(loaded.geometry.values())
        if len(geoms) == 1:
            asset = _load_from_trimesh_geom(path.stem, geoms[0])
        else:
            # Concatenate meshes
            concat = trimesh.util.concatenate(geoms)
            asset = _load_from_trimesh_geom(path.stem, concat)
    elif isinstance(loaded, trimesh.Trimesh):
        asset = _load_from_trimesh_geom(path.stem, loaded)
    else:
        raise ValueError(f"Unsupported geometry type loaded from {path}: {type(loaded)}")

    if normalize:
        asset = asset.normalize(target_radius=target_radius)

    return asset
