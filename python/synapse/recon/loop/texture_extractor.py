# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""UV texture extraction, seam repair, and OpenPBR packing (RFC 0062 §4.3).

Provides utilities for extracting PBR texture maps from optimised material
results and packing them into a standard OpenPBR-compatible texture set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PBRTextureSet:
    """Extracted PBR texture set.

    Attributes:
        albedo: Base colour map, ``(R, R, 3)`` float32 linear sRGB.
        normal: Normal map, ``(R, R, 3)`` float32 tangent-space.
        roughness: Roughness map, ``(R, R)`` float32.
        metallic: Metallic map, ``(R, R)`` float32.
        resolution: Texture resolution (width = height).
    """

    albedo: np.ndarray
    normal: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    resolution: int = 256


def extract_and_pack_pbr_textures(
    albedo_map: np.ndarray,
    normal_map: np.ndarray | None = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
    output_dir: str | Path | None = None,
) -> PBRTextureSet:
    """Extract and pack PBR textures from inverse optimisation results.

    Performs UV-space total-variation smoothing on the albedo map and
    constructs a minimal OpenPBR-compatible texture set.

    Args:
        albedo_map: ``(R, R, 3)`` float32 recovered albedo from the
            inverse optimiser.
        normal_map: ``(R, R, 3)`` float32 normal map, or ``None`` to
            generate a flat normal map.
        roughness: Uniform roughness value (default 0.5).
        metallic: Uniform metallic value (default 0.0).
        output_dir: Optional output directory for saving textures as
            NumPy files.

    Returns:
        :class:`PBRTextureSet` with the extracted textures.
    """
    res = albedo_map.shape[0]

    # Seam repair: apply TV smoothing to reduce UV seam artefacts
    albedo = _tv_smooth(np.asarray(albedo_map, dtype=np.float32), iterations=5)
    albedo = np.clip(albedo, 0.0, 1.0)

    # Normal map
    if normal_map is not None:
        normals = np.asarray(normal_map, dtype=np.float32)
    else:
        # Flat normal map (tangent space: [0, 0, 1])
        normals = np.zeros((res, res, 3), dtype=np.float32)
        normals[..., 2] = 1.0

    # Roughness and metallic maps
    rough_map = np.full((res, res), roughness, dtype=np.float32)
    metal_map = np.full((res, res), metallic, dtype=np.float32)

    texture_set = PBRTextureSet(
        albedo=albedo,
        normal=normals,
        roughness=rough_map,
        metallic=metal_map,
        resolution=res,
    )

    # Save if output directory provided
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(str(out / "albedo.npy"), albedo)
        np.save(str(out / "normal.npy"), normals)
        np.save(str(out / "roughness.npy"), rough_map)
        np.save(str(out / "metallic.npy"), metal_map)

    return texture_set


def _tv_smooth(
    img: np.ndarray, iterations: int = 5, weight: float = 0.1,
) -> np.ndarray:
    """Apply total-variation smoothing (Chambolle-like denoising)."""
    result = img.copy()
    for _ in range(iterations):
        grad = np.zeros_like(result, dtype=np.float64)
        grad[:-1, :, :] += (
            result[:-1, :, :].astype(np.float64)
            - result[1:, :, :].astype(np.float64)
        )
        grad[:, :-1, :] += (
            result[:, :-1, :].astype(np.float64)
            - result[:, 1:, :].astype(np.float64)
        )
        result = result - weight * grad.astype(np.float32)
    return result
