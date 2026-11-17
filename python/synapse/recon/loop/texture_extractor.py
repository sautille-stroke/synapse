# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""UV texture extraction, seam repair, and OpenPBR packing (RFC 0062 §4.3).

Provides utilities for extracting PBR texture maps from optimised material
results, baking tangent-space UV normal maps from world-space normals, and
packing them into a standard OpenPBR-compatible texture set.

The ``compute_tv_gradient`` function implements a *symmetric divergence*
total-variation gradient that sums to exactly zero across the UV domain
(conservation).  It is public so that the optimizer and unit tests can
share a single canonical implementation.
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


# ---------------------------------------------------------------------------
# Public: symmetric divergence TV gradient
# ---------------------------------------------------------------------------


def compute_tv_gradient(img: np.ndarray) -> np.ndarray:
    """Compute the symmetric divergence TV gradient for a 2-D image.

    For isotropic (L2) total variation the gradient at pixel ``(i, j)`` is
    the *negative divergence* of the normalised gradient field.  Using
    un-normalised TV, this reduces to the discrete Laplacian-like operator::

        div_x = (a[i,j] - a[i,j-1]) + (a[i,j] - a[i,j+1])
        div_y = (a[i,j] - a[i-1,j]) + (a[i,j] - a[i+1,j])

    Implemented via forward differences with symmetric scatter so that the
    global sum of the gradient is **exactly zero** (conservation / zero-sum).

    Args:
        img: ``(H, W, C)`` float image (any dtype convertible to float64).

    Returns:
        ``(H, W, C)`` float64 TV gradient with ``grad.sum() ≈ 0``.
    """
    alb_f64 = np.asarray(img, dtype=np.float64)
    tv_grad = np.zeros_like(alb_f64, dtype=np.float64)

    # Vertical (row) axis — forward differences along axis 0
    dy_fwd = alb_f64[:-1, :, :] - alb_f64[1:, :, :]
    tv_grad[:-1, :, :] += dy_fwd   # positive at source
    tv_grad[1:, :, :]  -= dy_fwd   # negative at destination

    # Horizontal (column) axis — forward differences along axis 1
    dx_fwd = alb_f64[:, :-1, :] - alb_f64[:, 1:, :]
    tv_grad[:, :-1, :] += dx_fwd   # positive at source
    tv_grad[:, 1:, :]  -= dx_fwd   # negative at destination

    return tv_grad


# ---------------------------------------------------------------------------
# UV normal map baking
# ---------------------------------------------------------------------------


def bake_uv_normal_map(
    world_normals: np.ndarray,
    uv_coords: np.ndarray,
    resolution: int = 256,
) -> np.ndarray:
    """Bake world-space normals into a tangent-space UV normal map.

    Performs nearest-neighbour UV splatting of per-pixel world-space normals
    from one or more views into a ``(resolution, resolution, 3)`` texture,
    then converts to tangent-space using a local finite-difference tangent
    frame.

    Args:
        world_normals: ``(N, H, W, 3)`` or ``(H, W, 3)`` float32
            world-space surface normals.
        uv_coords: ``(N, H, W, 2)`` or ``(H, W, 2)`` float32 UV coordinates.
        resolution: Output normal map resolution.

    Returns:
        ``(resolution, resolution, 3)`` float32 tangent-space normal map
        with values in ``[-1, 1]``.  Un-covered texels default to
        ``[0, 0, 1]`` (flat normal).
    """
    # Promote to 4-D if single-view
    if world_normals.ndim == 3:
        world_normals = world_normals[np.newaxis, ...]
    if uv_coords.ndim == 3:
        uv_coords = uv_coords[np.newaxis, ...]

    n_views = world_normals.shape[0]

    # Accumulation buffers
    accum_normal = np.zeros((resolution, resolution, 3), dtype=np.float64)
    accum_weight = np.zeros((resolution, resolution), dtype=np.float64)

    for v in range(n_views):
        uv = uv_coords[v].astype(np.float64)
        tx = np.clip(
            (uv[..., 0] * (resolution - 1)).astype(np.intp), 0, resolution - 1,
        )
        ty = np.clip(
            (uv[..., 1] * (resolution - 1)).astype(np.intp), 0, resolution - 1,
        )
        wn = world_normals[v].astype(np.float64)

        flat_idx = ty.ravel() * resolution + tx.ravel()
        wn_flat = wn.reshape(-1, 3)

        accum_flat = accum_normal.reshape(-1, 3)
        weight_flat = accum_weight.reshape(-1)
        for c in range(3):
            np.add.at(accum_flat[:, c], flat_idx, wn_flat[:, c])
        np.add.at(weight_flat, flat_idx, np.ones(flat_idx.shape[0]))

    # Normalise accumulated normals
    valid = accum_weight > 0.5
    for c in range(3):
        accum_normal[:, :, c][valid] /= accum_weight[valid]

    # Re-normalise to unit length
    norms = np.linalg.norm(accum_normal, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    accum_normal /= norms

    # Convert to tangent-space: approximate tangent frame from UV finite
    # differences.  For a flat surface the world-space Z maps directly to
    # tangent-space Z, so the conversion is simply the identity for planar
    # geometry.  For general surfaces we compute a per-texel tangent frame
    # from the UV gradient of the accumulated world-space position.
    #
    # Since we only have normals (not positions), we use the convention
    # that tangent-space = [right, up, normal] where right ≈ [1,0,0],
    # up ≈ [0,1,0] in object space and construct the tangent frame via
    # Gram–Schmidt orthonormalisation.
    tangent_map = _world_to_tangent_space(accum_normal)

    # Set uncovered texels to flat normal [0, 0, 1]
    tangent_map[~valid] = np.array([0.0, 0.0, 1.0])

    return tangent_map.astype(np.float32)


def _world_to_tangent_space(normals: np.ndarray) -> np.ndarray:
    """Convert world-space normals to tangent-space.

    Constructs a macro geometric reference frame from low-frequency surface
    normals (via spatial filtering), and projects the sampled normals
    into this local (T_geom, B_geom, N_geom) frame.

    Args:
        normals: ``(H, W, 3)`` float64 unit normals.

    Returns:
        ``(H, W, 3)`` float64 tangent-space normals in ``[-1, 1]``.
    """
    h, w, _ = normals.shape
    n = normals  # (H, W, 3)

    # Base macro geometric normal N_base via spatial smoothing
    pad = 4
    n_padded = np.pad(n, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    kernel_size = 2 * pad + 1
    n_base = np.zeros_like(n)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            n_base += n_padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
    n_base /= float(kernel_size * kernel_size)
    n_base_len = np.linalg.norm(n_base, axis=-1, keepdims=True)
    n_base = n_base / np.maximum(n_base_len, 1e-8)

    # Tangent and Bitangent via Frisvad's continuous tangent frame.
    # Avoids discontinuities from a global up vector; constructs T, B from N
    # alone by reflecting through the half-angle between N and (0, 0, 1).
    nz = n_base[..., 2:3]
    ny = n_base[..., 1:2]
    nx_arr = n_base[..., 0:1]

    # Frisvad: when nz ~ -1 the frame degenerates; handle via sign flip.
    neg_z = (nz < -0.9999)
    # For non-degenerate case: T = normalize((0 - nx*nz/(1+nz), 1 - ny*nz/(1+nz), -nz))
    # Simplified: direct cross-product construction
    tangent = np.empty_like(n_base)
    bitangent = np.empty_like(n_base)

    # Normal case: n.z > -0.9999
    denom = 1.0 + nz
    safe_denom = np.where(neg_z, 1.0, denom)
    a = -nx_arr * ny / safe_denom
    b_coeff = -nx_arr * nx_arr / safe_denom
    tangent[..., 0:1] = np.where(neg_z, 0.0, 1.0 + b_coeff)
    tangent[..., 1:2] = np.where(neg_z, -1.0, a)
    tangent[..., 2:3] = np.where(neg_z, 0.0, -nx_arr)

    bitangent[..., 0:1] = np.where(neg_z, -1.0, a)
    bitangent[..., 1:2] = np.where(neg_z, 0.0, 1.0 + (-ny * ny / safe_denom))
    bitangent[..., 2:3] = np.where(neg_z, 0.0, -ny)

    t_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent /= np.maximum(t_norm, 1e-8)

    b_norm = np.linalg.norm(bitangent, axis=-1, keepdims=True)
    bitangent /= np.maximum(b_norm, 1e-8)

    # Express the world normal in the geometric TBN frame
    ts_x = np.sum(n * tangent, axis=-1)
    ts_y = np.sum(n * bitangent, axis=-1)
    ts_z = np.sum(n * n_base, axis=-1)

    result = np.stack([ts_x, ts_y, ts_z], axis=-1)
    r_norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result /= np.maximum(r_norm, 1e-8)

    return result


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
    """Apply total-variation smoothing (Chambolle-like denoising).

    Uses the canonical :func:`compute_tv_gradient` for the TV gradient
    computation.
    """
    result = img.copy()
    for _ in range(iterations):
        grad = compute_tv_gradient(result)
        result = result - weight * grad.astype(np.float32)
    return np.asarray(result)
