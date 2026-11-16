# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Unified mathematical morphology primitives for compositing and fusion.

This module provides canonical implementations of binary dilation,
binary erosion, and circular structuring element construction.
Both ``synapse.composite.blend``, ``synapse.composite.mask``, and
``synapse.refine`` delegate to these functions to eliminate code duplication.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "circular_structuring_element",
    "binary_dilation",
    "binary_erosion",
    "exact_euclidean_distance_transform",
]


def circular_structuring_element(radius: int) -> np.ndarray:
    """Create a circular (disk) structuring element of given radius.

    Args:
        radius: Half-width of structuring disk in pixels.

    Returns:
        Boolean array of shape ``(2*radius + 1, 2*radius + 1)``.
    """
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return np.asarray((xx * xx + yy * yy) <= radius * radius, dtype=bool)


def binary_dilation(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask with a structuring element.

    Uses ``scipy.ndimage.binary_dilation`` when available, otherwise
    falls back to a pure-NumPy pad-shift-or implementation.

    Args:
        mask: 2D boolean or integer mask array.
        se: 2D boolean structuring element.

    Returns:
        Dilated boolean array with the same shape as *mask*.
    """
    try:
        from scipy.ndimage import binary_dilation as scipy_binary_dilation

        return np.asarray(scipy_binary_dilation(mask, structure=se).astype(bool))
    except ImportError:
        pass

    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    H, W = mask.shape
    padded = np.pad(mask.astype(bool), ((ph, ph), (pw, pw)), mode="constant", constant_values=False)
    result = np.zeros((H, W), dtype=bool)
    se_refl = se[::-1, ::-1]
    ys, xs = np.nonzero(se_refl)
    for sy, sx in zip(ys, xs):
        result |= padded[sy : sy + H, sx : sx + W]
    return result


def binary_erosion(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Erode a boolean mask with a structuring element.

    Uses ``scipy.ndimage.binary_erosion`` when available, otherwise
    falls back to dilation of the complement:
    ``Erode(A, B) = ~Dilate(~A, B)``.

    Args:
        mask: 2D boolean or integer mask array.
        se: 2D boolean structuring element.

    Returns:
        Eroded boolean array with the same shape as *mask*.
    """
    try:
        from scipy.ndimage import binary_erosion as scipy_binary_erosion

        return np.asarray(scipy_binary_erosion(mask, structure=se).astype(bool))
    except ImportError:
        pass

    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    H, W = mask.shape
    padded = np.pad(mask.astype(bool), ((ph, ph), (pw, pw)), mode="constant", constant_values=True)
    result = np.ones((H, W), dtype=bool)
    ys, xs = np.nonzero(se)
    for sy, sx in zip(ys, xs):
        result &= padded[sy : sy + H, sx : sx + W]
    return result


def _edt_1d_parabolas(f: np.ndarray) -> np.ndarray:
    """1D Euclidean distance transform squared via Felzenszwalb-Huttenlocher.

    Args:
        f: 1D array of squared distance values or large sentinel for background.

    Returns:
        1D array of minimum squared distances.
    """
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    v = np.zeros(n, dtype=np.intp)
    z = np.zeros(n + 1, dtype=np.float64)
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    k = 0

    for q in range(1, n):
        while k >= 0:
            vk = v[k]
            s = ((f[q] + float(q * q)) - (f[vk] + float(vk * vk))) / (2.0 * float(q - vk))
            if s <= z[k]:
                k -= 1
            else:
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = np.inf
                break

    out = np.zeros(n, dtype=np.float64)
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        vk = v[k]
        diff = float(q - vk)
        out[q] = diff * diff + f[vk]
    return out


def exact_euclidean_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Exact 2D Euclidean distance transform from False pixels to nearest True pixel.

    For pixels where ``mask == True``, the returned distance is ``0.0``.
    For pixels where ``mask == False``, the returned distance is the exact
    Euclidean distance to the closest pixel where ``mask == True``.

    Uses ``scipy.ndimage.distance_transform_edt`` when available, otherwise
    falls back to the Felzenszwalb-Huttenlocher O(HW) separable lower-envelope
    algorithm.

    Args:
        mask: 2D boolean or integer mask array.

    Returns:
        ``[H, W]`` float32 array of exact Euclidean distances.
    """
    mask_b = np.asarray(mask, dtype=bool)
    if mask_b.ndim != 2:
        raise ValueError(f"exact_euclidean_distance_transform requires 2D array, got {mask_b.shape}")

    h, w = mask_b.shape
    if not np.any(mask_b):
        return np.full((h, w), float(h + w), dtype=np.float32)
    if np.all(mask_b):
        return np.zeros((h, w), dtype=np.float32)

    try:
        from scipy.ndimage import distance_transform_edt

        return np.asarray(distance_transform_edt(~mask_b).astype(np.float32))
    except ImportError:
        pass

    # Pure-NumPy Felzenszwalb-Huttenlocher 2D separable lower-envelope algorithm
    f = np.where(mask_b, 0.0, 1e12).astype(np.float64)

    # 1D transform along columns
    col_dists = np.zeros((h, w), dtype=np.float64)
    for c in range(w):
        col_dists[:, c] = _edt_1d_parabolas(f[:, c])

    # 1D transform along rows
    d2 = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        d2[r, :] = _edt_1d_parabolas(col_dists[r, :])

    return np.sqrt(np.clip(d2, 0.0, None)).astype(np.float32)

