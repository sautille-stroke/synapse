# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Unified mathematical morphology primitives for compositing and fusion.

This module provides canonical implementations of binary dilation,
binary erosion, and circular structuring element construction.
Both ``synapse.composite.blend``, ``synapse.composite.mask``, and
``synapse.refine`` delegate to these functions to eliminate code duplication.
"""

from __future__ import annotations

import numpy as np


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
