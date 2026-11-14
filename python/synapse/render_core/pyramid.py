# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Unified Laplacian pyramid primitives for compositing and fusion.

This module provides the canonical implementations of Gaussian/Laplacian
pyramid construction, reconstruction, downsample, and upsample.  Both
``synapse.composite.blend`` and ``synapse.refine.fusion`` delegate to
these functions to eliminate code duplication.

All operations accept either ``float32`` or ``float64`` input and preserve
the caller's dtype throughout the pipeline.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-7


# ---------------------------------------------------------------------------
# 1-D Gaussian kernel
# ---------------------------------------------------------------------------


def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create a normalised 1-D Gaussian kernel.

    Args:
        sigma: Standard deviation.
        radius: Half-width of the kernel (kernel length = 2*radius + 1).

    Returns:
        Normalised 1-D kernel array with shape ``(2*radius + 1,)``.
    """
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(sigma, _EPS)) ** 2)
    return k / k.sum()


# ---------------------------------------------------------------------------
# Separable Gaussian blur
# ---------------------------------------------------------------------------


def separable_gaussian(
    img: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply separable Gaussian blur.

    Uses ``scipy.ndimage.gaussian_filter`` when available, otherwise
    falls back to a manual separable convolution.

    Args:
        img: ``(H, W)`` or ``(H, W, C)`` array.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred array with the same shape and dtype.
    """
    if sigma < 0.5:
        return img.copy()

    orig_dtype = img.dtype

    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            img.astype(np.float64), sigma=sigma, mode="reflect",
        ).astype(orig_dtype)
    except ImportError:
        pass

    # Pure NumPy separable convolution.
    radius = max(int(round(3.0 * sigma)), 1)
    kernel = gaussian_kernel_1d(sigma, radius)

    def _convolve_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        pad_widths: list[tuple[int, int]] = [(0, 0)] * arr.ndim
        pad_widths[axis] = (radius, radius)
        padded = np.pad(arr, pad_widths, mode="reflect")
        out_n = arr.shape[axis]
        result = np.zeros_like(arr)
        for i, w in enumerate(kernel):
            slices_src: list[slice] = [slice(None)] * arr.ndim
            slices_src[axis] = slice(i, i + out_n)
            result += w * padded[tuple(slices_src)]
        return result

    out = _convolve_axis(img, axis=0)
    out = _convolve_axis(out, axis=1)
    return out


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------


def downsample2x(img: np.ndarray) -> np.ndarray:
    """Downsample by 2× with Gaussian pre-filter."""
    blurred = separable_gaussian(img, sigma=1.0)
    return blurred[::2, ::2]


def upsample2x(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample by ~2× via bilinear interpolation to ``(target_h, target_w, ...)``.

    Args:
        img: ``(H, W)`` or ``(H, W, C)`` array.
        target_h: Output height.
        target_w: Output width.

    Returns:
        Upsampled array with shape ``(target_h, target_w, ...)`` and same dtype.
    """
    h, w = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros((target_h, target_w), dtype=img.dtype)
    else:
        out = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)

    row_idx = np.arange(target_h, dtype=np.float64) * (h - 1) / max(target_h - 1, 1)
    col_idx = np.arange(target_w, dtype=np.float64) * (w - 1) / max(target_w - 1, 1)

    r0 = np.floor(row_idx).astype(np.intp)
    r1 = np.minimum(r0 + 1, h - 1)
    rf = (row_idx - r0).astype(np.float64)

    c0 = np.floor(col_idx).astype(np.intp)
    c1 = np.minimum(c0 + 1, w - 1)
    cf = (col_idx - c0).astype(np.float64)

    if img.ndim == 2:
        for i in range(target_h):
            top = (1.0 - rf[i]) * img[r0[i], :] + rf[i] * img[r1[i], :]
            out[i, :] = (1.0 - cf) * top[c0] + cf * top[c1]
    else:
        for i in range(target_h):
            top = (1.0 - rf[i]) * img[r0[i], :, :] + rf[i] * img[r1[i], :, :]
            out[i, :, :] = (
                (1.0 - cf)[:, np.newaxis] * top[c0, :]
                + cf[:, np.newaxis] * top[c1, :]
            )
    return out


# ---------------------------------------------------------------------------
# Laplacian pyramid
# ---------------------------------------------------------------------------


def build_gaussian_pyramid(img: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Build a Gaussian pyramid with ``n_levels + 1`` entries."""
    pyramid: list[np.ndarray] = [img.copy()]
    cur = img.copy()
    for _ in range(n_levels):
        cur = downsample2x(cur)
        pyramid.append(cur)
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Build a Laplacian pyramid (last entry is the low-pass residual)."""
    gaussian = build_gaussian_pyramid(img, n_levels)
    lap: list[np.ndarray] = []
    for i in range(n_levels):
        up = upsample2x(gaussian[i + 1], gaussian[i].shape[0], gaussian[i].shape[1])
        lap.append(gaussian[i] - up)
    lap.append(gaussian[-1])
    return lap


def reconstruct_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from its Laplacian pyramid."""
    cur = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        up = upsample2x(cur, pyramid[i].shape[0], pyramid[i].shape[1])
        cur = pyramid[i] + up
    return cur
