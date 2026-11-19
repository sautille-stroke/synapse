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
    return np.asarray(k / k.sum())


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
    if sigma < 0.5 or min(img.shape[:2]) <= 1:
        return np.asarray(img.copy())

    orig_dtype = img.dtype

    try:
        from scipy.ndimage import gaussian_filter

        mode = "nearest" if min(img.shape[:2]) <= 1 else "reflect"
        return np.asarray(
            gaussian_filter(
                img.astype(np.float64),
                sigma=sigma,
                mode=mode,
            ).astype(orig_dtype)
        )
    except ImportError:
        pass

    # Pure NumPy separable convolution.
    radius = max(int(round(3.0 * sigma)), 1)
    kernel = gaussian_kernel_1d(sigma, radius)

    def _convolve_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        dim_len = arr.shape[axis]
        if dim_len <= 1:
            return arr.copy()
        pad_widths: list[tuple[int, int]] = [(0, 0)] * arr.ndim
        pad_widths[axis] = (radius, radius)
        if dim_len > radius:
            padded = np.pad(arr.astype(np.float64), pad_widths, mode="reflect")
        else:
            padded = np.pad(arr.astype(np.float64), pad_widths, mode="edge")
        out_n = arr.shape[axis]
        result = np.zeros(arr.shape, dtype=np.float64)
        for i, w in enumerate(kernel):
            slices_src: list[slice] = [slice(None)] * arr.ndim
            slices_src[axis] = slice(i, i + out_n)
            result += w * padded[tuple(slices_src)]
        return result.astype(arr.dtype)

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


def upsample2x(
    img: np.ndarray,
    target_h: int | tuple[int, ...],
    target_w: int | None = None,
) -> np.ndarray:
    """Upsample by ~2× via bilinear interpolation to ``(target_h, target_w, ...)``.

    Vectorized implementation without Python scanline loops.

    Args:
        img: ``(H, W)`` or ``(H, W, C)`` array.
        target_h: Output height or tuple shape ``(th, tw, ...)``.
        target_w: Output width (optional if *target_h* is a shape tuple).

    Returns:
        Upsampled array with target spatial dimensions and same dtype.
    """
    if isinstance(target_h, (tuple, list)):
        th, tw = int(target_h[0]), int(target_h[1])
    else:
        th = int(target_h)
        tw = int(target_w) if target_w is not None else th

    h, w = img.shape[:2]
    orig_dtype = img.dtype
    img_f = img.astype(np.float64) if np.issubdtype(orig_dtype, np.integer) else img

    row_idx = np.clip(
        (np.arange(th, dtype=np.float64) + 0.5) * (h / max(th, 1)) - 0.5, 0.0, float(h - 1)
    )
    col_idx = np.clip(
        (np.arange(tw, dtype=np.float64) + 0.5) * (w / max(tw, 1)) - 0.5, 0.0, float(w - 1)
    )

    r0 = np.floor(row_idx).astype(np.intp)
    r1 = np.minimum(r0 + 1, h - 1)
    rf = (row_idx - r0).astype(np.float64)

    c0 = np.floor(col_idx).astype(np.intp)
    c1 = np.minimum(c0 + 1, w - 1)
    cf = (col_idx - c0).astype(np.float64)

    if img.ndim == 2:
        rf_2d = rf[:, np.newaxis]
        cf_2d = cf[np.newaxis, :]
        top = (1.0 - cf_2d) * img_f[r0[:, None], c0[None, :]] + cf_2d * img_f[r0[:, None], c1[None, :]]
        bottom = (1.0 - cf_2d) * img_f[r1[:, None], c0[None, :]] + cf_2d * img_f[r1[:, None], c1[None, :]]
        out = (1.0 - rf_2d) * top + rf_2d * bottom
    else:
        rf_3d = rf[:, np.newaxis, np.newaxis]
        cf_3d = cf[np.newaxis, :, np.newaxis]
        top = (1.0 - cf_3d) * img_f[r0[:, None], c0[None, :], :] + cf_3d * img_f[r0[:, None], c1[None, :], :]
        bottom = (1.0 - cf_3d) * img_f[r1[:, None], c0[None, :], :] + cf_3d * img_f[r1[:, None], c1[None, :], :]
        out = (1.0 - rf_3d) * top + rf_3d * bottom

    if np.issubdtype(orig_dtype, np.integer):
        return np.asarray(np.clip(np.round(out), 0, 255).astype(orig_dtype))
    return np.asarray(out.astype(orig_dtype))


# ---------------------------------------------------------------------------
# Laplacian pyramid
# ---------------------------------------------------------------------------


def build_gaussian_pyramid(img: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Build a Gaussian pyramid with ``n_levels + 1`` entries."""
    work_img = img.astype(np.float32) if np.issubdtype(img.dtype, np.integer) else img
    pyramid: list[np.ndarray] = [work_img.copy()]
    cur = work_img.copy()
    for _ in range(n_levels):
        cur = downsample2x(cur)
        pyramid.append(cur)
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Build a Laplacian pyramid (last entry is the low-pass residual)."""
    work_img = img.astype(np.float32) if np.issubdtype(img.dtype, np.integer) else img
    gaussian = build_gaussian_pyramid(work_img, n_levels)
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
    return np.asarray(cur)
