"""Motion-adaptive surgical compositing masks.

Implements the surgical mask equation from the compositing pipeline:

    M_surgical(t) = Dilate_v(t)( M_3D_new(t) ∪ M_SAM2_old(t) )

The mask is dilated isotropically and then directionally along the
dominant motion vector, producing a soft alpha with cosine feathering
for seamless blending.

All operations are vectorised NumPy.  ``scipy.ndimage`` is used for
morphological dilation when available, with a pure-NumPy fallback.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _circular_structuring_element(radius: int) -> np.ndarray:
    """Create a circular (disk) structuring element of given radius.

    Returns a bool array of shape ``(2*radius+1, 2*radius+1)``.
    """
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _directional_structuring_element(
    angle_rad: float,
    length: int,
) -> np.ndarray:
    """Create a forward-biased line structuring element.

    The line extends from the centre primarily in the *angle_rad*
    direction, giving asymmetric (motion-forward) dilation.  A small
    backward extent (¼ of total length) is kept for seamless overlap.

    Returns a bool array large enough to contain the line, centred.
    """
    if length < 1:
        return np.ones((1, 1), dtype=bool)

    # Forward-biased: extend mostly in the positive direction.
    back = max(length // 4, 1)
    forward = length - back
    half = max(forward, back)
    size = 2 * half + 1
    se = np.zeros((size, size), dtype=bool)
    cx = cy = half

    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    for i in range(-back, forward + 1):
        x = int(round(cx + i * dx))
        y = int(round(cy + i * dy))
        if 0 <= x < size and 0 <= y < size:
            se[y, x] = True

    return se


def _binary_dilation(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Dilate a bool mask with a structuring element.

    Uses ``scipy.ndimage.binary_dilation`` when available, otherwise
    falls back to a pure-NumPy pad-shift-or implementation.
    """
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(
            mask, structure=se
        ).astype(bool)
    except ImportError:
        pass

    # Pure NumPy fallback: pad, then logical-OR shifted copies.
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    padded = np.pad(mask, ((ph, ph), (pw, pw)), constant_values=False)
    result = np.zeros_like(padded)
    ys, xs = np.nonzero(se)
    for dy, dx in zip(ys, xs):
        oy = dy - ph
        ox = dx - pw
        shifted = np.roll(np.roll(padded, oy, axis=0), ox, axis=1)
        result |= shifted
    # Crop back to original size.
    return result[ph:ph + mask.shape[0], pw:pw + mask.shape[1]]


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from each pixel to the nearest True pixel.

    Uses ``scipy.ndimage.distance_transform_edt`` when available.
    Falls back to a chamfer-style 2-pass approximation.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask).astype(np.float32)
    except ImportError:
        pass

    # Chamfer 2-pass approximation.
    h, w = mask.shape
    dist = np.full((h, w), dtype=np.float32, fill_value=float(h + w))
    dist[mask] = 0.0
    for y in range(h):
        for x in range(w):
            if y > 0:
                dist[y, x] = min(dist[y, x], dist[y - 1, x] + 1.0)
            if x > 0:
                dist[y, x] = min(dist[y, x], dist[y, x - 1] + 1.0)
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if y < h - 1:
                dist[y, x] = min(dist[y, x], dist[y + 1, x] + 1.0)
            if x < w - 1:
                dist[y, x] = min(dist[y, x], dist[y, x + 1] + 1.0)
    return dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_surgical_mask(
    mask_3d_new: np.ndarray,
    mask_sam2_old: np.ndarray | None = None,
    motion_vectors: np.ndarray | None = None,
    r_base: int = 8,
    motion_scale: float = 1.5,
    max_motion_kernel: int = 32,
    feather_width: float = 6.0,
) -> np.ndarray:
    """Generate a motion-adaptive surgical compositing mask.

    Implements::

        M_surgical(t) = Dilate_v(t)( M_3D_new(t) ∪ M_SAM2_old(t) )

    Args:
        mask_3d_new: ``[H, W]`` bool array — new 3D-rendered entity
            mask.
        mask_sam2_old: ``[H, W]`` bool array or ``None`` — SAM2-
            predicted mask of the old entity to be replaced.
        motion_vectors: ``[H, W, 2]`` float16/32 array or ``None``
            — screen-space motion vectors ``(dx, dy)`` per pixel.
        r_base: Isotropic dilation radius in pixels (default 8).
        motion_scale: Motion-vector-to-dilation scaling factor.
        max_motion_kernel: Maximum directional kernel length in
            pixels.
        feather_width: Cosine-falloff feather width in pixels.

    Returns:
        ``[H, W]`` float32 alpha mask in ``[0.0, 1.0]``.
    """
    # 1. Union of masks.
    combined = mask_3d_new.astype(bool).copy()
    if mask_sam2_old is not None:
        combined |= mask_sam2_old.astype(bool)

    # 2. Isotropic dilation with circular structuring element.
    if r_base > 0:
        se_circle = _circular_structuring_element(r_base)
        combined = _binary_dilation(combined, se_circle)

    # 3. Directional (motion-adaptive) dilation.
    if motion_vectors is not None and np.any(combined):
        mv = motion_vectors.astype(np.float32)
        # Replace NaN with 0 for safe mean computation.
        mv_safe = np.where(np.isfinite(mv), mv, 0.0)

        # Mean motion within the combined mask region.
        count = float(combined.sum())
        if count > 0:
            mean_dx = float((mv_safe[..., 0] * combined).sum() / count)
            mean_dy = float((mv_safe[..., 1] * combined).sum() / count)
            mag = math.sqrt(mean_dx * mean_dx + mean_dy * mean_dy)

            kernel_len = int(min(
                max_motion_kernel,
                round(mag * motion_scale),
            ))

            if kernel_len >= 2:
                angle = math.atan2(mean_dy, mean_dx)
                se_line = _directional_structuring_element(
                    angle, kernel_len
                )
                combined = _binary_dilation(combined, se_line)

    # 4. Soft alpha with cosine feathering.
    if feather_width <= 0 or not np.any(combined):
        return combined.astype(np.float32)

    dist = _distance_transform(combined)
    alpha = np.ones_like(dist)

    # Feather zone: outside the hard mask but within feather_width.
    feather_mask = (~combined) & (dist <= feather_width)
    alpha[feather_mask] = 0.5 * (
        1.0 + np.cos(np.pi * dist[feather_mask] / feather_width)
    )
    alpha[~combined & ~feather_mask] = 0.0
    alpha[combined] = 1.0

    return alpha.astype(np.float32)


def apply_surgical_mask(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Alpha-composite *foreground* over *background* using *alpha*.

    Args:
        background: ``[H, W, C]`` image array.
        foreground: ``[H, W, C]`` image array (same shape & dtype as
            *background*).
        alpha: ``[H, W]`` float32 blending mask in ``[0, 1]``.

    Returns:
        Composited ``[H, W, C]`` image with the same dtype as
        *background*.
    """
    src_dtype = background.dtype
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)
    a = alpha[..., np.newaxis].astype(np.float32)

    out = fg * a + bg * (1.0 - a)

    if np.issubdtype(src_dtype, np.integer):
        out = np.clip(out, 0, np.iinfo(src_dtype).max)
    return out.astype(src_dtype)
