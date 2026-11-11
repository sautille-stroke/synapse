"""Style-consistency metrics for compositing and lighting alignment.

Provides border-ring colour difference, light direction error, tone
curve RMSE, and histogram intersection — all pure NumPy.
"""

from __future__ import annotations

import numpy as np


def border_ring_ciede2000(
    composited: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    ring_width: int = 16,
) -> float:
    """Mean CIEDE2000 in the border ring around a composited region.

    Dilates the binary *mask* by *ring_width* pixels, subtracts the
    original mask, and computes the mean CIEDE2000 between
    *composited* and *background* in the resulting ring.

    Args:
        composited: ``[H, W, 3]`` sRGB image in [0, 1].
        background: ``[H, W, 3]`` sRGB image in [0, 1].
        mask: ``[H, W]`` bool mask of the composited foreground.
        ring_width: Dilation radius in pixels (default 16).

    Returns:
        Mean CIEDE2000 in the border ring (scalar float).
    """
    from synapse.style.aces_color import (
        ciede2000_delta_e,
        rgb_to_cielab,
        srgb_to_linear,
    )

    m = np.asarray(mask, dtype=bool)
    h, w = m.shape

    # Simple morphological dilation via shift-and-OR
    dilated = m.copy()
    for dy in range(-ring_width, ring_width + 1):
        for dx in range(-ring_width, ring_width + 1):
            if dy * dy + dx * dx > ring_width * ring_width:
                continue
            shifted = np.zeros_like(m)
            sy = slice(max(0, dy), min(h, h + dy))
            sx = slice(max(0, dx), min(w, w + dx))
            ty = slice(max(0, -dy), min(h, h - dy))
            tx = slice(max(0, -dx), min(w, w - dx))
            shifted[sy, sx] = m[ty, tx]
            dilated |= shifted

    ring = dilated & ~m
    if not ring.any():
        return 0.0

    comp = np.asarray(composited, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)

    lab_comp = rgb_to_cielab(srgb_to_linear(comp))
    lab_bg = rgb_to_cielab(srgb_to_linear(bg))

    de = ciede2000_delta_e(lab_comp, lab_bg)  # [H, W]
    return float(np.mean(de[ring]))


def light_dir_error_deg(
    pred_dir: np.ndarray,
    gt_dir: np.ndarray,
) -> float:
    """Angular error between two 3D light direction vectors.

    Args:
        pred_dir: Length-3 predicted light direction (need not be unit).
        gt_dir: Length-3 ground-truth light direction (need not be unit).

    Returns:
        Angular error in degrees (scalar float).
    """
    p = np.asarray(pred_dir, dtype=np.float64).ravel()
    g = np.asarray(gt_dir, dtype=np.float64).ravel()

    p_norm = np.linalg.norm(p)
    g_norm = np.linalg.norm(g)
    if p_norm < 1e-12 or g_norm < 1e-12:
        return 0.0

    cos_angle = np.dot(p, g) / (p_norm * g_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def tone_curve_rmse(
    pred_curve: np.ndarray,
    gt_curve: np.ndarray,
) -> float:
    """RMSE between two 1-D tone curves (256-entry LUTs).

    Args:
        pred_curve: ``[256]`` predicted tone-mapping LUT.
        gt_curve: ``[256]`` ground-truth tone-mapping LUT.

    Returns:
        RMSE (scalar float).
    """
    p = np.asarray(pred_curve, dtype=np.float64)
    g = np.asarray(gt_curve, dtype=np.float64)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def histogram_intersection(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
) -> float:
    """Normalised histogram intersection.

    ``sum(min(a, b)) / sum(a)``

    Args:
        hist_a: 1-D histogram (reference).
        hist_b: 1-D histogram (candidate).

    Returns:
        Intersection in [0, 1] (scalar float).
    """
    a = np.asarray(hist_a, dtype=np.float64)
    b = np.asarray(hist_b, dtype=np.float64)

    sum_a = a.sum()
    if sum_a < 1e-12:
        return 0.0

    return float(np.minimum(a, b).sum() / sum_a)
