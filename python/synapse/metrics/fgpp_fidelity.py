"""FGPP fidelity metrics — depth, normal, flow, and silhouette errors.

All functions accept NumPy arrays and return scalar floats.  Optional
``valid_mask`` parameters allow masking out sky, background, or invalid
pixels from the error computation.
"""

from __future__ import annotations

import numpy as np


def depth_rmse_m(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """RMSE of linear depth in meters.

    Args:
        pred_depth: ``[H, W]`` predicted linear depth (metres).
        gt_depth: ``[H, W]`` ground-truth linear depth (metres).
        valid_mask: Optional ``[H, W]`` bool mask (``True`` = valid).

    Returns:
        RMSE in metres (scalar float).
    """
    p = np.asarray(pred_depth, dtype=np.float64)
    g = np.asarray(gt_depth, dtype=np.float64)

    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        p, g = p[m], g[m]

    if p.size == 0:
        return 0.0

    return float(np.sqrt(np.mean((p - g) ** 2)))


def scale_invariant_depth_error(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Scale-invariant depth error (Eigen et al.).

    Computed in log-space::

        d_i = log(pred_i) - log(gt_i)
        SIDE = sqrt( mean(d^2) - mean(d)^2 )

    Args:
        pred: ``[H, W]`` predicted depth (positive values).
        gt: ``[H, W]`` ground-truth depth (positive values).
        valid_mask: Optional ``[H, W]`` bool mask (``True`` = valid).

    Returns:
        Scale-invariant depth error (scalar float).
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)

    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        p, g = p[m], g[m]

    # Clamp to tiny positive to avoid log(0)
    eps = 1e-8
    p = np.clip(p, eps, None)
    g = np.clip(g, eps, None)

    if p.size == 0:
        return 0.0

    d = np.log(p) - np.log(g)
    n = d.size
    side = np.sqrt(np.mean(d ** 2) - (np.sum(d) ** 2) / (n * n))
    return float(side)


def normal_angular_error_deg(
    pred_normals: np.ndarray,
    gt_normals: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Mean angular error between predicted and ground-truth normals.

    Args:
        pred_normals: ``[H, W, 3]`` predicted unit normals.
        gt_normals: ``[H, W, 3]`` ground-truth unit normals.
        valid_mask: Optional ``[H, W]`` bool mask (``True`` = valid).

    Returns:
        Mean angular error in degrees (scalar float).
    """
    p = np.asarray(pred_normals, dtype=np.float64)
    g = np.asarray(gt_normals, dtype=np.float64)

    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        p, g = p[m], g[m]
    else:
        p = p.reshape(-1, 3)
        g = g.reshape(-1, 3)

    if p.shape[0] == 0:
        return 0.0

    dot = np.sum(p * g, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    return float(np.mean(angles))


def flow_endpoint_error_px(
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Mean endpoint error between predicted and ground-truth flow.

    Args:
        pred_flow: ``[H, W, 2]`` predicted optical flow (pixels).
        gt_flow: ``[H, W, 2]`` ground-truth optical flow (pixels).
        valid_mask: Optional ``[H, W]`` bool mask (``True`` = valid).

    Returns:
        Mean EPE in pixels (scalar float).
    """
    p = np.asarray(pred_flow, dtype=np.float64)
    g = np.asarray(gt_flow, dtype=np.float64)

    diff = p - g
    epe = np.sqrt(np.sum(diff ** 2, axis=-1))  # [H, W]

    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        epe = epe[m]

    if epe.size == 0:
        return 0.0

    return float(np.mean(epe))


def silhouette_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Intersection over Union of binary silhouette masks.

    Args:
        pred_mask: ``[H, W]`` predicted binary mask.
        gt_mask: ``[H, W]`` ground-truth binary mask.

    Returns:
        IoU in [0, 1] (scalar float).
    """
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)

    intersection = np.sum(p & g)
    union = np.sum(p | g)

    if union == 0:
        return 1.0  # both empty → perfect agreement

    return float(intersection / union)
