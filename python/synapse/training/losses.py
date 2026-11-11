"""Training losses for LoRA fine-tuning.

Provides:
- ``flow_matching_loss``: Rectified flow velocity field MSE
- ``latent_edge_alignment_loss``: Sobel-gradient edge consistency
- ``diffusion_dpo_loss``: Flow-matching DPO preference loss
"""

from __future__ import annotations

import numpy as np


def flow_matching_loss(
    v_pred: np.ndarray,
    x_1: np.ndarray,
    x_0: np.ndarray,
) -> float:
    """Rectified flow velocity field mean squared error.

    Computes::

        E[||v_pred - (x_1 - x_0)||_2^2]

    Args:
        v_pred: Predicted velocity field, any shape.
        x_1: Target sample (noise-free data), same shape.
        x_0: Source sample (pure noise), same shape.

    Returns:
        Scalar MSE loss value.
    """
    v_pred = np.asarray(v_pred, dtype=np.float64)
    x_1 = np.asarray(x_1, dtype=np.float64)
    x_0 = np.asarray(x_0, dtype=np.float64)

    target = x_1 - x_0
    diff = v_pred - target
    return float(np.mean(diff ** 2))


def latent_edge_alignment_loss(
    z_pred: np.ndarray,
    depth_latent: np.ndarray,
) -> float:
    """Sobel-gradient edge alignment between prediction and depth.

    Computes 2D Sobel gradient magnitude on both inputs and
    returns the mean absolute difference.

    Args:
        z_pred: Predicted denoised latent ``[H, W]`` or
            ``[C, H, W]`` (averaged across channels if 3D).
        depth_latent: Min-pool depth latent, same spatial shape.

    Returns:
        Scalar edge alignment loss value.
    """
    z = np.asarray(z_pred, dtype=np.float64)
    d = np.asarray(depth_latent, dtype=np.float64)

    # If 3D, average across channel dimension
    if z.ndim == 3:
        z = z.mean(axis=0)
    if d.ndim == 3:
        d = d.mean(axis=0)

    grad_z = _sobel_magnitude(z)
    grad_d = _sobel_magnitude(d)

    return float(np.mean(np.abs(grad_z - grad_d)))


def diffusion_dpo_loss(
    v_theta_w: np.ndarray,
    v_theta_l: np.ndarray,
    v_ref_w: np.ndarray,
    v_ref_l: np.ndarray,
    beta: float = 500.0,
) -> float:
    """Flow-matching DPO preference loss.

    Computes the diffusion-DPO objective for flow matching::

        -log(sigmoid(beta * (||v_ref_w - v_theta_w||^2
                            - ||v_ref_l - v_theta_l||^2)))

    where ``w`` = chosen (winner) and ``l`` = rejected (loser).

    Args:
        v_theta_w: Model prediction on chosen sample.
        v_theta_l: Model prediction on rejected sample.
        v_ref_w: Reference model prediction on chosen sample.
        v_ref_l: Reference model prediction on rejected sample.
        beta: DPO temperature scaling factor.

    Returns:
        Scalar DPO loss value.
    """
    v_tw = np.asarray(v_theta_w, dtype=np.float64)
    v_tl = np.asarray(v_theta_l, dtype=np.float64)
    v_rw = np.asarray(v_ref_w, dtype=np.float64)
    v_rl = np.asarray(v_ref_l, dtype=np.float64)

    # Per-element squared error, then mean
    err_w = float(np.mean((v_rw - v_tw) ** 2))
    err_l = float(np.mean((v_rl - v_tl) ** 2))

    # DPO: chosen should have lower error than rejected
    logit = beta * (err_w - err_l)

    # -log(sigmoid(logit)) = log(1 + exp(-logit))
    # Use numerically stable version
    if logit > 20.0:
        return float(-logit)
    elif logit < -20.0:
        return 0.0
    else:
        return float(np.log(1.0 + np.exp(-logit)))


def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a 2D array."""
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    if w > 2 and h > 2:
        gx[1:-1, 1:-1] = (
            -img[:-2, :-2] + img[:-2, 2:]
            - 2 * img[1:-1, :-2] + 2 * img[1:-1, 2:]
            - img[2:, :-2] + img[2:, 2:]
        )
        gy[1:-1, 1:-1] = (
            -img[:-2, :-2] - 2 * img[:-2, 1:-1] - img[:-2, 2:]
            + img[2:, :-2] + 2 * img[2:, 1:-1] + img[2:, 2:]
        )

    return np.sqrt(gx * gx + gy * gy)
