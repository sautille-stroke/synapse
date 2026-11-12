"""Training losses for LoRA fine-tuning.

Provides:
- ``flow_matching_loss``: Rectified flow velocity field MSE
- ``latent_edge_alignment_loss``: Sobel-gradient edge consistency
- ``normal_reconstruction_loss``: Cosine angular distance on normals
- ``diffusion_dpo_loss``: Flow-matching DPO preference loss (with reference model)
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
    return float(np.mean(diff**2))


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


def normal_reconstruction_loss(
    n_pred: np.ndarray,
    n_gt: np.ndarray,
) -> float:
    """Cosine angular distance between predicted and ground-truth normals.

    Computes the per-pixel normal reconstruction loss::

        mean(1 - <n_pred, n_gt>)

    where both inputs are expected to be unit-length normal vectors
    from the FGPP ``WorldNormal.XYZ`` channels.

    Locked hyperparameter HP-04 from PLAN-ALGO-TRAINING-DATA-PIPELINE.md.

    Args:
        n_pred: Predicted normal field, shape ``[H, W, 3]`` or
            ``[3, H, W]``.  Will be L2-normalised internally.
        n_gt: Ground-truth normal field from FGPP, same shape.

    Returns:
        Scalar cosine angular distance loss in ``[0, 2]``.
    """
    n_p = np.asarray(n_pred, dtype=np.float64)
    n_g = np.asarray(n_gt, dtype=np.float64)

    # If channel-first [3, H, W], transpose to [H, W, 3]
    if n_p.ndim == 3 and n_p.shape[0] == 3:
        n_p = np.transpose(n_p, (1, 2, 0))
    if n_g.ndim == 3 and n_g.shape[0] == 3:
        n_g = np.transpose(n_g, (1, 2, 0))

    # L2-normalise to unit vectors (avoid div-by-zero)
    eps = 1e-8
    n_p = n_p / (np.linalg.norm(n_p, axis=-1, keepdims=True) + eps)
    n_g = n_g / (np.linalg.norm(n_g, axis=-1, keepdims=True) + eps)

    # Cosine similarity per pixel, clamp to [-1, 1]
    cos_sim = np.sum(n_p * n_g, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Loss = mean(1 - cos_sim)
    return float(np.mean(1.0 - cos_sim))


def diffusion_dpo_loss(
    v_theta_w: np.ndarray,
    v_theta_l: np.ndarray,
    v_ref_w: np.ndarray,
    v_ref_l: np.ndarray,
    beta: float = 500.0,
    target_w: np.ndarray | None = None,
    target_l: np.ndarray | None = None,
) -> float:
    """Flow-matching DPO preference loss.

    Computes the Diffusion-DPO objective (Wallace et al., CVPR 2024)::

        -log(sigmoid(-beta * ((err_theta_w - err_ref_w)
                              - (err_theta_l - err_ref_l))))

    where ``w`` = chosen (winner) and ``l`` = rejected (loser).
    When ``target_w`` and ``target_l`` are provided, ``err_theta_w = ||v_theta_w - target_w||^2``
    and ``err_ref_w = ||v_ref_w - target_w||^2`` (and analogously for ``l``),
    so that when ``v_theta == v_ref`` the loss is identically ``ln(2)``.
    When ``target_w`` / ``target_l`` are omitted, ``v_ref_w`` and ``v_ref_l``
    are treated as the ground-truth velocity targets ``v_target_w`` / ``v_target_l``
    (with ``err_ref_w = err_ref_l = 0``) for backward-compatible 4-argument calls.

    Args:
        v_theta_w: Policy model prediction on chosen sample.
        v_theta_l: Policy model prediction on rejected sample.
        v_ref_w: Reference model prediction on chosen sample (or target_w if target_w is None).
        v_ref_l: Reference model prediction on rejected sample (or target_l if target_l is None).
        beta: DPO temperature scaling factor.
        target_w: Ground-truth velocity ``x_1_w - x_0`` for chosen sample.
        target_l: Ground-truth velocity ``x_1_l - x_0`` for rejected sample.

    Returns:
        Scalar DPO loss value (non-negative).
    """
    v_tw = np.asarray(v_theta_w, dtype=np.float64)
    v_tl = np.asarray(v_theta_l, dtype=np.float64)
    v_rw = np.asarray(v_ref_w, dtype=np.float64)
    v_rl = np.asarray(v_ref_l, dtype=np.float64)

    if target_w is not None and target_l is not None:
        tw = np.asarray(target_w, dtype=np.float64)
        tl = np.asarray(target_l, dtype=np.float64)
        err_theta_w = float(np.mean((v_tw - tw) ** 2))
        err_theta_l = float(np.mean((v_tl - tl) ** 2))
        err_ref_w = float(np.mean((v_rw - tw) ** 2))
        err_ref_l = float(np.mean((v_rl - tl) ** 2))
        diff_w = err_theta_w - err_ref_w
        diff_l = err_theta_l - err_ref_l
        logit = -beta * (diff_w - diff_l)
    else:
        err_w = float(np.mean((v_rw - v_tw) ** 2))
        err_l = float(np.mean((v_rl - v_tl) ** 2))
        logit = beta * (err_l - err_w)

    # -log(sigmoid(logit)) = log(1 + exp(-logit))
    if logit < -20.0:
        return float(-logit)
    if logit > 20.0:
        return float(np.exp(-logit))
    return float(np.log(1.0 + np.exp(-logit)))


def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a 2D array."""
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    if w > 2 and h > 2:
        gx[1:-1, 1:-1] = (
            -img[:-2, :-2]
            + img[:-2, 2:]
            - 2 * img[1:-1, :-2]
            + 2 * img[1:-1, 2:]
            - img[2:, :-2]
            + img[2:, 2:]
        )
        gy[1:-1, 1:-1] = (
            -img[:-2, :-2]
            - 2 * img[:-2, 1:-1]
            - img[:-2, 2:]
            + img[2:, :-2]
            + 2 * img[2:, 1:-1]
            + img[2:, 2:]
        )

    return np.sqrt(gx * gx + gy * gy)
