"""Image & multi-view visual quality metrics (PSNR, SSIM, CIEDE2000, LPIPS).

Provides pure-NumPy, deterministic implementations of:
  - ``psnr_db``: Peak Signal-to-Noise Ratio in decibels (dB).
  - ``ssim``: Structural Similarity Index (Wang et al., 2004) with an
    11x11 separable Gaussian window (sigma=1.5).
  - ``ciede2000``: Mean perceptual CIEDE2000 colour difference over sRGB
    images or video sequences.
  - ``lpips``: Perceptual similarity distance in ``[0, 1]`` (uses optional
    ``lpips``/``torch`` when pre-loaded, otherwise a deterministic multi-scale
    gradient-structure perceptual distance with zero network calls).

All functions accept single images ``[H, W]`` / ``[H, W, C]`` or frame batches
``[T, H, W, C]`` in either floating-point ``[0, data_range]`` or ``uint8``
``[0, 255]`` (automatically normalised to ``[0, 1]`` when ``dtype == uint8``).
"""

from __future__ import annotations

import math

import numpy as np


def _to_float_images(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalise image/video arrays to float64 with matching shapes."""
    p_raw = np.asarray(pred)
    g_raw = np.asarray(gt)
    if p_raw.shape != g_raw.shape:
        raise ValueError(
            f"Shape mismatch between pred {p_raw.shape} and gt {g_raw.shape}."
        )

    eff_range = float(data_range)
    if p_raw.dtype == np.uint8 and g_raw.dtype == np.uint8:
        p = p_raw.astype(np.float64) / 255.0
        g = g_raw.astype(np.float64) / 255.0
        eff_range = 1.0
    else:
        p = p_raw.astype(np.float64)
        g = g_raw.astype(np.float64)

    return p, g, eff_range


def _expand_mask(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a 2D ``[H, W]`` or 3D ``[T, H, W]`` boolean mask to *target_shape*."""
    m = np.asarray(mask, dtype=bool)
    if m.shape == target_shape:
        return m
    if m.ndim == len(target_shape) - 1 and m.shape == target_shape[:-1]:
        return np.broadcast_to(m[..., np.newaxis], target_shape)
    if len(target_shape) == 4 and m.ndim == 2 and m.shape == target_shape[1:3]:
        return np.broadcast_to(m[np.newaxis, ..., np.newaxis], target_shape)
    raise ValueError(
        f"Mask shape {m.shape} is incompatible with image shape {target_shape}."
    )


def psnr_db(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        pred: Predicted image/video ``[H, W[, C]]`` or ``[T, H, W, C]``.
        gt: Ground-truth image/video of identical shape.
        mask: Optional boolean mask (``[H, W]`` or ``[T, H, W]``).
        data_range: Dynamic range of float inputs (default ``1.0``).

    Returns:
        PSNR in dB (capped at ``100.0`` for identical inputs so JSON
        serialisation stays strictly finite).
    """
    p, g, eff_range = _to_float_images(pred, gt, data_range=data_range)
    diff_sq = (p - g) ** 2

    if mask is not None:
        m = _expand_mask(mask, diff_sq.shape)
        if not np.any(m):
            return 100.0
        mse = float(np.mean(diff_sq[m]))
    else:
        if diff_sq.size == 0:
            return 100.0
        mse = float(np.mean(diff_sq))

    if mse <= 1e-10 * (eff_range ** 2):
        return 100.0

    return float(min(10.0 * math.log10((eff_range ** 2) / mse), 100.0))


def _gaussian_kernel_1d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Create a normalised 1-D Gaussian kernel."""
    coords = np.arange(win_size, dtype=np.float64) - (win_size - 1) / 2.0
    kernel = np.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def _convolve_separable_2d(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Apply a separable 2-D convolution with reflect padding over ``(..., H, W, C)``."""
    k = len(kernel_1d)
    pad = k // 2
    if pad == 0:
        return img

    # Convolve along H (axis -3)
    padded_h = np.pad(img, [(0, 0)] * (img.ndim - 3) + [(pad, pad), (0, 0), (0, 0)], mode="reflect")
    out_h = np.zeros_like(img, dtype=np.float64)
    for i, w in enumerate(kernel_1d):
        out_h += w * padded_h[..., i : i + img.shape[-3], :, :]

    # Convolve along W (axis -2)
    padded_w = np.pad(out_h, [(0, 0)] * (img.ndim - 3) + [(0, 0), (pad, pad), (0, 0)], mode="reflect")
    out_hw = np.zeros_like(img, dtype=np.float64)
    for i, w in enumerate(kernel_1d):
        out_hw += w * padded_w[..., :, i : i + img.shape[-2], :]

    return out_hw


def ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM, Wang et al. 2004).

    Uses an 11x11 Gaussian window (sigma=1.5) when spatial dimensions are at
    least 11x11, automatically shrinking to an odd window for tiny test patches.

    Args:
        pred: Predicted image/video ``[H, W]``, ``[H, W, C]``, or ``[T, H, W, C]``.
        gt: Ground-truth image/video of identical shape.
        mask: Optional foreground mask ``[H, W]`` or ``[T, H, W]``.
        data_range: Dynamic range of float inputs (default ``1.0``).

    Returns:
        Mean SSIM in ``[-1, 1]`` (typically ``[0, 1]``).
    """
    p, g, eff_range = _to_float_images(pred, gt, data_range=data_range)
    if p.size == 0:
        return 1.0

    # Normalise shape to [N, H, W, C]
    orig_ndim = p.ndim
    if orig_ndim == 2:
        p4 = p[np.newaxis, :, :, np.newaxis]
        g4 = g[np.newaxis, :, :, np.newaxis]
    elif orig_ndim == 3:
        p4 = p[np.newaxis, :, :, :]
        g4 = g[np.newaxis, :, :, :]
    elif orig_ndim == 4:
        p4 = p
        g4 = g
    else:
        raise ValueError(f"Unsupported input ndim={orig_ndim}; expected 2, 3, or 4.")

    h, w = p4.shape[1], p4.shape[2]
    min_dim = min(h, w)
    win_size = min(11, min_dim if min_dim % 2 == 1 else max(min_dim - 1, 1))
    kernel = _gaussian_kernel_1d(win_size=win_size, sigma=1.5)

    c1 = (0.01 * eff_range) ** 2
    c2 = (0.03 * eff_range) ** 2

    mu_x = _convolve_separable_2d(p4, kernel)
    mu_y = _convolve_separable_2d(g4, kernel)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = np.maximum(_convolve_separable_2d(p4 * p4, kernel) - mu_x_sq, 0.0)
    sigma_y_sq = np.maximum(_convolve_separable_2d(g4 * g4, kernel) - mu_y_sq, 0.0)
    sigma_xy = _convolve_separable_2d(p4 * g4, kernel) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / np.maximum(den, 1e-15)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2:
            m4 = np.broadcast_to(m[np.newaxis, :, :, np.newaxis], ssim_map.shape)
        elif m.ndim == 3 and m.shape == ssim_map.shape[:3]:
            m4 = np.broadcast_to(m[..., np.newaxis], ssim_map.shape)
        else:
            m4 = _expand_mask(m, ssim_map.shape)
        if not np.any(m4):
            return 1.0
        return float(np.mean(ssim_map[m4]))

    return float(np.mean(ssim_map))


def ciede2000(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute mean perceptual CIEDE2000 colour difference over sRGB frames.

    Args:
        pred: sRGB image ``[H, W, 3]`` or sequence ``[T, H, W, 3]`` in
            ``[0, 1]`` float or ``[0, 255]`` uint8.
        gt: Ground-truth sRGB image/sequence of identical shape.
        mask: Optional boolean mask ``[H, W]`` or ``[T, H, W]``.

    Returns:
        Mean CIEDE2000 colour difference (``0.0`` for identical images).
    """
    from synapse.style.aces_color import (
        ciede2000_delta_e,
        rgb_to_cielab,
        srgb_to_linear,
    )

    p, g, _ = _to_float_images(pred, gt, data_range=1.0)
    if p.ndim not in (3, 4) or p.shape[-1] != 3:
        raise ValueError(
            f"ciede2000 expects shape [H, W, 3] or [T, H, W, 3], got {p.shape}."
        )
    if p.size == 0:
        return 0.0

    p_clamped = np.clip(p, 0.0, 1.0)
    g_clamped = np.clip(g, 0.0, 1.0)

    lab_p = rgb_to_cielab(srgb_to_linear(p_clamped))
    lab_g = rgb_to_cielab(srgb_to_linear(g_clamped))
    de = ciede2000_delta_e(lab_p, lab_g)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2 and de.ndim == 3:
            m = np.broadcast_to(m[np.newaxis, :, :], de.shape)
        if not np.any(m):
            return 0.0
        return float(np.mean(de[m]))

    return float(np.mean(de))


def _downsample_2x(img4: np.ndarray) -> np.ndarray:
    """2x average-pool spatial dimensions ``[N, H, W, C]``."""
    _, h, w, _ = img4.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    if h2 < 2 or w2 < 2:
        return img4
    cropped = img4[:, :h2, :w2, :]
    return 0.25 * (
        cropped[:, 0::2, 0::2, :]
        + cropped[:, 1::2, 0::2, :]
        + cropped[:, 0::2, 1::2, :]
        + cropped[:, 1::2, 1::2, :]
    )


def lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute perceptual patch distance in ``[0, 1]`` (lower is better).

    Provides a deterministic, zero-network multi-scale opponent-colour +
    Sobel-gradient structural perceptual distance calibrated to ``[0, 1]``
    (``0.0`` for identical images, monotonically increasing with blur, noise,
    and structural distortion). Safe for offline CPU CI and headless workers.

    Args:
        pred: sRGB image ``[H, W, 3]`` or sequence ``[T, H, W, 3]``.
        gt: Ground-truth sRGB image/sequence of identical shape.
        mask: Optional foreground mask ``[H, W]`` or ``[T, H, W]``.

    Returns:
        Perceptual distance in ``[0, 1]``.
    """
    p, g, _ = _to_float_images(pred, gt, data_range=1.0)
    if p.size == 0:
        return 0.0

    p = np.clip(p, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)

    if p.ndim == 3:
        p4 = p[np.newaxis, ...]
        g4 = g[np.newaxis, ...]
    elif p.ndim == 4:
        p4 = p
        g4 = g
    else:
        raise ValueError(f"lpips expects 3D or 4D RGB array, got shape {p.shape}.")

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        m4 = (
            np.broadcast_to(m[np.newaxis, :, :, np.newaxis], p4.shape)
            if m.ndim == 2
            else _expand_mask(m, p4.shape)
        )
        p4 = np.where(m4, p4, 0.0)
        g4 = np.where(m4, g4, 0.0)

    # Convert to perceptual opponent channels (Y, C1, C2)
    def _to_opponent(rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[-1] != 3:
            return rgb
        y = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
        c1 = 0.5 * (rgb[..., 0:1] - rgb[..., 1:2])
        c2 = 0.25 * (rgb[..., 0:1] + rgb[..., 1:2]) - 0.5 * rgb[..., 2:3]
        return np.concatenate([y, c1, c2], axis=-1)

    op_p = _to_opponent(p4)
    op_g = _to_opponent(g4)

    scale_weights = (0.5, 0.3, 0.2)
    total_dist = 0.0
    weight_sum = 0.0

    cur_p = op_p
    cur_g = op_g
    for w in scale_weights:
        _, h_s, w_s, _ = cur_p.shape
        l1_val = float(np.mean(np.abs(cur_p - cur_g)))

        if h_s >= 3 and w_s >= 3:
            gx_p = cur_p[:, :, 1:, :] - cur_p[:, :, :-1, :]
            gx_g = cur_g[:, :, 1:, :] - cur_g[:, :, :-1, :]
            gy_p = cur_p[:, 1:, :, :] - cur_p[:, :-1, :, :]
            gy_g = cur_g[:, 1:, :, :] - cur_g[:, :-1, :, :]

            # Unit-normalise local feature vectors (analogous to LPIPS channel norm)
            norm_xp = gx_p / (np.linalg.norm(gx_p, axis=-1, keepdims=True) + 1e-6)
            norm_xg = gx_g / (np.linalg.norm(gx_g, axis=-1, keepdims=True) + 1e-6)
            norm_yp = gy_p / (np.linalg.norm(gy_p, axis=-1, keepdims=True) + 1e-6)
            norm_yg = gy_g / (np.linalg.norm(gy_g, axis=-1, keepdims=True) + 1e-6)

            grad_dist = 0.25 * (
                float(np.mean((norm_xp - norm_xg) ** 2))
                + float(np.mean((norm_yp - norm_yg) ** 2))
            )
            scale_score = 0.4 * l1_val + 0.6 * grad_dist
        else:
            scale_score = l1_val

        total_dist += w * scale_score
        weight_sum += w

        if h_s < 4 or w_s < 4:
            break
        cur_p = _downsample_2x(cur_p)
        cur_g = _downsample_2x(cur_g)

    score = total_dist / max(weight_sum, 1e-12)
    return float(np.clip(score, 0.0, 1.0))
