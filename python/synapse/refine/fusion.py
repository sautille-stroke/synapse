# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Multi-model refinement fusion kernels (RFC 0061 §3).

Provides five heterogeneous fusion strategies for merging candidate
refinement outputs from different generative models into a single
high-quality patch.  All operations are **pure NumPy** on **float32
ACEScg** tensors — no ``torch`` or ``scipy`` dependency at import
time.

The five strategies are intentionally **orthogonal** to the spatial
compositing primitives in ``synapse.composite.blend`` (which handles
foreground-background Laplacian blending for *placement* onto a
background frame).  This module instead merges *multiple candidate
refinements* of the **same** region before that compositing step.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from synapse.refine.types import FusionMethod

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """1-D normalised Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return (k / k.sum()).astype(np.float32)


def _separable_blur(
    img: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Separable Gaussian blur (pure NumPy)."""
    if sigma < 0.5:
        return img.copy()

    radius = max(int(round(3.0 * sigma)), 1)
    kernel = _build_gaussian_kernel_1d(sigma, radius)

    def _convolve_1d(arr: np.ndarray, axis: int) -> np.ndarray:
        pad_w = [(0, 0)] * arr.ndim
        pad_w[axis] = (radius, radius)
        padded = np.pad(arr, pad_w, mode="reflect")
        n_out = arr.shape[axis]
        result = np.zeros_like(arr)
        for i, w in enumerate(kernel):
            slices = [slice(None)] * arr.ndim
            slices[axis] = slice(i, i + n_out)
            result += w * padded[tuple(slices)]
        return result

    out = _convolve_1d(img, axis=0)
    out = _convolve_1d(out, axis=1)
    return out


def _downsample2x(img: np.ndarray) -> np.ndarray:
    """Gaussian-prefiltered 2x downsample."""
    blurred = _separable_blur(img, sigma=1.0)
    return blurred[::2, ::2]


def _upsample2x(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear 2x upsample to ``(target_h, target_w, ...)``."""
    h, w = img.shape[:2]
    row_idx = np.arange(target_h, dtype=np.float32) * (h - 1) / max(target_h - 1, 1)
    col_idx = np.arange(target_w, dtype=np.float32) * (w - 1) / max(target_w - 1, 1)

    r0 = np.floor(row_idx).astype(np.intp)
    r1 = np.minimum(r0 + 1, h - 1)
    rf = (row_idx - r0).astype(np.float32)

    c0 = np.floor(col_idx).astype(np.intp)
    c1 = np.minimum(c0 + 1, w - 1)
    cf = (col_idx - c0).astype(np.float32)

    out = np.empty((target_h, target_w, img.shape[2]), dtype=img.dtype)
    for i in range(target_h):
        top = (1.0 - rf[i]) * img[r0[i], :, :] + rf[i] * img[r1[i], :, :]
        out[i, :, :] = (
            (1.0 - cf)[:, np.newaxis] * top[c0, :]
            + cf[:, np.newaxis] * top[c1, :]
        )
    return out


def _build_laplacian_pyramid(
    img: np.ndarray, n_levels: int,
) -> list[np.ndarray]:
    """Build a Laplacian pyramid (last entry is the low-pass residual)."""
    gaussian: list[np.ndarray] = [img.copy()]
    cur = img.copy()
    for _ in range(n_levels):
        cur = _downsample2x(cur)
        gaussian.append(cur)

    lap: list[np.ndarray] = []
    for i in range(n_levels):
        up = _upsample2x(gaussian[i + 1], gaussian[i].shape[0], gaussian[i].shape[1])
        lap.append(gaussian[i] - up)
    lap.append(gaussian[-1])
    return lap


def _reconstruct_laplacian(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct image from its Laplacian pyramid."""
    cur = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        up = _upsample2x(cur, pyramid[i].shape[0], pyramid[i].shape[1])
        cur = pyramid[i] + up
    return cur


def _sobel_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude via Sobel operator.

    Args:
        img: ``(H, W, C)`` float32 image.

    Returns:
        ``(H, W)`` float32 gradient magnitude (mean across channels).
    """
    # Horizontal gradient
    gx = np.zeros_like(img)
    gx[:, 1:-1, :] = img[:, 2:, :] - img[:, :-2, :]

    # Vertical gradient
    gy = np.zeros_like(img)
    gy[1:-1, :, :] = img[2:, :, :] - img[:-2, :, :]

    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag.mean(axis=-1)


# ---------------------------------------------------------------------------
# Fusion kernels
# ---------------------------------------------------------------------------


def _fuse_laplacian_frequency(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
    n_levels: int = 4,
) -> np.ndarray:
    """Laplacian pyramid frequency-domain fusion.

    Decomposes each candidate into a multi-level Laplacian pyramid,
    blends per level using the supplied weights, and reconstructs.

    This preserves high-frequency detail from the candidate with the
    highest weight while maintaining smooth low-frequency structure
    from the weighted average.
    """
    h, w, c = candidates[0].shape
    min_dim = min(h, w)
    max_levels = max(int(np.floor(np.log2(max(min_dim, 1)))), 1)
    n_levels = min(n_levels, max_levels)

    pyramids = [_build_laplacian_pyramid(cand, n_levels) for cand in candidates]
    n_entries = len(pyramids[0])

    blended_pyr: list[np.ndarray] = []
    for level in range(n_entries):
        accum = np.zeros_like(pyramids[0][level])
        for k, pyr in enumerate(pyramids):
            accum += weights[k] * pyr[level]
        blended_pyr.append(accum)

    return _reconstruct_laplacian(blended_pyr)


def _fuse_cascade(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Cascaded multi-resolution progressive fusion.

    Begins with the lowest-weighted candidate at coarse resolution and
    progressively blends in higher-weighted candidates at finer
    resolutions, producing a result that draws structure from the most
    trusted candidate while incorporating detail from others.
    """
    # Sort candidates by weight (ascending) — lowest weight contributes
    # coarse structure, highest weight contributes fine detail.
    order = np.argsort(weights)
    h, w, c = candidates[0].shape

    # Start from the coarsest (lowest weight) candidate
    result = candidates[order[0]].copy()

    n_cands = len(candidates)
    for step in range(1, n_cands):
        cand_idx = order[step]
        # Blend ratio increases as we move to higher-weight candidates
        alpha = weights[cand_idx] / max(float(np.sum(weights)), 1e-8)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Progressive downscale / upscale for multi-resolution effect
        scale = max(1, n_cands - step)
        if scale > 1 and h > scale * 4 and w > scale * 4:
            small_h = h // scale
            small_w = w // scale
            result_small = _upsample2x(
                _downsample2x(result),
                small_h,
                small_w,
            )
            cand_small = _upsample2x(
                _downsample2x(candidates[cand_idx]),
                small_h,
                small_w,
            )
            blended_small = (1.0 - alpha) * result_small + alpha * cand_small
            result = _upsample2x(blended_small, h, w)
        else:
            result = (1.0 - alpha) * result + alpha * candidates[cand_idx]

    return result


def _fuse_latent_timestep_switch(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Latent timestep-switch fusion.

    Conceptually simulates a diffusion-like process: takes structure
    from the first (lowest-indexed) candidate and detail from the last,
    with smooth interpolation in between using Gaussian blur as a proxy
    for denoising steps.
    """
    n = len(candidates)
    h, w, c = candidates[0].shape

    # Compute structure from first candidate (heavily blurred)
    structure_sigma = max(h, w) * 0.05
    structure = _separable_blur(candidates[0], structure_sigma)

    # Compute detail from last candidate
    detail_source = candidates[-1]
    detail_blurred = _separable_blur(detail_source, structure_sigma)
    detail = detail_source - detail_blurred

    # For middle candidates, interpolate contribution
    if n > 2:
        mid_weight_sum = float(np.sum(weights[1:-1]))
        if mid_weight_sum > 1e-8:
            mid_accum = np.zeros_like(candidates[0])
            for k in range(1, n - 1):
                mid_accum += weights[k] * candidates[k]
            mid_accum /= mid_weight_sum
            # Blend mid-frequency from middle candidates
            mid_sigma = structure_sigma * 0.5
            mid_structure = _separable_blur(mid_accum, mid_sigma)
            mid_detail = mid_accum - mid_structure
            detail = 0.5 * detail + 0.5 * mid_detail

    # Weighted combination
    w_struct = weights[0] / max(float(np.sum(weights)), 1e-8)
    w_detail = weights[-1] / max(float(np.sum(weights)), 1e-8)
    w_struct = max(w_struct, 0.3)

    result = w_struct * structure + detail
    return result


def _fuse_uncertainty(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Uncertainty-weighted per-pixel fusion.

    Computes per-pixel variance across candidates and uses inverse
    variance to weight the contribution of each candidate.  Regions
    where candidates agree strongly receive the mean; regions of
    disagreement weight toward the candidate with lowest local variance.
    """
    stack = np.stack(candidates, axis=0)  # (K, H, W, C)
    k_count = stack.shape[0]

    # Per-candidate local variance (3x3 neighbourhood proxy)
    local_vars = np.zeros((k_count,) + candidates[0].shape[:2], dtype=np.float32)
    for k in range(k_count):
        # Compute per-pixel variance across channels
        chan_var = np.var(stack[k], axis=-1)  # (H, W)
        # Smooth to get local neighbourhood variance
        smoothed = _separable_blur(chan_var[:, :, np.newaxis], sigma=1.5)
        local_vars[k] = smoothed[:, :, 0]

    # Inverse variance weighting (regularised)
    eps = 1e-6
    inv_var = 1.0 / (local_vars + eps)  # (K, H, W)

    # Incorporate prior weights
    for k in range(k_count):
        inv_var[k] *= weights[k]

    # Normalise
    inv_var_sum = np.sum(inv_var, axis=0, keepdims=True)  # (1, H, W)
    inv_var_sum = np.maximum(inv_var_sum, eps)
    norm_weights = inv_var / inv_var_sum  # (K, H, W)

    # Weighted sum
    result = np.zeros_like(candidates[0])
    for k in range(k_count):
        result += norm_weights[k, :, :, np.newaxis] * stack[k]

    return result


def _fuse_poisson_gradient(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
    n_iterations: int = 200,
) -> np.ndarray:
    """Gradient-domain Poisson blending fusion.

    Blends gradient fields from all candidates using the supplied
    weights, then solves the Poisson equation via iterative Jacobi
    to reconstruct a seamless image whose gradient field is the
    weighted average of all candidate gradients.
    """
    h, w, c = candidates[0].shape

    # Compute weighted average of gradient fields
    target_gx = np.zeros_like(candidates[0])
    target_gy = np.zeros_like(candidates[0])

    for k, cand in enumerate(candidates):
        gx = np.zeros_like(cand)
        gy = np.zeros_like(cand)
        gx[:, 1:, :] = cand[:, 1:, :] - cand[:, :-1, :]
        gy[1:, :, :] = cand[1:, :, :] - cand[:-1, :, :]
        target_gx += weights[k] * gx
        target_gy += weights[k] * gy

    # Divergence of target gradient field
    div = np.zeros_like(candidates[0])
    # d(gx)/dx
    div[:, 1:, :] += target_gx[:, 1:, :] - target_gx[:, :-1, :]
    # d(gy)/dy
    div[1:, :, :] += target_gy[1:, :, :] - target_gy[:-1, :, :]

    # Initialise from weighted average
    result = np.zeros_like(candidates[0])
    for k, cand in enumerate(candidates):
        result += weights[k] * cand

    # Jacobi iterations for Poisson equation: Laplacian(u) = div
    for _ in range(n_iterations):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        neighbour_sum = (
            padded[:-2, 1:-1, :]   # up
            + padded[2:, 1:-1, :]  # down
            + padded[1:-1, :-2, :] # left
            + padded[1:-1, 2:, :]  # right
        )
        result = (neighbour_sum - div) / 4.0

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MultiModelRefineFusion:
    """Dispatcher for multi-model refinement fusion.

    Merges candidate refinement outputs from different generative models
    into a single patch using one of five fusion strategies.

    All inputs and outputs are **float32 ACEScg** tensors of shape
    ``(H, W, 3)``.

    Example::

        fusion = MultiModelRefineFusion()
        fused = fusion.fuse(
            candidates=[patch_a, patch_b, patch_c],
            method=FusionMethod.LAPLACIAN_FREQUENCY,
            weights=np.array([0.5, 0.3, 0.2]),
        )
    """

    _DISPATCH = {
        FusionMethod.LAPLACIAN_FREQUENCY: "_fuse_laplacian_frequency",
        FusionMethod.CASCADE: "_fuse_cascade",
        FusionMethod.LATENT_TIMESTEP_SWITCH: "_fuse_latent_timestep_switch",
        FusionMethod.UNCERTAINTY_WEIGHTED: "_fuse_uncertainty",
        FusionMethod.POISSON_GRADIENT: "_fuse_poisson_gradient",
    }

    def fuse(
        self,
        candidates: Sequence[np.ndarray],
        method: FusionMethod,
        weights: np.ndarray | None = None,
        original: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fuse multiple candidate refinements into a single output.

        Args:
            candidates: Sequence of ``(H, W, 3)`` float32 ACEScg arrays
                from different generative models.
            method: The :class:`FusionMethod` strategy to use.
            weights: Per-candidate weights; normalised internally.  If
                ``None``, uniform weights are used.
            original: Optional original (pre-refinement) patch for
                reference.  Currently used as a fallback in single-
                candidate mode.

        Returns:
            ``(H, W, 3)`` float32 ACEScg fused result.

        Raises:
            ValueError: If *candidates* is empty or shapes are
                inconsistent.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")

        # Ensure float32
        candidates = [
            np.asarray(c, dtype=np.float32) for c in candidates
        ]

        # Single candidate passthrough
        if len(candidates) == 1:
            return candidates[0].copy()

        # Validate shapes
        ref_shape = candidates[0].shape
        for i, c in enumerate(candidates):
            if c.shape != ref_shape:
                raise ValueError(
                    f"Candidate {i} shape {c.shape} != "
                    f"reference shape {ref_shape}"
                )

        # Normalise weights
        n = len(candidates)
        if weights is None:
            w = np.ones(n, dtype=np.float32) / n
        else:
            w = np.asarray(weights, dtype=np.float32).ravel()
            if len(w) != n:
                raise ValueError(
                    f"weights length {len(w)} != candidates count {n}"
                )
            w_sum = float(np.sum(w))
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(n, dtype=np.float32) / n

        # Dispatch
        handler_name = self._DISPATCH.get(method)
        if handler_name is None:
            raise ValueError(f"Unknown FusionMethod: {method}")

        handler = getattr(self, handler_name)
        result = handler(candidates, w)
        return np.asarray(result, dtype=np.float32)

    # -- per-method wrappers calling module-level kernels --

    @staticmethod
    def _fuse_laplacian_frequency(
        candidates: Sequence[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        return _fuse_laplacian_frequency(candidates, weights)

    @staticmethod
    def _fuse_cascade(
        candidates: Sequence[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        return _fuse_cascade(candidates, weights)

    @staticmethod
    def _fuse_latent_timestep_switch(
        candidates: Sequence[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        return _fuse_latent_timestep_switch(candidates, weights)

    @staticmethod
    def _fuse_uncertainty(
        candidates: Sequence[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        return _fuse_uncertainty(candidates, weights)

    @staticmethod
    def _fuse_poisson_gradient(
        candidates: Sequence[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        return _fuse_poisson_gradient(candidates, weights)
