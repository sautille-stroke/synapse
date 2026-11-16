# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Multi-model refinement fusion kernels (RFC 0061 §3).

Provides five heterogeneous fusion strategies for merging candidate
refinement outputs from different generative models into a single
high-quality patch.  All operations are **pure NumPy** on **float32
ACEScg** tensors  -- no ``torch`` or ``scipy`` dependency at import
time.

The five strategies are intentionally **orthogonal** to the spatial
compositing primitives in ``synapse.composite.blend`` (which handles
foreground-background Laplacian blending for *placement* onto a
background frame).  This module instead merges *multiple candidate
refinements* of the **same** region before that compositing step.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from synapse.refine.types import FusionMethod
from synapse.render_core.pyramid import (
    build_laplacian_pyramid,
    reconstruct_laplacian_pyramid,
    separable_gaussian,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



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
    return np.asarray(mag.mean(axis=-1))


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

    pyramids = [build_laplacian_pyramid(cand, n_levels) for cand in candidates]
    n_entries = len(pyramids[0])

    blended_pyr: list[np.ndarray] = []
    for level in range(n_entries):
        accum = np.zeros_like(pyramids[0][level])
        for k, pyr in enumerate(pyramids):
            accum += weights[k] * pyr[level]
        blended_pyr.append(accum)

    return reconstruct_laplacian_pyramid(blended_pyr)


def _fuse_cascade(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Cascaded multi-resolution progressive fusion.

    Begins with the lowest-weighted candidate at coarse resolution and
    progressively blends in higher-weighted candidates at finer
    resolutions using multiscale Gaussian pyramid residual synthesis.
    """
    order = np.argsort(weights)
    h, w, c = candidates[0].shape

    result = candidates[order[0]].copy()
    n_cands = len(candidates)
    if n_cands == 1:
        return result

    for step in range(1, n_cands):
        cand_idx = order[step]
        alpha = float(weights[cand_idx] / max(float(np.sum(weights)), 1e-8))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        cand = candidates[cand_idx]
        sigma = max(2.0 ** (n_cands - step), 1.0)
        coarse_res = separable_gaussian(result, sigma)
        coarse_cand = separable_gaussian(cand, sigma)

        blended_coarse = (1.0 - alpha) * coarse_res + alpha * coarse_cand
        detail_cand = cand - coarse_cand
        result = blended_coarse + alpha * detail_cand

    return np.asarray(result)


def _fuse_latent_timestep_switch(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Latent timestep-switch fusion.

    Takes base structure from the first candidate and high-frequency detail
    from subsequent candidates, preserving energy conservation
    (1.0 * structure + detail) to eliminate brightness attenuation.
    """
    n = len(candidates)
    h, w, c = candidates[0].shape

    structure_sigma = max(h, w) * 0.05
    structure = separable_gaussian(candidates[0], structure_sigma)

    detail_source = candidates[-1]
    detail_blurred = separable_gaussian(detail_source, structure_sigma)
    detail = detail_source - detail_blurred

    if n > 2:
        mid_weight_sum = float(np.sum(weights[1:-1]))
        if mid_weight_sum > 1e-8:
            mid_accum = np.zeros_like(candidates[0])
            for k in range(1, n - 1):
                mid_accum += weights[k] * candidates[k]
            mid_accum /= mid_weight_sum
            mid_sigma = structure_sigma * 0.5
            mid_structure = separable_gaussian(mid_accum, mid_sigma)
            mid_detail = mid_accum - mid_structure
            detail = 0.5 * detail + 0.5 * mid_detail

    # Energy-conserving combination: 1.0 * structure + detail
    result = 1.0 * structure + detail
    return np.asarray(result)


def _fuse_uncertainty(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Uncertainty-weighted per-pixel fusion.

    Computes spatial disagreement variance across candidate models and uses
    inverse variance to weight each candidate.
    """
    stack = np.stack(candidates, axis=0)  # (K, H, W, C)
    k_count = stack.shape[0]

    # Consensus mean across ensemble models
    consensus_mean = np.mean(stack, axis=0)  # (H, W, C)

    # Per-candidate spatial variance (disagreement from consensus)
    local_vars = np.zeros((k_count,) + candidates[0].shape[:2], dtype=np.float32)
    for k in range(k_count):
        disagreement = np.mean((stack[k] - consensus_mean) ** 2, axis=-1)  # (H, W)
        smoothed = separable_gaussian(disagreement[:, :, np.newaxis], sigma=1.5)
        local_vars[k] = smoothed[:, :, 0]

    eps = 1e-6
    inv_var = 1.0 / (local_vars + eps)

    for k in range(k_count):
        inv_var[k] *= weights[k]

    inv_var_sum = np.sum(inv_var, axis=0, keepdims=True)
    inv_var_sum = np.maximum(inv_var_sum, eps)
    norm_weights = inv_var / inv_var_sum

    result = np.zeros_like(candidates[0])
    for k in range(k_count):
        result += norm_weights[k, :, :, np.newaxis] * stack[k]

    return result


def _fuse_poisson_gradient(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
    n_iterations: int = 200,
) -> np.ndarray:
    """Gradient-domain Poisson blending fusion with aligned centered differences."""
    h, w, c = candidates[0].shape

    target_gx = np.zeros_like(candidates[0])
    target_gy = np.zeros_like(candidates[0])

    for k, cand in enumerate(candidates):
        gx = np.zeros_like(cand)
        gy = np.zeros_like(cand)
        # Centered difference for interior
        gx[:, 1:-1, :] = (cand[:, 2:, :] - cand[:, :-2, :]) / 2.0
        gy[1:-1, :, :] = (cand[2:, :, :] - cand[:-2, :, :]) / 2.0
        # One-sided for boundaries
        gx[:, 0, :] = cand[:, 1, :] - cand[:, 0, :]
        gx[:, -1, :] = cand[:, -1, :] - cand[:, -2, :]
        gy[0, :, :] = cand[1, :, :] - cand[0, :, :]
        gy[-1, :, :] = cand[-1, :, :] - cand[-2, :, :]

        target_gx += weights[k] * gx
        target_gy += weights[k] * gy

    # Centered divergence of target gradient field
    div = np.zeros_like(candidates[0])
    div[:, 1:-1, :] += (target_gx[:, 2:, :] - target_gx[:, :-2, :]) / 2.0
    div[1:-1, :, :] += (target_gy[2:, :, :] - target_gy[:-2, :, :]) / 2.0
    div[:, 0, :] += target_gx[:, 1, :] - target_gx[:, 0, :]
    div[:, -1, :] += target_gx[:, -1, :] - target_gx[:, -2, :]
    div[0, :, :] += target_gy[1, :, :] - target_gy[0, :, :]
    div[-1, :, :] += target_gy[-1, :, :] - target_gy[-2, :, :]

    # Initialise from weighted average
    result = np.zeros_like(candidates[0])
    for k, cand in enumerate(candidates):
        result += weights[k] * cand

    # Jacobi iterations for Poisson equation
    for _ in range(n_iterations):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        neighbour_sum = (
            padded[:-2, 1:-1, :]
            + padded[2:, 1:-1, :]
            + padded[1:-1, :-2, :]
            + padded[1:-1, 2:, :]
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

    _DISPATCH: ClassVar[dict[FusionMethod, str]] = {
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
            return np.asarray(candidates[0].copy())

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
