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

from typing import Any, ClassVar, Sequence

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

    mag = np.sqrt(gx**2 + gy**2)
    return np.asarray(mag.mean(axis=-1))


# ---------------------------------------------------------------------------
# Fusion kernels
# ---------------------------------------------------------------------------


def _fuse_laplacian_frequency(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
    n_levels: int = 4,
) -> np.ndarray:
    """Frequency-adaptive Laplacian pyramid fusion.

    Decomposes each candidate into a multi-level Laplacian pyramid and
    blends with **frequency-adaptive** weights: at fine (high-frequency)
    levels, sharper candidates receive higher weight; at coarse
    (low-frequency) levels, smoother candidates receive higher weight.

    Sharpness is measured per-candidate per-level as the L1 norm of the
    Laplacian band, which directly reflects the energy present at that
    frequency.  The per-level sharpness modulates the input ``weights``
    so that the fusion is no longer a uniform linear interpolation at
    every level.
    """
    h, w, c = candidates[0].shape
    min_dim = min(h, w)
    max_levels = max(int(np.floor(np.log2(max(min_dim, 1)))), 1)
    n_levels = min(n_levels, max_levels)

    pyramids = [build_laplacian_pyramid(cand, n_levels) for cand in candidates]
    n_entries = len(pyramids[0])
    n_cands = len(candidates)

    blended_pyr: list[np.ndarray] = []
    for level in range(n_entries):
        is_residual = level == n_entries - 1  # last entry is low-freq residual

        if is_residual:
            # Residual (coarsest): plain weighted average is correct
            accum = np.zeros_like(pyramids[0][level])
            for k in range(n_cands):
                accum += weights[k] * pyramids[k][level]
            blended_pyr.append(accum)
        else:
            # Laplacian band: frequency-adaptive weighting
            # Measure sharpness (L1 energy) of each candidate at this level
            sharpness = np.zeros(n_cands, dtype=np.float64)
            for k in range(n_cands):
                sharpness[k] = float(np.mean(np.abs(pyramids[k][level]))) + 1e-8

            # Fine levels (low level index) → favour sharp candidates
            # Coarse levels (high level index) → favour smooth candidates
            # Exponent: positive for fine (favour high energy), negative
            # for coarse (favour low energy).  Linear schedule from +1 to -1.
            if n_entries > 2:
                alpha = 1.0 - 2.0 * level / (n_entries - 2)
            else:
                alpha = 1.0
            modulated = sharpness ** alpha

            # Combine with input weights
            level_weights = np.asarray(weights, dtype=np.float64) * modulated
            level_sum = float(np.sum(level_weights))
            if level_sum > 1e-10:
                level_weights /= level_sum
            else:
                level_weights = np.asarray(weights, dtype=np.float64)

            accum = np.zeros_like(pyramids[0][level])
            for k in range(n_cands):
                accum += level_weights[k] * pyramids[k][level]
            blended_pyr.append(accum)

    return reconstruct_laplacian_pyramid(blended_pyr)


def _fuse_cascade(
    candidates: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Cascaded multi-resolution progressive fusion.

    Builds a true cascade: starts with the low-frequency content from
    the *worst* (lowest-weight) candidate, then progressively
    accumulates high-frequency residuals from better candidates at
    increasingly fine scales.

    Unlike a naïve blend (which degenerates to ``(1-α)*result + α*cand``
    because Gaussian blur/residual terms cancel), this cascade
    accumulates detail *additively* — each candidate contributes only
    its high-frequency band at the resolution tier assigned to it,
    producing genuine multi-resolution synthesis.
    """
    order = np.argsort(weights)
    h, w, c = candidates[0].shape

    n_cands = len(candidates)
    if n_cands == 1:
        return np.asarray(candidates[order[0]].copy())

    # Start with low-frequency baseline from the lowest-weight candidate.
    # Build telescoping frequency bands so identical candidates reconstruct
    # the exact identity (zero high-frequency energy loss):
    #   G(s_0, c_0) + sum_{k=1}^{K-2} (G(s_k, c_k) - G(s_{k-1}, c_k)) + (c_{K-1} - G(s_{K-2}, c_{K-1}))
    sigmas = [max(2.0 ** (n_cands - step), 1.0) for step in range(n_cands)]
    result = separable_gaussian(candidates[order[0]].copy(), sigmas[0])

    # Progressively add telescoping high-frequency detail from higher-weight candidates
    for step in range(1, n_cands):
        cand_idx = order[step]
        cand = candidates[cand_idx]
        sigma_prev = sigmas[step - 1]

        if step < n_cands - 1:
            sigma_curr = sigmas[step]
            coarse = separable_gaussian(cand, sigma_prev)
            fine = separable_gaussian(cand, sigma_curr)
            band_detail = fine - coarse
        else:
            # Finest tier: all detail above the previous tier's cutoff
            coarse = separable_gaussian(cand, sigma_prev)
            band_detail = cand - coarse

        result = result + band_detail

    return np.asarray(result, dtype=np.float32)


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

    if k_count == 2:
        # For K=2, consensus disagreement is symmetric (|c0 - mean|^2 == |c1 - mean|^2).
        # Break symmetry using local gradient sharpness so sharper candidates receive
        # lower uncertainty (higher weight), while identical candidates preserve exact weights.
        def _sharp(img: np.ndarray) -> np.ndarray:
            gy = np.diff(img, axis=0, prepend=img[:1, :])
            gx = np.diff(img, axis=1, prepend=img[:, :1])
            mag = np.mean(np.abs(gx) + np.abs(gy), axis=-1).astype(np.float32)
            return separable_gaussian(mag[:, :, np.newaxis], sigma=1.0)[:, :, 0]

        sharp0 = _sharp(candidates[0])
        sharp1 = _sharp(candidates[1])
        local_vars[0] = (local_vars[0] + 1e-3) / (sharp0 + 1.0)
        local_vars[1] = (local_vars[1] + 1e-3) / (sharp1 + 1.0)

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
    """Gradient-domain Poisson blending fusion with mixed gradients.

    Implements proper Poisson blending: for each pixel, selects the
    gradient (from any candidate) with the *largest magnitude* (mixed
    gradient approach), weighted by candidate weights.  Then solves the
    Poisson equation via Jacobi iteration with Dirichlet boundary
    conditions from the weighted average.

    The mixed-gradient approach ensures that the strongest edges from
    all candidates are preserved in the fused result, unlike simple
    weighted-average gradients which blur competing edges.
    """
    h, w, c = candidates[0].shape

    # Compute per-candidate forward-difference gradients
    n_cands = len(candidates)
    cand_gx = []
    cand_gy = []
    for cand in candidates:
        gx = np.zeros_like(cand)
        gy = np.zeros_like(cand)
        gx[:, :-1, :] = cand[:, 1:, :] - cand[:, :-1, :]
        gy[:-1, :, :] = cand[1:, :, :] - cand[:-1, :, :]
        cand_gx.append(gx)
        cand_gy.append(gy)

    # Mixed gradient selection: at each pixel, pick the gradient with
    # the largest weighted magnitude across candidates
    target_gx = np.zeros_like(candidates[0])
    target_gy = np.zeros_like(candidates[0])

    # Compute weighted gradient magnitudes per candidate
    gx_mag = np.zeros((n_cands, h, w), dtype=np.float64)
    gy_mag = np.zeros((n_cands, h, w), dtype=np.float64)
    for k in range(n_cands):
        gx_mag[k] = weights[k] * np.mean(np.abs(cand_gx[k]), axis=-1)
        gy_mag[k] = weights[k] * np.mean(np.abs(cand_gy[k]), axis=-1)

    # Select gradient from the candidate with strongest weighted magnitude
    best_gx_idx = np.argmax(gx_mag, axis=0)  # (H, W)
    best_gy_idx = np.argmax(gy_mag, axis=0)  # (H, W)

    for k in range(n_cands):
        mask_gx = best_gx_idx == k
        mask_gy = best_gy_idx == k
        target_gx[mask_gx] = cand_gx[k][mask_gx]
        target_gy[mask_gy] = cand_gy[k][mask_gy]

    # Backward-difference divergence (adjoint of forward-diff gradient)
    div = np.zeros_like(candidates[0])
    # Interior: backward difference of target_gx along x
    div[:, 1:, :] += target_gx[:, 1:, :] - target_gx[:, :-1, :]
    div[:, 0, :] += target_gx[:, 0, :]
    # Interior: backward difference of target_gy along y
    div[1:, :, :] += target_gy[1:, :, :] - target_gy[:-1, :, :]
    div[0, :, :] += target_gy[0, :, :]

    # Initialise from weighted average
    result = np.zeros_like(candidates[0])
    for k, cand in enumerate(candidates):
        result += weights[k] * cand

    dirichlet_boundary = result.copy()

    # Exact O(HW log(HW)) 2D Discrete Sine Transform (DST-I) direct Poisson solver
    # for interior Dirichlet problem L(u) = div, falling back to Jacobi if scipy is absent.
    if h > 2 and w > 2:
        try:
            from scipy.fft import dstn, idstn

            m_int, n_int = h - 2, w - 2
            rhs = div[1:-1, 1:-1, :].astype(np.float64, copy=True)
            rhs[0, :, :] -= dirichlet_boundary[0, 1:-1, :]
            rhs[-1, :, :] -= dirichlet_boundary[-1, 1:-1, :]
            rhs[:, 0, :] -= dirichlet_boundary[1:-1, 0, :]
            rhs[:, -1, :] -= dirichlet_boundary[1:-1, -1, :]

            py = np.arange(1, m_int + 1, dtype=np.float64)[:, None]
            px = np.arange(1, n_int + 1, dtype=np.float64)[None, :]
            eig = (2.0 * np.cos(np.pi * py / (m_int + 1)) - 2.0) + (
                2.0 * np.cos(np.pi * px / (n_int + 1)) - 2.0
            )
            u_hat = dstn(rhs, type=1, axes=(0, 1)) / eig[:, :, np.newaxis]
            u_int = idstn(u_hat, type=1, axes=(0, 1))
            result[1:-1, 1:-1, :] = u_int.astype(np.float32)
            return result
        except ImportError:
            pass

    # Fallback: Jacobi iterations for Poisson equation with fixed outer 1px Dirichlet boundary
    for _ in range(n_iterations):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        neighbour_sum = (
            padded[:-2, 1:-1, :] + padded[2:, 1:-1, :] + padded[1:-1, :-2, :] + padded[1:-1, 2:, :]
        )
        result[1:-1, 1:-1, :] = (neighbour_sum[1:-1, 1:-1, :] - div[1:-1, 1:-1, :]) / 4.0
        # Re-anchor outer 1px Dirichlet boundary
        result[0, :, :] = dirichlet_boundary[0, :, :]
        result[-1, :, :] = dirichlet_boundary[-1, :, :]
        result[:, 0, :] = dirichlet_boundary[:, 0, :]
        result[:, -1, :] = dirichlet_boundary[:, -1, :]

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
        candidates = [np.asarray(c, dtype=np.float32) for c in candidates]

        # Single candidate passthrough
        if len(candidates) == 1:
            return np.asarray(candidates[0].copy())

        # Validate shapes
        ref_shape = candidates[0].shape
        for i, c in enumerate(candidates):
            if c.shape != ref_shape:
                raise ValueError(f"Candidate {i} shape {c.shape} != reference shape {ref_shape}")

        # Normalise weights
        n = len(candidates)
        if weights is None:
            w = np.ones(n, dtype=np.float32) / n
        else:
            w = np.asarray(weights, dtype=np.float32).ravel()
            if len(w) != n:
                raise ValueError(f"weights length {len(w)} != candidates count {n}")
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


class AnatomicalModelEnsemble:
    """Multi-candidate model ensemble generating diverse candidates for refinement fusion.

    Integrates:
    - Base original crop (structure & color guidance)
    - Primary refiner (e.g. Hand/Face neural or geometric model)
    - Optional secondary refiner / detail enhancer (e.g. Foundary transport)
    """

    def __init__(
        self,
        refiners: Sequence[Any] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.refiners = list(refiners) if refiners else []
        self.weights = list(weights) if weights else []

    def generate_candidates(
        self,
        roi: Any,
    ) -> list[np.ndarray]:
        cands: list[np.ndarray] = [roi.crop_acescg]
        for refiner in self.refiners:
            if hasattr(refiner, "refine"):
                res = refiner.refine(roi)
                cands.append(res.refined_acescg if hasattr(res, "refined_acescg") else res)
            elif callable(refiner):
                res = refiner(roi)
                cands.append(res.refined_acescg if hasattr(res, "refined_acescg") else res)
        return cands


__all__ = [
    "AnatomicalModelEnsemble",
    "MultiModelRefineFusion",
]
