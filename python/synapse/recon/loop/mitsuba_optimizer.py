# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Dual-track inverse material optimizer (RFC 0062 §4).

Provides two optimisation backends:

1. **AnalyticUVInverseOptimizer** — pure NumPy CPU solver that uses
   analytic Jacobians of the Lambertian rendering equation to recover
   per-texel albedo from multi-view images.  Always available; suitable
   for CI and CPU-only environments.

2. **MitsubaInverseOptimizer** — optional GPU-accelerated differentiable
   rendering via Mitsuba 3 (``pip install 'synapse[invrender]'``).
   Lazily imported; ``import synapse.recon.loop`` always succeeds even
   without ``mitsuba`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.recon.loop.types import LoopConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InverseOptimConfig:
    """Configuration for inverse material optimisation.

    Attributes:
        max_iterations: Maximum number of gradient-descent steps.
        learning_rate: Step size for gradient descent.
        tv_weight: Total-variation regularisation weight.
        convergence_threshold: Early-stop L2 residual threshold.
        texture_resolution: UV texture map resolution (width = height).
    """

    max_iterations: int = 100
    learning_rate: float = 0.01
    tv_weight: float = 0.001
    convergence_threshold: float = 1e-4
    texture_resolution: int = 256


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class InverseOptimResult:
    """Result of inverse material optimisation.

    Attributes:
        albedo_map: Recovered albedo texture, shape ``(R, R, 3)`` float32
            in linear sRGB / ACEScg.
        psnr_before_db: PSNR before optimisation (initial guess vs target).
        psnr_after_db: PSNR after optimisation.
        ciede2000_after: Approximate CIE ΔE₀₀ after optimisation.
        iterations_run: Actual number of iterations before convergence
            or limit.
        converged: ``True`` if converged within the threshold.
    """

    albedo_map: np.ndarray  # (R, R, 3) float32
    psnr_before_db: float = 0.0
    psnr_after_db: float = 0.0
    ciede2000_after: float = 0.0
    iterations_run: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# AnalyticUVInverseOptimizer (always available, pure NumPy)
# ---------------------------------------------------------------------------


class AnalyticUVInverseOptimizer:
    """CPU-only analytic inverse material optimizer.

    Uses the Lambertian rendering equation ``I = albedo * max(N·L, 0)``
    to analytically solve for per-texel albedo given known geometry
    (normals) and lighting.

    For multi-view consistency, multiple observations of each texel are
    averaged with inverse-variance weighting.

    Args:
        config: Optimisation configuration.
    """

    def __init__(self, config: InverseOptimConfig | None = None) -> None:
        self.config = config or InverseOptimConfig()

    def optimize_material_from_video(
        self,
        frames: np.ndarray,
        normals: np.ndarray,
        uv_maps: np.ndarray,
        light_dirs: np.ndarray | None = None,
    ) -> InverseOptimResult:
        """Recover per-texel albedo from multi-view video frames.

        Args:
            frames: ``(N, H, W, 3)`` float32 observed images (linear).
            normals: ``(N, H, W, 3)`` float32 surface normals per view.
            uv_maps: ``(N, H, W, 2)`` float32 UV coordinates per pixel.
            light_dirs: ``(N, 3)`` float32 light directions per view.
                If ``None``, a frontal light ``[0, 0, 1]`` is assumed.

        Returns:
            :class:`InverseOptimResult` with the recovered albedo map.
        """
        n_views, h, w, _ = frames.shape
        res = self.config.texture_resolution

        if light_dirs is None:
            light_dirs = np.tile(
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                (n_views, 1),
            )

        # Initialise albedo map with mid-grey
        albedo = np.full((res, res, 3), 0.5, dtype=np.float32)

        # Compute initial PSNR
        initial_render = self._render(albedo, normals, uv_maps, light_dirs)
        psnr_before = _compute_psnr(initial_render, frames)

        # Accumulation buffers for per-texel inverse solve
        accum_color = np.zeros((res, res, 3), dtype=np.float64)
        accum_weight = np.zeros((res, res), dtype=np.float64)

        for v in range(n_views):
            l_dir = light_dirs[v].astype(np.float64)
            n_map = normals[v].astype(np.float64)

            # N·L clamped shading
            ndotl = np.sum(n_map * l_dir[np.newaxis, np.newaxis, :], axis=-1)
            ndotl = np.clip(ndotl, 0.01, 1.0)  # avoid division by zero

            # UV -> texel index
            uv = uv_maps[v].astype(np.float64)
            tx = np.clip((uv[..., 0] * (res - 1)).astype(np.intp), 0, res - 1)
            ty = np.clip((uv[..., 1] * (res - 1)).astype(np.intp), 0, res - 1)

            # Solve albedo = I / (N·L) per pixel, accumulate
            frame_f = frames[v].astype(np.float64)
            solved = frame_f / ndotl[..., np.newaxis]
            solved = np.clip(solved, 0.0, 2.0)

            # Weight by N·L (higher confidence when well-lit)
            for row in range(h):
                for col in range(w):
                    ti, tj = ty[row, col], tx[row, col]
                    w_val = ndotl[row, col]
                    accum_color[ti, tj] += w_val * solved[row, col]
                    accum_weight[ti, tj] += w_val

        # Finalise albedo
        valid = accum_weight > 1e-8
        for c in range(3):
            albedo[:, :, c][valid] = (
                accum_color[:, :, c][valid] / accum_weight[valid]
            ).astype(np.float32)

        albedo = np.clip(albedo, 0.0, 1.0)

        # Iterative TV regularisation refinement
        converged = False
        iters_run = 0
        for it in range(self.config.max_iterations):
            iters_run = it + 1

            # TV gradient (finite differences)
            tv_grad = np.zeros_like(albedo, dtype=np.float64)
            tv_grad[:-1, :, :] += (
                albedo[:-1, :, :].astype(np.float64)
                - albedo[1:, :, :].astype(np.float64)
            )
            tv_grad[:, :-1, :] += (
                albedo[:, :-1, :].astype(np.float64)
                - albedo[:, 1:, :].astype(np.float64)
            )
            tv_grad *= self.config.tv_weight

            # Apply gradient
            albedo = albedo - self.config.learning_rate * tv_grad.astype(np.float32)
            albedo = np.clip(albedo, 0.0, 1.0)

            # Check convergence
            grad_norm = float(np.sqrt(np.mean(tv_grad ** 2)))
            if grad_norm < self.config.convergence_threshold:
                converged = True
                break

        # Compute final metrics
        final_render = self._render(albedo, normals, uv_maps, light_dirs)
        psnr_after = _compute_psnr(final_render, frames)
        ciede_after = _approx_ciede2000(final_render, frames)

        return InverseOptimResult(
            albedo_map=albedo,
            psnr_before_db=psnr_before,
            psnr_after_db=psnr_after,
            ciede2000_after=ciede_after,
            iterations_run=iters_run,
            converged=converged,
        )

    def _render(
        self,
        albedo: np.ndarray,
        normals: np.ndarray,
        uv_maps: np.ndarray,
        light_dirs: np.ndarray,
    ) -> np.ndarray:
        """Render views from albedo + normals + UVs."""
        n_views, h, w, _ = normals.shape
        res = albedo.shape[0]
        rendered = np.zeros((n_views, h, w, 3), dtype=np.float32)

        for v in range(n_views):
            l_dir = light_dirs[v]
            n_map = normals[v]
            ndotl = np.clip(
                np.sum(n_map * l_dir[np.newaxis, np.newaxis, :], axis=-1),
                0.0, 1.0,
            )

            uv = uv_maps[v]
            tx = np.clip((uv[..., 0] * (res - 1)).astype(np.intp), 0, res - 1)
            ty = np.clip((uv[..., 1] * (res - 1)).astype(np.intp), 0, res - 1)

            sampled_albedo = albedo[ty, tx, :]
            rendered[v] = sampled_albedo * ndotl[..., np.newaxis]

        return rendered


# ---------------------------------------------------------------------------
# MitsubaInverseOptimizer (optional, lazy import)
# ---------------------------------------------------------------------------


class MitsubaInverseOptimizer:
    """GPU-accelerated inverse material optimizer via Mitsuba 3.

    Requires ``pip install 'synapse[invrender]'`` (mitsuba >= 3.5.0,
    drjit >= 0.4.5).  Raises :class:`~synapse.errors.BackendUnavailableError`
    if mitsuba is not installed.
    """

    def __init__(self) -> None:
        try:
            import mitsuba  # noqa: F401
        except ImportError:
            from synapse.errors import BackendUnavailableError

            raise BackendUnavailableError(
                "Mitsuba 3 is not installed.  "
                "Install with: pip install 'synapse[invrender]'",
                agent_fix_hint=(
                    "pip install 'mitsuba>=3.5.0' 'drjit>=0.4.5' or "
                    "use AnalyticUVInverseOptimizer for CPU-only mode."
                ),
            )

    def optimize(self, **kwargs: Any) -> InverseOptimResult:
        """Run Mitsuba-based differentiable inverse rendering.

        This is a placeholder that validates the import path.
        Full implementation requires a GPU runtime.
        """
        from synapse.errors import NotImplementedStageError

        raise NotImplementedStageError(
            "MitsubaInverseOptimizer.optimize() requires a GPU runtime "
            "with Mitsuba 3 and is not available in CPU-only CI.",
            agent_fix_hint=(
                "Use AnalyticUVInverseOptimizer for CPU-only environments "
                "or run on a GPU node with mitsuba installed."
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between two float arrays (assuming [0, 1] range)."""
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _approx_ciede2000(pred: np.ndarray, target: np.ndarray) -> float:
    """Approximate CIE ΔE₀₀ as scaled L2 distance."""
    diff = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2))) * 100.0
