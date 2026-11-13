"""Temporal consistency metrics — upstream-first with pure NumPy fallback.

When ``fulgur_metrics`` (the aether C++/pybind11 native extension) is
installed, temporal warp error and flicker score delegate to the
SIMD-accelerated implementation from ``@aether//metrics/temporal.h``.
Otherwise the module falls back to the local pure-NumPy implementations.

``fvd_feature_distance`` is a synapse-original contribution (scipy-based
Frechet Video Distance) with no upstream equivalent.

See: https://github.com/sautille-stroke/synapse/issues/34
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "temporal_warp_error",
    "flicker_score",
    "fvd_feature_distance",
]

# ---------------------------------------------------------------------------
# Upstream native extension probe
# ---------------------------------------------------------------------------

_DEPRECATION_MSG = (
    "synapse.metrics.temporal: using pure-NumPy fallback for temporal "
    "metrics.  Install 'fulgur-metrics' for production-grade "
    "SIMD-accelerated temporal analysis.  This fallback will be removed "
    "in synapse v2.0.  "
    "See https://github.com/sautille-stroke/synapse/issues/34"
)

try:
    from fulgur_metrics.metrics import (  # type: ignore[import-not-found]
        temporal_flicker_score as _native_flicker_score,
        temporal_warp_error as _native_temporal_warp_error,
    )

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False
    _native_temporal_warp_error = None
    _native_flicker_score = None

_warned = False


def _warn_fallback() -> None:
    """Emit a single deprecation warning per process lifetime."""
    global _warned  # noqa: PLW0603
    if not _warned:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
        _warned = True


# ===================================================================
#  Pure-NumPy fallback implementations
# ===================================================================


def _fallback_temporal_warp_error(
    frames: np.ndarray,
    motion_vectors: np.ndarray,
    occlusion_masks: np.ndarray | None = None,
) -> float:
    """Warp each frame *t* to *t+1* using motion vectors and return RMSE."""
    frames = np.asarray(frames, dtype=np.float32)
    motion_vectors = np.asarray(motion_vectors, dtype=np.float32)

    t_count = frames.shape[0] - 1
    if t_count <= 0:
        return 0.0

    h, w = frames.shape[1], frames.shape[2]

    # Build base coordinate grids
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)  # [H, W] each

    rmse_sum = 0.0
    for t in range(t_count):
        mv = motion_vectors[t]  # [H, W, 2]
        src_x = gx + mv[..., 0]  # source x in frame t
        src_y = gy + mv[..., 1]  # source y in frame t

        # Nearest-neighbour sampling (clamp to border)
        sx = np.clip(np.round(src_x).astype(np.int32), 0, w - 1)
        sy = np.clip(np.round(src_y).astype(np.int32), 0, h - 1)

        warped = frames[t][sy, sx]  # [H, W, 3]
        diff_sq = (warped - frames[t + 1]) ** 2  # [H, W, 3]
        pixel_mse = diff_sq.mean(axis=-1)  # [H, W]

        if occlusion_masks is not None:
            weight = 1.0 - np.asarray(
                occlusion_masks[t], dtype=np.float32,
            )
            weight_sum = weight.sum()
            if weight_sum > 0:
                rmse = np.sqrt((pixel_mse * weight).sum() / weight_sum)
            else:
                rmse = 0.0
        else:
            rmse = np.sqrt(pixel_mse.mean())

        rmse_sum += float(rmse)

    return rmse_sum / t_count


def _fallback_flicker_score(frames: np.ndarray) -> float:
    """Second-order temporal Laplacian energy (flicker metric)."""
    frames = np.asarray(frames, dtype=np.float32)
    t_count = frames.shape[0]
    if t_count < 3:
        return 0.0

    # Vectorised Laplacian over all interior frames
    lap = frames[:-2] - 2.0 * frames[1:-1] + frames[2:]  # [T-2, H, W, 3]
    energy = (lap ** 2).mean()
    return float(energy)


# ===================================================================
#  Public API — delegates to native when available
# ===================================================================


def temporal_warp_error(
    frames: np.ndarray,
    motion_vectors: np.ndarray,
    occlusion_masks: np.ndarray | None = None,
) -> float:
    """Warp each frame *t* to *t+1* using motion vectors and return RMSE.

    Args:
        frames: ``[T, H, W, 3]`` float32 RGB frames in [0, 1].
        motion_vectors: ``[T-1, H, W, 2]`` per-pixel displacement
            (dx, dy) from frame *t* to frame *t+1*.
        occlusion_masks: Optional ``[T-1, H, W]`` float masks where
            1 = occluded.  Non-occluded pixels are weighted by
            ``(1 - mask)``.

    Returns:
        Mean RMSE across all frame pairs (scalar float).
    """
    if _HAS_NATIVE:
        return _native_temporal_warp_error(  # type: ignore[misc]
            frames, motion_vectors, occlusion_masks,
        )
    _warn_fallback()
    return _fallback_temporal_warp_error(frames, motion_vectors, occlusion_masks)


def flicker_score(frames: np.ndarray) -> float:
    """Second-order temporal Laplacian energy (flicker metric).

    For each interior frame *t* computes
    ``|f(t-1) - 2*f(t) + f(t+1)|**2`` per pixel and averages across all
    pixels and valid frames.

    Args:
        frames: ``[T, H, W, 3]`` float32 RGB frames.

    Returns:
        Scalar float (lower is better; 0 for perfectly static content).
    """
    if _HAS_NATIVE:
        return _native_flicker_score(frames)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_flicker_score(frames)


def fvd_feature_distance(
    features_real: np.ndarray,
    features_gen: np.ndarray,
) -> float:
    """Frechet Video Distance between two sets of feature vectors.

    Computes the Frechet distance between two multivariate Gaussians
    fitted to the real and generated feature distributions::

        FVD = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 * sqrtm(S1 @ S2))

    This is a synapse-original contribution with no upstream equivalent.

    Args:
        features_real: ``[N, D]`` feature vectors from real videos.
        features_gen: ``[M, D]`` feature vectors from generated videos.

    Returns:
        Scalar float FVD (lower is better; 0 for identical
        distributions).

    Note:
        Requires ``scipy`` (lazy-imported).
    """
    from scipy.linalg import sqrtm  # lazy import

    fr = np.asarray(features_real, dtype=np.float64)
    fg = np.asarray(features_gen, dtype=np.float64)

    mu1 = fr.mean(axis=0)
    mu2 = fg.mean(axis=0)
    sigma1 = np.cov(fr, rowvar=False)
    sigma2 = np.cov(fg, rowvar=False)

    # Ensure covariance matrices are 2-D even for D=1 edge case
    if sigma1.ndim == 0:
        sigma1 = sigma1.reshape(1, 1)
    if sigma2.ndim == 0:
        sigma2 = sigma2.reshape(1, 1)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    # Numerical cleanup — drop small imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = float(
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covmean)
    )
    return max(fvd, 0.0)
