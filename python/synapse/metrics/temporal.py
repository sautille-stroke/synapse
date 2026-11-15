"""Temporal consistency metrics — upstream-first with pure NumPy fallback.

When native bindings from ``@aether//metrics`` are available, temporal warp
error and flicker score delegate to the SIMD-accelerated implementation.
Otherwise the module falls back to the local pure-NumPy implementations
without emitting runtime warnings.

``fvd_feature_distance`` is a synapse-original contribution (scipy-based
Frechet Video Distance) with no upstream equivalent.

See: https://github.com/sautille-stroke/synapse/issues/34
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "temporal_warp_error",
    "flicker_score",
    "fvd_feature_distance",
    "compute_fvd_from_videos",
    "FVDResult",
]


@dataclass(frozen=True)
class FVDResult:
    """Result container for FVD computation with proxy metadata.

    Attributes:
        value: The scalar FVD score (lower is better).
        is_proxy_metric: ``True`` if statistical proxy features were used
            instead of a pretrained neural feature extractor.
    """

    value: float
    is_proxy_metric: bool = False

# ---------------------------------------------------------------------------
# Upstream native extension probe (@aether//metrics)
# ---------------------------------------------------------------------------

try:
    from aether.metrics import (
        temporal_flicker_score as _native_flicker_score,
    )
    from aether.metrics import (
        temporal_warp_error as _native_temporal_warp_error,
    )

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False
    _native_temporal_warp_error = None
    _native_flicker_score = None


# ===================================================================
#  Pure-NumPy fallback implementations
# ===================================================================


def _fallback_temporal_warp_error(
    frames: np.ndarray,
    motion_vectors: np.ndarray,
    occlusion_masks: np.ndarray | None = None,
    metric: Literal["l1", "rmse"] = "l1",
) -> float:
    """Warp each frame *t* to *t+1* using motion vectors and return error (L1 or RMSE)."""
    frames = np.asarray(frames, dtype=np.float32)
    motion_vectors = np.asarray(motion_vectors, dtype=np.float32)

    t_count = frames.shape[0] - 1
    if t_count <= 0:
        return 0.0

    h, w = frames.shape[1], frames.shape[2]

    # Build base coordinate grids
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)  # [H, W] each

    error_sum = 0.0
    for t in range(t_count):
        mv = motion_vectors[t]  # [H, W, 2]
        src_x = gx + mv[..., 0]  # source x in frame t
        src_y = gy + mv[..., 1]  # source y in frame t

        # Nearest-neighbour sampling (clamp to border)
        sx = np.clip(np.round(src_x).astype(np.int32), 0, w - 1)
        sy = np.clip(np.round(src_y).astype(np.int32), 0, h - 1)

        warped = frames[t][sy, sx]  # [H, W, 3]
        target = frames[t + 1]      # [H, W, 3]

        if metric == "l1":
            pixel_err = np.abs(warped - target).mean(axis=-1)  # [H, W]
        elif metric == "rmse":
            pixel_err = ((warped - target) ** 2).mean(axis=-1)  # [H, W]
        else:
            raise ValueError(f"Unsupported warp error metric: '{metric}'. Choose 'l1' or 'rmse'.")

        if occlusion_masks is not None:
            weight = 1.0 - np.asarray(
                occlusion_masks[t], dtype=np.float32,
            )
            weight_sum = weight.sum()
            if weight_sum > 0:
                if metric == "l1":
                    err_val = (pixel_err * weight).sum() / weight_sum
                else:
                    err_val = np.sqrt((pixel_err * weight).sum() / weight_sum)
            else:
                err_val = 0.0
        else:
            if metric == "l1":
                err_val = pixel_err.mean()
            else:
                err_val = np.sqrt(pixel_err.mean())

        error_sum += float(err_val)

    return float(error_sum / t_count)


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
    metric: Literal["l1", "rmse"] = "l1",
) -> float:
    """Warp each frame *t* to *t+1* using motion vectors and return warp error.

    Args:
        frames: ``[T, H, W, 3]`` float32 RGB frames in [0, 1].
        motion_vectors: ``[T-1, H, W, 2]`` per-pixel displacement
            (dx, dy) from frame *t* to frame *t+1*.
        occlusion_masks: Optional ``[T-1, H, W]`` float masks where
            1 = occluded.  Non-occluded pixels are weighted by
            ``(1 - mask)``.
        metric: Error metric to evaluate: ``"l1"`` (default, per RFC-0063
            §4.4 / §5 Track 2 specification) or ``"rmse"`` (legacy RMSE).

    Returns:
        Mean error across all frame pairs (scalar float).
    """
    if _HAS_NATIVE and metric == "rmse":
        return float(
            _native_temporal_warp_error(
                frames, motion_vectors, occlusion_masks,
            )
        )
    return _fallback_temporal_warp_error(
        frames, motion_vectors, occlusion_masks, metric=metric,
    )


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
        return float(_native_flicker_score(frames))
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

        Uses adaptive Tikhonov regularization (epsilon * I) on covariance
        matrices to ensure numerical stability and eliminate
        ``LinAlgWarning: Matrix is singular`` from ``scipy.linalg.sqrtm``.
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

    # Adaptive Tikhonov regularization: epsilon * I to stabilize sqrtm
    # Use max of trace-derived epsilon and a fixed floor to handle
    # singular/near-singular covariance matrices robustly.
    # The product sigma1 @ sigma2 can still be singular/ill-conditioned
    # even when both inputs are positive definite, so we regularize
    # aggressively enough that sqrtm does not trigger LinAlgWarning.
    d = sigma1.shape[0]
    eps_floor = 1e-7
    eps1 = max(float(np.trace(sigma1)) * 1e-6, eps_floor)
    eps2 = max(float(np.trace(sigma2)) * 1e-6, eps_floor)
    sigma1 = sigma1 + eps1 * np.eye(d, dtype=np.float64)
    sigma2 = sigma2 + eps2 * np.eye(d, dtype=np.float64)

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


def compute_fvd_from_videos(
    real_videos: np.ndarray | list[np.ndarray],
    gen_videos: np.ndarray | list[np.ndarray],
    feature_extractor: Any = None,
    *,
    require_pretrained: bool = False,
    allow_statistical_proxy: bool = True,
) -> FVDResult:
    """Compute Fréchet Video Distance (FVD) directly from video sequences.

    Args:
        real_videos: Reference video batch as array ``[N, T, H, W, 3]`` or list of ``[T, H, W, 3]``.
        gen_videos: Generated video batch as array ``[M, T, H, W, 3]`` or list of ``[T, H, W, 3]``.
        feature_extractor: Optional callable taking a video sequence and returning ``[N, D]``
            feature embeddings. If None, attempts to use pretrained torchvision R3D-18.
        require_pretrained: If ``True`` and no pretrained model is available (torch/torchvision
            not installed), raise ``BackendUnavailableError`` instead of falling back.
        allow_statistical_proxy: If ``True`` (default) and no neural extractor is available,
            use statistical proxy features. If ``False`` and neural extraction is unavailable,
            raise ``BackendUnavailableError``.

    Returns:
        An :class:`FVDResult` containing the scalar FVD score and proxy metadata.

    Raises:
        BackendUnavailableError: If ``require_pretrained=True`` or
            ``allow_statistical_proxy=False`` and PyTorch/torchvision is not installed.
        ValueError: If input video batches have fewer than 2 videos for covariance estimation.
    """
    is_proxy = False

    if feature_extractor is None:
        # Try to build a real pretrained 3D video feature extractor (R3D-18)
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            feature_extractor = _build_r3d18_extractor()
        except ImportError:
            if require_pretrained or not allow_statistical_proxy:
                from synapse.errors import BackendUnavailableError
                raise BackendUnavailableError(
                    "PyTorch and torchvision are required for neural FVD video feature extraction. "
                    "Install synapse[gpu] or provide a custom feature_extractor callable.",
                    agent_fix_hint="pip install torch torchvision or pass a feature_extractor callable",
                ) from None

            # Fall back to statistical proxy
            is_proxy = True
            feature_extractor = _statistical_proxy_extractor

    feats_real = np.asarray(feature_extractor(real_videos), dtype=np.float64)
    feats_gen = np.asarray(feature_extractor(gen_videos), dtype=np.float64)

    if feats_real.ndim == 1:
        feats_real = feats_real[np.newaxis, :]
    if feats_gen.ndim == 1:
        feats_gen = feats_gen[np.newaxis, :]

    if len(feats_real) < 2 or len(feats_gen) < 2:
        raise ValueError(
            f"FVD requires at least 2 video samples per distribution to compute covariance; "
            f"got {len(feats_real)} real and {len(feats_gen)} generated."
        )

    fvd_value = fvd_feature_distance(feats_real, feats_gen)
    return FVDResult(value=fvd_value, is_proxy_metric=is_proxy)


def _build_r3d18_extractor() -> Any:
    """Build a pretrained R3D-18 penultimate-layer feature extractor.

    Returns:
        A callable that takes ``[N, T, H, W, 3]`` numpy video batch and
        returns ``[N, 512]`` numpy feature vectors.
    """
    import torch
    import torchvision.models.video as video_models
    from torchvision import transforms
    # Load pretrained R3D-18 and strip the final FC layer
    model = video_models.r3d_18(weights="DEFAULT")
    # Remove the final fully connected layer to get penultimate features
    model.fc = torch.nn.Identity()
    model.eval()

    # Standard video preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
        ),
    ])

    def _extract(vids: np.ndarray | list[np.ndarray]) -> np.ndarray:
        arr = np.asarray(vids, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[np.newaxis, ...]

        all_feats = []
        with torch.no_grad():
            for i in range(arr.shape[0]):
                v = arr[i]  # [T, H, W, C]
                # Convert to [C, T, H, W] tensor
                t_video = torch.from_numpy(v).permute(3, 0, 1, 2).float()
                # Apply preprocessing per-channel across spatial dims
                t_video = preprocess(t_video)
                # Add batch dim [1, C, T, H, W]
                t_video = t_video.unsqueeze(0)
                feats = model(t_video)  # [1, 512]
                all_feats.append(feats.squeeze(0).cpu().numpy())

        return np.array(all_feats, dtype=np.float64)

    return _extract


def _statistical_proxy_extractor(
    vids: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """Statistical proxy feature extractor (no neural network required).

    Computes per-video spatial-temporal statistics as a feature vector.
    This is a proxy metric and should be flagged as such.

    Args:
        vids: ``[N, T, H, W, C]`` or list of ``[T, H, W, C]`` video arrays.

    Returns:
        ``[N, D]`` feature vectors.
    """
    arr = np.asarray(vids, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[np.newaxis, ...]
    n_samples = arr.shape[0]
    feats = []
    for i in range(n_samples):
        v = arr[i]  # [T, H, W, C]
        # Spatial-temporal mean and std as robust baseline statistics
        m = v.mean(axis=(0, 1, 2))  # [C]
        s = v.std(axis=(0, 1, 2))   # [C]
        # Temporal diff features
        tdiff = np.abs(v[1:] - v[:-1]).mean(axis=(0, 1, 2)) if v.shape[0] > 1 else m
        feats.append(np.concatenate([m, s, tdiff]))
    return np.array(feats, dtype=np.float64)
