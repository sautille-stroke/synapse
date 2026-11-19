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
from typing import Any, Callable, Literal

import numpy as np

from synapse.bridge.aether_bridge import get_aether_metrics

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
# Upstream native extension probe (@aether//metrics via aether_bridge)
# ---------------------------------------------------------------------------

_aether_mod = get_aether_metrics()
_native_temporal_warp_error: Callable[..., Any] | None = getattr(
    _aether_mod, "temporal_warp_error", None
)
_native_flicker_score: Callable[..., Any] | None = getattr(_aether_mod, "temporal_flicker_score", None)
_HAS_NATIVE = (_native_temporal_warp_error is not None) and (_native_flicker_score is not None)


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
    raw_frames = np.asarray(frames)
    if raw_frames.dtype == np.uint8:
        frames_f = raw_frames.astype(np.float32) / 255.0
    else:
        frames_f = raw_frames.astype(np.float32)

    motion_vectors = np.asarray(motion_vectors, dtype=np.float32)

    t_count = frames_f.shape[0] - 1
    if t_count <= 0:
        return 0.0

    if motion_vectors.shape[0] < t_count:
        raise ValueError(
            f"Expected at least {t_count} motion vector frames for {frames_f.shape[0]} video frames, "
            f"got {motion_vectors.shape[0]}."
        )

    h, w = frames_f.shape[1], frames_f.shape[2]

    # Build base coordinate grids
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)  # [H, W] each

    error_sum = 0.0
    for t in range(t_count):
        mv = motion_vectors[t]  # [H, W, 2] or [H_mv, W_mv, 2]
        if mv.shape[0] != h or mv.shape[1] != w:
            sx = float(w) / float(mv.shape[1])
            sy = float(h) / float(mv.shape[0])
            src_mv_x = np.clip(gx / sx, 0.0, float(mv.shape[1] - 1))
            src_mv_y = np.clip(gy / sy, 0.0, float(mv.shape[0] - 1))
            x0_mv = np.floor(src_mv_x).astype(np.int32)
            x1_mv = np.minimum(x0_mv + 1, mv.shape[1] - 1)
            y0_mv = np.floor(src_mv_y).astype(np.int32)
            y1_mv = np.minimum(y0_mv + 1, mv.shape[0] - 1)
            fx_mv = (src_mv_x - x0_mv)[..., np.newaxis]
            fy_mv = (src_mv_y - y0_mv)[..., np.newaxis]
            mv_scaled = (
                (1.0 - fx_mv) * (1.0 - fy_mv) * mv[y0_mv, x0_mv]
                + (1.0 - fx_mv) * fy_mv * mv[y1_mv, x0_mv]
                + fx_mv * (1.0 - fy_mv) * mv[y0_mv, x1_mv]
                + fx_mv * fy_mv * mv[y1_mv, x1_mv]
            )
            mv_scaled[..., 0] *= sx
            mv_scaled[..., 1] *= sy
            mv = mv_scaled

        # Reverse motion vector -mv per MotionVector.XY convention
        raw_src_x = gx - mv[..., 0]
        raw_src_y = gy - mv[..., 1]
        valid_in_bounds = (
            (raw_src_x >= 0.0)
            & (raw_src_x <= float(w - 1))
            & (raw_src_y >= 0.0)
            & (raw_src_y <= float(h - 1))
        )
        src_x = np.clip(raw_src_x, 0.0, float(w - 1))
        src_y = np.clip(raw_src_y, 0.0, float(h - 1))

        # Bilinear interpolation
        x0 = np.floor(src_x).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(src_y).astype(np.int32)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = (src_x - x0)[..., np.newaxis]
        fy = (src_y - y0)[..., np.newaxis]

        wa = (1.0 - fx) * (1.0 - fy)
        wb = (1.0 - fx) * fy
        wc = fx * (1.0 - fy)
        wd = fx * fy

        warped = (
            wa * frames_f[t][y0, x0]
            + wb * frames_f[t][y1, x0]
            + wc * frames_f[t][y0, x1]
            + wd * frames_f[t][y1, x1]
        )
        target = frames_f[t + 1]  # [H, W, 3]

        if metric == "l1":
            pixel_err = np.abs(warped - target).mean(axis=-1)  # [H, W]
        elif metric == "rmse":
            pixel_err = ((warped - target) ** 2).mean(axis=-1)  # [H, W]
        else:
            raise ValueError(f"Unsupported warp error metric: '{metric}'. Choose 'l1' or 'rmse'.")

        valid_weight = valid_in_bounds.astype(np.float32)
        if occlusion_masks is not None:
            valid_weight *= 1.0 - np.asarray(occlusion_masks[t], dtype=np.float32)

        weight_sum = valid_weight.sum()
        if weight_sum > 0:
            if metric == "l1":
                err_val = (pixel_err * valid_weight).sum() / weight_sum
            else:
                err_val = np.sqrt((pixel_err * valid_weight).sum() / weight_sum)
        else:
            err_val = 0.0

        error_sum += float(err_val)

    return float(error_sum / t_count)


def _fallback_flicker_score(frames: np.ndarray) -> float:
    """Second-order temporal Laplacian energy (flicker metric)."""
    raw_frames = np.asarray(frames)
    if raw_frames.dtype == np.uint8:
        frames_f = raw_frames.astype(np.float32) / 255.0
    else:
        frames_f = raw_frames.astype(np.float32)

    t_count = frames_f.shape[0]
    if t_count < 3:
        return 0.0

    # Vectorised Laplacian over all interior frames
    lap = frames_f[:-2] - 2.0 * frames_f[1:-1] + frames_f[2:]  # [T-2, H, W, 3]
    energy = (lap**2).mean()
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
        frames: ``[T, H, W, 3]`` float32 RGB frames in [0, 1] or uint8 in [0, 255].
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
    raw_frames = np.asarray(frames)
    if raw_frames.dtype == np.uint8:
        frames_f = raw_frames.astype(np.float32) / 255.0
    else:
        frames_f = raw_frames.astype(np.float32)

    if _HAS_NATIVE and callable(_native_temporal_warp_error) and metric == "rmse":
        return float(
            _native_temporal_warp_error(
                frames_f,
                motion_vectors,
                occlusion_masks,
            )
        )
    return _fallback_temporal_warp_error(
        frames_f,
        motion_vectors,
        occlusion_masks,
        metric=metric,
    )


def flicker_score(frames: np.ndarray) -> float:
    """Second-order temporal Laplacian energy (flicker metric).

    For each interior frame *t* computes
    ``|f(t-1) - 2*f(t) + f(t+1)|**2`` per pixel and averages across all
    pixels and valid frames.

    Args:
        frames: ``[T, H, W, 3]`` float32 RGB frames or uint8.

    Returns:
        Scalar float (lower is better; 0 for perfectly static content).
    """
    raw_frames = np.asarray(frames)
    if raw_frames.dtype == np.uint8:
        frames_f = raw_frames.astype(np.float32) / 255.0
    else:
        frames_f = raw_frames.astype(np.float32)

    if _HAS_NATIVE and callable(_native_flicker_score):
        return float(_native_flicker_score(frames_f))
    return _fallback_flicker_score(frames_f)


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

    Raises:
        ContractValidationError: If N < 2 or M < 2 (insufficient samples for covariance).
    """
    from synapse.errors import ContractValidationError

    fr = np.asarray(features_real, dtype=np.float64)
    fg = np.asarray(features_gen, dtype=np.float64)

    if fr.ndim == 1:
        fr = fr.reshape(1, -1)
    if fg.ndim == 1:
        fg = fg.reshape(1, -1)

    if fr.shape[0] < 2 or fg.shape[0] < 2:
        raise ContractValidationError(
            f"FVD requires at least 2 video samples per distribution to compute covariance; "
            f"got {fr.shape[0]} real and {fg.shape[0]} generated samples.",
            agent_fix_hint="Provide at least 2 video feature samples for FVD covariance estimation",
        )

    mu1 = fr.mean(axis=0)
    mu2 = fg.mean(axis=0)
    sigma1 = np.cov(fr, rowvar=False)
    sigma2 = np.cov(fg, rowvar=False)

    # Ensure covariance matrices are 2-D even for D=1 edge case
    if sigma1.ndim == 0:
        sigma1 = sigma1.reshape(1, 1)
    if sigma2.ndim == 0:
        sigma2 = sigma2.reshape(1, 1)

    # Adaptive symmetric Tikhonov regularization: epsilon * I to stabilize square root
    # Unified epsilon eliminates null-space asymmetry bias when N < D.
    d = sigma1.shape[0]
    eps_floor = 1e-7
    eps = max(max(float(np.trace(sigma1)), float(np.trace(sigma2))) * 1e-6, eps_floor)
    sigma1 = sigma1 + eps * np.eye(d, dtype=np.float64)
    sigma2 = sigma2 + eps * np.eye(d, dtype=np.float64)

    diff = mu1 - mu2

    # Robust symmetric matrix square root via eigh (zero complex/imaginary error):
    # Tr(sqrt(sigma1 @ sigma2)) = Tr(sqrt(sigma1^(1/2) @ sigma2 @ sigma1^(1/2)))
    vals1, vecs1 = np.linalg.eigh(sigma1)
    s1_sqrt = vecs1 @ np.diag(np.sqrt(np.maximum(vals1, 0.0))) @ vecs1.T
    m = s1_sqrt @ sigma2 @ s1_sqrt
    m = 0.5 * (m + m.T)
    tr_covmean = float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(m), 0.0))))

    fvd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
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
        ContractValidationError: If input video batches have fewer than 2 videos for covariance estimation.
    """
    is_proxy = False

    if feature_extractor is None:
        try:
            from synapse.errors import BackendUnavailableError

            feature_extractor = _build_r3d18_extractor()
            if feature_extractor is None:
                raise BackendUnavailableError("No neural feature extractor available.")
        except (ImportError, BackendUnavailableError):
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
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            f"FVD requires at least 2 video samples per distribution to compute covariance; "
            f"got {len(feats_real)} real and {len(feats_gen)} generated.",
            agent_fix_hint="Provide at least 2 video samples per distribution for FVD covariance estimation",
        )

    fvd_value = fvd_feature_distance(feats_real, feats_gen)
    return FVDResult(value=fvd_value, is_proxy_metric=is_proxy)


def _build_r3d18_extractor(
    weights: Any = "DEFAULT",
    model: Any = None,
) -> Any:
    """Build a pretrained R3D-18 penultimate-layer feature extractor.

    Returns:
        A callable that takes ``[N, T, H, W, 3]`` numpy video batch and
        returns ``[N, 512]`` numpy feature vectors.
    """
    import os

    from synapse.errors import BackendUnavailableError

    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            f"PyTorch is required for R3D-18 feature extraction: {exc}",
            agent_fix_hint="pip install torch",
        ) from exc

    if model is None:
        try:
            import torchvision.models.video as video_models

            model = video_models.r3d_18(weights=weights)
        except (ImportError, OSError, Exception) as exc:
            torch_home = os.environ.get("TORCH_HOME", "~/.cache/torch")
            raise BackendUnavailableError(
                f"Failed to load R3D-18 video model/weights (TORCH_HOME={torch_home}): {exc}",
                agent_fix_hint="Install torchvision and ensure weights are cached or internet is available",
            ) from exc

    if hasattr(model, "fc") and not isinstance(model.fc, torch.nn.Identity):
        model.fc = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Normalization parameters on [3, 1, 1, 1] channel axis
    mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1).to(device)
    std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1).to(device)

    def _extract(vids: np.ndarray | list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        arr = np.asarray(vids)
        if arr.ndim == 4:
            arr = arr[np.newaxis, ...]

        all_feats: list[np.ndarray] = []
        n_videos = arr.shape[0]
        with torch.no_grad():
            for b_start in range(0, n_videos, batch_size):
                b_end = min(b_start + batch_size, n_videos)
                batch_tensors = []
                for i in range(b_start, b_end):
                    v = arr[i]  # [T, H, W, C]
                    t_video = torch.from_numpy(v.copy() if not v.flags.writeable else v)
                    if t_video.dtype == torch.uint8:
                        t_video = t_video.float() / 255.0
                    else:
                        t_video = t_video.float()

                    # Permute to [C, T, H, W]
                    t_video = t_video.permute(3, 0, 1, 2)  # [3, T, H, W]

                    # Resize spatial dims to 112x112 via trilinear interpolation if needed
                    if t_video.shape[2] != 112 or t_video.shape[3] != 112:
                        t_video = torch.nn.functional.interpolate(
                            t_video.unsqueeze(0),
                            size=(t_video.shape[1], 112, 112),
                            mode="trilinear",
                            align_corners=False,
                        ).squeeze(0)

                    # Channel-wise normalization along dim=0
                    t_video = (t_video.to(device) - mean) / std
                    batch_tensors.append(t_video)

                batch_input = torch.stack(batch_tensors, dim=0)  # [B, C, T, H, W]
                feats = model(batch_input)
                all_feats.append(feats.cpu().numpy())

        if not all_feats:
            return np.zeros((0, 512), dtype=np.float64)
        return np.concatenate(all_feats, axis=0).astype(np.float64)

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
        s = v.std(axis=(0, 1, 2))  # [C]
        # Temporal diff features
        tdiff = np.abs(v[1:] - v[:-1]).mean(axis=(0, 1, 2)) if v.shape[0] > 1 else m
        feats.append(np.concatenate([m, s, tdiff]))
    return np.array(feats, dtype=np.float64)
