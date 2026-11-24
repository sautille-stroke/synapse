"""Perceptual similarity metrics (DINOv2 cosine similarity, LPIPS-VGG, Patch-LPIPS).

Heavy neural backends (``torch``, ``torchvision``, ``lpips``, ``timm``,
``transformers``) are strictly lazy-loaded ONLY when explicitly requested via
``backend="dinov2"`` or ``backend="vgg"``.  The default ``backend="auto"`` mode
provides deterministic, zero-torch pure-NumPy structural/perceptual oracles so
unit tests and CPU verification run fast with zero network or GPU overhead.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from synapse.metrics.image_quality import _to_float_images, lpips

__all__ = [
    "dino_cosine_sim",
    "lpips_vgg",
    "patch_lpips",
]


def _normalize_batch_rgb(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise inputs to float64 ``[T, H, W, C]`` in ``[0, 1]``."""
    a, b, eff_range = _to_float_images(img_a, img_b, data_range=1.0)
    if eff_range > 0.0 and eff_range != 1.0:
        a = a / eff_range
        b = b / eff_range
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[np.newaxis, ..., np.newaxis], 3, axis=-1)
        b = np.repeat(b[np.newaxis, ..., np.newaxis], 3, axis=-1)
    elif a.ndim == 3:
        if a.shape[2] == 1:
            a = np.repeat(a, 3, axis=-1)
            b = np.repeat(b, 3, axis=-1)
        a = a[np.newaxis, ...]
        b = b[np.newaxis, ...]
    elif a.ndim == 4:
        if a.shape[3] == 1:
            a = np.repeat(a, 3, axis=-1)
            b = np.repeat(b, 3, axis=-1)
    else:
        raise ValueError(f"Expected 2D/3D/4D image tensor, got shape {a.shape}.")
    return a, b


def _extract_hog_spatial_descriptor(img: np.ndarray, grid_levels: tuple[int, ...] = (4, 8)) -> np.ndarray:
    """Compute a deterministic L2-normalized multi-scale HOG + color descriptor."""
    h, w, c = img.shape
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2] if c >= 3 else img[..., 0]

    # Spatial image gradients
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    if w > 1:
        gx[:, :-1] = luma[:, 1:] - luma[:, :-1]
    if h > 1:
        gy[:-1, :] = luma[1:, :] - luma[:-1, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # [0, 1]
    bins = np.clip((ang * 8.0).astype(np.int64), 0, 7)

    feats: list[np.ndarray] = []
    for g in grid_levels:
        gh = min(g, max(1, h))
        gw = min(g, max(1, w))
        y_edges = np.linspace(0, h, gh + 1, dtype=np.int64)
        x_edges = np.linspace(0, w, gw + 1, dtype=np.int64)
        for iy in range(gh):
            y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
            for ix in range(gw):
                x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
                patch_rgb = img[y0:y1, x0:x1, :3]
                patch_mag = mag[y0:y1, x0:x1]
                patch_bin = bins[y0:y1, x0:x1]
                mean_rgb = patch_rgb.mean(axis=(0, 1)) if patch_rgb.size > 0 else np.zeros(3)
                std_rgb = patch_rgb.std(axis=(0, 1)) if patch_rgb.size > 0 else np.zeros(3)
                hog = np.bincount(patch_bin.ravel(), weights=patch_mag.ravel(), minlength=8)
                feats.append(mean_rgb)
                feats.append(std_rgb)
                feats.append(hog / max(float(patch_mag.size), 1.0))

    vec = np.concatenate(feats, axis=0).astype(np.float64)
    # Include a small uniform bias anchor so identical flat zero images still align (1.0)
    # while a zero image vs a non-zero image separates cleanly
    vec = np.concatenate([vec, np.array([1e-3], dtype=np.float64)])
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-12)


def dino_cosine_sim(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    backend: Literal["auto", "hog_oracle", "dinov2"] = "auto",
) -> float:
    """Compute perceptual semantic cosine similarity in ``[-1, 1]`` (1.0 = identical).

    When ``backend="dinov2"``, lazy-imports ``torch`` and ``transformers`` /
    ``timm`` to extract ViT DINOv2 CLS embeddings.  When ``backend="auto"`` or
    ``"hog_oracle"``, computes a deterministic multi-scale HOG + spatial color
    descriptor cosine similarity with zero heavy framework imports.
    """
    a, b = _normalize_batch_rgb(img_a, img_b)
    if a.shape[0] == 0:
        return 1.0

    if backend not in ("auto", "hog_oracle", "dinov2"):
        raise ValueError(
            f"Unsupported backend {backend!r}; expected 'auto', 'hog_oracle', or 'dinov2'."
        )

    if backend == "dinov2":
        from synapse.errors import BackendUnavailableError

        try:
            import torch
            import transformers
        except ImportError as exc:
            raise BackendUnavailableError(
                f"DINOv2 perceptual metric requires torch and transformers: {exc}",
                agent_fix_hint="Install synapse[gpu] or pass backend='auto' for CPU HOG oracle.",
            ) from exc

        try:
            processor = transformers.AutoImageProcessor.from_pretrained(
                "facebook/dinov2-small", local_files_only=True
            )
            model = transformers.AutoModel.from_pretrained(
                "facebook/dinov2-small", local_files_only=True
            )
            model.eval()
            with torch.no_grad():
                u8_a = [np.clip(frame * 255.0, 0, 255).astype(np.uint8) for frame in a]
                u8_b = [np.clip(frame * 255.0, 0, 255).astype(np.uint8) for frame in b]
                in_a = processor(images=u8_a, return_tensors="pt")
                in_b = processor(images=u8_b, return_tensors="pt")
                emb_a = model(**in_a).last_hidden_state[:, 0, :]
                emb_b = model(**in_b).last_hidden_state[:, 0, :]
                emb_a = torch.nn.functional.normalize(emb_a, dim=-1)
                emb_b = torch.nn.functional.normalize(emb_b, dim=-1)
                sims = (emb_a * emb_b).sum(dim=-1)
                return float(sims.mean().item())
        except Exception as exc:
            raise BackendUnavailableError(
                f"DINOv2 local weights unavailable: {exc}",
                agent_fix_hint=(
                    "Cache 'facebook/dinov2-small' locally or run with backend='auto'."
                ),
            ) from exc

    # Zero-torch multi-scale HOG + spatial color descriptor cosine similarity
    sims_list: list[float] = []
    for fa, fb in zip(a, b, strict=True):
        if np.array_equal(fa, fb):
            sims_list.append(1.0)
            continue
        da = _extract_hog_spatial_descriptor(fa)
        db = _extract_hog_spatial_descriptor(fb)
        sims_list.append(float(np.clip(np.dot(da, db), -1.0, 1.0)))
    return float(np.mean(sims_list))


def lpips_vgg(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    backend: Literal["auto", "proxy", "vgg"] = "auto",
) -> float:
    """Compute perceptual distance LPIPS-VGG in ``[0, 1]`` (0.0 = identical).

    When ``backend="vgg"``, lazy-imports ``torch`` and the ``lpips`` library
    using VGG16 weights (raising ``BackendUnavailableError`` exit 30 if missing).
    When ``backend="auto"`` or ``"proxy"``, delegates to the zero-torch
    multi-scale gradient perceptual metric :func:`synapse.metrics.image_quality.lpips`.
    """
    if backend not in ("auto", "proxy", "vgg"):
        raise ValueError(
            f"Unsupported backend {backend!r}; expected 'auto', 'proxy', or 'vgg'."
        )

    if backend == "vgg":
        from synapse.errors import BackendUnavailableError

        a, b = _normalize_batch_rgb(img_a, img_b)
        if a.shape[0] == 0:
            return 0.0
        try:
            import lpips as lpips_pkg  # type: ignore[import-not-found]
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                f"VGG LPIPS requires torch and the 'lpips' package: {exc}",
                agent_fix_hint="Install 'lpips' and torch (synapse[gpu]), or pass backend='auto'.",
            ) from exc

        try:
            net: Any = lpips_pkg.LPIPS(net="vgg")
            net.eval()
            ta = torch.from_numpy(a.transpose(0, 3, 1, 2).copy()).float() * 2.0 - 1.0
            tb = torch.from_numpy(b.transpose(0, 3, 1, 2).copy()).float() * 2.0 - 1.0
            with torch.no_grad():
                val = net(ta, tb)
            return float(np.clip(val.mean().item(), 0.0, 1.0))
        except Exception as exc:
            raise BackendUnavailableError(
                f"VGG LPIPS evaluation failed: {exc}",
                agent_fix_hint="Ensure VGG16 pretrained weights are cached or pass backend='auto'.",
            ) from exc

    return float(lpips(img_a, img_b))


def patch_lpips(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    patch_size: int = 64,
    stride: int = 32,
    aggregation: Literal["mean", "max", "max_quartile"] = "max_quartile",
    backend: Literal["auto", "proxy", "vgg"] = "auto",
) -> float:
    """Compute localized sliding-patch LPIPS to catch seam/texture artifacts.

    Args:
        img_a: Predicted image/batch ``[H, W, C]`` or ``[T, H, W, C]``.
        img_b: Reference image/batch of matching shape.
        patch_size: Sliding window side length in pixels (clamped to image dims).
        stride: Step size in pixels between windows.
        aggregation: ``"max_quartile"`` (mean of worst 25% patches, default),
            ``"max"`` (worst single patch), or ``"mean"`` (all patches).
        backend: LPIPS backend passed to :func:`lpips_vgg`.

    Returns:
        Scalar perceptual distance in ``[0, 1]``.
    """
    a, b = _normalize_batch_rgb(img_a, img_b)
    if a.shape[0] == 0:
        return 0.0

    _, h, w, _ = a.shape
    eff_ph = max(1, min(patch_size, h))
    eff_pw = max(1, min(patch_size, w))
    eff_stride = max(1, stride)

    if h <= eff_ph and w <= eff_pw:
        return lpips_vgg(a, b, backend=backend)

    ys = list(range(0, max(1, h - eff_ph + 1), eff_stride))
    if ys[-1] + eff_ph < h:
        ys.append(h - eff_ph)
    xs = list(range(0, max(1, w - eff_pw + 1), eff_stride))
    if xs[-1] + eff_pw < w:
        xs.append(w - eff_pw)

    patch_scores: list[float] = []
    for t in range(a.shape[0]):
        for y in ys:
            for x in xs:
                pa = a[t, y : y + eff_ph, x : x + eff_pw, :]
                pb = b[t, y : y + eff_ph, x : x + eff_pw, :]
                patch_scores.append(lpips_vgg(pa, pb, backend=backend))

    arr = np.asarray(patch_scores, dtype=np.float64)
    if aggregation == "mean":
        return float(np.mean(arr))
    if aggregation == "max":
        return float(np.max(arr))
    if aggregation == "max_quartile":
        k = max(1, int(np.ceil(0.25 * len(arr))))
        top_k = np.sort(arr)[-k:]
        return float(np.mean(top_k))
    raise ValueError(
        f"Unsupported aggregation {aggregation!r}; expected 'mean', 'max', or 'max_quartile'."
    )
