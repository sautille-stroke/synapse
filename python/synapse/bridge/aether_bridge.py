"""Bridge to @aether//metrics and @aether//color native C++20 acceleration.

Provides turnkey auto-detection and graceful fallback for upstream Aether
computational kernels:
1. Probes for pre-installed ``aether_metrics`` or builds in sibling ``../aether``.
2. Wraps SIMD-accelerated image quality metrics (PSNR, masked PSNR, SSIM, CIEDE2000, FLIP).
3. Wraps high-performance color space conversions and tone mapping.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Upstream aether_metrics probe & lazy loader
# ---------------------------------------------------------------------------

_aether_metrics_mod: Any = None
_aether_color_mod: Any = None
_PROBED = False


def _find_and_import_aether_metrics() -> Any:
    """Locate and import the aether_metrics extension module."""
    global _PROBED, _aether_metrics_mod, _aether_color_mod
    if _PROBED:
        return _aether_metrics_mod

    _PROBED = True

    # 1. Try standard import (site-packages or existing PYTHONPATH)
    try:
        aether_metrics = importlib.import_module("aether_metrics")
        _aether_metrics_mod = aether_metrics
        _aether_color_mod = getattr(aether_metrics, "color", None)
        return _aether_metrics_mod
    except ImportError:
        pass

    # 2. Try explicit AETHER_ROOT override if defined
    if "AETHER_ROOT" in os.environ:
        cand_root = Path(os.environ["AETHER_ROOT"]).resolve()
        for sub in ["bazel-bin/python", "python"]:
            cand = cand_root / sub
            if cand.is_dir() and (
                (cand / "aether_metrics.so").is_file()
                or (cand / "aether_metrics.pyd").is_file()
            ):
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
                try:
                    aether_metrics = importlib.import_module("aether_metrics")
                    _aether_metrics_mod = aether_metrics
                    _aether_color_mod = getattr(aether_metrics, "color", None)
                    return _aether_metrics_mod
                except ImportError:
                    pass

    # 3. Try sibling directory checkouts (../aether/bazel-bin/python)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    for parent in [repo_root.parent, repo_root]:
        cand = parent / "aether" / "bazel-bin" / "python"
        if cand.is_dir() and (
            (cand / "aether_metrics.so").is_file()
            or (cand / "aether_metrics.pyd").is_file()
        ):
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            try:
                aether_metrics = importlib.import_module("aether_metrics")
                _aether_metrics_mod = aether_metrics
                _aether_color_mod = getattr(aether_metrics, "color", None)
                return _aether_metrics_mod
            except ImportError:
                pass

    return None


def get_aether_metrics() -> Any:
    """Return the loaded aether_metrics module or None."""
    return _find_and_import_aether_metrics()


def has_aether_metrics() -> bool:
    """Return True if aether_metrics native module is accessible."""
    return get_aether_metrics() is not None


def has_aether_color() -> bool:
    """Return True if aether_metrics.color submodule is accessible."""
    _find_and_import_aether_metrics()
    return _aether_color_mod is not None


# ---------------------------------------------------------------------------
# Native Color Science Adapters
# ---------------------------------------------------------------------------


def native_srgb_to_linear(srgb: np.ndarray) -> np.ndarray | None:
    """Convert sRGB to linear RGB via Aether SIMD, preserving shape/dtype and odd symmetry."""
    _find_and_import_aether_metrics()
    if _aether_color_mod is None:
        return None

    arr = np.asarray(srgb)
    orig_shape = arr.shape
    orig_dtype = arr.dtype

    try:
        has_neg = bool(np.any(arr < 0.0))
        if has_neg:
            sign = np.sign(arr)
            abs_flat = np.abs(arr).reshape(-1, 1).astype(np.float32)
            buf = _aether_color_mod.srgb_to_linear_image(abs_flat)
            res = (sign * np.asarray(buf).reshape(orig_shape))
        else:
            flat_2d = arr.reshape(-1, 1).astype(np.float32)
            buf = _aether_color_mod.srgb_to_linear_image(flat_2d)
            res = np.asarray(buf).reshape(orig_shape)
        return res.astype(orig_dtype if orig_dtype in (np.float32, np.float64) else np.float32)
    except Exception:
        return None


def native_linear_to_srgb(linear: np.ndarray) -> np.ndarray | None:
    """Convert linear RGB to sRGB via Aether SIMD, preserving shape/dtype and odd symmetry."""
    _find_and_import_aether_metrics()
    if _aether_color_mod is None:
        return None

    arr = np.asarray(linear)
    orig_shape = arr.shape
    orig_dtype = arr.dtype

    try:
        has_neg = bool(np.any(arr < 0.0))
        if has_neg:
            sign = np.sign(arr)
            abs_flat = np.abs(arr).reshape(-1, 1).astype(np.float32)
            buf = _aether_color_mod.linear_to_srgb_image(abs_flat)
            res = (sign * np.asarray(buf).reshape(orig_shape))
        else:
            flat_2d = arr.reshape(-1, 1).astype(np.float32)
            buf = _aether_color_mod.linear_to_srgb_image(flat_2d)
            res = np.asarray(buf).reshape(orig_shape)
        return res.astype(orig_dtype if orig_dtype in (np.float32, np.float64) else np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Native Image Quality Metric Adapters
# ---------------------------------------------------------------------------


def native_compute_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    peak: float = 1.0,
    auto_hdr: bool = False,
) -> float | None:
    """Compute PSNR using aether_metrics.compute_psnr if available."""
    mod = get_aether_metrics()
    if mod is None:
        return None

    p = np.asarray(pred, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)

    if p.shape != g.shape:
        return None

    # aether requires 2D (H,W) or 3D (H,W,C)
    if p.ndim == 2 or (p.ndim == 3 and p.shape[2] in (1, 3, 4)):
        try:
            return float(mod.compute_psnr(p, g, peak=float(peak), auto_hdr=bool(auto_hdr)))
        except Exception:
            return None

    return None


def native_compute_masked_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    peak: float = 1.0,
    mask_threshold: float = 0.5,
) -> float | None:
    """Compute Masked PSNR using aether_metrics.compute_masked_psnr if available."""
    mod = get_aether_metrics()
    if mod is None:
        return None

    p = np.asarray(pred, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)

    if p.shape != g.shape or p.shape[:2] != m.shape[:2]:
        return None

    if (p.ndim in (2, 3)) and (m.ndim in (2, 3)):
        try:
            m_2d = m.squeeze()
            if m_2d.ndim != 2:
                return None
            return float(
                mod.compute_masked_psnr(
                    p, g, m_2d, mask_threshold=float(mask_threshold), peak=float(peak)
                )
            )
        except Exception:
            return None

    return None


def native_compute_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    zero_pad: bool = False,
    dynamic_range: float = 1.0,
    auto_hdr: bool = False,
) -> float | None:
    """Compute SSIM using aether_metrics.compute_ssim if available."""
    mod = get_aether_metrics()
    if mod is None:
        return None

    p = np.asarray(pred, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)

    if p.shape != g.shape:
        return None

    if p.ndim == 2 or (p.ndim == 3 and p.shape[2] in (1, 3, 4)):
        try:
            return float(
                mod.compute_ssim(
                    p,
                    g,
                    window_size=int(window_size),
                    sigma=float(sigma),
                    zero_pad=bool(zero_pad),
                    dynamic_range=float(dynamic_range),
                    auto_hdr=bool(auto_hdr),
                )
            )
        except Exception:
            return None

    return None


def native_compute_ciede2000(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Compute CIEDE2000 using aether_metrics.compute_ciede2000 if available."""
    mod = get_aether_metrics()
    if mod is None:
        return None

    p = np.asarray(pred, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)

    if p.shape != g.shape or p.ndim != 3 or p.shape[2] != 3:
        return None

    try:
        return float(mod.compute_ciede2000(p, g))
    except Exception:
        return None
