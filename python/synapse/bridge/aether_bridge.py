"""Bridge to @aether//metrics, @aether//color, and @aether//geo native C++20 acceleration.

Provides turnkey auto-detection for upstream Aether computational kernels:
1. Imports via ``aether_py`` umbrella or installed ``aether_*`` modules (optional ``AETHER_ROOT`` override).
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
# Upstream aether extension probe & lazy loader
# ---------------------------------------------------------------------------

_aether_metrics_mod: Any = None
_aether_color_mod: Any = None
_PROBED = False

_AETHER_PY_SUBMODULES: dict[str, str] = {
    "aether_geo": "geo",
    "aether_metrics": "metrics",
    "aether_spectra": "spectra",
    "aether_scene": "scene",
}


def _candidate_aether_python_dirs() -> list[Path]:
    """Return candidate directories from explicit ``AETHER_ROOT`` override only."""
    dirs: list[Path] = []
    if "AETHER_ROOT" in os.environ:
        cand_root = Path(os.environ["AETHER_ROOT"]).resolve()
        for sub in ("bazel-bin/python", "python"):
            cand = cand_root / sub
            if cand.is_dir() and cand not in dirs:
                dirs.append(cand)
    return dirs


def _has_extension(directory: Path, stem: str) -> bool:
    """Return True if *directory* contains a built shared library for *stem*."""
    return (
        (directory / f"{stem}.so").is_file()
        or (directory / f"{stem}.pyd").is_file()
        or any(directory.glob(f"{stem}.cpython-*.so"))
        or any(directory.glob(f"{stem}.cp*-*.pyd"))
    )


def _import_aether_extension(stem: str) -> Any:
    """Locate and import a compiled aether Python extension module."""
    if stem in sys.modules:
        return sys.modules[stem]
    sub_attr = _AETHER_PY_SUBMODULES.get(stem)
    if sub_attr is not None:
        try:
            aether_py = importlib.import_module("aether_py")
            mod = getattr(aether_py, sub_attr, None)
            if mod is not None:
                return mod
        except Exception:
            pass
    try:
        return importlib.import_module(stem)  # dag-guard: external-only
    except ImportError:
        pass
    for cand in _candidate_aether_python_dirs():
        if _has_extension(cand, stem):
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            try:
                return importlib.import_module(stem)  # dag-guard: external-only
            except ImportError:
                pass
    return None


def _find_and_import_aether_metrics() -> Any:
    """Locate and import the aether_metrics extension module."""
    global _PROBED, _aether_metrics_mod, _aether_color_mod
    if _PROBED:
        return _aether_metrics_mod

    _PROBED = True
    _aether_metrics_mod = _import_aether_extension("aether_metrics")
    if _aether_metrics_mod is not None:
        _aether_color_mod = getattr(_aether_metrics_mod, "color", None)
    return _aether_metrics_mod


def get_aether_geo() -> Any:
    """Return the loaded aether_geo extension module or None."""
    return _import_aether_extension("aether_geo")


def has_aether_geo() -> bool:
    """Return True if the aether_geo native module is accessible."""
    return get_aether_geo() is not None


def require_aether_geo() -> Any:
    """Return the loaded aether_geo module or raise BackendUnavailableError."""
    mod = get_aether_geo()
    if mod is None:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "aether_geo C++20 extension is not available",
            agent_fix_hint="Install aether_geo or run: python scripts/build_native.py",
        )
    return mod


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
