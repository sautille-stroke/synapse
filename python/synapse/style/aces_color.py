"""ACEScg color science utilities — upstream-first with pure NumPy fallback.

When native bindings from ``@aether//color`` and ``@aether//metrics``
are available, all colour-space conversions and CIEDE2000 delegate to the
SIMD-accelerated implementation. Otherwise the module falls back to the
local pure-NumPy implementations without emitting runtime warnings.

See: https://github.com/sautille-stroke/synapse/issues/34
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "linear_srgb_to_acescg",
    "acescg_to_linear_srgb",
    "soft_knee_inverse_tonemap",
    "rgb_to_cielab",
    "ciede2000_delta_e",
]

# ---------------------------------------------------------------------------
# Upstream native extension probe (@aether//color)
# ---------------------------------------------------------------------------

try:
    from aether.color import (  # type: ignore[import-not-found]
        acescg_to_linear_srgb as _native_acescg_to_linear_srgb,
    )
    from aether.color import (
        ciede2000_delta_e as _native_ciede2000_delta_e,
    )
    from aether.color import (
        linear_srgb_to_acescg as _native_linear_srgb_to_acescg,
    )
    from aether.color import (
        linear_to_srgb as _native_linear_to_srgb,
    )
    from aether.color import (
        rgb_to_cielab as _native_rgb_to_cielab,
    )
    from aether.color import (
        srgb_to_linear as _native_srgb_to_linear,
    )

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False
    _native_srgb_to_linear = None
    _native_linear_to_srgb = None
    _native_linear_srgb_to_acescg = None
    _native_acescg_to_linear_srgb = None
    _native_rgb_to_cielab = None
    _native_ciede2000_delta_e = None


def _warn_fallback() -> None:
    """Fallback notification helper (silent to avoid runtime warning noise)."""
    pass


# ===================================================================
#  Pure-NumPy fallback implementations
#  (preserved verbatim for environments without @aether//color native binding)
# ===================================================================

# --- sRGB ↔ Linear (IEC 61966-2-1 piecewise) ----------------------

_SRGB_THRESHOLD: float = 0.04045
_SRGB_LINEAR_SLOPE: float = 12.92
_SRGB_GAMMA: float = 2.4
_SRGB_A: float = 0.055


def _fallback_srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    s = np.asarray(srgb, dtype=np.float64)
    low = s / _SRGB_LINEAR_SLOPE
    high = ((s + _SRGB_A) / (1.0 + _SRGB_A)) ** _SRGB_GAMMA
    return np.where(s <= _SRGB_THRESHOLD, low, high)


def _fallback_linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    lin = np.asarray(linear, dtype=np.float64)
    lin_clipped = np.clip(lin, 0.0, None)
    threshold_linear = _SRGB_THRESHOLD / _SRGB_LINEAR_SLOPE
    low = lin_clipped * _SRGB_LINEAR_SLOPE
    high = (1.0 + _SRGB_A) * np.power(lin_clipped, 1.0 / _SRGB_GAMMA) - _SRGB_A
    return np.where(lin_clipped <= threshold_linear, low, high)


# --- Linear sRGB ↔ ACEScg (Bradford D65→D60 AP1) ------------------

_M_SRGB_TO_ACESCG: np.ndarray = np.array([
    [0.6131324, 0.3395380, 0.0474166],
    [0.0701243, 0.9163940, 0.0134853],
    [0.0206155, 0.1095698, 0.8697794],
], dtype=np.float64)

_M_ACESCG_TO_SRGB: np.ndarray = np.linalg.inv(_M_SRGB_TO_ACESCG)


def _fallback_linear_srgb_to_acescg(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return arr @ _M_SRGB_TO_ACESCG.T


def _fallback_acescg_to_linear_srgb(acescg: np.ndarray) -> np.ndarray:
    arr = np.asarray(acescg, dtype=np.float64)
    return arr @ _M_ACESCG_TO_SRGB.T


# --- Soft-knee inverse tonemapping ---------------------------------

_SOFT_KNEE_THRESHOLD: float = 4.0
_SOFT_KNEE_RANGE: float = 8.0


def _fallback_soft_knee_inverse_tonemap(ldr_linear: np.ndarray) -> np.ndarray:
    x = np.asarray(ldr_linear, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    excess = (x - _SOFT_KNEE_THRESHOLD) / _SOFT_KNEE_RANGE
    high = _SOFT_KNEE_THRESHOLD + _SOFT_KNEE_RANGE * np.tanh(excess)
    result = np.where(x <= _SOFT_KNEE_THRESHOLD, x, high)
    return result


# --- CIE Lab and CIEDE2000 -----------------------------------------

_M_SRGB_TO_XYZ: np.ndarray = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_D65_X: float = 0.95047
_D65_Y: float = 1.00000
_D65_Z: float = 1.08883

_LAB_DELTA: float = 6.0 / 29.0
_LAB_DELTA_SQ: float = _LAB_DELTA ** 2
_LAB_DELTA_CU: float = _LAB_DELTA ** 3


def _lab_f(t: np.ndarray) -> np.ndarray:
    """CIE Lab forward nonlinearity."""
    return np.where(
        t > _LAB_DELTA_CU,
        np.cbrt(t),
        t / (3.0 * _LAB_DELTA_SQ) + 4.0 / 29.0,
    )


def _fallback_rgb_to_cielab(rgb_linear: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_linear, dtype=np.float64)
    xyz = arr @ _M_SRGB_TO_XYZ.T
    xn = xyz[..., 0] / _D65_X
    yn = xyz[..., 1] / _D65_Y
    zn = xyz[..., 2] / _D65_Z
    fx = _lab_f(xn)
    fy = _lab_f(yn)
    fz = _lab_f(zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _fallback_ciede2000_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Full CIEDE2000 ΔE₀₀ including S_L, S_C, S_H weighting and R_T."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1_ab = np.sqrt(a1 ** 2 + b1 ** 2)
    C2_ab = np.sqrt(a2 ** 2 + b2 ** 2)
    C_ab_mean = (C1_ab + C2_ab) / 2.0

    C_ab_mean_7 = C_ab_mean ** 7
    _25_7 = 25.0 ** 7
    G = 0.5 * (1.0 - np.sqrt(C_ab_mean_7 / (C_ab_mean_7 + _25_7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = np.zeros_like(h1_prime)
    product_C = C1_prime * C2_prime

    mask_zero = product_C == 0
    diff_h = h2_prime - h1_prime

    mask_le = np.abs(diff_h) <= 180.0
    mask_gt_pos = diff_h > 180.0
    mask_gt_neg = diff_h < -180.0

    dh_prime = np.where(mask_zero, 0.0, dh_prime)
    dh_prime = np.where(~mask_zero & mask_le, diff_h, dh_prime)
    dh_prime = np.where(~mask_zero & mask_gt_pos, diff_h - 360.0, dh_prime)
    dh_prime = np.where(~mask_zero & mask_gt_neg, diff_h + 360.0, dh_prime)

    dH_prime = 2.0 * np.sqrt(product_C) * np.sin(np.radians(dh_prime / 2.0))

    L_prime_mean = (L1 + L2) / 2.0
    C_prime_mean = (C1_prime + C2_prime) / 2.0

    h_prime_sum = h1_prime + h2_prime
    h_prime_abs_diff = np.abs(h1_prime - h2_prime)

    h_prime_mean = np.zeros_like(h1_prime)
    h_prime_mean = np.where(mask_zero, h_prime_sum, h_prime_mean)
    h_prime_mean = np.where(
        ~mask_zero & (h_prime_abs_diff <= 180.0),
        h_prime_sum / 2.0,
        h_prime_mean,
    )
    h_prime_mean = np.where(
        ~mask_zero & (h_prime_abs_diff > 180.0) & (h_prime_sum < 360.0),
        (h_prime_sum + 360.0) / 2.0,
        h_prime_mean,
    )
    h_prime_mean = np.where(
        ~mask_zero & (h_prime_abs_diff > 180.0) & (h_prime_sum >= 360.0),
        (h_prime_sum - 360.0) / 2.0,
        h_prime_mean,
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_prime_mean - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_prime_mean))
        + 0.32 * np.cos(np.radians(3.0 * h_prime_mean + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_prime_mean - 63.0))
    )

    L_prime_mean_50_sq = (L_prime_mean - 50.0) ** 2
    S_L = 1.0 + 0.015 * L_prime_mean_50_sq / np.sqrt(20.0 + L_prime_mean_50_sq)
    S_C = 1.0 + 0.045 * C_prime_mean
    S_H = 1.0 + 0.015 * C_prime_mean * T

    C_prime_mean_7 = C_prime_mean ** 7
    R_C = 2.0 * np.sqrt(C_prime_mean_7 / (C_prime_mean_7 + _25_7))
    d_theta = 30.0 * np.exp(-((h_prime_mean - 275.0) / 25.0) ** 2)
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    term_L = dL_prime / S_L
    term_C = dC_prime / S_C
    term_H = dH_prime / S_H

    dE = np.sqrt(
        term_L ** 2
        + term_C ** 2
        + term_H ** 2
        + R_T * term_C * term_H
    )
    return dE


# ===================================================================
#  Public API — delegates to native when available
# ===================================================================


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values to linear RGB using IEC 61966-2-1 piecewise.

    Args:
        srgb: Input array in [0, 1] sRGB space.

    Returns:
        Linear RGB array with the same shape.
    """
    if _HAS_NATIVE:
        return _native_srgb_to_linear(srgb)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_srgb_to_linear(srgb)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB using IEC 61966-2-1 piecewise inverse.

    Args:
        linear: Input array in linear space.

    Returns:
        sRGB array with the same shape.
    """
    if _HAS_NATIVE:
        return _native_linear_to_srgb(linear)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_linear_to_srgb(linear)


def linear_srgb_to_acescg(rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB (D65 Rec.709) to ACEScg (D60 AP1).

    Args:
        rgb: Linear sRGB array with shape (..., 3).

    Returns:
        ACEScg array with the same shape.
    """
    if _HAS_NATIVE:
        return _native_linear_srgb_to_acescg(rgb)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_linear_srgb_to_acescg(rgb)


def acescg_to_linear_srgb(acescg: np.ndarray) -> np.ndarray:
    """Convert ACEScg (D60 AP1) to linear sRGB (D65 Rec.709).

    Args:
        acescg: ACEScg array with shape (..., 3).

    Returns:
        Linear sRGB array with the same shape.
    """
    if _HAS_NATIVE:
        return _native_acescg_to_linear_srgb(acescg)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_acescg_to_linear_srgb(acescg)


def soft_knee_inverse_tonemap(ldr_linear: np.ndarray) -> np.ndarray:
    """Apply soft-knee inverse tonemapping.

    - When L <= 4.0: output = L (identity)
    - When L > 4.0: output = 4.0 + 8.0 * tanh((L - 4.0) / 8.0)
    - C1 continuous at L = 4.0, max output strictly <= 12.0, zero NaN/Inf.

    Args:
        ldr_linear: Input linear values. Any shape.

    Returns:
        Tonemapped array, same shape. Max value strictly < 12.0.
    """
    # soft_knee_inverse_tonemap has no direct upstream equivalent — always
    # use the local implementation.  No deprecation warning needed.
    return _fallback_soft_knee_inverse_tonemap(ldr_linear)


def rgb_to_cielab(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB (D65) to CIE L*a*b*.

    Args:
        rgb_linear: Linear sRGB array with shape (..., 3).

    Returns:
        CIE Lab array with shape (..., 3) where channels are [L, a, b].
    """
    if _HAS_NATIVE:
        return _native_rgb_to_cielab(rgb_linear)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_rgb_to_cielab(rgb_linear)


def ciede2000_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 colour difference ΔE₀₀.

    Full implementation including S_L, S_C, S_H weighting and the
    blue-region R_T rotation term.

    Args:
        lab1: First CIE Lab array with shape (..., 3).
        lab2: Second CIE Lab array with shape (..., 3).

    Returns:
        ΔE₀₀ array with shape (...).
    """
    if _HAS_NATIVE:
        return _native_ciede2000_delta_e(lab1, lab2)  # type: ignore[misc]
    _warn_fallback()
    return _fallback_ciede2000_delta_e(lab1, lab2)
