"""Radiometric conversion utilities and VAE phase-locking helpers.

Functions
---------
- ``validate_frame_count_phase`` / ``enforce_phase_lock`` — ensure the total
  frame count satisfies the T = 4k+1 constraint required by the 3-D Causal
  VAE temporal down-sampling factor of 4.
- ``linear_to_acescg`` / ``acescg_to_linear`` — colour-space conversion between
  linear sRGB (Rec 709 primaries) and ACEScg (AP1 primaries).  Delegates to
  :mod:`synapse.style.aces_color` when available to eliminate matrix
  duplication; falls back to inline matrices otherwise.
- ``apply_kappa_reanchor`` — κ-weighted hard-cut re-anchoring at the preroll →
  production boundary.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Delegate colour-space matrices to style.aces_color to eliminate duplication.
# Falls back to inline matrices if the style module is not available.
# ---------------------------------------------------------------------------

_USE_STYLE_MODULE = False
_SRGB_TO_AP1_EXT: np.ndarray | None
_AP1_TO_SRGB_EXT: np.ndarray | None
try:
    from synapse.style.aces_color import AP1_TO_SRGB as _AP1_TO_SRGB_EXT
    from synapse.style.aces_color import SRGB_TO_AP1 as _SRGB_TO_AP1_EXT

    _USE_STYLE_MODULE = True
except ImportError:
    _SRGB_TO_AP1_EXT = None
    _AP1_TO_SRGB_EXT = None

# Inline fallback matrices (used only when style.aces_color is unavailable)
_SRGB_TO_AP1_INLINE: np.ndarray = np.array(
    [
        [0.613097, 0.339523, 0.047379],
        [0.070194, 0.916354, 0.013452],
        [0.020616, 0.109570, 0.869815],
    ],
    dtype=np.float64,
)
_AP1_TO_SRGB_INLINE: np.ndarray = np.linalg.inv(_SRGB_TO_AP1_INLINE)

# Select active matrices
_SRGB_TO_AP1: np.ndarray = _SRGB_TO_AP1_EXT if _USE_STYLE_MODULE else _SRGB_TO_AP1_INLINE
_AP1_TO_SRGB: np.ndarray = _AP1_TO_SRGB_EXT if _USE_STYLE_MODULE else _AP1_TO_SRGB_INLINE


# -- Phase-locking -----------------------------------------------------------


def validate_frame_count_phase(total_frames: int) -> bool:
    """Check that *total_frames* satisfies ``T = 4k + 1``.

    The 3-D Causal VAE uses a temporal down-sampling factor of 4 and expects
    the very first frame to be encoded independently (the "+1").

    Returns:
        ``True`` when the constraint is satisfied.
    """
    if total_frames < 1:
        return False
    return (total_frames - 1) % 4 == 0


def enforce_phase_lock(requested_frames: int) -> int:
    """Round *requested_frames* **up** to the nearest valid ``T = 4k + 1``.

    Examples:
        >>> enforce_phase_lock(96)
        97
        >>> enforce_phase_lock(10)
        13
        >>> enforce_phase_lock(4)
        5
    """
    if requested_frames < 1:
        return 1  # Minimum valid count.
    if validate_frame_count_phase(requested_frames):
        return requested_frames
    # Next valid: ceil((requested_frames - 1) / 4) * 4 + 1
    k = ((requested_frames - 1) + 3) // 4  # ceil division
    return 4 * k + 1


# -- Colour-space conversion -------------------------------------------------


def linear_to_acescg(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to ACEScg (AP1 primaries).

    Args:
        linear_rgb: Array of shape ``(..., 3)`` in linear sRGB.

    Returns:
        Array of the same shape in ACEScg colour space.
    """
    return np.asarray(
        np.einsum("...c,rc->...r", linear_rgb.astype(np.float64), _SRGB_TO_AP1).astype(
            linear_rgb.dtype
        )
    )


def acescg_to_linear(acescg: np.ndarray) -> np.ndarray:
    """Convert ACEScg (AP1 primaries) back to linear sRGB.

    Args:
        acescg: Array of shape ``(..., 3)`` in ACEScg.

    Returns:
        Array of the same shape in linear sRGB colour space.
    """
    return np.asarray(
        np.einsum("...c,rc->...r", acescg.astype(np.float64), _AP1_TO_SRGB).astype(acescg.dtype)
    )


# -- Kappa re-anchoring ------------------------------------------------------


def apply_kappa_reanchor(
    preroll_frames: np.ndarray,
    production_first_frame: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """Apply hard-cut re-anchoring at the preroll → production boundary.

    When ``kappa = 1.0`` the last preroll frame is completely replaced by
    *production_first_frame*, preventing studio-lighting contamination
    (5500 K warm shift) from leaking into the production colour grade.

    For ``0 < kappa < 1`` the result is a linear blend::

        out = (1 - κ) · preroll_last + κ · production_first

    Args:
        preroll_frames: Array of shape ``(N, H, W, 3)`` — the preroll segment.
        production_first_frame: Array of shape ``(H, W, 3)`` — the first
            production key-frame.
        kappa: Blending strength in ``[0, 1]``.  Default is ``1.0`` (full
            replacement).

    Returns:
        Modified copy of *preroll_frames* with the last frame re-anchored.
    """
    if preroll_frames.ndim != 4 or production_first_frame.ndim != 3:
        raise ValueError(
            "preroll_frames must be (N,H,W,3) and production_first_frame "
            f"must be (H,W,3); got shapes {preroll_frames.shape} and "
            f"{production_first_frame.shape}."
        )
    result = preroll_frames.copy()
    if kappa >= 1.0:
        result[-1] = production_first_frame
    elif kappa > 0.0:
        result[-1] = (1.0 - kappa) * result[-1] + kappa * production_first_frame
    return np.asarray(result)
