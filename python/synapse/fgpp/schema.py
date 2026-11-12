"""FGPP wire format schema — 15 scalar channel contract mirroring @aether//fgpp.

This module is the Python-side Single Source of Truth for the FGPP
channel layout.  Every constant here must stay bit-identical with
``aether/fgpp/include/aether/fgpp/fgpp_types.h``.  The cross-language
parity is enforced by ``tools/wire_guard.py --check``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Pixel storage type enum — mirrors aether::fgpp::PixelType
# ---------------------------------------------------------------------------


class PixelType(IntEnum):
    """EXR pixel storage types, matching ``aether::fgpp::PixelType``."""

    UINT32 = 0
    HALF = 1
    FLOAT = 2


# ---------------------------------------------------------------------------
# 15-channel spec — ASCII-sorted, (name, pixel_type, logical_group)
# ---------------------------------------------------------------------------

FGPP_CHANNEL_SPECS: Final[list[tuple[str, PixelType, str]]] = [
    ("BaseColor.B", PixelType.HALF, "base_color"),
    ("BaseColor.G", PixelType.HALF, "base_color"),
    ("BaseColor.R", PixelType.HALF, "base_color"),
    ("CameraLinearDepth.Z", PixelType.FLOAT, "depth"),
    ("Cryptomatte.ID", PixelType.UINT32, "cryptomatte_id"),
    ("DirectRadiance.B", PixelType.HALF, "direct_radiance"),
    ("DirectRadiance.G", PixelType.HALF, "direct_radiance"),
    ("DirectRadiance.R", PixelType.HALF, "direct_radiance"),
    ("MotionVector.X", PixelType.HALF, "motion_vector"),
    ("MotionVector.Y", PixelType.HALF, "motion_vector"),
    ("RoughnessMetallic.G", PixelType.HALF, "roughness_metallic"),
    ("RoughnessMetallic.R", PixelType.HALF, "roughness_metallic"),
    ("WorldNormal.X", PixelType.HALF, "world_normal"),
    ("WorldNormal.Y", PixelType.HALF, "world_normal"),
    ("WorldNormal.Z", PixelType.HALF, "world_normal"),
]

# Derived constants
FGPP_NUM_CHANNELS: Final[int] = 15
FGPP_BYTES_PER_PIXEL: Final[int] = 34  # 10×HALF(2B) + 1×FLOAT(4B) + 1×UINT32(4B)
FGPP_SCHEMA_VERSION: Final[str] = "fgpp.v1.0"

# Whitebox proxy guard — RFC 0044 §4.4
MIN_CHARACTER_WHITEBOX_VERTICES: Final[int] = 1500

# Physical anchor frame interval
DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES: Final[int] = 48

# ---------------------------------------------------------------------------
# Byte-size lookup
# ---------------------------------------------------------------------------

_PIXEL_TYPE_BYTES: Final[dict[PixelType, int]] = {
    PixelType.UINT32: 4,
    PixelType.HALF: 2,
    PixelType.FLOAT: 4,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_channel_names() -> list[str]:
    """Return the 15 ASCII-sorted FGPP channel names.

    >>> len(get_channel_names())
    15
    """
    return [name for name, _, _ in FGPP_CHANNEL_SPECS]


def get_logical_groups() -> dict[str, list[str]]:
    """Group the 15 channels by their logical group name.

    Returns a dict mapping logical group (e.g. ``"base_color"``) to the
    list of EXR channel names belonging to that group.
    """
    groups: dict[str, list[str]] = {}
    for name, _, group in FGPP_CHANNEL_SPECS:
        groups.setdefault(group, []).append(name)
    return groups


def computed_bytes_per_pixel() -> int:
    """Compute bytes-per-pixel from the channel specs (self-check)."""
    return sum(_PIXEL_TYPE_BYTES[pt] for _, pt, _ in FGPP_CHANNEL_SPECS)


# ---------------------------------------------------------------------------
# Depth conversion — log-normalised ↔ linear
# ---------------------------------------------------------------------------


def linear_to_log_normalized_depth(
    z_m: np.ndarray | float,
    z_near_m: float,
    z_far_m: float,
) -> np.ndarray:
    """Convert linear depth (metres) to log-normalised depth in [0, 1].

    Uses::

        z_norm = log(z / z_near) / log(z_far / z_near)

    Parameters
    ----------
    z_m:
        Linear depth values in metres.
    z_near_m:
        Near plane distance.  Clamped to ``1e-7`` if ``<= 0``.
    z_far_m:
        Far plane distance.  Clamped to ``z_near_m + 1e-7`` if
        ``<= z_near_m``.

    Returns
    -------
    np.ndarray
        Normalised depth in ``[0, 1]``.
    """
    eps = 1e-7
    z_near_m = max(z_near_m, eps)
    z_far_m = max(z_far_m, z_near_m + eps)

    z = np.asarray(z_m, dtype=np.float64)
    # Clamp NaN/Inf to z_far before the log
    z = np.where(np.isfinite(z), z, z_far_m)
    z = np.clip(z, z_near_m, z_far_m)

    log_ratio = np.log(z_far_m / z_near_m)
    result = np.log(z / z_near_m) / log_ratio
    return np.clip(result, 0.0, 1.0)


def log_normalized_to_linear_depth(
    z_norm: np.ndarray | float,
    z_near_m: float,
    z_far_m: float,
) -> np.ndarray:
    """Convert log-normalised depth [0, 1] back to linear depth (metres).

    Exact inverse of :func:`linear_to_log_normalized_depth`.

    Parameters
    ----------
    z_norm:
        Normalised depth values in ``[0, 1]``.
    z_near_m:
        Near plane distance.  Clamped to ``1e-7`` if ``<= 0``.
    z_far_m:
        Far plane distance.  Clamped to ``z_near_m + 1e-7`` if
        ``<= z_near_m``.

    Returns
    -------
    np.ndarray
        Linear depth in metres.
    """
    eps = 1e-7
    z_near_m = max(z_near_m, eps)
    z_far_m = max(z_far_m, z_near_m + eps)

    z_n = np.asarray(z_norm, dtype=np.float64)
    log_ratio = np.log(z_far_m / z_near_m)
    return z_near_m * np.exp(z_n * log_ratio)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Whitebox & anchor guards
# ---------------------------------------------------------------------------


def validate_whitebox_coarseness_floor(
    vertex_count: int,
    has_articulated_limbs: bool,
    is_character: bool,
) -> bool:
    """Validate whitebox proxy meets RFC 0044 §4.4 coarseness floor.

    Character assets require ``vertex_count >= 1500`` **and**
    ``has_articulated_limbs``.  Non-character assets always pass.

    Parameters
    ----------
    vertex_count:
        Number of mesh vertices.
    has_articulated_limbs:
        Whether the mesh has articulated (jointed) limbs.
    is_character:
        Whether this asset is a character proxy.

    Returns
    -------
    bool
        ``True`` if the whitebox passes the coarseness guard.
    """
    if not is_character:
        return True
    return vertex_count >= MIN_CHARACTER_WHITEBOX_VERTICES and has_articulated_limbs


def is_physical_anchor_frame(
    frame_idx: int,
    interval: int = DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
) -> bool:
    """Return ``True`` if *frame_idx* is a physical anchor frame.

    Physical anchor frames are spaced at regular intervals starting from
    frame 0 to ensure physics consistency.

    Parameters
    ----------
    frame_idx:
        Zero-based frame index.
    interval:
        Anchor spacing.  Defaults to 48.
    """
    return frame_idx % interval == 0
