"""Unified camera mathematics for Synapse.

Thin delegation layer over ``aether_geo`` (C++20 double-precision SSOT, RFC 0071
PR-U22b). Preserves existing numpy array contracts and exception mapping.

Coordinate conventions:
  - Right-handed, Y-up
  - Camera looks along -Z
  - Projection uses OpenGL NDC (depth in [-1, 1])
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from synapse.bridge.aether_bridge import require_aether_geo
from synapse.errors import ContractValidationError


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a 4x4 world-to-camera (view) matrix via aether_geo.look_at_rh.

    Raises:
        ContractValidationError: If ``eye`` and ``target`` are coincident.
        BackendUnavailableError: If ``aether_geo`` is not available.
    """
    geo = require_aether_geo()
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_arr = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64)
    )
    try:
        return np.asarray(geo.look_at_rh(eye_arr, target_arr, up_arr), dtype=np.float64)
    except ValueError as exc:
        raise ContractValidationError(
            str(exc),
            agent_fix_hint="Ensure eye != target and up is valid when calling look_at().",
        ) from exc


def look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a 3x3 look-at rotation matrix [right; up; -forward] via aether_geo."""
    geo = require_aether_geo()
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_arr = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64)
    )
    return np.asarray(geo.look_at_rotation(eye_arr, target_arr, up_arr), dtype=np.float64)


def perspective_projection(
    fov_deg: float,
    width: int,
    height: int,
    z_near: float = 0.1,
    z_far: float = 1000.0,
) -> np.ndarray:
    """Build a 4x4 symmetric perspective projection matrix (OpenGL NDC [-1, 1])."""
    geo = require_aether_geo()
    fov_rad = math.radians(float(fov_deg))
    aspect = float(width) / max(float(height), 1.0)
    return np.asarray(
        geo.perspective_rh_no(fov_rad, aspect, float(z_near), float(z_far)),
        dtype=np.float64,
    )


def view_matrix_from_pose(
    position: Sequence[float] | np.ndarray,
    rotation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert position [3] and rotation ([9] col-major or [4] quat xyzw) to a 4x4 view matrix."""
    geo = require_aether_geo()
    pos_arr = np.asarray(position, dtype=np.float64)
    rot_arr = np.asarray(rotation, dtype=np.float64)
    return np.asarray(geo.view_matrix_from_pose(pos_arr, rot_arr), dtype=np.float64)


def pose_from_view_matrix(
    view_matrix: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Extract (position [3], column-major rotation [9]) from a 4x4 view matrix."""
    geo = require_aether_geo()
    pos, rot_col = geo.pose_from_view_matrix(np.asarray(view_matrix, dtype=np.float64))
    return [float(v) for v in pos], [float(v) for v in rot_col]


def quat_to_matrix(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix via aether_geo."""
    geo = require_aether_geo()
    return np.asarray(geo.quat_to_matrix(np.asarray(q, dtype=np.float64)), dtype=np.float64)


# Backward-compatible alias
_quat_to_matrix = quat_to_matrix
