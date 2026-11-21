"""Unified camera mathematics for Synapse.

Thin delegation layer over ``aether_geo`` (C++20 double-precision SSOT, RFC 0071
PR-U22b) with pure NumPy fallback when native extension is not available.

Coordinate conventions:
  - Right-handed, Y-up
  - Camera looks along -Z
  - Projection uses OpenGL NDC (depth in [-1, 1])
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np

from synapse.bridge.aether_bridge import get_aether_geo
from synapse.errors import ContractValidationError


def _require_native(backend: str) -> None:
    """Raise BackendUnavailableError when ``backend="native"`` but aether_geo is missing."""
    if backend == "native" and get_aether_geo() is None:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "aether_geo C++20 extension is not available and backend='native' was explicitly requested",
            agent_fix_hint="Install aether_geo or run: python scripts/build_native.py",
        )


# ---------------------------------------------------------------------------
# Pure NumPy fallback implementations
# ---------------------------------------------------------------------------


def _np_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Pure NumPy right-handed look-at view matrix."""
    forward = target - eye
    fwd_len = float(np.linalg.norm(forward))
    if fwd_len < 1e-12:
        raise ContractValidationError(
            "eye and target are coincident",
            agent_fix_hint="Ensure eye != target and up is valid when calling look_at().",
        )
    forward = forward / fwd_len
    right = np.cross(forward, up)
    right_len = float(np.linalg.norm(right))
    if right_len < 1e-12:
        raise ContractValidationError(
            "forward and up vectors are parallel",
            agent_fix_hint="Ensure eye != target and up is valid when calling look_at().",
        )
    right = right / right_len
    new_up = np.cross(right, forward)

    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = right
    mat[1, :3] = new_up
    mat[2, :3] = -forward
    mat[0, 3] = -np.dot(right, eye)
    mat[1, 3] = -np.dot(new_up, eye)
    mat[2, 3] = np.dot(forward, eye)
    return mat


def _np_look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Pure NumPy 3x3 look-at rotation matrix [right; up; -forward]."""
    forward = target - eye
    fwd_len = float(np.linalg.norm(forward))
    if fwd_len < 1e-12:
        raise ContractValidationError(
            "eye and target are coincident",
            agent_fix_hint="Ensure eye != target and up is valid when calling look_at_rotation().",
        )
    forward = forward / fwd_len
    right = np.cross(forward, up)
    right_len = float(np.linalg.norm(right))
    if right_len < 1e-12:
        raise ContractValidationError(
            "forward and up vectors are parallel",
            agent_fix_hint="Ensure eye != target and up is valid when calling look_at_rotation().",
        )
    right = right / right_len
    new_up = np.cross(right, forward)
    rot = np.empty((3, 3), dtype=np.float64)
    rot[0, :] = right
    rot[1, :] = new_up
    rot[2, :] = -forward
    return rot


def _np_perspective_projection(
    fov_rad: float,
    aspect: float,
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """Pure NumPy symmetric perspective matrix (OpenGL NDC [-1, 1])."""
    f = 1.0 / math.tan(fov_rad / 2.0)
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (z_far + z_near) / (z_near - z_far)
    mat[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    mat[3, 2] = -1.0
    return mat


def _np_quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Pure NumPy quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    mat = np.empty((3, 3), dtype=np.float64)
    mat[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mat[0, 1] = 2.0 * (x * y - z * w)
    mat[0, 2] = 2.0 * (x * z + y * w)
    mat[1, 0] = 2.0 * (x * y + z * w)
    mat[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mat[1, 2] = 2.0 * (y * z - x * w)
    mat[2, 0] = 2.0 * (x * z - y * w)
    mat[2, 1] = 2.0 * (y * z + x * w)
    mat[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mat


def _np_view_matrix_from_pose(
    pos: np.ndarray,
    rot: np.ndarray,
) -> np.ndarray:
    """Pure NumPy pose (position, rotation) to 4x4 view matrix.

    rotation can be [9] column-major 3x3 or [4] quaternion xyzw.
    """
    if rot.size == 4:
        r = _np_quat_to_matrix(rot)
    elif rot.size == 9:
        r = rot.reshape(3, 3, order="F")  # column-major
    else:
        raise ContractValidationError(
            f"rotation must be [4] (quat xyzw) or [9] (col-major 3x3), got {rot.shape}",
            agent_fix_hint="Pass rotation as [4] quaternion or [9] column-major rotation.",
        )
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = r.T
    view[:3, 3] = -r.T @ pos
    return view


def _np_pose_from_view_matrix(
    view_matrix: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Pure NumPy extract (position [3], column-major rotation [9]) from 4x4 view."""
    v = np.asarray(view_matrix, dtype=np.float64)
    r = v[:3, :3].T  # view R^T -> world rotation
    pos = -r @ v[:3, 3]
    rot_col = r.flatten(order="F")
    return [float(x) for x in pos], [float(x) for x in rot_col]


# ---------------------------------------------------------------------------
# Public API — try native, fall back to NumPy
# ---------------------------------------------------------------------------


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> np.ndarray:
    """Compute a 4x4 world-to-camera (view) matrix.

    Raises:
        ContractValidationError: If ``eye`` and ``target`` are coincident.
        BackendUnavailableError: If ``backend="native"`` but ``aether_geo`` is not available.
    """
    _require_native(backend)
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_arr = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64)
    )

    geo = get_aether_geo()
    if geo is not None:
        try:
            return np.asarray(geo.look_at_rh(eye_arr, target_arr, up_arr), dtype=np.float64)
        except ValueError as exc:
            raise ContractValidationError(
                str(exc),
                agent_fix_hint="Ensure eye != target and up is valid when calling look_at().",
            ) from exc

    return _np_look_at(eye_arr, target_arr, up_arr)


def look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> np.ndarray:
    """Compute a 3x3 look-at rotation matrix [right; up; -forward]."""
    _require_native(backend)
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_arr = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64)
    )

    geo = get_aether_geo()
    if geo is not None:
        return np.asarray(geo.look_at_rotation(eye_arr, target_arr, up_arr), dtype=np.float64)

    return _np_look_at_rotation(eye_arr, target_arr, up_arr)


def perspective_projection(
    fov_deg: float,
    width: int,
    height: int,
    z_near: float = 0.1,
    z_far: float = 1000.0,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> np.ndarray:
    """Build a 4x4 symmetric perspective projection matrix (OpenGL NDC [-1, 1])."""
    _require_native(backend)
    fov_rad = math.radians(float(fov_deg))
    aspect = float(width) / max(float(height), 1.0)

    geo = get_aether_geo()
    if geo is not None:
        return np.asarray(
            geo.perspective_rh_no(fov_rad, aspect, float(z_near), float(z_far)),
            dtype=np.float64,
        )

    return _np_perspective_projection(fov_rad, aspect, float(z_near), float(z_far))


def view_matrix_from_pose(
    position: Sequence[float] | np.ndarray,
    rotation: Sequence[float] | np.ndarray,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> np.ndarray:
    """Convert position [3] and rotation ([9] col-major or [4] quat xyzw) to a 4x4 view matrix."""
    _require_native(backend)
    pos_arr = np.asarray(position, dtype=np.float64)
    rot_arr = np.asarray(rotation, dtype=np.float64)

    geo = get_aether_geo()
    if geo is not None:
        return np.asarray(geo.view_matrix_from_pose(pos_arr, rot_arr), dtype=np.float64)

    return _np_view_matrix_from_pose(pos_arr, rot_arr)


def pose_from_view_matrix(
    view_matrix: np.ndarray,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> tuple[list[float], list[float]]:
    """Extract (position [3], column-major rotation [9]) from a 4x4 view matrix."""
    _require_native(backend)

    geo = get_aether_geo()
    if geo is not None:
        pos, rot_col = geo.pose_from_view_matrix(np.asarray(view_matrix, dtype=np.float64))
        return [float(v) for v in pos], [float(v) for v in rot_col]

    return _np_pose_from_view_matrix(np.asarray(view_matrix, dtype=np.float64))


def quat_to_matrix(
    q: Sequence[float] | np.ndarray,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    _require_native(backend)
    q_arr = np.asarray(q, dtype=np.float64)

    geo = get_aether_geo()
    if geo is not None:
        return np.asarray(geo.quat_to_matrix(q_arr), dtype=np.float64)

    return _np_quat_to_matrix(q_arr)


# Backward-compatible alias
_quat_to_matrix = quat_to_matrix
