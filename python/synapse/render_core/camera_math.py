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


def _np_check_vec3(vec: np.ndarray, name: str) -> np.ndarray:
    """Validate finite 3-element float64 vector matching aether_geo."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ContractValidationError(
            f"{name} must contain 3 elements, got {arr.size}.",
            agent_fix_hint=f"Pass a finite 3-element vector for {name}.",
        )
    if not bool(np.all(np.isfinite(arr))):
        raise ContractValidationError(
            f"{name} must contain finite coordinates.",
            agent_fix_hint=f"Remove NaN/Inf values from {name}.",
        )
    return arr


def _np_compute_look_at_basis(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute orthonormal (right, cam_up, forward, coincident) matching aether_geo."""
    eye_v = _np_check_vec3(eye, "eye")
    target_v = _np_check_vec3(target, "target")
    up_v = _np_check_vec3(up, "up")

    diff = target_v - eye_v
    fwd_len = float(np.linalg.norm(diff))
    if fwd_len < 1e-12:
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            True,
        )
    forward = diff / fwd_len

    up_len = float(np.linalg.norm(up_v))
    up_norm = (up_v / up_len) if up_len > 1e-12 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(forward, up_norm)
    right_len = float(np.linalg.norm(right))
    if abs(float(np.dot(forward, up_norm))) > 0.9999 or right_len < 1e-12:
        if abs(float(forward[1])) < 0.9999:
            alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            alt_up = np.array([0.0, 0.0, -1.0 if float(forward[1]) > 0.0 else 1.0], dtype=np.float64)
        right = np.cross(forward, alt_up)
        right_len = float(np.linalg.norm(right))
        if right_len < 1e-12:
            right = np.cross(forward, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            right_len = float(np.linalg.norm(right))
    right = (right / right_len) if right_len > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cam_up = np.cross(right, forward)
    cam_up_len = float(np.linalg.norm(cam_up))
    if cam_up_len > 1e-12:
        cam_up = cam_up / cam_up_len
    return right, cam_up, forward, False


def _np_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Pure NumPy right-handed look-at view matrix matching aether_geo.look_at_rh."""
    right, cam_up, forward, coincident = _np_compute_look_at_basis(eye, target, up)
    if coincident:
        raise ContractValidationError(
            "look_at_rh: eye and target are coincident (zero-length forward vector).",
            agent_fix_hint="Ensure eye != target and up is valid when calling look_at().",
        )

    eye_v = np.asarray(eye, dtype=np.float64).reshape(3)
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = right
    mat[1, :3] = cam_up
    mat[2, :3] = -forward
    mat[0, 3] = -float(np.dot(right, eye_v))
    mat[1, 3] = -float(np.dot(cam_up, eye_v))
    mat[2, 3] = float(np.dot(forward, eye_v))
    return mat


def _np_look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Pure NumPy 3x3 look-at rotation matrix [right; up; -forward] matching aether_geo."""
    right, cam_up, forward, coincident = _np_compute_look_at_basis(eye, target, up)
    if coincident:
        return np.eye(3, dtype=np.float64)
    rot = np.empty((3, 3), dtype=np.float64)
    rot[0, :] = right
    rot[1, :] = cam_up
    rot[2, :] = -forward
    return rot


def _np_perspective_projection(
    fov_rad: float,
    aspect: float,
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """Pure NumPy symmetric perspective matrix (OpenGL NDC [-1, 1]) matching aether_geo."""
    if (
        not math.isfinite(fov_rad)
        or not math.isfinite(aspect)
        or not math.isfinite(z_near)
        or not math.isfinite(z_far)
        or fov_rad <= 0.0
        or fov_rad >= math.pi
        or aspect <= 0.0
        or z_near <= 0.0
        or z_far <= z_near
    ):
        raise ContractValidationError(
            "perspective_rh_no requires finite 0 < fov_y_rad < pi, aspect > 0, 0 < near_z < far_z.",
            agent_fix_hint="Check fov_deg in (0, 180), width/height > 0, and 0 < z_near < z_far.",
        )
    f = 1.0 / math.tan(fov_rad * 0.5)
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = -(z_far + z_near) / (z_far - z_near)
    mat[2, 3] = -(2.0 * z_far * z_near) / (z_far - z_near)
    mat[3, 2] = -1.0
    return mat


def _np_quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Pure NumPy quaternion [x, y, z, w] to 3x3 rotation matrix matching aether_geo."""
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if q_arr.size != 4:
        raise ContractValidationError(
            "quat_to_matrix expects 4-element quaternion [x, y, z, w].",
            agent_fix_hint="Pass a 4-element [x, y, z, w] quaternion.",
        )
    if not bool(np.all(np.isfinite(q_arr))):
        raise ContractValidationError(
            "quat_to_matrix expects finite quaternion components.",
            agent_fix_hint="Remove NaN/Inf values from the quaternion.",
        )
    norm = float(np.linalg.norm(q_arr))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    inv = 1.0 / norm
    x, y, z, w = float(q_arr[0] * inv), float(q_arr[1] * inv), float(q_arr[2] * inv), float(q_arr[3] * inv)
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
    """Pure NumPy pose (position, rotation) to 4x4 view matrix matching aether_geo.

    rotation can be (3, 3) row-major matrix, [9] column-major 3x3, or [4] quaternion xyzw.
    """
    pos_v = _np_check_vec3(pos, "position")
    rot_arr = np.asarray(rot, dtype=np.float64)
    if rot_arr.ndim == 2 and rot_arr.shape == (3, 3):
        if not bool(np.all(np.isfinite(rot_arr))):
            raise ContractValidationError(
                "view_matrix_from_pose rotation matrix must be finite.",
                agent_fix_hint="Remove NaN/Inf values from rotation.",
            )
        r = rot_arr
    elif rot_arr.size == 9:
        if not bool(np.all(np.isfinite(rot_arr))):
            raise ContractValidationError(
                "view_matrix_from_pose rotation matrix must be finite.",
                agent_fix_hint="Remove NaN/Inf values from rotation.",
            )
        r = rot_arr.reshape(3, 3, order="F")  # column-major -> row-major
    elif rot_arr.size == 4:
        r = _np_quat_to_matrix(rot_arr)
    else:
        raise ContractValidationError(
            f"rotation must be (3, 3), [4] (quat xyzw), or [9] (col-major 3x3), got {rot_arr.shape}",
            agent_fix_hint="Pass rotation as (3, 3), [4] quaternion, or [9] column-major rotation.",
        )
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = r
    view[:3, 3] = -(r @ pos_v)
    return view


def _np_pose_from_view_matrix(
    view_matrix: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Pure NumPy extract (position [3], column-major rotation [9]) from 4x4 view matching aether_geo."""
    v = np.asarray(view_matrix, dtype=np.float64)
    if v.ndim != 2 or v.shape != (4, 4):
        raise ContractValidationError(
            f"pose_from_view_matrix expects shape (4, 4), got {v.shape}.",
            agent_fix_hint="Pass a finite (4, 4) view matrix.",
        )
    if not bool(np.all(np.isfinite(v))):
        raise ContractValidationError(
            "pose_from_view_matrix expects finite matrix entries.",
            agent_fix_hint="Remove NaN/Inf values from the view matrix.",
        )
    r = v[:3, :3]
    pos = -(r.T @ v[:3, 3])
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
        try:
            return np.asarray(geo.look_at_rotation(eye_arr, target_arr, up_arr), dtype=np.float64)
        except ValueError as exc:
            raise ContractValidationError(
                str(exc),
                agent_fix_hint="Ensure eye, target, and up are finite 3-element vectors.",
            ) from exc

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
        try:
            return np.asarray(
                geo.perspective_rh_no(fov_rad, aspect, float(z_near), float(z_far)),
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ContractValidationError(
                str(exc),
                agent_fix_hint="Check fov_deg in (0, 180), width/height > 0, and 0 < z_near < z_far.",
            ) from exc

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
        try:
            return np.asarray(geo.view_matrix_from_pose(pos_arr, rot_arr), dtype=np.float64)
        except ValueError as exc:
            raise ContractValidationError(
                str(exc),
                agent_fix_hint="Pass finite position [3] and rotation (3x3, [9] col-major, or [4] quat).",
            ) from exc

    return _np_view_matrix_from_pose(pos_arr, rot_arr)


def pose_from_view_matrix(
    view_matrix: np.ndarray,
    *,
    backend: Literal["auto", "native"] = "auto",
) -> tuple[list[float], list[float]]:
    """Extract (position [3], column-major rotation [9]) from a 4x4 view matrix."""
    _require_native(backend)
    v_arr = np.asarray(view_matrix, dtype=np.float64)

    geo = get_aether_geo()
    if geo is not None:
        try:
            pos, rot_col = geo.pose_from_view_matrix(v_arr)
            return [float(v) for v in pos], [float(v) for v in rot_col]
        except ValueError as exc:
            raise ContractValidationError(
                str(exc),
                agent_fix_hint="Pass a finite (4, 4) view matrix.",
            ) from exc

    return _np_pose_from_view_matrix(v_arr)


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
        try:
            return np.asarray(geo.quat_to_matrix(q_arr), dtype=np.float64)
        except ValueError as exc:
            raise ContractValidationError(
                str(exc),
                agent_fix_hint="Pass a finite 4-element [x, y, z, w] quaternion.",
            ) from exc

    return _np_quat_to_matrix(q_arr)


# Backward-compatible alias
_quat_to_matrix = quat_to_matrix
