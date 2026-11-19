"""Unified camera mathematics for Synapse.

Consolidates look-at, projection, and view-matrix utilities that were
previously duplicated across ``synapse.render_core.camera`` and
``synapse.inspect.align``.

Coordinate conventions:
  - Right-handed, Y-up
  - Camera looks along −Z
  - Projection uses OpenGL NDC (depth in [−1, 1])
"""

from __future__ import annotations

import math

import numpy as np

from synapse.errors import ContractValidationError

# ---------------------------------------------------------------------------
# Look-at
# ---------------------------------------------------------------------------


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a 4×4 world-to-camera (view) matrix.

    Right-handed, Y-up, camera looks along −Z.

    Includes a zenith/nadir singularity guard: when *forward* is nearly
    parallel to *up*, a fallback up-vector is used so the cross product
    remains well-conditioned.

    Parameters
    ----------
    eye : np.ndarray
        ``[3]`` camera position in world space.
    target : np.ndarray
        ``[3]`` look-at target in world space.
    up : np.ndarray | None
        ``[3]`` up vector; defaults to ``[0, 1, 0]``.

    Returns
    -------
    np.ndarray
        ``[4, 4]`` world-to-camera view matrix.

    Raises
    ------
    ContractValidationError
        If ``eye`` and ``target`` are coincident (zero-length forward vector).
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward = target - eye
    f_len = float(np.linalg.norm(forward))
    if f_len < 1e-12:
        raise ContractValidationError(
            "look_at: eye and target are coincident (zero-length forward vector).",
            agent_fix_hint="Ensure eye != target when calling look_at().",
        )
    forward = forward / f_len

    up_norm = up / (np.linalg.norm(up) + 1e-12)

    # Zenith / nadir singularity guard
    if abs(float(np.dot(forward, up_norm))) > 0.9999:
        up_fallback = np.array(
            [0.0, 0.0, -1.0 if forward[1] > 0 else 1.0],
            dtype=np.float64,
        )
        right = np.cross(forward, up_fallback)
    else:
        right = np.cross(forward, up_norm)

    right = right / (np.linalg.norm(right) + 1e-12)
    new_up = np.cross(right, forward)
    new_up = new_up / (np.linalg.norm(new_up) + 1e-12)  # normalize

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = new_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye

    # Sanity: rotation sub-matrix must be orthonormal (det ≈ 1)
    det = float(np.linalg.det(view[:3, :3]))
    if abs(det - 1.0) >= 1e-4:
        raise ContractValidationError(
            f"look_at produced non-orthonormal rotation (det={det})",
            agent_fix_hint="Verify eye, target, and up vectors are linearly independent.",
        )
    return view


def look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a 3×3 rotation matrix for a look-at camera.

    Returns a row-major rotation where:
    - Row 0 = right
    - Row 1 = camera up
    - Row 2 = −forward (camera looks along −Z)

    Parameters
    ----------
    eye, target, up :
        Camera position, look-at target, and up vector (defaults to Y-up).

    Returns
    -------
    np.ndarray
        ``[3, 3]`` rotation matrix.
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward = target - eye
    f_len = float(np.linalg.norm(forward))
    if f_len < 1e-12:
        return np.eye(3, dtype=np.float64)
    forward = forward / f_len

    right = np.cross(forward, up)
    r_len = float(np.linalg.norm(right))
    if r_len < 1e-12:
        # Singularity: forward parallel to up — use Z-axis fallback
        # (consistent with look_at() above)
        alt_up = np.array(
            [0.0, 0.0, -1.0 if forward[1] > 0 else 1.0],
            dtype=np.float64,
        )
        right = np.cross(forward, alt_up)
        r_len = float(np.linalg.norm(right))
        if r_len < 1e-12:
            return np.eye(3, dtype=np.float64)
    right = right / r_len

    cam_up = np.cross(right, forward)
    cam_up = cam_up / float(np.linalg.norm(cam_up))

    return np.array([right, cam_up, -forward], dtype=np.float64)


# ---------------------------------------------------------------------------
# Perspective projection (OpenGL NDC, depth in [-1, 1])
# ---------------------------------------------------------------------------


def perspective_projection(
    fov_deg: float,
    width: int,
    height: int,
    z_near: float = 0.1,
    z_far: float = 1000.0,
) -> np.ndarray:
    """Build a 4×4 symmetric perspective projection matrix.

    Uses OpenGL NDC conventions with depth mapped to [−1, 1].

    Parameters
    ----------
    fov_deg : float
        Vertical field of view in degrees.
    width, height : int
        Viewport dimensions.
    z_near, z_far : float
        Near and far clip planes.

    Returns
    -------
    np.ndarray
        ``[4, 4]`` projection matrix.
    """
    fov_rad = math.radians(fov_deg)
    aspect = width / max(height, 1)
    f = 1.0 / math.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = -(z_far + z_near) / (z_far - z_near)
    proj[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
    proj[3, 2] = -1.0
    return proj


# ---------------------------------------------------------------------------
# View-matrix ↔ CameraPose conversion
# ---------------------------------------------------------------------------


def view_matrix_from_pose(
    position: list[float],
    rotation: list[float],
) -> np.ndarray:
    """Convert position + rotation arrays to a 4×4 view matrix.

    Parameters
    ----------
    position : list[float]
        ``[3]`` camera position in world space.
    rotation : list[float]
        Column-major 3×3 (9 floats) or quaternion ``[x, y, z, w]`` (4 floats).

    Returns
    -------
    np.ndarray
        ``[4, 4]`` view (world-to-camera) matrix.
    """
    pos = np.array(position, dtype=np.float64)

    if len(rotation) == 9:
        rot_col = np.array(rotation, dtype=np.float64).reshape(3, 3)
        rot = rot_col.T  # column-major → row-major
    elif len(rotation) == 4:
        rot = quat_to_matrix(rotation)
    else:
        rot = np.eye(3, dtype=np.float64)

    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = rot
    view[:3, 3] = -rot @ pos
    return view


def pose_from_view_matrix(
    view_matrix: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Extract position and column-major rotation from a 4×4 view matrix.

    Returns
    -------
    tuple of (position, rotation)
        position: ``[3]`` floats, rotation: ``[9]`` column-major floats.
    """
    view = np.asarray(view_matrix, dtype=np.float64)
    rot = view[:3, :3]
    t = view[:3, 3]

    pos = (-rot.T @ t).tolist()
    rot_col = rot.T.flatten().tolist()
    return pos, rot_col


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def quat_to_matrix(q: list[float]) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to 3×3 rotation matrix.

    Normalizes the quaternion before conversion to handle
    near-unit inputs robustly.
    """
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# Backward-compatible alias
_quat_to_matrix = quat_to_matrix
