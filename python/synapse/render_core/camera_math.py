"""Unified camera mathematics for Synapse.

Canonical implementations of look-at view matrices, quaternion conversion,
projection builders, and camera pose ↔ view matrix roundtrip.  All consumers
(``camera.py``, ``align.py``, test helpers) should import from here.

Convention: right-handed, Y-up, camera looks along −Z.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def look_at_view_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Compute a 4×4 world-to-camera (view) matrix.

    Right-handed, Y-up, camera looks along −Z.

    Includes a zenith/nadir singularity guard: when *forward* is nearly
    parallel to *up*, a fallback up-vector is used so the cross product
    remains well-conditioned.

    Args:
        eye: Camera position in world space ``[3]``.
        target: Look-at target in world space ``[3]``.
        up: World up vector ``[3]``.  Defaults to ``[0, 1, 0]``.
        strict: If ``True`` (default), raise on degenerate input.
            If ``False``, return identity on degenerate input (useful
            in optimizer loops where exceptions crash the solver).

    Returns:
        ``[4, 4]`` view matrix.

    Raises:
        ContractValidationError: When ``strict=True`` and rotation is
            non-orthonormal (det ≠ 1).
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    f_len = float(np.linalg.norm(forward))
    if f_len < 1e-12:
        if not strict:
            return np.eye(4, dtype=np.float64)
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            "look_at_view_matrix: eye and target are coincident",
            agent_fix_hint="Ensure eye and target positions differ.",
        )
    forward = forward / f_len

    up_norm = np.asarray(up, dtype=np.float64)
    up_norm = up_norm / (float(np.linalg.norm(up_norm)) + 1e-12)

    # Zenith / nadir singularity guard
    dot_up = abs(float(np.dot(forward, up_norm)))
    if dot_up > 0.9999:
        up_fallback = np.array(
            [0.0, 0.0, -1.0 if forward[1] > 0 else 1.0],
            dtype=np.float64,
        )
        right = np.cross(forward, up_fallback)
        r_len = float(np.linalg.norm(right))
        if r_len < 1e-12:
            # Even the fallback failed — try X axis
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            right = np.cross(forward, alt_up)
            r_len = float(np.linalg.norm(right))
            if r_len < 1e-12:
                if not strict:
                    return np.eye(4, dtype=np.float64)
                from synapse.errors import ContractValidationError

                raise ContractValidationError(
                    "look_at_view_matrix: cannot construct orthonormal basis",
                    agent_fix_hint="Verify eye, target, and up vectors are linearly independent.",
                )
    else:
        right = np.cross(forward, up_norm)

    right = right / (float(np.linalg.norm(right)) + 1e-12)
    new_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = new_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ np.asarray(eye, dtype=np.float64)

    # Orthonormality check
    det = float(np.linalg.det(view[:3, :3]))
    if abs(det - 1.0) >= 1e-4:
        if not strict:
            return np.eye(4, dtype=np.float64)
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            f"look_at_view_matrix produced non-orthonormal rotation (det={det})",
            agent_fix_hint="Verify eye, target, and up vectors are linearly independent.",
        )
    return view


def look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Compute a 3×3 rotation matrix from a look-at specification.

    Convenience wrapper extracting the upper-left 3×3 from
    :func:`look_at_view_matrix`.

    Args:
        eye: Camera position ``[3]``.
        target: Look-at target ``[3]``.
        up: World up vector ``[3]``.  Defaults to ``[0, 1, 0]``.
        strict: Raise on degenerate input (default ``False`` for
            backward compatibility with optimiser usage).

    Returns:
        ``[3, 3]`` rotation matrix (row-major: right, up, −forward).
    """
    return look_at_view_matrix(eye, target, up, strict=strict)[:3, :3]


def quat_to_matrix(q: list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to a 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_projection(
    fov_deg: float,
    width: int,
    height: int,
    z_near: float = 0.1,
    z_far: float = 1000.0,
) -> np.ndarray:
    """Build a 4×4 OpenGL-convention perspective projection matrix.

    Uses ``[-1, 1]`` depth range (compatible with software rasterizer NDC).
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


def camera_pose_to_view_matrix(pose: Any) -> np.ndarray:
    """Convert a ``CameraPose`` to a 4×4 view (world-to-camera) matrix.

    Accepts 9-element (column-major 3×3) or 4-element (quaternion
    ``[x, y, z, w]``) rotation representations.
    """
    pos = np.array(pose.position, dtype=np.float64)

    rot_vals = pose.rotation
    if len(rot_vals) == 9:
        # Column-major 3×3 → row-major by transposing
        rot_col = np.array(rot_vals, dtype=np.float64).reshape(3, 3)
        rot = rot_col.T
    elif len(rot_vals) == 4:
        rot = quat_to_matrix(rot_vals)
    else:
        rot = np.eye(3, dtype=np.float64)

    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = rot
    view[:3, 3] = -rot @ pos
    return view


def view_matrix_to_camera_pose(
    view_matrix: np.ndarray,
    fov_deg: float,
) -> Any:
    """Convert a 4×4 view matrix back to a ``CameraPose``.

    Lazy-imports ``CameraPose`` to avoid circular dependency.
    """
    from synapse.inspect.schemas import CameraPose

    view = np.asarray(view_matrix, dtype=np.float64)
    rot = view[:3, :3]
    t = view[:3, 3]

    pos = (-rot.T @ t).tolist()
    rot_col = rot.T.flatten().tolist()

    return CameraPose(
        position=pos,
        rotation=rot_col,
        fov_deg=fov_deg,
    )
