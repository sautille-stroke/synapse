"""Camera trajectory generators for Synapse-Bench.

Each generator produces a sequence of :class:`~synapse.fgpp.CameraParams`
frames that define the virtual camera path used to render GT data and
to serve as FGPP guide signals.

Supported trajectories:

* **orbit** — 360° (or partial) orbit around a target
* **tracking** — Linear dolly tracking shot
* **dolly** — Push-in / pull-out along camera forward axis
* **static** — Fixed camera with optional slow drift
* **arc** — Vertical arc (elevation sweep)
* **pan** — Horizontal pan (azimuth sweep without orbiting)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from synapse.fgpp.frame import CameraParams


class CameraTrajectoryType(str, Enum):
    """Supported camera trajectory types."""

    ORBIT = "orbit"
    TRACKING = "tracking"
    DOLLY = "dolly"
    STATIC = "static"
    ARC = "arc"
    PAN = "pan"


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Compute a 4×4 world-to-camera (view) matrix.

    Right-handed, Y-up, camera looks along -Z.

    Includes a zenith/nadir singularity guard: when *forward* is nearly
    parallel to *up*, a fallback up-vector is used so the cross product
    remains well-conditioned.
    """
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    up_norm = up / (np.linalg.norm(up) + 1e-12)

    # Zenith / nadir singularity guard
    if abs(float(np.dot(forward, up_norm))) > 0.9999:
        up_fallback = np.array(
            [0.0, 0.0, -1.0 if forward[1] > 0 else 1.0], dtype=np.float64,
        )
        right = np.cross(forward, up_fallback)
    else:
        right = np.cross(forward, up_norm)

    right = right / (np.linalg.norm(right) + 1e-12)
    new_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = new_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye

    # Sanity: rotation sub-matrix must be orthonormal (det ≈ 1)
    assert abs(np.linalg.det(view[:3, :3]) - 1.0) < 1e-6, (
        f"_look_at produced non-orthonormal rotation "
        f"(det={np.linalg.det(view[:3, :3])})"
    )
    return view


def _make_camera(
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    z_near: float = 0.1,
    z_far: float = 1000.0,
    aperture_fstop: float = 0.0,
    shutter_angle_deg: float = 180.0,
) -> CameraParams:
    """Build a CameraParams from eye/target positions."""
    view = _look_at(eye, target)
    world = np.linalg.inv(view)

    # Intrinsics from vertical FOV
    fy = (height / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fx = fy  # Square pixels
    cx = width / 2.0
    cy = height / 2.0

    return CameraParams(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        fov_degrees=fov_deg,
        view_matrix=view,
        world_matrix=world,
        z_near=z_near,
        z_far=z_far,
        aperture_fstop=aperture_fstop,
        shutter_angle_deg=shutter_angle_deg,
    )


def _generate_orbit(
    num_frames: int = 120,
    radius: float = 3.0,
    elevation_deg: float = 15.0,
    azimuth_start_deg: float = 0.0,
    azimuth_range_deg: float = 360.0,
    target: Optional[list[float]] = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    endpoint: bool | None = None,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate an orbital camera trajectory around a target point.

    Parameters
    ----------
    endpoint:
        If ``True``, the last frame revisits the start azimuth (inclusive
        endpoint). If ``False``, frames are spaced so that concatenating
        the sequence would produce a seamless loop. When ``None``
        (default), ``endpoint`` is inferred: it is ``False`` for a full
        360° sweep and ``True`` otherwise.
    """
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    el_rad = math.radians(elevation_deg)

    # Auto-detect endpoint: full 360° loops should NOT include endpoint
    if endpoint is None:
        endpoint = abs(azimuth_range_deg) < 360.0

    cameras: list[CameraParams] = []

    for i in range(num_frames):
        if endpoint:
            t = i / max(num_frames - 1, 1)
        else:
            t = i / max(num_frames, 1)
        az_deg = azimuth_start_deg + t * azimuth_range_deg
        az_rad = math.radians(az_deg)

        x = radius * math.cos(el_rad) * math.sin(az_rad)
        y = radius * math.sin(el_rad)
        z = radius * math.cos(el_rad) * math.cos(az_rad)

        eye = target_pt + np.array([x, y, z])
        cameras.append(
            _make_camera(eye, target_pt, width, height, fov_deg)
        )

    return cameras


def _generate_tracking(
    num_frames: int = 120,
    start: Optional[list[float]] = None,
    end: Optional[list[float]] = None,
    target: Optional[list[float]] = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a linear tracking shot (camera moves, looks at target)."""
    start_pt = np.array(start or [-3.0, 1.5, 0.0], dtype=np.float64)
    end_pt = np.array(end or [3.0, 1.5, 0.0], dtype=np.float64)
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        eye = start_pt + t * (end_pt - start_pt)
        cameras.append(
            _make_camera(eye, target_pt, width, height, fov_deg)
        )

    return cameras


def _generate_dolly(
    num_frames: int = 120,
    start_distance: float = 5.0,
    end_distance: float = 1.5,
    elevation_deg: float = 15.0,
    azimuth_deg: float = 0.0,
    target: Optional[list[float]] = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a dolly push-in / pull-out shot."""
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    el_rad = math.radians(elevation_deg)
    az_rad = math.radians(azimuth_deg)
    cameras: list[CameraParams] = []

    direction = np.array([
        math.cos(el_rad) * math.sin(az_rad),
        math.sin(el_rad),
        math.cos(el_rad) * math.cos(az_rad),
    ])

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        dist = start_distance + t * (end_distance - start_distance)
        eye = target_pt + direction * dist
        cameras.append(
            _make_camera(eye, target_pt, width, height, fov_deg)
        )

    return cameras


def _generate_static(
    num_frames: int = 120,
    position: Optional[list[float]] = None,
    target: Optional[list[float]] = None,
    drift_amplitude: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a static (or slow-drift) camera."""
    pos = np.array(position or [0.0, 1.5, 3.0], dtype=np.float64)
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        drift = np.array([
            drift_amplitude * math.sin(2 * math.pi * t),
            drift_amplitude * 0.3 * math.sin(4 * math.pi * t),
            0.0,
        ])
        eye = pos + drift
        cameras.append(
            _make_camera(eye, target_pt, width, height, fov_deg)
        )

    return cameras


def _generate_arc(
    num_frames: int = 120,
    radius: float = 3.0,
    azimuth_deg: float = 0.0,
    elevation_start_deg: float = -15.0,
    elevation_end_deg: float = 75.0,
    target: Optional[list[float]] = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a vertical arc (elevation sweep at fixed azimuth)."""
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    az_rad = math.radians(azimuth_deg)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        el_deg = elevation_start_deg + t * (elevation_end_deg - elevation_start_deg)
        el_rad = math.radians(el_deg)

        x = radius * math.cos(el_rad) * math.sin(az_rad)
        y = radius * math.sin(el_rad)
        z = radius * math.cos(el_rad) * math.cos(az_rad)

        eye = target_pt + np.array([x, y, z])
        cameras.append(
            _make_camera(eye, target_pt, width, height, fov_deg)
        )

    return cameras


def _generate_pan(
    num_frames: int = 120,
    position: Optional[list[float]] = None,
    pan_start_deg: float = -30.0,
    pan_end_deg: float = 30.0,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a horizontal pan (camera rotates in place)."""
    pos = np.array(position or [0.0, 1.5, 3.0], dtype=np.float64)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        pan_deg = pan_start_deg + t * (pan_end_deg - pan_start_deg)
        pan_rad = math.radians(pan_deg)

        target = pos + np.array([
            math.sin(pan_rad) * 5.0,
            0.0,
            -math.cos(pan_rad) * 5.0,
        ])
        cameras.append(
            _make_camera(pos, target, width, height, fov_deg)
        )

    return cameras


_GENERATORS = {
    CameraTrajectoryType.ORBIT: _generate_orbit,
    CameraTrajectoryType.TRACKING: _generate_tracking,
    CameraTrajectoryType.DOLLY: _generate_dolly,
    CameraTrajectoryType.STATIC: _generate_static,
    CameraTrajectoryType.ARC: _generate_arc,
    CameraTrajectoryType.PAN: _generate_pan,
}


def generate_cameras(
    scenario_or_type: "Scenario | CameraTrajectoryType | str",
    **override_params: object,
) -> list[CameraParams]:
    """Generate camera trajectory for a scenario or trajectory type."""
    if isinstance(scenario_or_type, str):
        traj_type = CameraTrajectoryType(scenario_or_type)
        params = dict(override_params)
    elif isinstance(scenario_or_type, CameraTrajectoryType):
        traj_type = scenario_or_type
        params = dict(override_params)
    else:
        scenario = scenario_or_type
        traj_type = CameraTrajectoryType(scenario.camera_type)
        params = dict(scenario.camera_params)
        params["width"] = scenario.width
        params["height"] = scenario.height
        params.update(override_params)

    generator = _GENERATORS.get(traj_type)
    if generator is None:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return generator(**params)
