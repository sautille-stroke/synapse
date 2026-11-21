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
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np

from synapse.fgpp.frame import CameraParams


class CameraTrajectoryType(StrEnum):
    """Supported camera trajectory types."""

    ORBIT = "orbit"
    TRACKING = "tracking"
    DOLLY = "dolly"
    STATIC = "static"
    FIXED = "fixed"
    ARC = "arc"
    CRANE = "crane"
    PAN = "pan"


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a 4×4 world-to-camera (view) matrix.

    Delegates to :func:`synapse.render_core.camera_math.look_at`.
    """
    from synapse.render_core.camera_math import look_at

    return look_at(eye, target, up)


def _extract_cam_kw(kwargs: dict[str, Any] | Any) -> dict[str, Any]:
    """Extract intrinsic and optical camera parameters to forward to _make_camera."""
    cam_kw: dict[str, Any] = {}
    if isinstance(kwargs, dict):
        if "z_near" in kwargs and kwargs["z_near"] is not None:
            cam_kw["z_near"] = float(kwargs["z_near"])
        if "z_far" in kwargs and kwargs["z_far"] is not None:
            cam_kw["z_far"] = float(kwargs["z_far"])
        if "aperture_fstop" in kwargs and kwargs["aperture_fstop"] is not None:
            cam_kw["aperture_fstop"] = float(kwargs["aperture_fstop"])
        if "shutter_angle_deg" in kwargs and kwargs["shutter_angle_deg"] is not None:
            cam_kw["shutter_angle_deg"] = float(kwargs["shutter_angle_deg"])
    return cam_kw


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
    target: list[float] | None = None,
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

    cam_kw = _extract_cam_kw(kwargs)
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
        cameras.append(_make_camera(eye, target_pt, width, height, fov_deg, **cam_kw))

    return cameras


def _generate_tracking(
    num_frames: int = 120,
    start: list[float] | None = None,
    end: list[float] | None = None,
    target: list[float] | None = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a linear tracking shot (camera moves, looks at target)."""
    start_pt = np.array(start or [-3.0, 1.5, 0.0], dtype=np.float64)
    end_pt = np.array(end or [3.0, 1.5, 0.0], dtype=np.float64)
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    cam_kw = _extract_cam_kw(kwargs)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        eye = start_pt + t * (end_pt - start_pt)
        cameras.append(_make_camera(eye, target_pt, width, height, fov_deg, **cam_kw))

    return cameras


def _generate_dolly(
    num_frames: int = 120,
    start_distance: float = 5.0,
    end_distance: float = 1.5,
    elevation_deg: float = 15.0,
    azimuth_deg: float = 0.0,
    target: list[float] | None = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a dolly push-in / pull-out shot."""
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    el_rad = math.radians(elevation_deg)
    az_rad = math.radians(azimuth_deg)
    cam_kw = _extract_cam_kw(kwargs)
    cameras: list[CameraParams] = []

    direction = np.array(
        [
            math.cos(el_rad) * math.sin(az_rad),
            math.sin(el_rad),
            math.cos(el_rad) * math.cos(az_rad),
        ]
    )

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        dist = start_distance + t * (end_distance - start_distance)
        eye = target_pt + direction * dist
        cameras.append(_make_camera(eye, target_pt, width, height, fov_deg, **cam_kw))

    return cameras


def _generate_static(
    num_frames: int = 120,
    position: list[float] | None = None,
    target: list[float] | None = None,
    drift_amplitude: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a static (or slow-drift) camera."""
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    if position is None:
        dist = kwargs.get("distance", kwargs.get("radius"))
        elev = kwargs.get("elevation_deg")
        azim = kwargs.get("azimuth_deg")
        if dist is not None or elev is not None or azim is not None:
            r = float(dist) if dist is not None else 3.0  # type: ignore[arg-type]
            el_rad = math.radians(float(elev) if elev is not None else 15.0)  # type: ignore[arg-type]
            az_rad = math.radians(float(azim) if azim is not None else 0.0)  # type: ignore[arg-type]
            x = r * math.cos(el_rad) * math.sin(az_rad)
            y = r * math.sin(el_rad)
            z = r * math.cos(el_rad) * math.cos(az_rad)
            pos = target_pt + np.array([x, y, z], dtype=np.float64)
        else:
            pos = np.array([0.0, 1.5, 3.0], dtype=np.float64)
    else:
        pos = np.array(position, dtype=np.float64)
    cam_kw = _extract_cam_kw(kwargs)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        drift = np.array(
            [
                drift_amplitude * math.sin(2 * math.pi * t),
                drift_amplitude * 0.3 * math.sin(4 * math.pi * t),
                0.0,
            ]
        )
        eye = pos + drift
        cameras.append(_make_camera(eye, target_pt, width, height, fov_deg, **cam_kw))

    return cameras


def _generate_arc(
    num_frames: int = 120,
    radius: float = 3.0,
    azimuth_deg: float = 0.0,
    elevation_start_deg: float = -15.0,
    elevation_end_deg: float = 75.0,
    target: list[float] | None = None,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a vertical arc (elevation sweep at fixed azimuth)."""
    if "distance" in kwargs and kwargs["distance"] is not None:
        radius = float(kwargs["distance"])  # type: ignore[arg-type]
    target_pt = np.array(target or [0.0, 0.0, 0.0], dtype=np.float64)
    az_rad = math.radians(azimuth_deg)
    cam_kw = _extract_cam_kw(kwargs)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        el_deg = elevation_start_deg + t * (elevation_end_deg - elevation_start_deg)
        el_rad = math.radians(el_deg)

        x = radius * math.cos(el_rad) * math.sin(az_rad)
        y = radius * math.sin(el_rad)
        z = radius * math.cos(el_rad) * math.cos(az_rad)

        eye = target_pt + np.array([x, y, z])
        cameras.append(_make_camera(eye, target_pt, width, height, fov_deg, **cam_kw))

    return cameras


def _generate_pan(
    num_frames: int = 120,
    position: list[float] | None = None,
    pan_start_deg: float = -30.0,
    pan_end_deg: float = 30.0,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 60.0,
    **kwargs: object,
) -> list[CameraParams]:
    """Generate a horizontal pan (camera rotates in place)."""
    if "azimuth_start_deg" in kwargs and kwargs["azimuth_start_deg"] is not None:
        pan_start_deg = float(kwargs["azimuth_start_deg"])  # type: ignore[arg-type]
    elif "azimuth_start" in kwargs and kwargs["azimuth_start"] is not None:
        pan_start_deg = float(kwargs["azimuth_start"])  # type: ignore[arg-type]
    if "azimuth_end_deg" in kwargs and kwargs["azimuth_end_deg"] is not None:
        pan_end_deg = float(kwargs["azimuth_end_deg"])  # type: ignore[arg-type]
    elif "azimuth_end" in kwargs and kwargs["azimuth_end"] is not None:
        pan_end_deg = float(kwargs["azimuth_end"])  # type: ignore[arg-type]

    target_pt = np.array(kwargs.get("target") or [0.0, 0.0, 0.0], dtype=np.float64)
    if position is None:
        dist = kwargs.get("distance", kwargs.get("radius"))
        elev = kwargs.get("elevation_deg")
        azim = kwargs.get("azimuth_deg")
        if dist is not None or elev is not None or azim is not None:
            r = float(dist) if dist is not None else 3.0  # type: ignore[arg-type]
            el_rad = math.radians(float(elev) if elev is not None else 15.0)  # type: ignore[arg-type]
            az_rad = math.radians(float(azim) if azim is not None else 0.0)  # type: ignore[arg-type]
            x = r * math.cos(el_rad) * math.sin(az_rad)
            y = r * math.sin(el_rad)
            z = r * math.cos(el_rad) * math.cos(az_rad)
            pos = target_pt + np.array([x, y, z], dtype=np.float64)
        else:
            pos = np.array([0.0, 1.5, 3.0], dtype=np.float64)
    else:
        pos = np.array(position, dtype=np.float64)

    cam_kw = _extract_cam_kw(kwargs)
    cameras: list[CameraParams] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        pan_deg = pan_start_deg + t * (pan_end_deg - pan_start_deg)
        pan_rad = math.radians(pan_deg)
        el_rad = math.radians(float(kwargs.get("elevation_deg", 0.0)))  # type: ignore[arg-type]

        target = pos + np.array(
            [
                math.sin(pan_rad) * math.cos(el_rad) * 5.0,
                math.sin(el_rad) * 5.0,
                -math.cos(pan_rad) * math.cos(el_rad) * 5.0,
            ]
        )
        cameras.append(_make_camera(pos, target, width, height, fov_deg, **cam_kw))

    return cameras


_GENERATORS: dict[CameraTrajectoryType, Callable[..., list[CameraParams]]] = {
    CameraTrajectoryType.ORBIT: _generate_orbit,
    CameraTrajectoryType.TRACKING: _generate_tracking,
    CameraTrajectoryType.DOLLY: _generate_dolly,
    CameraTrajectoryType.STATIC: _generate_static,
    CameraTrajectoryType.FIXED: _generate_static,
    CameraTrajectoryType.ARC: _generate_arc,
    CameraTrajectoryType.CRANE: _generate_arc,
    CameraTrajectoryType.PAN: _generate_pan,
}


def generate_cameras(
    scenario_or_type: Any,
    **override_params: Any,
) -> list[CameraParams]:
    """Generate camera trajectory for a scenario or trajectory type.

    Supports parameter aliasing so that scenario definitions using
    alternative names (e.g. ``distance`` → ``radius``) are correctly
    routed to the underlying generator functions.
    """
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
        for attr in ("z_near", "z_far", "aperture_fstop", "shutter_angle_deg"):
            if hasattr(scenario, attr) and getattr(scenario, attr) is not None:
                params.setdefault(attr, getattr(scenario, attr))
        params.update(override_params)

    # --- Parameter aliasing for cross-scenario compatibility ---
    _ALIASES: dict[str, str] = {
        "distance": "radius",
        "azimuth_start": "azimuth_start_deg",
        "azimuth_end": "azimuth_end_deg",
        "elevation": "elevation_deg",
        "azimuth": "azimuth_deg",
    }
    for alias, canonical in _ALIASES.items():
        if alias in params and canonical not in params:
            params[canonical] = params.pop(alias)
        elif alias in params:
            # Drop alias if canonical already present
            params.pop(alias)

    # Derive azimuth_range_deg from start/end if both present
    if "azimuth_start_deg" in params and "azimuth_end_deg" in params:
        if "azimuth_range_deg" not in params:
            params["azimuth_range_deg"] = float(params["azimuth_end_deg"]) - float(
                params["azimuth_start_deg"]
            )

    generator = _GENERATORS.get(traj_type)
    if generator is None:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return generator(**params)
