"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.camera`` instead.
"""

from synapse.render_core.camera import (  # noqa: F401
    CameraTrajectoryType,
    generate_cameras,
    _look_at,
    _make_camera,
    _generate_orbit,
    _generate_tracking,
    _generate_dolly,
    _generate_static,
    _generate_arc,
    _generate_pan,
    _GENERATORS,
)
