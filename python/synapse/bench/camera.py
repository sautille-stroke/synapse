"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.camera`` instead.
"""

from synapse.render_core.camera import (  # noqa: F401
    _GENERATORS,
    CameraTrajectoryType,
    _generate_arc,
    _generate_dolly,
    _generate_orbit,
    _generate_pan,
    _generate_static,
    _generate_tracking,
    _look_at,
    _make_camera,
    generate_cameras,
)
