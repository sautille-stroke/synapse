"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.procedural`` instead.
"""

from synapse.render_core.procedural import (  # noqa: F401
    ProceduralMeshType,
    ProceduralMesh,
    generate_procedural_mesh,
    generate_humanoid_proxy,
    _generate_cube,
    _generate_sphere,
    _generate_cylinder,
    _generate_torus,
    _generate_plane,
    _GENERATORS,
    _HUMANOID_SEGMENTS,
    _make_rotation_x,
    _make_rotation_y,
    _make_rotation_z,
    _generate_segment_mesh,
    _compute_walk_pose,
)
