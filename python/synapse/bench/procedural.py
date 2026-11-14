"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.procedural`` instead.
"""

from synapse.render_core.procedural import (  # noqa: F401
    _GENERATORS,
    _HUMANOID_SEGMENTS,
    ProceduralMesh,
    ProceduralMeshType,
    _compute_walk_pose,
    _generate_cube,
    _generate_cylinder,
    _generate_plane,
    _generate_segment_mesh,
    _generate_sphere,
    _generate_torus,
    _make_rotation_x,
    _make_rotation_y,
    _make_rotation_z,
    generate_humanoid_proxy,
    generate_procedural_mesh,
)
