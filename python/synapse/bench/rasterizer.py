"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.rasterizer`` instead.
"""

from synapse.render_core.rasterizer import (  # noqa: F401
    SoftwareRasterizer,
    _murmurhash3_32,
    entity_hash,
)
