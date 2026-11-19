"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.rasterizer`` instead.
"""

__all__ = ["SoftwareRasterizer", "entity_hash"]

from synapse.render_core.rasterizer import (  # noqa: F401
    SoftwareRasterizer,
    _cryptomatte_name_hash,
    entity_hash,
)
