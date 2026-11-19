"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.rasterizer`` instead.
"""

from __future__ import annotations

import warnings

from synapse.render_core.rasterizer import (  # noqa: F401
    SoftwareRasterizer,
    _cryptomatte_name_hash,
    entity_hash,
)

warnings.warn(
    "synapse.bench.rasterizer is deprecated; use synapse.render_core.rasterizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SoftwareRasterizer", "entity_hash"]
