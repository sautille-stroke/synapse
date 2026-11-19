"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.rasterizer`` instead.
"""

from __future__ import annotations

import warnings
from typing import Any

from synapse.render_core.rasterizer import (
    SoftwareRasterizer,
    entity_hash,
)


def __getattr__(name: str) -> Any:
    """Lazy deprecation warning — only fires when attributes are actually accessed."""
    if name in ("SoftwareRasterizer", "entity_hash"):
        warnings.warn(
            "synapse.bench.rasterizer is deprecated; use synapse.render_core.rasterizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SoftwareRasterizer", "entity_hash"]
