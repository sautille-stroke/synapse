"""synapse.generators — Multi-channel dataset generation engine.

.. deprecated:: ``SoftwareRasterizer`` and ``RenderResult``
    The ``synapse.generators.rasterizer`` module is now a compatibility shim
    delegating to ``synapse.render_core.rasterizer``.  New code should import
    from ``synapse.render_core`` directly.
    See https://github.com/sautille-stroke/synapse/issues/34
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from synapse.generators.dataset import DatasetConfig, DatasetGenerator
from synapse.generators.visualizer import (
    colorize_depth,
    colorize_flow,
    colorize_mask,
    colorize_normal,
    make_contact_sheet,
)

if TYPE_CHECKING:
    from synapse.generators.rasterizer import RenderResult, SoftwareRasterizer

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "RenderResult",
    "SoftwareRasterizer",
    "colorize_depth",
    "colorize_flow",
    "colorize_mask",
    "colorize_normal",
    "make_contact_sheet",
]


def __getattr__(name: str) -> Any:
    """Lazy import of deprecated rasterizer symbols to avoid module-level DeprecationWarning."""
    if name in ("RenderResult", "SoftwareRasterizer"):
        from synapse.generators.rasterizer import (
            RenderResult as _RenderResult,
        )
        from synapse.generators.rasterizer import (
            SoftwareRasterizer as _SoftwareRasterizer,
        )

        _lazy = {"RenderResult": _RenderResult, "SoftwareRasterizer": _SoftwareRasterizer}
        return _lazy[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
