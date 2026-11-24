"""Zero-warning compatibility shim (PR-213) consolidated into :mod:`synapse.bench`.

Direct consumers should import ``DatasetConfig`` and ``DatasetGenerator`` from
``synapse.bench.dataset_gen``, colorizers from ``synapse.bench.dataset_visualizer``,
and ``SoftwareRasterizer`` from ``synapse.render_core``. Planned for formal removal
in ``v6.0.0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from synapse.bench.dataset_gen import DatasetConfig, DatasetGenerator
from synapse.bench.dataset_visualizer import (
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
