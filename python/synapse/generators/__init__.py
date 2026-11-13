"""synapse.generators — Multi-channel dataset generation engine.

.. deprecated:: ``SoftwareRasterizer`` and ``RenderResult``
    The ``synapse.generators.rasterizer`` module is now a compatibility shim
    delegating to ``synapse.bench.rasterizer``.  New code should import
    from ``synapse.bench`` directly.
    See https://github.com/sautille-stroke/synapse/issues/34
"""

from synapse.generators.dataset import DatasetConfig, DatasetGenerator
from synapse.generators.rasterizer import RenderResult, SoftwareRasterizer
from synapse.generators.visualizer import (
    colorize_depth,
    colorize_flow,
    colorize_mask,
    colorize_normal,
    make_contact_sheet,
)

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
