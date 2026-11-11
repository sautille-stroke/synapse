"""synapse.generators — Multi-channel software rasterizer and dataset generation engine."""

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
    "SoftwareRasterizer",
    "RenderResult",
    "colorize_depth",
    "colorize_normal",
    "colorize_flow",
    "colorize_mask",
    "make_contact_sheet",
]
