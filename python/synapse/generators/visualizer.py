"""Zero-warning compatibility shim (PR-213) — consolidated into :mod:`synapse.bench.dataset_visualizer`.

Planned for formal removal in ``v6.0.0``.
"""

from __future__ import annotations

from synapse.bench.dataset_visualizer import (
    colorize_depth,
    colorize_flow,
    colorize_mask,
    colorize_normal,
    make_contact_sheet,
)

__all__ = [
    "colorize_depth",
    "colorize_flow",
    "colorize_mask",
    "colorize_normal",
    "make_contact_sheet",
]
