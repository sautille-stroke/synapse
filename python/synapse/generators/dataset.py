"""Zero-warning compatibility shim (PR-213) — consolidated into :mod:`synapse.bench.dataset_gen`.

Planned for formal removal in ``v6.0.0``.
"""

from __future__ import annotations

from synapse.bench.dataset_gen import (
    DatasetConfig,
    DatasetGenerator,
    _generate_hemisphere_cameras,
    _generate_spiral_cameras,
)

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "_generate_hemisphere_cameras",
    "_generate_spiral_cameras",
]
