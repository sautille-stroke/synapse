"""synapse.composite — Selective 3D re-rendering & surgical compositing."""

from synapse.composite.blend import (
    border_ring_lab_match,
    composite_optical_increments,
    composite_surgical_sequence,
    laplacian_pyramid_blend,
)
from synapse.composite.detector import (
    VideoDefectRegion,
    detect_video_artifacts,
)
from synapse.composite.mask import (
    apply_surgical_mask,
    generate_surgical_mask,
)

__all__ = [
    "VideoDefectRegion",
    "apply_surgical_mask",
    "border_ring_lab_match",
    "composite_optical_increments",
    "composite_surgical_sequence",
    "detect_video_artifacts",
    "generate_surgical_mask",
    "laplacian_pyramid_blend",
]
