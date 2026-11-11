"""synapse.fgpp — FGPP wire format types, schema, and I/O.

Re-exports the core data classes and schema constants for convenient
single-import access.
"""

from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.io import (
    export_fgpp,
    import_fgpp,
    resize_fgpp_frame,
    sanitize_fgpp_frame,
)
from synapse.fgpp.schema import (
    DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
    FGPP_BYTES_PER_PIXEL,
    FGPP_CHANNEL_SPECS,
    FGPP_NUM_CHANNELS,
    FGPP_SCHEMA_VERSION,
    MIN_CHARACTER_WHITEBOX_VERTICES,
    PixelType,
    get_channel_names,
    get_logical_groups,
    is_physical_anchor_frame,
    linear_to_log_normalized_depth,
    log_normalized_to_linear_depth,
    validate_whitebox_coarseness_floor,
)
from synapse.fgpp.torch_bridge import frame_to_tensors, tensors_to_frame

__all__ = [
    # Frame types
    "CameraParams",
    "FGPPFrame",
    "FGPPSequence",
    # Schema
    "PixelType",
    "FGPP_CHANNEL_SPECS",
    "FGPP_NUM_CHANNELS",
    "FGPP_BYTES_PER_PIXEL",
    "FGPP_SCHEMA_VERSION",
    "MIN_CHARACTER_WHITEBOX_VERTICES",
    "DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES",
    "get_channel_names",
    "get_logical_groups",
    "linear_to_log_normalized_depth",
    "log_normalized_to_linear_depth",
    "validate_whitebox_coarseness_floor",
    "is_physical_anchor_frame",
    # I/O
    "export_fgpp",
    "import_fgpp",
    "sanitize_fgpp_frame",
    "resize_fgpp_frame",
    # Torch bridge
    "frame_to_tensors",
    "tensors_to_frame",
]
