"""Pipeline stage adapters for training, RSI, 3D inspection, reconstruction,
and anatomical refinement.

Consolidates ``@register_stage`` wiring inside ``synapse.pipeline.stages``
so domain modules (``synapse.training``, ``synapse.rsi``, ``synapse.inspect``,
``synapse.recon``, ``synapse.refine``) remain pure, side-effect-free libraries
that never import ``synapse.pipeline``.
"""

from __future__ import annotations

from synapse.pipeline.stages.inspect import (
    handle_asset_inspect_3d,
    handle_inspect_export_rl,
    handle_inspect_refine,
)
from synapse.pipeline.stages.recon import handle_video_to_3d_invert
from synapse.pipeline.stages.refine import handle_anatomical_refine
from synapse.pipeline.stages.rsi import handle_rsi_dual_verify
from synapse.pipeline.stages.training import (
    handle_lora_package,
    handle_lora_preencode,
    handle_lora_train,
)

__all__ = [
    "handle_anatomical_refine",
    "handle_asset_inspect_3d",
    "handle_inspect_export_rl",
    "handle_inspect_refine",
    "handle_lora_package",
    "handle_lora_preencode",
    "handle_lora_train",
    "handle_rsi_dual_verify",
    "handle_video_to_3d_invert",
]
