"""LoRA training pipeline --- feature pre-encoding, losses, and trainer.

Submodules:
    feature_store   Offline latent pre-encoding and WebDataset shard builder
    losses          Flow-matching, edge alignment, and DPO loss functions
    trainer         Stateless DiT LoRA trainer with LoRA injection
    packager        LoRA bundle packager with quality gate enforcement
"""

from __future__ import annotations

from synapse.training.feature_store import (
    FeatureStoreBuilder,
    LatentEncoderProtocol,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
    TextEncoderProtocol,
    format_structured_caption,
    iter_feature_shards,
)
from synapse.training.losses import (
    diffusion_dpo_loss,
    flow_matching_loss,
    latent_edge_alignment_loss,
)
from synapse.training.packager import (
    LoRAEvaluationReport,
    LoRAGateThresholds,
    package_and_gate_lora_bundle,
)
from synapse.training.trainer import (
    LoRATrainConfig,
    StatelessLoRATrainer,
)

__all__ = [
    "FeatureStoreBuilder",
    "LatentEncoderProtocol",
    "LinearProjectionEncoder",
    "LinearProjectionTextEncoder",
    "TextEncoderProtocol",
    "format_structured_caption",
    "iter_feature_shards",
    "flow_matching_loss",
    "latent_edge_alignment_loss",
    "diffusion_dpo_loss",
    "LoRATrainConfig",
    "StatelessLoRATrainer",
    "LoRAGateThresholds",
    "LoRAEvaluationReport",
    "package_and_gate_lora_bundle",
]
