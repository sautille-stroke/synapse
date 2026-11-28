"""LoRA training pipeline --- feature pre-encoding, losses, and trainer.

Submodules:
    feature_store   Offline latent pre-encoding and WebDataset shard builder
    losses          Flow-matching, edge alignment, and DPO loss functions
    trainer         Stateless DiT LoRA trainer with LoRA injection
    packager        LoRA bundle packager with quality gate enforcement
"""

from __future__ import annotations

from synapse.training.backbone import (
    CATEGORY_DEFAULT_TARGETS,
    DIFFUSERS_WAN_CATEGORY_TARGETS,
    build_mock_dit_backbone,
    load_real_dit_backbone,
    resolve_target_modules,
    resolve_training_backbone,
)
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
    DiffusionDPOLoss,
    EdgePreservingLoss,
    FlowMatchingLoss,
    LatentGeometryProbe,
    NormalReconstructionLoss,
    diffusion_dpo_loss,
    flow_matching_loss,
    latent_edge_alignment_loss,
    normal_reconstruction_loss,
)
from synapse.training.bucketing import (
    AutoTuneRecommendation,
    FrameBucketBatch,
    FrameBucketKey,
    FrameBucketSample,
    FrameBucketSampler,
    auto_tune_lora_config,
    detect_cuda_vram_gb,
    snap_to_frame_bucket,
)
from synapse.training.packager import (
    LoRAEvaluationReport,
    LoRAGateThresholds,
    package_and_gate_lora_bundle,
)
from synapse.training.qlora import (
    NF4_CODEBOOK,
    NF4QuantizedTensor,
    create_qlora_linear,
    dequantize_nf4_blockwise,
    quantize_nf4_blockwise,
)
from synapse.training.trainer import (
    LoRATrainConfig,
    StatelessLoRATrainer,
)

__all__ = [
    "CATEGORY_DEFAULT_TARGETS",
    "DIFFUSERS_WAN_CATEGORY_TARGETS",
    "build_mock_dit_backbone",
    "load_real_dit_backbone",
    "resolve_target_modules",
    "resolve_training_backbone",
    "FeatureStoreBuilder",
    "LatentEncoderProtocol",
    "LinearProjectionEncoder",
    "LinearProjectionTextEncoder",
    "TextEncoderProtocol",
    "format_structured_caption",
    "iter_feature_shards",
    "FlowMatchingLoss",
    "LatentGeometryProbe",
    "EdgePreservingLoss",
    "NormalReconstructionLoss",
    "DiffusionDPOLoss",
    "flow_matching_loss",
    "latent_edge_alignment_loss",
    "normal_reconstruction_loss",
    "diffusion_dpo_loss",
    "NF4_CODEBOOK",
    "NF4QuantizedTensor",
    "quantize_nf4_blockwise",
    "dequantize_nf4_blockwise",
    "create_qlora_linear",
    "LoRATrainConfig",
    "StatelessLoRATrainer",
    "AutoTuneRecommendation",
    "FrameBucketBatch",
    "FrameBucketKey",
    "FrameBucketSample",
    "FrameBucketSampler",
    "auto_tune_lora_config",
    "detect_cuda_vram_gb",
    "snap_to_frame_bucket",
    "LoRAGateThresholds",
    "LoRAEvaluationReport",
    "package_and_gate_lora_bundle",
]
