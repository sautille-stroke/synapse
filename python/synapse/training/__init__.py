"""LoRA training pipeline --- feature pre-encoding, losses, and trainer.

Submodules:
    feature_store   Offline latent pre-encoding and WebDataset shard builder
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

__all__ = [
    "FeatureStoreBuilder",
    "LatentEncoderProtocol",
    "LinearProjectionEncoder",
    "LinearProjectionTextEncoder",
    "TextEncoderProtocol",
    "format_structured_caption",
    "iter_feature_shards",
]
