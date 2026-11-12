"""Training pipeline stage handlers (lora_preencode, lora_train, lora_package)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from synapse.errors import (
    ContractValidationError,
    NotImplementedStageError,
)
from synapse.pipeline.bundle_worker import (
    BundleManifest,
    register_stage,
)
from synapse.training.feature_store import (
    FeatureStoreBuilder,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
    format_structured_caption,
)
from synapse.training.packager import package_and_gate_lora_bundle
from synapse.training.trainer import LoRATrainConfig


@register_stage("lora_preencode")
def handle_lora_preencode(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Pre-encode FGPP + video frames into feature shards."""
    params = dict(manifest.params)
    num_frames = int(params.get("num_frames", 17))
    height = int(params.get("height", 64))
    width = int(params.get("width", 64))
    num_samples = int(params.get("num_samples", 1))

    output_dir = Path(
        manifest.outputs.get("shards_dir", "output/shards"),
    )

    enc = LinearProjectionEncoder()
    txt_enc = LinearProjectionTextEncoder()
    builder = FeatureStoreBuilder(enc, txt_enc)

    for i in range(num_samples):
        rng = np.random.RandomState(i)
        frames = rng.randint(
            0,
            256,
            size=(num_frames, height, width, 3),
            dtype=np.uint8,
        )
        caption = format_structured_caption(
            trigger_token="sks",
            subject_desc=f"sample {i}",
            lighting_env="studio",
            camera_motion="static",
            art_style_tag="photorealistic",
        )
        builder.add_sample(
            sample_id=f"sample_{i:04d}",
            video_frames=frames,
            caption=caption,
        )

    builder.build_shards(output_dir)

    return {
        "output_paths": {
            "shards_dir": str(output_dir),
        },
    }


@register_stage("lora_train")
def handle_lora_train(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Execute a LoRA training run from a job bundle manifest."""
    params = dict(manifest.params)
    raw_cfg = params.get("train_config")
    if not isinstance(raw_cfg, dict) or "category" not in raw_cfg:
        raise ContractValidationError(
            "lora_train stage requires 'train_config' dict with "
            "explicit 'category' field in manifest.params.",
            agent_fix_hint=(
                "Provide train_config={'category': 'style', ...} "
                "in bundle_manifest.json params."
            ),
        )
    try:
        config = LoRATrainConfig(**raw_cfg)
    except Exception as exc:
        raise ContractValidationError(
            f"Invalid LoRATrainConfig in lora_train stage: {exc}",
            agent_fix_hint="Fix train_config fields per LoRATrainConfig schema.",
        ) from exc

    raise NotImplementedStageError(
        "Stage 'lora_train' end-to-end backbone loop is not yet "
        "wired; refusing to emit silent success.",
        agent_fix_hint=(
            "Use StatelessLoRATrainer API directly with an injected "
            "torch.nn.Module in Python."
        ),
        details={"validated_config": config.model_dump()},
    )


@register_stage("lora_package")
def handle_lora_package(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Package LoRA weights from a job bundle manifest."""
    params = dict(manifest.params)
    output_dir = Path(
        params.get("output_dir", "/tmp/lora_bundle"),
    )
    weights_path = Path(
        params.get("weights_path", "adapter_model.safetensors"),
    )

    result = package_and_gate_lora_bundle(
        weights_path=weights_path,
        output_dir=output_dir,
        category=params.get("category", "style"),
        rank=params.get("rank", 8),
        alpha=params.get("alpha", 1.0),
        bench_metrics=params.get("bench_metrics"),
        skip_bench=bool(params.get("skip_bench", True)),
    )
    return {
        "output_paths": {
            "bundle": str(output_dir),
        },
        **result,
    }
