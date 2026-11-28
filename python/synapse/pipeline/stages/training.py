"""Training pipeline stage handlers (lora_preencode, lora_train, lora_package)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from synapse.errors import (
    ContractValidationError,
    QualityGateError,
)
from synapse.pipeline.bundle_worker import (
    BundleManifest,
    register_stage,
)
from synapse.training.backbone import (
    resolve_target_modules,
    resolve_training_backbone,
)
from synapse.training.feature_store import (
    FeatureStoreBuilder,
    LinearProjectionEncoder,
    LinearProjectionTextEncoder,
    format_structured_caption,
    iter_feature_shards,
)
from synapse.training.packager import package_and_gate_lora_bundle
from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer


@register_stage("lora_preencode")
def handle_lora_preencode(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Pre-encode FGPP + video frames into feature shards."""
    params = dict(manifest.params)
    int(params.get("num_frames", 17))

    output_dir = Path(
        manifest.outputs.get("shards_dir", "output/shards"),
    )

    # Load real dataset — no synthetic fallback
    dataset_dir_str = manifest.inputs.get("dataset_dir", "")
    if not dataset_dir_str or not Path(dataset_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Dataset directory not found: {dataset_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['dataset_dir'] pointing to "
                "a directory containing video frame .npy files and "
                "caption .txt files."
            ),
        )

    dataset_dir = Path(dataset_dir_str)
    enc = LinearProjectionEncoder()
    txt_enc = LinearProjectionTextEncoder()
    builder = FeatureStoreBuilder(enc, txt_enc)

    # Each .npy file in the dataset directory is a sample
    sample_files = sorted(dataset_dir.glob("*.npy"))
    if not sample_files:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .npy sample files found in {dataset_dir}",
            agent_fix_hint=(
                "Ensure dataset_dir contains .npy video frame files of shape (T, H, W, 3) uint8."
            ),
        )

    for i, npy_path in enumerate(sample_files):
        video_frames = np.load(str(npy_path))
        caption_path = npy_path.with_suffix(".txt")
        if caption_path.is_file():
            caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            caption = format_structured_caption(
                trigger_token="sks",
                subject_desc=npy_path.stem,
                lighting_env="studio",
                camera_motion="static",
                art_style_tag="photorealistic",
            )
        builder.add_sample(
            sample_id=f"sample_{i:04d}",
            video_frames=video_frames,
            caption=caption,
        )

    # Fail-closed: 0 accepted samples → exit 40
    accepted = len(sample_files) - builder.rejected_count
    if accepted == 0:
        raise QualityGateError(
            f"All {len(sample_files)} samples rejected by ingestion gates; "
            "0 samples passed. Fail-closed (exit 40).",
            agent_fix_hint=(
                "Review input video quality, FGPP conditions, and camera matrices. "
                "At least one sample must pass all 5 ingestion gates."
            ),
        )

    builder.build_shards(output_dir)

    return {
        "output_paths": {
            "shards_dir": str(output_dir),
        },
        "accepted_samples": accepted,
        "rejected_samples": builder.rejected_count,
    }


@register_stage("lora_train")
def handle_lora_train(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Execute a LoRA training run from a job bundle manifest.

    Wires ``FeatureStoreBuilder`` shards → ``StatelessLoRATrainer`` →
    ``export_checkpoint``.  Requires PyTorch (exit 30 if missing).
    """
    params = dict(manifest.params)
    raw_cfg = params.get("train_config")
    if not isinstance(raw_cfg, dict) or "category" not in raw_cfg:
        raise ContractValidationError(
            "lora_train stage requires 'train_config' dict with "
            "explicit 'category' field in manifest.params.",
            agent_fix_hint=(
                "Provide train_config={'category': 'style', ...} in bundle_manifest.json params."
            ),
        )
    try:
        config = LoRATrainConfig(**raw_cfg)
    except Exception as exc:
        raise ContractValidationError(
            f"Invalid LoRATrainConfig in lora_train stage: {exc}",
            agent_fix_hint="Fix train_config fields per LoRATrainConfig schema.",
        ) from exc

    # Require shards directory
    shards_dir_str = manifest.inputs.get("shards_dir", "")
    if not shards_dir_str or not Path(shards_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Shards directory not found: {shards_dir_str!r}",
            agent_fix_hint=(
                "Run lora_preencode stage first to create feature shards, "
                "then set manifest.inputs['shards_dir']."
            ),
        )

    shards_dir = Path(shards_dir_str)
    shard_paths = sorted(shards_dir.glob("shard-*.tar"))
    if not shard_paths:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No shard-*.tar files found in {shards_dir}",
            agent_fix_hint="Run lora_preencode to generate .tar shards first.",
        )

    # Build DiT model with whitelist-compatible block architecture
    try:
        import torch
    except ImportError as exc:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "PyTorch is required for lora_train stage",
            agent_fix_hint="Install torch via 'pip install torch'.",
        ) from exc

    target_modules = params.get("target_modules")
    model_dim = int(params.get("model_dim", 64))
    base_model_path = params.get("base_model")
    use_mock_backbone = bool(params.get("mock_backbone", False))

    model, backbone_kind = resolve_training_backbone(
        base_model_path=base_model_path,
        mock_backbone=use_mock_backbone,
        model_dim=model_dim,
    )
    resolved_targets = resolve_target_modules(
        model,
        config.category,
        explicit_targets=list(target_modules) if target_modules else None,
    )

    trainer = StatelessLoRATrainer(config)
    trainer.inject_lora(model, resolved_targets)
    trainer.freeze_backbone(model)
    trainer.enable_gradient_checkpointing(model)

    output_dir = Path(manifest.outputs.get("weights_dir", "output/weights"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop over shard data — cycle shards until config.max_steps is reached
    step = 0
    final_loss = 0.0
    cached_samples = list(iter_feature_shards(shard_paths))
    if not cached_samples:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No valid samples found in shards under {shards_dir}",
            agent_fix_hint="Re-run lora_preencode to populate valid training shards.",
        )

    while step < config.max_steps:
        sample = cached_samples[step % len(cached_samples)]
        latent = torch.as_tensor(sample["latent"], dtype=torch.float32)
        noise = torch.randn_like(latent)
        batch: dict[str, Any] = {"latent": latent.unsqueeze(0), "noise": noise.unsqueeze(0)}
        if sample.get("fgpp_cond") is not None:
            batch["fgpp_cond"] = torch.as_tensor(
                sample["fgpp_cond"], dtype=torch.float32,
            ).unsqueeze(0)
        result = trainer.train_step(model, batch, step)
        final_loss = result["loss"]
        step += 1

    trainer.flush_gradients(model)

    # Export checkpoint
    weights_path = trainer.export_checkpoint(
        model, output_dir / "adapter_model.safetensors",
    )

    return {
        "output_paths": {"weights_dir": str(output_dir)},
        "steps_completed": step,
        "final_loss": final_loss,
        "weights_path": str(weights_path),
        "backbone_kind": backbone_kind,
        "mock_backbone": backbone_kind == "mock",
    }


@register_stage("lora_package")
def handle_lora_package(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Package LoRA weights from a job bundle manifest."""
    params = dict(manifest.params)
    output_dir = Path(
        manifest.outputs.get("bundle", params.get("output_dir", "output/lora_bundle")),
    )
    weights_path = Path(
        manifest.inputs.get("weights", params.get("weights_path", "adapter_model.safetensors")),
    )

    result = package_and_gate_lora_bundle(
        weights_path=weights_path,
        output_dir=output_dir,
        category=params.get("category", "style"),
        rank=params.get("rank", 8),
        alpha=params.get("alpha", 1.0),
        bench_metrics=params.get("bench_metrics"),
        skip_bench=bool(params.get("skip_bench", False)),
    )
    return {
        "output_paths": {
            "bundle": str(output_dir),
        },
        **result,
    }
