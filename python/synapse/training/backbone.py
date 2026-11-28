"""Shared LoRA DiT backbone loading and target module resolution.

Provides a single source of truth for both :mod:`synapse.pipeline.stages.training`
and :mod:`synapse.cli._lora`:

- :func:`load_real_dit_backbone`: fail-closed loader for local Diffusers
  transformer checkpoints (``<root>/transformer/config.json`` or standalone
  ``<dir>/config.json``). Never falls back to a mock toy model.
- :func:`build_mock_dit_backbone`: explicit 1-block synthetic DiT used **only**
  when ``mock_backbone=True`` / ``--mock-backbone`` is requested.
- :func:`resolve_training_backbone`: mutual-exclusion and fail-closed router.
- :func:`resolve_target_modules`: resolves category-default LoRA target globs
  aligned with both Diffusers ``WanTransformer3DModel`` (``attn1``/``attn2``)
  and custom/mock ``spatial_attn``/``cross_attn``/``temporal_attn`` blocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    ContractValidationError,
    ExitCode,
    SynapseError,
)

__all__ = [
    "CATEGORY_DEFAULT_TARGETS",
    "DIFFUSERS_WAN_CATEGORY_TARGETS",
    "build_mock_dit_backbone",
    "load_real_dit_backbone",
    "prepare_lora_training_batch",
    "resolve_target_modules",
    "resolve_training_backbone",
    "run_lora_shard_training",
]

CATEGORY_DEFAULT_TARGETS: dict[str, list[str]] = {
    "style": ["blocks.*.cross_attn.to_k", "blocks.*.cross_attn.to_v", "blocks.*.ffn.*"],
    "identity": [
        "blocks.*.spatial_attn.*",
        "blocks.*.cross_attn.to_k",
        "blocks.*.cross_attn.to_v",
    ],
    "control": ["blocks.*.spatial_attn.*", "blocks.*.ffn.*"],
    "whitebox": ["blocks.*.spatial_attn.*", "blocks.*.ffn.*"],
    "motion": ["blocks.*.temporal_attn.*"],
}

DIFFUSERS_WAN_CATEGORY_TARGETS: dict[str, list[str]] = {
    "style": ["blocks.*.attn2.to_k", "blocks.*.attn2.to_v", "blocks.*.ffn.*"],
    "identity": [
        "blocks.*.attn1.*",
        "blocks.*.attn2.to_k",
        "blocks.*.attn2.to_v",
    ],
    "control": ["blocks.*.attn1.*", "blocks.*.ffn.*"],
    "whitebox": ["blocks.*.attn1.*", "blocks.*.ffn.*"],
    "motion": ["blocks.*.attn1.*"],
}


def build_mock_dit_backbone(dim: int = 64) -> Any:
    """Build the synthetic 1-block DiT backbone for explicit ``--mock-backbone`` smoke runs."""
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "PyTorch is required to build the DiT backbone.",
            agent_fix_hint="Install torch via 'pip install torch'.",
        ) from exc

    class _WhitelistDiT(torch.nn.Module):
        """Mock Wan DiT backbone with all 4 attention/FFN blocks wired in forward()."""

        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self._dim = hidden_dim
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "spatial_attn": torch.nn.ModuleDict(
                                {
                                    "to_q": torch.nn.Linear(hidden_dim, hidden_dim),
                                    "to_k": torch.nn.Linear(hidden_dim, hidden_dim),
                                    "to_v": torch.nn.Linear(hidden_dim, hidden_dim),
                                }
                            ),
                            "cross_attn": torch.nn.ModuleDict(
                                {
                                    "to_k": torch.nn.Linear(hidden_dim, hidden_dim),
                                    "to_v": torch.nn.Linear(hidden_dim, hidden_dim),
                                }
                            ),
                            "temporal_attn": torch.nn.ModuleDict(
                                {
                                    "to_q": torch.nn.Linear(hidden_dim, hidden_dim),
                                    "to_k": torch.nn.Linear(hidden_dim, hidden_dim),
                                    "to_v": torch.nn.Linear(hidden_dim, hidden_dim),
                                }
                            ),
                            "ffn": torch.nn.ModuleDict(
                                {
                                    "linear": torch.nn.Linear(hidden_dim, hidden_dim),
                                }
                            ),
                        }
                    ),
                ]
            )
            self.out = torch.nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: Any, t: Any = None, **kwargs: Any) -> Any:
            _ = (t, kwargs)
            orig_shape = x.shape
            if x.shape[-1] == self._dim:
                h = x
                adapted = False
            else:
                flat = x.reshape(orig_shape[0], -1)
                n_feat = flat.shape[1]
                pad = (self._dim - (n_feat % self._dim)) % self._dim
                if pad > 0:
                    flat = torch.nn.functional.pad(flat, (0, pad))
                h = flat.reshape(orig_shape[0], -1, self._dim)
                adapted = True

            for block in self.blocks:
                blk: Any = block
                sp = (
                    blk["spatial_attn"]["to_q"](h)
                    + blk["spatial_attn"]["to_k"](h)
                    + blk["spatial_attn"]["to_v"](h)
                ) / 3.0
                h = h + sp
                cr = (blk["cross_attn"]["to_k"](h) + blk["cross_attn"]["to_v"](h)) / 2.0
                h = h + cr
                tm = (
                    blk["temporal_attn"]["to_q"](h)
                    + blk["temporal_attn"]["to_k"](h)
                    + blk["temporal_attn"]["to_v"](h)
                ) / 3.0
                h = h + tm
                h = h + blk["ffn"]["linear"](h)
            out = self.out(h)
            if adapted:
                n_orig = int(np.prod(orig_shape[1:]))
                return out.reshape(orig_shape[0], -1)[:, :n_orig].reshape(orig_shape)
            return out

    return _WhitelistDiT(dim)


def load_real_dit_backbone(
    base_model_path: str | Path,
    *,
    torch_dtype: Any | None = None,
    device: str | None = None,
) -> Any:
    """Load a pretrained local Diffusers DiT backbone, failing closed on any error."""
    path = Path(base_model_path)
    if not path.exists():
        raise AssetNotFoundError(
            f"Base model path not found: {base_model_path}",
            agent_fix_hint="Provide a valid local Diffusers checkpoint directory via --base-model.",
        )

    if not path.is_dir():
        raise ContractValidationError(
            f"Base model path must be a Diffusers checkpoint directory, got file: {path}",
            agent_fix_hint=(
                "Point --base-model to a Diffusers pipeline directory (containing transformer/config.json) "
                "or a transformer model directory (containing config.json)."
            ),
        )

    transformer_dir = path / "transformer" if (path / "transformer").is_dir() else path
    config_file = transformer_dir / "config.json"
    if not config_file.is_file():
        raise ContractValidationError(
            f"Invalid DiT checkpoint directory '{path}': missing transformer config.json at {config_file}. "
            "Refusing to fall back to a mock backbone.",
            agent_fix_hint=(
                "Ensure --base-model points to a valid local Diffusers Wan directory with config.json."
            ),
        )

    try:
        cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(
            f"Failed to parse transformer config.json at {config_file}: {exc}",
            agent_fix_hint="Verify config.json is valid UTF-8 JSON.",
        ) from exc

    if not isinstance(cfg_data, dict):
        raise ContractValidationError(
            f"Expected JSON object in {config_file}, got {type(cfg_data).__name__}.",
            agent_fix_hint="Provide a valid Diffusers transformer config.json.",
        )

    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "PyTorch is required to load a pretrained DiT backbone.",
            agent_fix_hint="Install torch via 'pip install torch'.",
        ) from exc

    try:
        import diffusers
    except ImportError as exc:
        raise BackendUnavailableError(
            "diffusers is required to load a real pretrained --base-model checkpoint "
            "(refusing to fall back to mock _WhitelistDiT).",
            agent_fix_hint="Install GPU/Diffusers dependencies: pip install 'synapse[gpu]'",
        ) from exc

    class_name = cfg_data.get("_class_name")
    model_cls: Any = None
    if isinstance(class_name, str) and class_name:
        model_cls = getattr(diffusers, class_name, None)
    if model_cls is None:
        model_cls = getattr(diffusers, "WanTransformer3DModel", None)
    if model_cls is None or not hasattr(model_cls, "from_pretrained"):
        raise BackendUnavailableError(
            f"Installed diffusers does not provide transformer loader class {class_name or 'WanTransformer3DModel'!r}.",
            agent_fix_hint="Upgrade diffusers >= 0.30.0 or inspect config.json _class_name.",
        )

    load_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    try:
        model = model_cls.from_pretrained(str(transformer_dir), **load_kwargs)
    except Exception as exc:
        raise BackendUnavailableError(
            f"Failed to load pretrained DiT backbone from '{transformer_dir}': {exc}",
            agent_fix_hint="Verify safetensors/model weights and diffusers version compatibility.",
        ) from exc

    if not isinstance(model, torch.nn.Module):
        raise ContractValidationError(
            f"Loaded backbone from '{transformer_dir}' is not a torch.nn.Module "
            f"(got {type(model).__name__}).",
            agent_fix_hint="Ensure from_pretrained returns a torch.nn.Module transformer.",
        )

    if device is not None:
        model = model.to(device)
    return model


def resolve_training_backbone(
    *,
    base_model_path: str | Path | None,
    mock_backbone: bool,
    model_dim: int = 64,
    torch_dtype: Any | None = None,
    device: str | None = None,
) -> tuple[Any, Literal["real", "mock"]]:
    """Resolve either a real pretrained DiT backbone or the explicit mock backbone."""
    has_base_model = base_model_path is not None and str(base_model_path).strip() != ""
    if has_base_model and mock_backbone:
        raise ContractValidationError(
            "Cannot specify both base_model and mock_backbone=True.",
            agent_fix_hint="Pass either --base-model <dir> OR --mock-backbone, not both.",
        )
    if not has_base_model and not mock_backbone:
        raise SynapseError(
            "Production LoRA training requires --base-model (or explicit --mock-backbone for testing).",
            ExitCode.NOT_IMPLEMENTED,
            agent_fix_hint=(
                "Pass --base-model <path-to-wan-dit> for real training, or "
                "--mock-backbone to run the synthetic DiT training smoke loop."
            ),
        )
    if has_base_model:
        assert base_model_path is not None
        model = load_real_dit_backbone(
            base_model_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        return model, "real"
    return build_mock_dit_backbone(dim=model_dim), "mock"


def resolve_target_modules(
    model: Any,
    category: str,
    explicit_targets: list[str] | None = None,
) -> list[str]:
    """Resolve LoRA injection target patterns matching the loaded backbone's module tree."""
    if explicit_targets:
        return list(explicit_targets)

    named = [name for name, _ in model.named_modules()] if hasattr(model, "named_modules") else []
    has_wan_custom = any(
        ".cross_attn." in n or ".spatial_attn." in n or ".temporal_attn." in n for n in named
    )
    has_diffusers_wan = any(".attn1." in n or ".attn2." in n for n in named)

    if has_diffusers_wan and not has_wan_custom:
        return list(DIFFUSERS_WAN_CATEGORY_TARGETS.get(category, ["blocks.*.ffn.*"]))
    return list(CATEGORY_DEFAULT_TARGETS.get(category, ["blocks.*.ffn.*"]))


def prepare_lora_training_batch(sample: dict[str, Any], torch_mod: Any) -> dict[str, Any]:
    """Build a flow-matching training batch from an ``iter_feature_shards`` sample.

    Propagates ``prompt`` and 3D ``encoder_hidden_states`` ``[B, L, D]`` plus
    optional ``fgpp_cond`` so Diffusers ``WanTransformer3DModel`` and mock
    backbones consume identical shard batches without silent parameter drops.
    """
    latent = torch_mod.as_tensor(sample["latent"], dtype=torch_mod.float32)
    noise = torch_mod.randn_like(latent)
    batch: dict[str, Any] = {
        "latent": latent.unsqueeze(0),
        "noise": noise.unsqueeze(0),
    }

    prompt_raw = sample.get("prompt")
    if prompt_raw is not None:
        prompt_t = torch_mod.as_tensor(prompt_raw, dtype=torch_mod.float32)
        batch["prompt"] = prompt_t.unsqueeze(0) if prompt_t.ndim == 1 else prompt_t
        enc = prompt_t
        while enc.ndim < 3:
            enc = enc.unsqueeze(0)
        batch["encoder_hidden_states"] = enc

    fgpp_raw = sample.get("fgpp_cond")
    if fgpp_raw is not None:
        batch["fgpp_cond"] = torch_mod.as_tensor(fgpp_raw, dtype=torch_mod.float32).unsqueeze(0)
    return batch


def run_lora_shard_training(
    trainer: Any,
    model: Any,
    shard_paths: list[Path],
    max_steps: int,
) -> tuple[int, float]:
    """Cycle WebDataset feature shards for *max_steps* and return ``(steps, final_loss)``."""
    import torch

    from synapse.training.feature_store import iter_feature_shards

    cached_samples = list(iter_feature_shards(shard_paths))
    if not cached_samples:
        raise AssetNotFoundError(
            "No valid samples found in feature shards.",
            agent_fix_hint="Run 'synapse lora prepare-data' first to populate valid .tar shards.",
        )

    steps = 0
    final_loss = 0.0
    while steps < max_steps:
        sample = cached_samples[steps % len(cached_samples)]
        batch = prepare_lora_training_batch(sample, torch)
        result = trainer.train_step(model, batch, steps)
        final_loss = float(result["loss"])
        steps += 1
    return steps, final_loss
