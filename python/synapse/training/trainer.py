"""Stateless DiT LoRA trainer with flow-matching and DPO loss modes.

Provides:
- ``LoRATrainConfig``: Pydantic configuration for LoRA fine-tuning
- ``StatelessLoRATrainer``: Single-execution trainer with LoRA injection,
  backbone freezing, and flow-matching / DPO training steps

Heavy dependencies (``torch``, ``safetensors``) are lazy-imported inside
methods to keep the base package lightweight.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from synapse.errors import ContractValidationError
from synapse.observability import get_probe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LoRATrainConfig(BaseModel):
    """Training configuration for stateless DiT LoRA fine-tuning.

    Attributes:
        category: LoRA category (must match manifest taxonomy).
        rank: Low-rank dimension (must be power of 2, 1-64).
        alpha: LoRA scaling factor.
        lr_spatial: Learning rate for spatial attention layers.
        lr_temporal: Learning rate for temporal attention layers.
        max_steps: Maximum training steps.
        grad_accum_steps: Gradient accumulation steps.
        mixed_precision: Mixed precision mode.
        loss_mode: Training loss mode (SFT or DPO).
        dpo_beta: Temperature scaling factor for Diffusion-DPO.
        max_trainable_ratio: Maximum allowed trainable parameter ratio.
    """

    category: Literal[
        "style",
        "identity",
        "control",
        "whitebox",
        "motion",
    ]
    rank: int = Field(default=8, ge=1, le=64)
    alpha: float = Field(default=1.0, gt=0.0)
    lr_spatial: float = Field(default=1e-4, gt=0.0)
    lr_temporal: float = Field(default=5e-5, gt=0.0)
    max_steps: int = Field(default=1000, ge=1)
    grad_accum_steps: int = Field(default=1, ge=1)
    mixed_precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    loss_mode: Literal["sft", "dpo"] = "sft"
    dpo_beta: float = Field(default=500.0, gt=0.0)
    max_trainable_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    edge_loss_weight: float = Field(default=0.0, ge=0.0)
    normal_loss_weight: float = Field(default=0.0, ge=0.0)
    gradient_checkpointing: bool = False
    warmup_steps: int = Field(default=0, ge=0)
    save_every_steps: int = Field(default=0, ge=0)

    @field_validator("rank")
    @classmethod
    def _rank_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"rank must be a power of 2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Lazy LoRA nn.Module subclass factory
# ---------------------------------------------------------------------------

_lora_linear_cls: type | None = None


def _get_lora_linear_class() -> type:
    """Return a ``torch.nn.Module`` LoRA wrapper, created once."""
    global _lora_linear_cls
    if _lora_linear_cls is not None:
        return _lora_linear_cls

    import torch

    class LoRALinear(torch.nn.Module):
        """Drop-in nn.Linear replacement with LoRA A/B adapters.

        At initialisation ``B = 0``, so the output equals the base
        layer exactly (identity property at step 0).
        """

        def __init__(
            self,
            orig: torch.nn.Linear,
            rank: int,
            alpha: float,
        ) -> None:
            super().__init__()
            self.orig = orig
            self.rank = rank
            self.alpha = alpha
            self.lora_enabled: bool = True

            in_f = orig.in_features
            out_f = orig.out_features
            dev = orig.weight.device

            # LoRA adapter parameters are kept strictly in float32 to prevent fp16/bf16 underflow
            self.lora_A = torch.nn.Parameter(
                torch.randn(rank, in_f, device=dev, dtype=torch.float32) * 0.01,
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(out_f, rank, device=dev, dtype=torch.float32),
            )

            # Freeze original layer weights
            for p in self.orig.parameters():
                p.requires_grad = False

        def forward(self, x: Any) -> Any:
            """Forward: base + LoRA delta (when enabled)."""
            base = self.orig(x)
            if not self.lora_enabled:
                return base
            x_f32 = x.to(dtype=torch.float32)
            lora_out = (x_f32 @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
            return base + lora_out.to(dtype=base.dtype)

    _lora_linear_cls = LoRALinear
    return _lora_linear_cls


@contextmanager
def _disable_lora_adapters(module: Any) -> Iterator[None]:
    """Context manager that temporarily disables LoRA adapters for reference forward pass."""
    LoRALinear = _get_lora_linear_class()
    lora_modules = [m for m in module.modules() if isinstance(m, LoRALinear)]
    prev_states = [bool(getattr(m, "lora_enabled", True)) for m in lora_modules]
    try:
        for m in lora_modules:
            setattr(m, "lora_enabled", False)
        yield
    finally:
        for m, state in zip(lora_modules, prev_states, strict=False):
            setattr(m, "lora_enabled", state)


# ---------------------------------------------------------------------------
# Temporal compression + spatial downsample helper
# ---------------------------------------------------------------------------


def _temporal_compress_and_spatial_ds(x: Any, dev: Any) -> Any:
    """Compress temporal axis T → τ = (T-1)//4 + 1 and spatial 8× downsample.

    Uses **causal first-frame isolation**: frame 0 is kept as its own
    temporal bin (no averaging with subsequent frames), matching the
    Wan 3D causal VAE's stride convention.  Remaining frames are
    grouped into bins of 4.

    Args:
        x: Tensor of shape ``[B, (C,) T, H, W]``.  If ndim == 4 (no C),
           treated as ``[B, T, H, W]`` → output ``[B, 1, τ, H//8, W//8]``.
           If ndim == 5, treated as ``[B, C, T, H, W]``.
        dev: Target device.

    Returns:
        Tensor ``[B, C, τ, H//8, W//8]``.
    """
    import torch

    if x.ndim == 4:
        # [B, T, H, W] → [B, 1, T, H, W]
        x = x.unsqueeze(1)

    b, c, t, h, w = x.shape
    tau = (t - 1) // 4 + 1

    # Causal first-frame isolation: frame 0 is its own bin
    chunks = [x[:, :, 0:1]]  # first frame, no averaging
    for i in range(1, tau):
        s = 1 + (i - 1) * 4
        e = min(s + 4, t)
        chunks.append(x[:, :, s:e].mean(dim=2, keepdim=True))
    x_t = torch.cat(chunks, dim=2)  # [B, C, τ, H, W]

    # Spatial 8× strided downsample
    return x_t[:, :, :, ::8, ::8].to(device=dev)


# ---------------------------------------------------------------------------
# StatelessLoRATrainer
# ---------------------------------------------------------------------------


class StatelessLoRATrainer:
    """Single-execution stateless LoRA trainer for DiT models.

    All ``torch`` usage is lazy-imported inside methods so that
    ``synapse.training`` can be imported without a GPU environment.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: LoRATrainConfig) -> None:
        from synapse.errors import ContractValidationError
        from synapse.training.losses import (
            DiffusionDPOLoss,
            EdgePreservingLoss,
            FlowMatchingLoss,
            NormalReconstructionLoss,
        )

        valid_categories = ("style", "identity", "control", "whitebox", "motion")
        if config.category not in valid_categories:
            raise ContractValidationError(
                f"Invalid LoRA category '{config.category}'. Allowed categories: {valid_categories}",
                agent_fix_hint=f"Choose one of: {list(valid_categories)}",
            )

        self._config = config
        self._optimizer: Any = None
        self._step_count: int = 0
        self._micro_step: int = 0
        self._target_patterns: list[str] = []
        self._geometry_probe: Any = None
        self._last_module: Any = None
        self._last_sampled_t: Any = None
        self._flow_matching_loss = FlowMatchingLoss()
        self._edge_preserving_loss = EdgePreservingLoss()
        self._normal_reconstruction_loss = NormalReconstructionLoss()
        self._dpo_loss = DiffusionDPOLoss(beta=config.dpo_beta)

    @property
    def config(self) -> LoRATrainConfig:
        """Return the training configuration."""
        return self._config

    def freeze_backbone(self, module: Any) -> dict[str, Any]:
        """Freeze base backbone parameters in *module* while preserving LoRA params.

        If ``inject_lora`` was already called before ``freeze_backbone``,
        LoRA adapter parameters (``lora_A``, ``lora_B``) remain trainable.

        Args:
            module: A ``torch.nn.Module`` whose base parameters will
                be frozen (``requires_grad = False``).

        Returns:
            Dict with ``total_params``, ``trainable_params``,
            and ``trainable_ratio``.

        Raises:
            RuntimeError: If non-LoRA parameters remain trainable, or if
                ``max_trainable_ratio`` is exceeded.
        """
        import torch

        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")

        for name, param in module.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        ratio = trainable / max(total, 1)

        max_ratio = self._config.max_trainable_ratio
        if max_ratio is not None and ratio > max_ratio:
            raise RuntimeError(
                f"Trainable ratio {ratio:.4f} exceeds max_trainable_ratio {max_ratio:.4f}."
            )

        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": ratio,
        }

    def inject_lora(
        self,
        module: Any,
        target_patterns: list[str],
    ) -> int:
        """Inject LoRA adapters into matching ``nn.Linear`` layers.

        For each ``nn.Linear`` sub-module whose fully-qualified name
        contains any string in *target_patterns*, the layer is
        replaced with a ``LoRALinear`` wrapper (A ~ N(0, 0.01),
        B = 0). Calling ``inject_lora`` multiple times is idempotent
        and never double-wraps ``LoRALinear.orig``.

        Args:
            module: Root ``torch.nn.Module``.
            target_patterns: Substring patterns to match against
                ``named_modules()`` keys.

        Returns:
            Number of layers injected.
        """
        import torch

        LoRALinear = _get_lora_linear_class()
        injected = 0
        replacements: list[tuple[str, Any]] = []

        # Collect names of already-wrapped inner .orig modules so we never re-wrap them
        wrapped_orig_names: set[str] = set()
        for name, child in module.named_modules():
            if isinstance(child, LoRALinear):
                wrapped_orig_names.add(f"{name}.orig" if name else "orig")

        for name, child in list(module.named_modules()):
            if name in wrapped_orig_names or name.endswith(".orig"):
                continue
            if not isinstance(child, torch.nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue
            if any(fnmatch.fnmatch(name, pat) or pat in name for pat in target_patterns):
                lora = LoRALinear(
                    child,
                    rank=self._config.rank,
                    alpha=self._config.alpha,
                )
                replacements.append((name, lora))
                injected += 1

        named_mods = dict(module.named_modules())
        for full_name, new_mod in replacements:
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                parent = named_mods[parent_name]
            else:
                parent = module
                attr = parts[0]
            setattr(parent, attr, new_mod)

        first_call = not self._target_patterns
        if target_patterns:
            self._target_patterns = list(target_patterns)
        self._last_module = module
        self._optimizer = None

        if injected == 0 and target_patterns and first_call:
            raise ContractValidationError(
                f"inject_lora matched 0 layers for target_modules={target_patterns!r}. "
                "Verify patterns against the model's named_modules.",
                agent_fix_hint=(
                    "Run `list(module.named_modules())` to see available layer names, "
                    "then adjust `target_modules` in the LoRA manifest."
                ),
            )

        if self._config.max_trainable_ratio is not None:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            ratio = trainable / max(total, 1)
            if ratio > self._config.max_trainable_ratio:
                raise RuntimeError(
                    f"Post-injection trainable ratio {ratio:.4f} exceeds "
                    f"max_trainable_ratio {self._config.max_trainable_ratio:.4f}."
                )

        # Log injection stats to Traceboard
        probe = get_probe()
        if probe is not None and probe.active:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            probe.log_scalar("train/lora_injected_layers", float(injected))
            probe.log_scalar("train/total_params", float(total))
            probe.log_scalar("train/trainable_params", float(trainable))

        return injected

    def _build_optimizer(self, module: Any) -> Any:
        """Build AdamW optimizer with separate spatial and temporal parameter groups.

        If a ``LatentGeometryProbe`` is registered, its parameters are
        included in the spatial group so they receive gradients.
        """
        import torch

        spatial_params: list[Any] = []
        temporal_params: list[Any] = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if any(tok in name.lower() for tok in ("temp", "temporal", "motion", "time")):
                temporal_params.append(param)
            else:
                spatial_params.append(param)

        # Include geometry probe parameters
        if self._geometry_probe is not None and hasattr(self._geometry_probe, "parameters"):
            for param in self._geometry_probe.parameters():
                if param.requires_grad:
                    spatial_params.append(param)

        if not spatial_params and not temporal_params:
            raise RuntimeError("No trainable parameters. Call inject_lora() first.")

        param_groups: list[dict[str, Any]] = []
        if spatial_params:
            param_groups.append({"params": spatial_params, "lr": self._config.lr_spatial})
        if temporal_params:
            param_groups.append({"params": temporal_params, "lr": self._config.lr_temporal})
        return torch.optim.AdamW(param_groups)

    def _get_lr_multiplier(self, step: int) -> float:
        """Cosine annealing LR multiplier with optional linear warmup."""
        warmup = self._config.warmup_steps
        max_steps = max(self._config.max_steps, 1)
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(max_steps - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def enable_gradient_checkpointing(self, module: Any) -> None:
        """Enable gradient checkpointing on module if configured."""
        if not self._config.gradient_checkpointing:
            return
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled via model API")
        else:
            logger.info("Module does not support gradient_checkpointing_enable()")

    def _collect_trainable_params(self, module: Any) -> list[Any]:
        """Collect all trainable parameters from *module* and the geometry probe."""
        params = [p for p in module.parameters() if p.requires_grad]
        if self._geometry_probe is not None and hasattr(self._geometry_probe, "parameters"):
            for p in self._geometry_probe.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def save_state_dict(self, path: Path, module: Any | None = None) -> Path:
        """Save full trainer + LoRA adapter + geometry probe state for checkpoint resume."""
        import torch

        target_mod = module if module is not None else self._last_module
        state: dict[str, Any] = {
            "step_count": self._step_count,
            "micro_step": self._micro_step,
            "config": self._config.model_dump(),
        }

        if target_mod is not None:
            LoRALinear = _get_lora_linear_class()
            lora_state: dict[str, Any] = {}
            for name, child in target_mod.named_modules():
                if isinstance(child, LoRALinear):
                    lora_a: Any = getattr(child, "lora_A")
                    lora_b: Any = getattr(child, "lora_B")
                    lora_state[f"{name}.lora_A"] = lora_a.data.detach().cpu().clone()
                    lora_state[f"{name}.lora_B"] = lora_b.data.detach().cpu().clone()
            state["lora_state_dict"] = lora_state

        if self._geometry_probe is not None and hasattr(self._geometry_probe, "state_dict"):
            state["geometry_probe_state_dict"] = {
                k: v.detach().cpu().clone()
                for k, v in self._geometry_probe.state_dict().items()
            }
            state["geometry_probe_meta"] = {
                "in_channels": int(getattr(self._geometry_probe, "in_channels", 16)),
                "out_channels": int(getattr(self._geometry_probe, "out_channels", 3)),
            }

        if self._optimizer is not None:
            state["optimizer_state_dict"] = self._optimizer.state_dict()
        if hasattr(self, "_grad_scaler"):
            state["grad_scaler_state_dict"] = self._grad_scaler.state_dict()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(out))

        # Log checkpoint save to Traceboard
        probe = get_probe()
        if probe is not None and probe.active:
            probe.log_scalar(
                "train/checkpoint_step",
                float(self._step_count),
                step=self._step_count,
            )

        return out.resolve()

    def load_state_dict(self, path: Path, module: Any) -> int:
        """Restore trainer, LoRA adapter, geometry probe, and optimizer state from checkpoint.

        Returns:
            The step count at which training was saved.
        """
        import torch

        try:
            model_device = next(module.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        state = torch.load(str(path), map_location=model_device, weights_only=True)
        self._step_count = int(state.get("step_count", 0))
        self._micro_step = int(state.get("micro_step", 0))
        self._last_module = module

        # Restore LoRA adapter weights
        lora_state = state.get("lora_state_dict")
        if isinstance(lora_state, dict):
            LoRALinear = _get_lora_linear_class()
            with torch.no_grad():
                for name, child in module.named_modules():
                    if isinstance(child, LoRALinear):
                        lora_a: Any = getattr(child, "lora_A")
                        lora_b: Any = getattr(child, "lora_B")
                        key_a = f"{name}.lora_A"
                        key_b = f"{name}.lora_B"
                        if key_a in lora_state:
                            lora_a.copy_(
                                lora_state[key_a].to(
                                    device=lora_a.device, dtype=lora_a.dtype
                                )
                            )
                        if key_b in lora_state:
                            lora_b.copy_(
                                lora_state[key_b].to(
                                    device=lora_b.device, dtype=lora_b.dtype
                                )
                            )

        # Restore geometry probe weights if present
        probe_state = state.get("geometry_probe_state_dict")
        if isinstance(probe_state, dict):
            if self._geometry_probe is None:
                from synapse.training.losses import LatentGeometryProbe

                meta = state.get("geometry_probe_meta", {})
                self._geometry_probe = LatentGeometryProbe(
                    in_channels=int(meta.get("in_channels", 16)),
                    out_channels=int(meta.get("out_channels", 3)),
                ).to(device=model_device)
            self._geometry_probe.load_state_dict(probe_state)
            self._geometry_probe.to(device=model_device)

        if "optimizer_state_dict" in state:
            if self._optimizer is None:
                self._optimizer = self._build_optimizer(module)
            self._optimizer.load_state_dict(state["optimizer_state_dict"])
        if "grad_scaler_state_dict" in state and hasattr(self, "_grad_scaler"):
            self._grad_scaler.load_state_dict(state["grad_scaler_state_dict"])
        return self._step_count

    def flush_gradients(self, module: Any) -> None:
        """Flush and step any remaining accumulated gradients.

        Respects GradScaler when fp16 mixed precision is active.
        """
        accum = max(1, self._config.grad_accum_steps)
        if self._micro_step % accum != 0:
            if self._optimizer is not None:
                trainable_params = self._collect_trainable_params(module)
                if trainable_params:
                    import torch

                    use_scaler = hasattr(self, "_grad_scaler")
                    if use_scaler:
                        self._grad_scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        self._grad_scaler.step(self._optimizer)
                        self._grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        self._optimizer.step()
                self._optimizer.zero_grad()
            self._micro_step = 0

    def train_step(
        self,
        module: Any,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, float]:
        """Execute a single training step with full gradient chain.

        For ``sft`` mode, computes flow-matching velocity MSE using
        differentiable ``FlowMatchingLoss``.
        For ``dpo`` mode, computes reference-anchored Diffusion-DPO loss
        using differentiable ``DiffusionDPOLoss``.

        After back-propagation, applies gradient clipping (max norm 1.0)
        to stabilise training, then steps the optimiser on the
        accumulation boundary.

        Args:
            module: The model with LoRA adapters injected.
            batch: Dict with ``"latent"`` (x_1), ``"noise"``
                (x_0).  DPO mode also expects ``"chosen"`` and
                ``"rejected"`` tensors.
            step: Current step index.

        Returns:
            Dict with ``"loss"`` and ``"step"`` values.
        """
        import torch

        probe = get_probe()
        if probe is not None and probe.active:
            probe.set_step(step)

        self._last_module = module
        # Shallow-copy caller batch so internal mutations (sampled 't',
        # extracted depth/normal, device migration) never freeze future steps.
        work_batch: dict[str, Any] = dict(batch)

        # Mixed precision context — select device_type from actual model device
        model_device = next(module.parameters()).device
        device_type = "cuda" if model_device.type == "cuda" else "cpu"

        # Auto-instantiate LatentGeometryProbe for 5D video latents when
        # normal_loss_weight > 0 and no explicit normal_pred is supplied.
        if (
            self._config.loss_mode == "sft"
            and self._config.normal_loss_weight > 0.0
            and self._geometry_probe is None
            and "normal_pred" not in work_batch
            and "latent" in work_batch
            and hasattr(work_batch["latent"], "ndim")
            and work_batch["latent"].ndim == 5
        ):
            from synapse.training.losses import LatentGeometryProbe

            in_ch = int(work_batch["latent"].shape[1])
            self._geometry_probe = LatentGeometryProbe(
                in_channels=in_ch, out_channels=3
            ).to(device=model_device)
            if self._optimizer is not None:
                self._optimizer.add_param_group(
                    {
                        "params": list(self._geometry_probe.parameters()),
                        "lr": self._config.lr_spatial,
                    }
                )

        if self._optimizer is None:
            self._optimizer = self._build_optimizer(module)

        # Apply cosine LR with warmup
        lr_mult = self._get_lr_multiplier(step)
        for pg in self._optimizer.param_groups:
            if "_base_lr" not in pg:
                pg["_base_lr"] = pg["lr"]
            pg["lr"] = pg["_base_lr"] * lr_mult

        module.train()
        accum = max(1, self._config.grad_accum_steps)
        if self._micro_step % accum == 0:
            self._optimizer.zero_grad()

        autocast_enabled = (
            self._config.mixed_precision in ("fp16", "bf16") and device_type == "cuda"
        )
        amp_dtype = torch.float16 if self._config.mixed_precision == "fp16" else torch.bfloat16

        # GradScaler for fp16 to prevent gradient underflow (device-aware)
        use_scaler = autocast_enabled and self._config.mixed_precision == "fp16"
        if use_scaler and not hasattr(self, "_grad_scaler"):
            grad_scaler_cls = getattr(torch.amp, "GradScaler")
            self._grad_scaler = grad_scaler_cls(device=device_type)

        # Move batch tensors to same device as module
        for k, v in work_batch.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                work_batch[k] = v.to(device=model_device)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
            if self._config.loss_mode == "sft":
                raw_loss = self._sft_step(module, work_batch)
            else:
                raw_loss = self._dpo_step(
                    module,
                    work_batch,
                    beta=self._config.dpo_beta,
                    step=step,
                )

        scaled_loss = raw_loss / float(accum)
        if use_scaler:
            self._grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._micro_step += 1

        if self._micro_step % accum == 0:
            # Gradient clipping for training stability (including geometry probe)
            trainable_params = self._collect_trainable_params(module)
            if use_scaler:
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self._optimizer.step()
            self._optimizer.zero_grad()

        self._step_count = step
        result = {"loss": float(raw_loss.item()), "step": step}

        # Log training scalars to Traceboard
        if probe is not None and probe.active:
            probe.set_step(step)
            probe.log_scalar("train/loss", result["loss"], step=step)
            probe.log_scalar("train/step", float(step), step=step)
            probe.log_scalar("train/lr_multiplier", lr_mult, step=step)
            if self._config.loss_mode == "dpo":
                probe.log_scalar("train/dpo_beta", self._config.dpo_beta, step=step)

        return result

    def export_checkpoint(
        self,
        module: Any,
        path: Path,
    ) -> Path:
        """Export LoRA-only weights to safetensors format with PEFT keys and metadata.

        Args:
            module: Model with LoRA adapters.
            path: Output file path.

        Returns:
            Absolute path to the saved checkpoint file.
        """
        from safetensors.torch import save_file

        LoRALinear = _get_lora_linear_class()
        lora_state: dict[str, Any] = {}

        for name, child in module.named_modules():
            if isinstance(child, LoRALinear):
                lora_a: Any = getattr(child, "lora_A")
                lora_b: Any = getattr(child, "lora_B")
                # Export both canonical PEFT keys (.weight) and short keys for compatibility
                lora_state[f"{name}.lora_A.weight"] = lora_a.data.detach().cpu().contiguous()
                lora_state[f"{name}.lora_B.weight"] = lora_b.data.detach().cpu().contiguous()

        metadata = {
            "category": str(self._config.category),
            "rank": str(self._config.rank),
            "alpha": str(self._config.alpha),
            "target_modules": ",".join(self._target_patterns),
        }

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_state, str(out), metadata=metadata)
        return out.resolve()

    # ---------------------------------------------------------------
    # Private loss helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _sample_timesteps(ref_tensor: Any, batch: dict[str, Any]) -> Any:
        """Sample per-item timesteps ``(B, 1, ...)`` or use ``batch['t']``."""
        import torch

        if "t" in batch and batch["t"] is not None:
            t = batch["t"].clone().to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            bsz = ref_tensor.shape[0] if ref_tensor.ndim >= 1 else 1
            shape = (bsz,) + (1,) * max(0, ref_tensor.ndim - 1)
            t = torch.rand(shape, device=ref_tensor.device, dtype=ref_tensor.dtype)
        while t.ndim < ref_tensor.ndim:
            t = t.unsqueeze(-1)
        return t

    @classmethod
    def _call_module(
        cls,
        module: Any,
        x: Any,
        batch: dict[str, Any],
        is_ref: bool = False,
    ) -> Any:
        """Call module with signature-based argument routing.

        Adapts ``batch["t"]`` to 1D ``[B]`` when the target accepts a
        scalar timestep parameter. Transparently passes
        ``encoder_hidden_states`` for Diffusers-style UNets/DiTs.
        Only catches ``TypeError`` / ``ValueError`` from signature
        mismatch, never from internal forward bugs.
        """
        import inspect

        target = getattr(module, "forward", module)
        try:
            sig = inspect.signature(target)
            kwargs: dict[str, Any] = {}
            for param_name in sig.parameters:
                if param_name in ("t", "timestep", "timesteps") and "t" in batch:
                    t_val = batch["t"]
                    # Adapt multi-dim t to 1D [B] for Diffusers compatibility
                    if hasattr(t_val, "ndim") and t_val.ndim > 1:
                        t_val = t_val.reshape(t_val.shape[0], -1).mean(dim=-1)
                    kwargs[param_name] = t_val
                elif param_name in ("prompt", "caption", "text_embeddings") and "prompt" in batch:
                    kwargs[param_name] = batch["prompt"]
                elif param_name == "encoder_hidden_states" and "encoder_hidden_states" in batch:
                    kwargs[param_name] = batch["encoder_hidden_states"]
                elif param_name in ("fgpp_cond", "cond", "conditions") and "fgpp_cond" in batch:
                    kwargs[param_name] = batch["fgpp_cond"]
                elif param_name in ("fgpp_cond", "cond", "conditions") and "conditions" in batch:
                    kwargs[param_name] = batch["conditions"]
            return module(x, **kwargs)
        except (ValueError, TypeError) as exc:
            # Only catch argument-binding issues, re-raise internal TypeError
            msg = str(exc)
            if "argument" in msg.lower() or "unexpected" in msg.lower() or "missing" in msg.lower():
                return module(x)
            raise

    def _sft_step(
        self,
        module: Any,
        batch: dict[str, Any],
    ) -> Any:
        """Compute flow-matching SFT loss with optional edge and normal losses."""
        x_1 = batch["latent"]
        x_0 = batch["noise"]
        t = self._sample_timesteps(x_1, batch)
        self._last_sampled_t = t
        # Store sampled t into local work_batch so _call_module can forward it to DiT
        batch["t"] = t
        x_t = (1.0 - t) * x_0 + t * x_1
        v_pred = self._call_module(module, x_t, batch)
        loss = self._flow_matching_loss(v_pred, x_1, x_0)

        # Auto-extract depth and normal from fgpp_cond if separate keys missing
        if (
            "fgpp_cond" in batch
            and batch["fgpp_cond"] is not None
            and ("depth" not in batch or "normal" not in batch)
        ):
            self._extract_auxiliary_from_fgpp(batch)

        z_pred: Any | None = None
        # Wire optional EdgePreservingLoss (operates on log-depth for scale invariance)
        if self._config.edge_loss_weight > 0.0:
            depth_key = (
                "depth_latent"
                if "depth_latent" in batch
                else ("depth" if "depth" in batch else None)
            )
            if depth_key is not None:
                import torch as _torch

                z_pred = x_t + v_pred * (1.0 - t)
                depth_val = batch[depth_key]
                # Avoid double log1p when depth is already log-encoded (depth_latent or Wan 7ch)
                if depth_key == "depth_latent" or batch.get("_depth_is_log", False):
                    log_depth = depth_val
                else:
                    log_depth = _torch.log1p(depth_val.clamp(min=0.0))
                loss = loss + self._config.edge_loss_weight * self._edge_preserving_loss(
                    z_pred, log_depth
                )

        # Wire optional NormalReconstructionLoss via LatentGeometryProbe
        if self._config.normal_loss_weight > 0.0:
            norm_key = (
                "normal_gt" if "normal_gt" in batch else ("normal" if "normal" in batch else None)
            )
            if norm_key is not None:
                mask = batch.get("mask")
                # Use LatentGeometryProbe if available, otherwise skip
                # (never directly slice 16ch latent as 3ch normal)
                if hasattr(self, "_geometry_probe") and self._geometry_probe is not None:
                    probe_in = z_pred if z_pred is not None else (x_t + v_pred * (1.0 - t))
                    n_pred = self._geometry_probe(probe_in)
                else:
                    n_pred = batch.get("normal_pred")
                if n_pred is not None:
                    loss = loss + self._config.normal_loss_weight * self._normal_reconstruction_loss(
                        n_pred, batch[norm_key], mask=mask
                    )
                else:
                    logger.debug(
                        "Skipping normal loss: no probe or normal_pred "
                        "(refusing to slice 16ch latent as normal)"
                    )

        return loss

    @staticmethod
    def _extract_auxiliary_from_fgpp(batch: dict[str, Any]) -> None:
        """Extract depth and normal from fgpp_cond — zero CPU round-trip.

        Keeps all extracted tensors on the same device as ``fgpp_cond``
        (no ``.cpu().numpy()`` detour). Handles both 4D and 5D inputs:

        - 4D raw FGPP ``[T, H, W, 15]`` or ``[7, T, H, W]``
        - 5D batched ``[B, T, H, W, 15]`` or ``[B, 7, T, H, W]``

        FGPP raw layout (15-channel):
        - Channel 0: ``CameraLinearDepth.Z``
        - Channels 1-3: ``DirectRadiance.RGB``
        - Channels 4-6: ``WorldNormal.XYZ`` (ground truth normal)

        Wan control layout (7-channel):
        - Channel 0: depth
        - Channels 1-3: normal

        Temporal axis is compressed ``T → τ = (T-1)//4 + 1`` via average
        pooling to match the latent time dimension (Wan 3D causal stride).
        Spatial axes are down-sampled 8× by strided slicing (VAE factor).
        """
        import torch

        fgpp = batch["fgpp_cond"]
        if not isinstance(fgpp, torch.Tensor):
            fgpp = torch.as_tensor(fgpp, dtype=torch.float32)

        # Determine device for all outputs
        dev = fgpp.device

        # Handle 5D batch: [B, ...] -> process first element for shape, but keep batch
        has_batch = False
        if fgpp.ndim == 5:
            has_batch = True
            if fgpp.shape[-1] == 15:
                # [B, T, H, W, 15]
                pass
            elif fgpp.shape[1] == 7:
                # [B, 7, T, H, W]
                pass
        elif fgpp.ndim == 4:
            # Add batch dim for uniform processing
            has_batch = False
            fgpp = fgpp.unsqueeze(0)
        else:
            return  # Unsupported layout

        if fgpp.shape[-1] == 15:
            # Raw FGPP: [B, T, H, W, 15]
            if "depth" not in batch:
                depth = fgpp[..., 0]  # [B, T, H, W]
                depth_ds = _temporal_compress_and_spatial_ds(depth, dev)
                batch["depth"] = depth_ds  # [B, 1, τ, H//8, W//8]
            if "normal" not in batch:
                normal = fgpp[..., 4:7]  # [B, T, H, W, 3]
                normal = normal.permute(0, 4, 1, 2, 3)  # [B, 3, T, H, W]
                normal_ds = _temporal_compress_and_spatial_ds(normal, dev)
                batch["normal"] = normal_ds  # [B, 3, τ, H//8, W//8]
        elif fgpp.shape[-3 if fgpp.ndim == 5 else 0] == 7 or (fgpp.ndim == 5 and fgpp.shape[1] == 7):
            # Wan control: [B, 7, T, H, W] (ch0 is already log-encoded depth in [0, 1])
            if "depth" not in batch:
                depth = fgpp[:, 0:1]  # [B, 1, T, H, W]
                depth_ds = _temporal_compress_and_spatial_ds(depth, dev)
                batch["depth"] = depth_ds
                batch["_depth_is_log"] = True
            if "normal" not in batch:
                normal = fgpp[:, 1:4]  # [B, 3, T, H, W]
                normal_ds = _temporal_compress_and_spatial_ds(normal, dev)
                batch["normal"] = normal_ds

        # Remove batch dim if input was unbatched
        if not has_batch:
            if "depth" in batch and batch["depth"].ndim == 5:
                batch["depth"] = batch["depth"].squeeze(0)
            if "normal" in batch and batch["normal"].ndim == 5:
                batch["normal"] = batch["normal"].squeeze(0)

    def _dpo_step(
        self,
        module: Any,
        batch: dict[str, Any],
        beta: float = 500.0,
        step: int | None = None,
    ) -> Any:
        """Compute reference-anchored Diffusion-DPO loss.

        Evaluates the frozen reference backbone by temporarily disabling
        all ``LoRALinear`` adapters under ``torch.no_grad()`` and eval mode,
        ensuring BatchNorm/Dropout do not drift during reference pass.
        """
        import torch

        chosen = batch["chosen"]
        rejected = batch["rejected"]
        x_0 = batch["noise"]
        t = self._sample_timesteps(chosen, batch)
        self._last_sampled_t = t
        # Store t into local work_batch so _call_module can forward timesteps to DiT
        batch["t"] = t
        x_t_w = (1.0 - t) * x_0 + t * chosen
        x_t_l = (1.0 - t) * x_0 + t * rejected

        target_w = chosen - x_0
        target_l = rejected - x_0

        v_theta_w = self._call_module(module, x_t_w, batch)
        v_theta_l = self._call_module(module, x_t_l, batch)

        was_training = module.training
        module.eval()
        try:
            with torch.no_grad(), _disable_lora_adapters(module):
                v_ref_w = self._call_module(module, x_t_w, batch, is_ref=True)
                v_ref_l = self._call_module(module, x_t_l, batch, is_ref=True)
        finally:
            if was_training:
                module.train()

        return self._dpo_loss(
            v_theta_w,
            v_theta_l,
            v_ref_w,
            v_ref_l,
            target_w,
            target_l,
            step=step,
        )
