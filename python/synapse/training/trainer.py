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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from synapse.errors import ContractValidationError

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
        "style", "identity", "control", "whitebox", "motion",
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

    @field_validator("rank")
    @classmethod
    def _rank_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(
                f"rank must be a power of 2, got {v}"
            )
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
            lora_out = (
                x_f32 @ self.lora_A.T @ self.lora_B.T
            ) * (self.alpha / self.rank)
            return base + lora_out.to(dtype=base.dtype)

    _lora_linear_cls = LoRALinear
    return _lora_linear_cls


@contextmanager
def _disable_lora_adapters(module: Any) -> Iterator[None]:
    """Context manager that temporarily disables LoRA adapters for reference forward pass."""
    LoRALinear = _get_lora_linear_class()
    lora_modules = [
        m for m in module.modules() if isinstance(m, LoRALinear)
    ]
    prev_states = [bool(getattr(m, "lora_enabled", True)) for m in lora_modules]
    try:
        for m in lora_modules:
            setattr(m, "lora_enabled", False)
        yield
    finally:
        for m, state in zip(lora_modules, prev_states, strict=False):
            setattr(m, "lora_enabled", state)


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
        trainable = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        ratio = trainable / max(total, 1)

        max_ratio = self._config.max_trainable_ratio
        if max_ratio is not None and ratio > max_ratio:
            raise RuntimeError(
                f"Trainable ratio {ratio:.4f} exceeds max_trainable_ratio "
                f"{max_ratio:.4f}."
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
            trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            ratio = trainable / max(total, 1)
            if ratio > self._config.max_trainable_ratio:
                raise RuntimeError(
                    f"Post-injection trainable ratio {ratio:.4f} exceeds "
                    f"max_trainable_ratio {self._config.max_trainable_ratio:.4f}."
                )

        return injected

    def _build_optimizer(self, module: Any) -> Any:
        """Build AdamW optimizer with separate spatial and temporal parameter groups."""
        import torch

        spatial_params: list[Any] = []
        temporal_params: list[Any] = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if any(
                tok in name.lower()
                for tok in ("temp", "temporal", "motion", "time")
            ):
                temporal_params.append(param)
            else:
                spatial_params.append(param)

        if not spatial_params and not temporal_params:
            raise RuntimeError(
                "No trainable parameters. Call inject_lora() first."
            )

        param_groups: list[dict[str, Any]] = []
        if spatial_params:
            param_groups.append(
                {"params": spatial_params, "lr": self._config.lr_spatial}
            )
        if temporal_params:
            param_groups.append(
                {"params": temporal_params, "lr": self._config.lr_temporal}
            )
        return torch.optim.AdamW(param_groups)

    def flush_gradients(self, module: Any) -> None:
        """Flush and step any remaining accumulated gradients."""
        accum = max(1, self._config.grad_accum_steps)
        if self._micro_step % accum != 0:
            if self._optimizer is not None:
                trainable_params = [
                    p for p in module.parameters() if p.requires_grad
                ]
                if trainable_params:
                    import torch

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

        if self._optimizer is None:
            self._optimizer = self._build_optimizer(module)

        module.train()
        accum = max(1, self._config.grad_accum_steps)
        if self._micro_step % accum == 0:
            self._optimizer.zero_grad()

        # Mixed precision context
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        autocast_enabled = (
            self._config.mixed_precision in ("fp16", "bf16")
            and device_type == "cuda"
        )
        amp_dtype = (
            torch.float16 if self._config.mixed_precision == "fp16" else torch.bfloat16
        )

        # GradScaler for fp16 to prevent gradient underflow
        use_scaler = (
            autocast_enabled
            and self._config.mixed_precision == "fp16"
        )
        if use_scaler and not hasattr(self, "_grad_scaler"):
            self._grad_scaler = torch.amp.GradScaler()  # type: ignore[attr-defined]

        # Move batch tensors to same device as module
        for k, v in batch.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                batch[k] = v.to(device=next(module.parameters()).device)

        with torch.autocast(
            device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled
        ):
            if self._config.loss_mode == "sft":
                raw_loss = self._sft_step(module, batch)
            else:
                raw_loss = self._dpo_step(
                    module, batch, beta=self._config.dpo_beta,
                )

        scaled_loss = raw_loss / float(accum)
        if use_scaler:
            self._grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._micro_step += 1

        if self._micro_step % accum == 0:
            # Gradient clipping for training stability
            trainable_params = [
                p for p in module.parameters() if p.requires_grad
            ]
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
        return {"loss": float(raw_loss.item()), "step": step}

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
            t = batch["t"].to(device=ref_tensor.device, dtype=ref_tensor.dtype)
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
        """Call module with signature-based argument routing."""
        import inspect

        target = getattr(module, "forward", module)
        try:
            sig = inspect.signature(target)
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in ("t", "timestep", "timesteps") and "t" in batch:
                    kwargs[param_name] = batch["t"]
                elif (
                    param_name in ("prompt", "caption", "text_embeddings")
                    and "prompt" in batch
                ):
                    kwargs[param_name] = batch["prompt"]
                elif (
                    param_name in ("fgpp_cond", "cond", "conditions")
                    and "fgpp_cond" in batch
                ):
                    kwargs[param_name] = batch["fgpp_cond"]
                elif (
                    param_name in ("fgpp_cond", "cond", "conditions")
                    and "conditions" in batch
                ):
                    kwargs[param_name] = batch["conditions"]
            return module(x, **kwargs)
        except (ValueError, TypeError):
            return module(x)

    def _sft_step(
        self,
        module: Any,
        batch: dict[str, Any],
    ) -> Any:
        """Compute flow-matching SFT loss with optional edge and normal losses."""
        x_1 = batch["latent"]
        x_0 = batch["noise"]
        t = self._sample_timesteps(x_1, batch)
        # Store sampled t back into batch so _call_module can forward it to DiT
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

        # Wire optional EdgePreservingLoss
        if self._config.edge_loss_weight > 0.0:
            depth_key = (
                "depth_latent"
                if "depth_latent" in batch
                else ("depth" if "depth" in batch else None)
            )
            if depth_key is not None:
                z_pred = x_t + v_pred * (1.0 - t)
                loss = loss + self._config.edge_loss_weight * self._edge_preserving_loss(
                    z_pred, batch[depth_key]
                )

        # Wire optional NormalReconstructionLoss
        if self._config.normal_loss_weight > 0.0:
            norm_key = (
                "normal_gt"
                if "normal_gt" in batch
                else ("normal" if "normal" in batch else None)
            )
            if norm_key is not None:
                mask = batch.get("mask")
                n_pred = batch.get("normal_pred", v_pred)
                loss = loss + self._config.normal_loss_weight * self._normal_reconstruction_loss(
                    n_pred, batch[norm_key], mask=mask
                )

        return loss

    @staticmethod
    def _extract_auxiliary_from_fgpp(batch: dict[str, Any]) -> None:
        """Extract depth and normal from fgpp_cond when separate keys are absent.

        FGPP raw layout ``[T, H, W, 15]``:
        - Channel 0: ``CameraLinearDepth.Z``
        - Channels 1-3: ``DirectRadiance.RGB``
        - Channels 4-6: ``WorldNormal.XYZ`` (used as normal ground truth)

        Wan control layout ``[7, T, H, W]``:
        - Channel 0: depth
        - Channels 1-3: normal

        The extracted arrays are down-sampled 8× to match the latent spatial
        resolution (VAE factor) using simple strided slicing.
        """
        import torch

        fgpp = batch["fgpp_cond"]
        if isinstance(fgpp, torch.Tensor):
            fgpp_np = fgpp.detach().cpu().numpy()
        else:
            fgpp_np = fgpp

        import numpy as np

        if fgpp_np.ndim == 4 and fgpp_np.shape[-1] == 15:
            # Raw FGPP: [T, H, W, 15]
            if "depth" not in batch:
                depth = fgpp_np[..., 0]  # [T, H, W]
                depth_ds = depth[:, ::8, ::8]  # downsample 8x
                batch["depth"] = torch.as_tensor(
                    depth_ds, dtype=torch.float32,
                ).unsqueeze(0)  # [1, T, H//8, W//8]
            if "normal" not in batch:
                normal = fgpp_np[..., 4:7]  # [T, H, W, 3]
                normal_ds = normal[:, ::8, ::8, :]
                # Transpose to [3, T, H//8, W//8] for channel-first
                batch["normal"] = torch.as_tensor(
                    np.ascontiguousarray(normal_ds.transpose(3, 0, 1, 2)),
                    dtype=torch.float32,
                ).unsqueeze(0)  # [1, 3, T, H//8, W//8]
        elif fgpp_np.ndim == 4 and fgpp_np.shape[0] == 7:
            # Wan control: [7, T, H, W]
            if "depth" not in batch:
                depth = fgpp_np[0]  # [T, H, W]
                depth_ds = depth[:, ::8, ::8]
                batch["depth"] = torch.as_tensor(
                    depth_ds, dtype=torch.float32,
                ).unsqueeze(0)
            if "normal" not in batch:
                normal = fgpp_np[1:4]  # [3, T, H, W]
                normal_ds = normal[:, :, ::8, ::8]
                batch["normal"] = torch.as_tensor(
                    np.ascontiguousarray(normal_ds),
                    dtype=torch.float32,
                ).unsqueeze(0)  # [1, 3, T, H//8, W//8]

    def _dpo_step(
        self,
        module: Any,
        batch: dict[str, Any],
        beta: float = 500.0,
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
            v_theta_w, v_theta_l, v_ref_w, v_ref_l, target_w, target_l
        )
