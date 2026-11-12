"""Stateless DiT LoRA trainer with flow-matching and DPO loss modes.

Provides:
- ``LoRATrainConfig``: Pydantic configuration for LoRA fine-tuning
- ``StatelessLoRATrainer``: Single-execution trainer with LoRA injection,
  backbone freezing, and flow-matching / DPO training steps

Heavy dependencies (``torch``, ``safetensors``) are lazy-imported inside
methods to keep the base package lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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

            in_f = orig.in_features
            out_f = orig.out_features

            self.lora_A = torch.nn.Parameter(
                torch.randn(rank, in_f) * 0.01,
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(out_f, rank),
            )

            # Freeze original layer weights
            for p in self.orig.parameters():
                p.requires_grad = False

        def forward(self, x: Any) -> Any:
            """Forward: base + LoRA delta."""
            base = self.orig(x)
            lora_out = (
                x @ self.lora_A.T @ self.lora_B.T
            ) * (self.alpha / self.rank)
            return base + lora_out

    _lora_linear_cls = LoRALinear
    return _lora_linear_cls


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
        self._config = config
        self._optimizer: Any = None
        self._step_count: int = 0

    @property
    def config(self) -> LoRATrainConfig:
        """Return the training configuration."""
        return self._config

    def freeze_backbone(self, module: Any) -> dict[str, Any]:
        """Freeze all parameters in *module* and verify ratio.

        Args:
            module: A ``torch.nn.Module`` whose parameters will
                be frozen (``requires_grad = False``).

        Returns:
            Dict with ``total_params``, ``trainable_params``,
            and ``trainable_ratio``.

        Raises:
            RuntimeError: If more than 3 pct of parameters remain
                trainable after freezing.
        """
        import torch

        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")

        for param in module.parameters():
            param.requires_grad = False

        total = sum(p.numel() for p in module.parameters())
        trainable = sum(
            p.numel() for p in module.parameters()
            if p.requires_grad
        )
        ratio = trainable / max(total, 1)

        if ratio > 0.03:
            raise RuntimeError(
                f"Trainable ratio {ratio:.4f} exceeds 3 pct "
                "after freezing; backbone freeze incomplete."
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
        B = 0).

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

        for name, child in list(module.named_modules()):
            if not isinstance(child, torch.nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue
            if any(pat in name for pat in target_patterns):
                lora = LoRALinear(
                    child,
                    rank=self._config.rank,
                    alpha=self._config.alpha,
                )
                replacements.append((name, lora))
                injected += 1

        for full_name, new_mod in replacements:
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                parent = dict(module.named_modules())[
                    parent_name
                ]
            else:
                parent = module
                attr = parts[0]
            setattr(parent, attr, new_mod)

        return injected

    def train_step(
        self,
        module: Any,
        batch: dict[str, Any],
        step: int,
    ) -> dict[str, float]:
        """Execute a single training step.

        For ``sft`` mode, computes flow-matching velocity MSE.
        For ``dpo`` mode, computes diffusion DPO loss.

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
            trainable = [
                p for p in module.parameters()
                if p.requires_grad
            ]
            if not trainable:
                raise RuntimeError(
                    "No trainable parameters. "
                    "Call inject_lora() first."
                )
            self._optimizer = torch.optim.AdamW(
                trainable, lr=self._config.lr_spatial,
            )

        module.train()
        self._optimizer.zero_grad()

        if self._config.loss_mode == "sft":
            loss = self._sft_step(module, batch)
        else:
            loss = self._dpo_step(module, batch)

        loss.backward()
        self._optimizer.step()
        self._step_count = step

        return {"loss": float(loss.item()), "step": step}

    def export_checkpoint(
        self,
        module: Any,
        path: Path,
    ) -> Path:
        """Export LoRA-only weights to safetensors format.

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
                lora_state[f"{name}.lora_A"] = (
                    lora_a.data.clone()
                )
                lora_state[f"{name}.lora_B"] = (
                    lora_b.data.clone()
                )

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_state, str(out))
        return out.resolve()

    # ---------------------------------------------------------------
    # Private loss helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _sft_step(
        module: Any, batch: dict[str, Any],
    ) -> Any:
        """Compute flow-matching SFT loss."""
        import torch

        x_1 = batch["latent"]
        x_0 = batch["noise"]
        t = torch.rand(
            1, device=x_1.device, dtype=x_1.dtype,
        )
        x_t = (1.0 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        v_pred = module(x_t)
        return torch.mean((v_pred - v_target) ** 2)

    @staticmethod
    def _dpo_step(
        module: Any, batch: dict[str, Any],
    ) -> Any:
        """Compute diffusion DPO loss."""
        import torch

        chosen = batch["chosen"]
        rejected = batch["rejected"]
        x_0 = batch["noise"]
        t = torch.rand(
            1, device=chosen.device, dtype=chosen.dtype,
        )
        x_t_w = (1.0 - t) * x_0 + t * chosen
        x_t_l = (1.0 - t) * x_0 + t * rejected

        v_w = module(x_t_w)
        v_l = module(x_t_l)

        err_w = torch.mean((v_w - (chosen - x_0)) ** 2)
        err_l = torch.mean((v_l - (rejected - x_0)) ** 2)

        beta = 500.0
        logit = beta * (err_l - err_w)

        # -log(sigmoid(logit)) = softplus(-logit)
        loss = torch.nn.functional.softplus(-logit)
        return loss

