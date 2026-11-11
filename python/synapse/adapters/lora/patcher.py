"""LoRA forward-hook patcher for spatial mask-gated inference.

Provides ``LoRAPatchHook`` for attaching mask-aware LoRA hooks to
``nn.Linear`` modules, and ``patch_wan_adapter_with_lora`` for
integrating LoRA bundles with the ``WanAdapter``.

Heavy dependencies (``torch``) are lazy-imported inside methods.
"""

from __future__ import annotations

from typing import Any

from synapse.adapters.lora.manifest import LoRAManifest
from synapse.adapters.lora.manager import MultiLoRAManager

# ---------------------------------------------------------------------------
# LoRAPatchHook
# ---------------------------------------------------------------------------


class LoRAPatchHook:
    """Forward hook that applies spatial mask-gated LoRA output.

    When a mask is provided the LoRA contribution is scaled per-token
    (spatial position) by the mask values, enabling entity-aware
    selective LoRA activation.

    Args:
        lora_delta: Pre-computed LoRA delta weight tensor (``torch.Tensor``
            or ``np.ndarray``).  Shape must be broadcastable to the
            linear layer's output.
    """

    def __init__(self, lora_delta: Any) -> None:
        self._delta = lora_delta
        self._handle: Any | None = None

    def attach(self, module: Any, mask: Any | None = None) -> None:
        """Register a forward hook on *module*.

        Args:
            module: A ``torch.nn.Linear`` (or compatible) module.
            mask: Optional spatial mask tensor broadcastable to the
                module output.  Values in ``[0, 1]`` scale the LoRA
                contribution per token.
        """
        import torch  # lazy

        delta = self._delta
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta, dtype=torch.float32)

        def _hook(
            mod: Any,
            input: Any,
            output: torch.Tensor,
        ) -> torch.Tensor:
            # x @ delta^T gives the LoRA contribution
            lora_out = output.new_zeros(output.shape)
            # Simple additive: add delta applied to input
            inp = input[0] if isinstance(input, tuple) else input
            lora_contrib = inp @ delta.T.to(output.device)
            if mask is not None:
                m = mask
                if not isinstance(m, torch.Tensor):
                    m = torch.tensor(
                        m, dtype=output.dtype, device=output.device,
                    )
                # Broadcast mask to match lora_contrib shape
                while m.dim() < lora_contrib.dim():
                    m = m.unsqueeze(-1)
                lora_contrib = lora_contrib * m
            return output + lora_contrib

        self._handle = module.register_forward_hook(_hook)

    def detach(self, module: Any) -> None:
        """Remove the forward hook from *module*.

        Args:
            module: The module to detach from (unused but kept for
                API symmetry).
        """
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# WanAdapter integration
# ---------------------------------------------------------------------------


def patch_wan_adapter_with_lora(
    wan_adapter: Any,
    lora_bundles: list[tuple[LoRAManifest, float]],
) -> MultiLoRAManager:
    """Connect a ``MultiLoRAManager`` to a WanAdapter's LoRA slot.

    Creates a ``MultiLoRAManager`` from the given bundles and assigns
    it to ``wan_adapter._lora_manager``.

    Args:
        wan_adapter: A ``WanAdapter`` instance.
        lora_bundles: List of ``(manifest, scale)`` tuples.

    Returns:
        The newly created ``MultiLoRAManager``.
    """
    manager = MultiLoRAManager(lora_bundles=lora_bundles)
    wan_adapter._lora_manager = manager
    return manager
