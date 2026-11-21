"""LoRA forward-hook patcher for spatial mask-gated inference.

Provides ``LoRAPatchHook`` for attaching mask-aware LoRA hooks to
``nn.Linear`` modules, and ``patch_wan_adapter_with_lora`` for
integrating LoRA bundles with the ``WanAdapter``.

Heavy dependencies (``torch``) are lazy-imported inside methods.
"""

from __future__ import annotations

from typing import Any, cast

from synapse.adapters.lora.manager import MultiLoRAManager
from synapse.adapters.lora.manifest import LoRAManifest
from synapse.errors import ContractValidationError

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

    def attach(
        self,
        module: Any,
        mask: Any | None = None,
        module_name: str = "",
    ) -> None:
        """Register a forward hook on *module*.

        The delta tensor is dynamically cast to the target module's
        ``weight.dtype`` and ``device`` (supports fp16/bf16/fp32),
        ensuring no dtype promotion mismatches at inference time.

        Args:
            module: A ``torch.nn.Linear`` (or compatible) module.
            mask: Optional spatial mask tensor broadcastable to the
                module output.  Values in ``[0, 1]`` scale the LoRA
                contribution per token.
            module_name: Name of the module within the model hierarchy,
                used to skip text cross-attention layers.
        """
        import torch  # lazy

        # Clean up existing handle to avoid double-attaching leak
        if self._handle is not None:
            self.detach(module)

        delta = self._delta
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta, dtype=torch.float32)

        # Cast delta to the target module's dtype and device for
        # fp16/bf16/fp32 compatibility (H6-P1-03 fix)
        target_dtype = module.weight.dtype
        target_device = module.weight.device
        delta = delta.to(device=target_device, dtype=target_dtype)

        # Skip spatial masking for text cross-attention layers (e.g. cross_attn.to_k/to_v, attn2.to_k/to_v)
        # where token dimensions reflect text sequences rather than video spatial tokens.
        _CROSS_ATTN_SUBS = (
            "cross_attn.to_k",
            "cross_attn.to_v",
            "attn2.to_k",
            "attn2.to_v",
            "add_k_proj",
            "add_v_proj",
        )
        if mask is not None and any(sub in module_name for sub in _CROSS_ATTN_SUBS):
            mask = None

        # Pre-cache mask to target device and dtype once at attach time
        _cached_mask: Any | None = None
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                _cached_mask = torch.tensor(
                    mask,
                    dtype=target_dtype,
                    device=target_device,
                )
            else:
                _cached_mask = mask.to(device=target_device, dtype=target_dtype)

        # Capture the cast delta in the closure
        _delta_cast = delta

        def _hook(
            mod: Any,
            input: Any,
            output: torch.Tensor,
        ) -> torch.Tensor:
            # x @ delta^T gives the LoRA contribution
            # Simple additive: add delta applied to input
            inp = input[0] if isinstance(input, tuple) else input
            # Cast input to match delta dtype for the matmul
            inp_cast = inp.to(dtype=_delta_cast.dtype)
            lora_contrib = inp_cast @ _delta_cast.T
            # Cast contribution back to output dtype
            lora_contrib = lora_contrib.to(dtype=output.dtype)

            if _cached_mask is not None:
                m = _cached_mask
                # Robust 3D DiT token reshaping [B, L, D]
                if lora_contrib.dim() == 3:
                    b_dim, l_dim, _ = lora_contrib.shape
                    # Check if m is a 3D video-space mask [T, Hm, Wm]
                    # (skip when already shaped as token mask [1/B, L, 1])
                    if m.dim() == 3 and not (m.shape[1] == l_dim and m.shape[2] == 1 and m.shape[0] in (1, b_dim)):
                        t_m, h_m, w_m = m.shape
                        tau = (t_m - 1) // 4 + 1
                        h16 = max(h_m // 2, 1)
                        w16 = max(w_m // 2, 1)
                        expected_l = tau * h16 * w16
                        if expected_l == l_dim:
                            m_5d = m.unsqueeze(0).unsqueeze(0)
                            m = torch.nn.functional.adaptive_avg_pool3d(
                                m_5d, (tau, h16, w16)
                            ).reshape(1, l_dim, 1)
                        elif t_m * h_m * w_m == l_dim:
                            m = m.reshape(1, l_dim, 1)
                        elif t_m * h_m * w_m == b_dim * l_dim:
                            m = m.reshape(b_dim, l_dim, 1)
                        else:
                            # True fail-closed: never silently broadcast ones on a
                            # mismatched 3D spatial mask (would leak identity/style
                            # across masked-out regions).
                            raise ContractValidationError(
                                f"3D video spatial mask shape {tuple(m.shape)} "
                                f"(pooled tokens={expected_l}, raw tokens={t_m * h_m * w_m}) "
                                f"does not match DiT token count L={l_dim} (B={b_dim}).",
                                agent_fix_hint=(
                                    "Ensure the Cryptomatte mask resolution aligns with "
                                    "the DiT spatial/temporal patch grid."
                                ),
                            )

                    if m.dim() == 1 and m.shape[0] == l_dim:
                        m = m.unsqueeze(0).unsqueeze(-1)
                    elif m.dim() == 2 and (m.shape == (b_dim, l_dim) or m.shape == (1, l_dim)):
                        m = m.unsqueeze(-1)
                    elif m.dim() >= 3 and m.numel() == b_dim * l_dim:
                        m = m.reshape(b_dim, l_dim, 1)
                    elif m.dim() >= 3 and m.numel() == l_dim:
                        m = m.reshape(1, l_dim, 1)
                    elif m.dim() == 3 and m.shape[1] == l_dim and m.shape[2] == 1:
                        pass
                    else:
                        # Fail-closed: dimension mismatch — raise instead of
                        # silently broadcasting ones
                        raise ContractValidationError(
                            f"LoRA spatial mask shape {tuple(m.shape)} cannot be "
                            f"broadcast to lora_contrib shape ({b_dim}, {l_dim}, ...)",
                            agent_fix_hint=(
                                "Verify Cryptomatte mask resolution matches the "
                                "DiT's spatial token grid."
                            ),
                        )

                    # CFG broadcast: if b_dim is a multiple of m's batch dim,
                    # expand mask to cover duplicated unconditional+conditional
                    if m.shape[0] == 1 and b_dim > 1:
                        m = m.expand(b_dim, -1, -1)
                else:
                    while m.dim() < lora_contrib.dim():
                        m = m.unsqueeze(-1)

                lora_contrib = lora_contrib * m
            return cast(torch.Tensor, output + lora_contrib)

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
    *,
    test_synthetic: bool = False,
) -> MultiLoRAManager:
    """Connect a ``MultiLoRAManager`` to a WanAdapter's LoRA slot.

    Creates a ``MultiLoRAManager`` from the given bundles and assigns
    it to ``wan_adapter._lora_manager``.

    Args:
        wan_adapter: A ``WanAdapter`` instance.
        lora_bundles: List of ``(manifest, scale)`` tuples.
        test_synthetic: Forwarded to ``MultiLoRAManager``.

    Returns:
        The newly created ``MultiLoRAManager``.
    """
    manager = MultiLoRAManager(
        lora_bundles=lora_bundles,
        test_synthetic=test_synthetic,
    )
    wan_adapter._lora_manager = manager
    return manager
