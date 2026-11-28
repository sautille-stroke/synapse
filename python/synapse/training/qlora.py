"""NF4 4-bit NormalFloat QLoRA blockwise quantization & frozen-base adapters.

Implements pure-PyTorch Dettmers et al. (2023) 16-level NormalFloat (NF4)
blockwise quantization/dequantization and ``QLoRALinear`` drop-in linear
modules with frozen 4-bit packed base weights (``uint8`` pairs + fp32
``absmax`` per block) and trainable float32 ``lora_A`` / ``lora_B`` factors.

Heavy dependencies (``torch``) are lazy-imported so this module remains
import-clean in zero-GPU environments.  Optional ``bitsandbytes`` acceleration
is probed safely via ``importlib.util.find_spec`` without failing when absent.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

# Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
# 16 asymmetric NormalFloat quantiles over [-1.0, 1.0].
NF4_CODEBOOK: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230182647705,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

_qlora_linear_cls: type | None = None


@dataclass(frozen=True)
class NF4QuantizedTensor:
    """Packed 4-bit NF4 quantized tensor with per-block ``absmax`` scales.

    Attributes:
        packed_uint8: Byte tensor packing two 4-bit NF4 indices per byte
            (high nibble holds even index, low nibble holds odd index).
        absmax: Per-block absolute maximum scaling factors in float32, shape
            ``(num_blocks, 1)``.
        numel: Original unpadded scalar element count.
        orig_shape: Original tensor shape prior to block flattening.
        block_size: Number of elements per quantization block.
    """

    packed_uint8: Any
    absmax: Any
    numel: int
    orig_shape: tuple[int, ...]
    block_size: int = 64

    def memory_bytes(self) -> int:
        """Return total storage footprint in bytes (packed nibbles + absmax)."""
        packed_bytes = int(self.packed_uint8.numel()) * int(
            self.packed_uint8.element_size()
        )
        scale_bytes = int(self.absmax.numel()) * int(self.absmax.element_size())
        return packed_bytes + scale_bytes

    def compression_ratio_vs_fp16(self) -> float:
        """Compute compression ratio relative to dense fp16 storage."""
        fp16_bytes = max(self.numel * 2, 1)
        return float(fp16_bytes) / float(max(self.memory_bytes(), 1))


def has_bitsandbytes() -> bool:
    """Return True when ``bitsandbytes`` is importable in the runtime."""
    return importlib.util.find_spec("bitsandbytes") is not None


def _validate_block_size(block_size: int) -> int:
    if block_size < 2 or block_size % 2 != 0:
        raise ValueError(
            f"qlora_block_size must be an even integer >= 2, got {block_size}"
        )
    return block_size


def quantize_nf4_blockwise(
    weight: Any,
    block_size: int = 64,
) -> NF4QuantizedTensor:
    """Quantize a tensor to 4-bit NF4 with per-block ``absmax`` scaling.

    Args:
        weight: Source weight tensor of arbitrary shape.
        block_size: Even number of weights per quantization block (default 64).

    Returns:
        Packed :class:`NF4QuantizedTensor` holding 4-bit indices and scales.
    """
    import torch

    bs = _validate_block_size(block_size)
    w = torch.as_tensor(weight).detach().to(dtype=torch.float32)
    orig_shape = tuple(int(d) for d in w.shape)
    flat = w.reshape(-1)
    numel = int(flat.numel())

    pad_len = (bs - (numel % bs)) % bs
    if pad_len > 0:
        flat_padded = torch.nn.functional.pad(flat, (0, pad_len), value=0.0)
    else:
        flat_padded = flat

    blocks = flat_padded.reshape(-1, bs)
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    safe_scale = absmax.clamp(min=1e-8)
    normed = torch.where(absmax > 0.0, blocks / safe_scale, torch.zeros_like(blocks))

    codebook = torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device=w.device)
    distances = (normed.unsqueeze(-1) - codebook).abs()
    indices = torch.argmin(distances, dim=-1).reshape(-1).to(dtype=torch.uint8)

    hi = indices[0::2]
    lo = indices[1::2]
    packed = (hi << 4) | lo

    return NF4QuantizedTensor(
        packed_uint8=packed.contiguous(),
        absmax=absmax.contiguous(),
        numel=numel,
        orig_shape=orig_shape,
        block_size=bs,
    )


def dequantize_nf4_blockwise(
    qtensor: NF4QuantizedTensor,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    """Dequantize an :class:`NF4QuantizedTensor` back to dense floating-point."""
    import torch

    target_dev = device if device is not None else qtensor.packed_uint8.device
    packed = qtensor.packed_uint8.to(device=target_dev)
    absmax = qtensor.absmax.to(device=target_dev, dtype=torch.float32)

    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    indices = torch.stack((hi, lo), dim=-1).reshape(-1).to(dtype=torch.long)

    codebook = torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device=target_dev)
    vals = codebook[indices].reshape(-1, qtensor.block_size)
    dense = (vals * absmax).reshape(-1)[: qtensor.numel].reshape(qtensor.orig_shape)

    if dtype is not None:
        dense = dense.to(dtype=dtype)
    return dense


def _get_qlora_linear_class() -> type:
    """Return a cached ``torch.nn.Module`` QLoRA 4-bit wrapper class."""
    global _qlora_linear_cls
    if _qlora_linear_cls is not None:
        return _qlora_linear_cls

    import torch

    class QLoRALinear(torch.nn.Module):
        """Drop-in ``nn.Linear`` replacement with frozen NF4 base weights.

        Base weights are stored compressed as 4-bit packed ``uint8`` buffers
        with per-block float32 ``absmax`` scales (never receiving gradients),
        while ``lora_A`` and ``lora_B`` stay trainable in float32.
        """

        is_qlora_linear: bool = True

        def __init__(
            self,
            orig: torch.nn.Linear,
            rank: int,
            alpha: float,
            *,
            block_size: int = 64,
            quantization_bits: int = 4,
        ) -> None:
            super().__init__()
            if quantization_bits not in (4, 8):
                raise ValueError(
                    f"QLoRALinear supports 4-bit or 8-bit quantization, got {quantization_bits}"
                )
            self.in_features = int(orig.in_features)
            self.out_features = int(orig.out_features)
            self.rank = int(rank)
            self.alpha = float(alpha)
            self.block_size = _validate_block_size(int(block_size))
            self.quantization_bits = int(quantization_bits)
            self.lora_enabled: bool = True
            self.uses_bitsandbytes: bool = False

            dev = orig.weight.device
            self.orig_weight_shape: tuple[int, int] = (
                self.out_features,
                self.in_features,
            )
            self.weight_numel: int = int(orig.weight.numel())

            qtensor = quantize_nf4_blockwise(
                orig.weight.data, block_size=self.block_size
            )
            self.register_buffer("qweight", qtensor.packed_uint8.to(device=dev))
            self.register_buffer("weight_absmax", qtensor.absmax.to(device=dev))

            if orig.bias is not None:
                self.register_buffer(
                    "bias", orig.bias.data.detach().clone().to(device=dev)
                )
            else:
                self.register_buffer("bias", None)

            # Match LoRALinear parameter naming & float32 precision
            self.lora_A = torch.nn.Parameter(
                torch.randn(self.rank, self.in_features, device=dev, dtype=torch.float32)
                * 0.01
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(self.out_features, self.rank, device=dev, dtype=torch.float32)
            )

        def dequantize_base_weight(self) -> Any:
            """Reconstruct float32 base weight tensor from packed NF4 buffers."""
            qw: Any = self.qweight
            am: Any = self.weight_absmax
            qtensor = NF4QuantizedTensor(
                packed_uint8=qw,
                absmax=am,
                numel=self.weight_numel,
                orig_shape=self.orig_weight_shape,
                block_size=self.block_size,
            )
            return dequantize_nf4_blockwise(qtensor)

        def weight_memory_bytes(self) -> int:
            """Return byte size of frozen quantized base weight + scales."""
            qw: Any = self.qweight
            am: Any = self.weight_absmax
            q_bytes = int(qw.numel()) * int(qw.element_size())
            s_bytes = int(am.numel()) * int(am.element_size())
            return q_bytes + s_bytes

        def forward(self, x: Any) -> Any:
            """Compute dequantized frozen linear projection + LoRA delta."""
            base_w = self.dequantize_base_weight().to(dtype=x.dtype, device=x.device)
            bias_buf: Any = self.bias
            bias: Any = (
                bias_buf.to(dtype=x.dtype, device=x.device)
                if bias_buf is not None
                else None
            )
            base = torch.nn.functional.linear(x, base_w, bias)
            if not self.lora_enabled:
                return base
            lora_a: Any = self.lora_A
            lora_b: Any = self.lora_B
            x_f32 = x.to(dtype=torch.float32)
            lora_out = (x_f32 @ lora_a.T @ lora_b.T) * (
                self.alpha / self.rank
            )
            return base + lora_out.to(dtype=base.dtype)

    _qlora_linear_cls = QLoRALinear
    return _qlora_linear_cls


def create_qlora_linear(
    orig: Any,
    rank: int,
    alpha: float,
    *,
    block_size: int = 64,
    quantization_bits: int = 4,
) -> Any:
    """Wrap an existing ``torch.nn.Linear`` in a :class:`QLoRALinear` adapter."""
    cls = _get_qlora_linear_class()
    return cls(
        orig,
        rank=rank,
        alpha=alpha,
        block_size=block_size,
        quantization_bits=quantization_bits,
    )
