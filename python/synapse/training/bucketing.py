"""Resolution / frame-count bucketing and VRAM-aware LoRA hyperparameter auto-tuning.

Provides:
- :func:`snap_to_frame_bucket` & :class:`FrameBucketSampler`: snaps arbitrary
  training clips to Wan-VAE-aligned ``(height // 8 * 8, width // 8 * 8, 4k + 1)``
  buckets and interleaves homogeneous batches deterministically across buckets
  to avoid wasted cross-resolution padding.
- :func:`auto_tune_lora_config`: recommends LoRA rank (``32..64``), category-aware
  learning rates, linear warmup + cosine schedule steps, QLoRA 4-bit enablement
  (automatically enabled when ``vram_gb <= 48``), and frame bucket caps.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from synapse.training.trainer import LoRATrainConfig

LoRACategory = Literal["style", "identity", "control", "whitebox", "motion"]


@dataclass(frozen=True, order=True)
class FrameBucketKey:
    """Discrete training batch shape bucket ``(height, width, num_frames)``."""

    height: int
    width: int
    num_frames: int


@dataclass(frozen=True)
class FrameBucketSample:
    """Metadata descriptor for a sample grouped by :class:`FrameBucketSampler`."""

    sample_id: str
    height: int
    width: int
    num_frames: int
    payload: Any = None


@dataclass(frozen=True)
class FrameBucketBatch:
    """Homogeneous batch emitted by :class:`FrameBucketSampler`."""

    bucket: FrameBucketKey
    sample_ids: tuple[str, ...]
    indices: tuple[int, ...]


def snap_to_frame_bucket(
    height: int,
    width: int,
    num_frames: int,
    *,
    spatial_multiple: int = 8,
    min_spatial: int = 8,
    max_frames: int = 97,
) -> FrameBucketKey:
    """Snap dimensions to Wan VAE spatial multiples and temporal ``4k + 1``."""
    if spatial_multiple < 1:
        raise ValueError(f"spatial_multiple must be >= 1, got {spatial_multiple}")
    if min_spatial < spatial_multiple:
        min_spatial = spatial_multiple
    h_snap = max(min_spatial, (max(height, min_spatial) // spatial_multiple) * spatial_multiple)
    w_snap = max(min_spatial, (max(width, min_spatial) // spatial_multiple) * spatial_multiple)

    clamped_t = max(1, min(int(num_frames), int(max_frames)))
    k = max(0, (clamped_t - 1) // 4)
    t_snap = 4 * k + 1
    return FrameBucketKey(height=int(h_snap), width=int(w_snap), num_frames=int(t_snap))


def _coerce_sample_descriptor(idx: int, item: Any) -> FrameBucketSample:
    if isinstance(item, FrameBucketSample):
        return item
    if isinstance(item, dict):
        return FrameBucketSample(
            sample_id=str(item.get("sample_id", f"sample_{idx}")),
            height=int(item["height"]),
            width=int(item["width"]),
            num_frames=int(item.get("num_frames", item.get("frames", 1))),
            payload=item,
        )
    if isinstance(item, (tuple, list)) and len(item) >= 3:
        return FrameBucketSample(
            sample_id=f"sample_{idx}",
            height=int(item[0]),
            width=int(item[1]),
            num_frames=int(item[2]),
            payload=item,
        )
    h = int(getattr(item, "height"))
    w = int(getattr(item, "width"))
    t = int(getattr(item, "num_frames"))
    sid = str(getattr(item, "sample_id", f"sample_{idx}"))
    return FrameBucketSample(sample_id=sid, height=h, width=w, num_frames=t, payload=item)


class FrameBucketSampler:
    """Deterministic resolution & ``4k+1`` temporal bucket sampler."""

    def __init__(
        self,
        samples: Sequence[Any],
        *,
        batch_size: int = 1,
        spatial_multiple: int = 8,
        max_frames: int = 97,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._batch_size = int(batch_size)
        self._spatial_multiple = int(spatial_multiple)
        self._max_frames = int(max_frames)
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._epoch = 0

        self._buckets: dict[FrameBucketKey, list[tuple[int, str]]] = defaultdict(list)
        for idx, raw in enumerate(samples):
            desc = _coerce_sample_descriptor(idx, raw)
            key = snap_to_frame_bucket(
                desc.height,
                desc.width,
                desc.num_frames,
                spatial_multiple=self._spatial_multiple,
                max_frames=self._max_frames,
            )
            self._buckets[key].append((idx, desc.sample_id))

        self._batches: list[FrameBucketBatch] = []
        self._rebuild_batches()

    def set_epoch(self, epoch: int) -> None:
        """Advance epoch seed and rebuild interleaved homogeneous batches."""
        self._epoch = int(epoch)
        self._rebuild_batches()

    def bucket_counts(self) -> dict[FrameBucketKey, int]:
        """Return sample count per discrete :class:`FrameBucketKey`."""
        return {k: len(v) for k, v in sorted(self._buckets.items())}

    def _rebuild_batches(self) -> None:
        rng = random.Random(self._seed + self._epoch)
        per_bucket_chunks: list[list[FrameBucketBatch]] = []
        for key in sorted(self._buckets.keys()):
            items = list(self._buckets[key])
            if self._shuffle:
                rng.shuffle(items)
            chunks: list[FrameBucketBatch] = []
            for start in range(0, len(items), self._batch_size):
                chunk = items[start : start + self._batch_size]
                if self._drop_last and len(chunk) < self._batch_size:
                    continue
                chunks.append(
                    FrameBucketBatch(
                        bucket=key,
                        sample_ids=tuple(sid for _, sid in chunk),
                        indices=tuple(idx for idx, _ in chunk),
                    )
                )
            if chunks:
                per_bucket_chunks.append(chunks)

        # Round-robin interleave across buckets so batches stay shape-pure and balanced
        interleaved: list[FrameBucketBatch] = []
        max_len = max((len(c) for c in per_bucket_chunks), default=0)
        for pos in range(max_len):
            for bucket_batches in per_bucket_chunks:
                if pos < len(bucket_batches):
                    interleaved.append(bucket_batches[pos])
        self._batches = interleaved

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[FrameBucketBatch]:
        return iter(self._batches)


@dataclass(frozen=True)
class AutoTuneRecommendation:
    """Hyperparameter recommendation emitted by :func:`auto_tune_lora_config`."""

    category: LoRACategory
    vram_gb: float
    dataset_size: int
    rank: int
    alpha: float
    lr_spatial: float
    lr_temporal: float
    warmup_steps: int
    max_steps: int
    grad_accum_steps: int
    use_qlora: bool
    quantization_bits: Literal[4, 16]
    gradient_checkpointing: bool
    mixed_precision: Literal["fp32", "fp16", "bf16"]
    max_frames_bucket: int
    reason: str

    def to_config_dict(
        self,
        *,
        mask_aware: bool = False,
        mask_background_weight: float = 0.0,
    ) -> dict[str, Any]:
        """Serialize into keyword arguments compatible with :class:`LoRATrainConfig`."""
        return {
            "category": self.category,
            "rank": self.rank,
            "alpha": self.alpha,
            "lr_spatial": self.lr_spatial,
            "lr_temporal": self.lr_temporal,
            "max_steps": self.max_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "warmup_steps": self.warmup_steps,
            "mask_aware": bool(mask_aware),
            "mask_background_weight": float(mask_background_weight),
            "use_qlora": self.use_qlora,
            "quantization_bits": self.quantization_bits,
        }

    def to_lora_train_config(
        self,
        *,
        mask_aware: bool = False,
        mask_background_weight: float = 0.0,
    ) -> LoRATrainConfig:
        """Construct a validated :class:`LoRATrainConfig` from this recommendation."""
        from synapse.training.trainer import LoRATrainConfig

        return LoRATrainConfig(
            **self.to_config_dict(
                mask_aware=mask_aware,
                mask_background_weight=mask_background_weight,
            )
        )


def detect_cuda_vram_gb(default_gb: float = 24.0) -> float:
    """Detect device 0 VRAM capacity in GiB cleanly without raising on CPU hosts."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(float(props.total_memory) / float(1024**3), 2)
    except Exception:
        pass
    return float(default_gb)


def auto_tune_lora_config(
    category: LoRACategory,
    *,
    vram_gb: float | None = None,
    dataset_size: int = 100,
    target_steps: int | None = None,
) -> AutoTuneRecommendation:
    """Recommend rank 32-64, warmup+cosine LR, QLoRA, and bucket limits from VRAM & data."""
    vram = float(vram_gb) if vram_gb is not None else detect_cuda_vram_gb()
    if vram <= 0.0:
        raise ValueError(f"vram_gb must be positive, got {vram}")
    n_samples = max(int(dataset_size), 1)

    if vram <= 48.0:
        use_qlora = True
        q_bits: Literal[4, 16] = 4
        grad_ckpt = True
        if vram <= 16.0:
            rank = 32
            grad_accum = 4
            max_frames_bucket = 49
            tier_desc = "low-vram (<=16GB): 4-bit NF4 QLoRA + grad checkpointing + 49f cap"
        elif vram < 32.0:
            rank = 32
            grad_accum = 2
            max_frames_bucket = 65
            tier_desc = "mid-vram (<32GB): 4-bit NF4 QLoRA + grad checkpointing + 65f cap"
        else:
            rank = 64
            grad_accum = 2
            max_frames_bucket = 81
            tier_desc = "workstation (32-48GB): rank 64 4-bit NF4 QLoRA + 81f cap"
    else:
        use_qlora = False
        q_bits = 16
        grad_ckpt = False
        rank = 64
        grad_accum = 1
        max_frames_bucket = 97
        tier_desc = "high-vram (>48GB): rank 64 bf16 full precision + 97f cap"

    if category == "motion":
        lr_spatial, lr_temporal = 5e-5, 1e-4
    elif category in ("control", "whitebox"):
        lr_spatial, lr_temporal = 8e-5, 5e-5
    else:
        lr_spatial, lr_temporal = 1e-4, 5e-5

    max_steps = (
        max(int(target_steps), 2)
        if target_steps is not None
        else max(200, min(4000, int(math.ceil(n_samples * 10 / 50.0) * 50)))
    )
    warmup_steps = max(1, min(max_steps - 1, int(round(0.05 * max_steps))))

    return AutoTuneRecommendation(
        category=category,
        vram_gb=vram,
        dataset_size=n_samples,
        rank=rank,
        alpha=float(rank),
        lr_spatial=lr_spatial,
        lr_temporal=lr_temporal,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        grad_accum_steps=grad_accum,
        use_qlora=use_qlora,
        quantization_bits=q_bits,
        gradient_checkpointing=grad_ckpt,
        mixed_precision="bf16",
        max_frames_bucket=max_frames_bucket,
        reason=tier_desc,
    )
