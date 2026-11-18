"""W3C trace context propagation and GPU memory forecaster.

Provides lightweight telemetry primitives for CLI and worker runtimes:

- ``TraceContext``: W3C ``traceparent`` / ``tracestate`` parsing and
  generation, suitable for propagating distributed trace identity
  through ``render-farm`` job bundles.
- ``MemoryForecaster``: estimates peak GPU memory for DiT-based video
  generation models and selects the appropriate fallback level to avoid
  OOM.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass, field
from enum import IntEnum

# W3C Trace Context Regex (traceparent: 00-32hex-16hex-2hex)
W3C_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


@dataclass
class TraceContext:
    """W3C Trace Context carrier (traceparent + tracestate).

    Attributes:
        trace_id: 32 hex-char trace identifier.
        span_id: 16 hex-char span identifier.
        trace_flags: 2 hex-char flags (``"01"`` = sampled).
        tracestate: Opaque vendor key-value pairs.
    """

    trace_id: str
    span_id: str
    trace_flags: str = "01"
    tracestate: str = ""

    def __post_init__(self) -> None:
        # Force lowercase normalisation for W3C compliance
        self.trace_id = self.trace_id.lower()
        self.span_id = self.span_id.lower()
        self.trace_flags = self.trace_flags.lower()
        if not re.match(r"^[0-9a-f]{32}$", self.trace_id) or self.trace_id == "0" * 32:
            self.trace_id = secrets.token_hex(16)
        if not re.match(r"^[0-9a-f]{16}$", self.span_id) or self.span_id == "0" * 16:
            self.span_id = secrets.token_hex(8)
        if not re.match(r"^[0-9a-f]{2}$", self.trace_flags):
            self.trace_flags = "01"

    @classmethod
    def current(cls) -> TraceContext:
        """Obtain the current trace context from environment or generate fresh IDs."""
        return cls.from_env_or_dict()

    @classmethod
    def from_env_or_dict(
        cls,
        env: dict[str, str] | None = None,
    ) -> TraceContext:
        """Parse W3C ``traceparent`` from *env* or generate fresh IDs.

        The ``traceparent`` header has the format::

            00-<trace_id 32hex>-<span_id 16hex>-<flags 2hex>

        If *env* is ``None``, ``os.environ`` is inspected.
        """
        if env is None:
            env = dict(os.environ)

        traceparent = env.get("traceparent", env.get("TRACEPARENT", "")).strip()
        tracestate = env.get("tracestate", env.get("TRACESTATE", "")).strip()

        if traceparent:
            m = W3C_TRACEPARENT_RE.match(traceparent.lower())
            if m:
                t_id, s_id, flags = m.groups()
                if t_id != "0" * 32 and s_id != "0" * 16:
                    return cls(
                        trace_id=t_id,
                        span_id=s_id,
                        trace_flags=flags,
                        tracestate=tracestate,
                    )

        # Generate random IDs when not provided or invalid
        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            trace_flags="01",
            tracestate=tracestate,
        )

    def to_traceparent(self) -> str:
        """Format as W3C ``traceparent`` header value."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"

    def to_headers(self) -> dict[str, str]:
        """Return HTTP-style header dict for downstream propagation."""
        headers: dict[str, str] = {"traceparent": self.to_traceparent()}
        if self.tracestate:
            headers["tracestate"] = self.tracestate
        return headers


# ---------------------------------------------------------------------------
# Memory Forecaster
# ---------------------------------------------------------------------------


class FallbackLevel(IntEnum):
    """GPU memory fallback tiers for DiT video generation."""

    LEVEL_0_FULL = 0
    LEVEL_1_SPATIAL_TILE_2X2 = 1
    LEVEL_2_CAUSAL_SHORT_WINDOW_49F = 2
    REJECT_OOM = 3


@dataclass
class MemoryForecast:
    """Result of a peak-memory estimation.

    Attributes:
        peak_gb: Estimated peak GPU memory in GiB.
        fallback_level: Recommended fallback tier.
        breakdown: Per-component memory breakdown in GiB.
    """

    peak_gb: float
    fallback_level: FallbackLevel
    breakdown: dict[str, float] = field(default_factory=dict)


# Known model weight sizes (GiB, FP16 on disk → FP16 in VRAM)
_MODEL_WEIGHT_GB: dict[str, float] = {
    "wan2.1_14b": 28.0,
    "wan2.1_1.3b": 5.0,
}

# Approximate channels multiplier for latent activations
_CHANNELS_FACTOR: dict[str, int] = {
    "wan2.1_14b": 16,
    "wan2.1_1.3b": 16,
}

# VRAM overhead factor (memory allocator fragmentation, cuDNN workspace)
_OVERHEAD_FACTOR: float = 1.15

# Fallback thresholds (in GiB of peak VRAM)
_THRESHOLD_L0: float = 24.0
_THRESHOLD_L1: float = 48.0
_THRESHOLD_L2: float = 80.0


class MemoryForecaster:
    """Estimate peak GPU memory for DiT-based video generation."""

    @staticmethod
    def estimate_dit_peak_gb(
        num_frames: int,
        height: int,
        width: int,
        model_variant: str = "wan2.1_14b",
        dtype_bytes: int = 2,
        num_layers: int | None = None,
        available_vram_gb: float | None = None,
    ) -> MemoryForecast:
        """Estimate peak GPU memory for a single inference pass.

        The formula accounts for three primary consumers:

        1. **Model weights** — looked up from a table of known variants.
        2. **Latent activations** — proportional to the spatiotemporal
           volume ``num_frames × H × W``.
        3. **Attention KV cache** — quadratic in the number of
           spatiotemporal tokens across all layers.

        A 15 % overhead factor is applied for allocator fragmentation
        and cuDNN workspace.

        Args:
            num_frames: Number of output video frames.
            height: Output frame height in pixels.
            width: Output frame width in pixels.
            model_variant: Model name key (e.g. ``"wan2.1_14b"``).
            dtype_bytes: Bytes per element (2 for FP16, 4 for FP32).
            num_layers: Number of DiT layers (default 40 for 14B, 30 for 1.3B).
            available_vram_gb: If specified, dynamically select fallback level
                based on available GPU memory.

        Returns:
            A ``MemoryForecast`` with peak estimate and fallback level.
        """
        weight_gb = _MODEL_WEIGHT_GB.get(model_variant, 28.0)
        channels = _CHANNELS_FACTOR.get(model_variant, 16)

        # Activation memory: N × H × W × C × dtype / 1 GiB
        act_bytes = num_frames * height * width * channels * dtype_bytes
        act_gb = act_bytes / (1024**3)

        # Spatiotemporal attention tokens.
        # Wan 2.1 VAE: 4× temporal compression (causal, so +1),
        # 16× spatial compression (two stages of 4× + 4×).
        temporal_tokens = max((num_frames - 1) // 4 + 1, 1)
        spatial_h = max(height // 16, 1)
        spatial_w = max(width // 16, 1)
        total_tokens = temporal_tokens * spatial_h * spatial_w

        # KV cache across all layers: 2 (K+V) × tokens × head_dim × num_heads × num_layers × dtype
        head_dim = 128
        num_heads = 40 if "14b" in model_variant else 16
        if num_layers is None:
            num_layers = 40 if "14b" in model_variant else 30
        kv_bytes = 2 * total_tokens * head_dim * num_heads * num_layers * dtype_bytes
        kv_gb = kv_bytes / (1024**3)

        raw_peak = weight_gb + act_gb + kv_gb
        peak_gb = raw_peak * _OVERHEAD_FACTOR

        # Determine fallback level
        if available_vram_gb is not None:
            if weight_gb > available_vram_gb:
                # Model weights alone exceed VRAM — recommend CPU offload
                # instead of useless spatial tiling
                level = FallbackLevel.REJECT_OOM
            elif peak_gb <= available_vram_gb:
                level = FallbackLevel.LEVEL_0_FULL
            elif peak_gb <= available_vram_gb * 1.5:
                level = FallbackLevel.LEVEL_1_SPATIAL_TILE_2X2
            elif peak_gb <= available_vram_gb * 2.5:
                level = FallbackLevel.LEVEL_2_CAUSAL_SHORT_WINDOW_49F
            else:
                level = FallbackLevel.REJECT_OOM
        else:
            if peak_gb <= _THRESHOLD_L0:
                level = FallbackLevel.LEVEL_0_FULL
            elif peak_gb <= _THRESHOLD_L1:
                level = FallbackLevel.LEVEL_1_SPATIAL_TILE_2X2
            elif peak_gb <= _THRESHOLD_L2:
                level = FallbackLevel.LEVEL_2_CAUSAL_SHORT_WINDOW_49F
            else:
                level = FallbackLevel.REJECT_OOM

        return MemoryForecast(
            peak_gb=round(peak_gb, 2),
            fallback_level=level,
            breakdown={
                "weights_gb": round(weight_gb, 2),
                "activations_gb": round(act_gb, 2),
                "kv_cache_gb": round(kv_gb, 2),
                "num_layers": num_layers,
                "overhead_factor": _OVERHEAD_FACTOR,
            },
        )
