"""synapse.adapters — Video generation model adapters.

This package provides the ``VideoModelAdapter`` plugin interface and a
global registry for discovering and instantiating adapters by name.

The ``MockVideoAdapter`` is always available and registered at import
time for deterministic CPU-only testing.
"""

from __future__ import annotations

from synapse.adapters.base import (
    AdapterTier,
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoModelAdapter,
    VideoResult,
    create_adapter,
    list_adapters,
    register_adapter,
)

# Import mock module to trigger @register_adapter side-effect.
from synapse.adapters.mock import DefectInjectionConfig, MockVideoAdapter

# Import wan module to trigger @register_adapter side-effect.
from synapse.adapters.wan import (
    FoundaryHttpTransport,
    LocalCallableBackend,
    WanAdapter,
    WanBackendTransport,
)

# Import cloud module to trigger @register_adapter side-effects.
from synapse.adapters.cloud import (
    CloudApiTransport,
    DegradedSegment,
    KlingAdapter,
    MockCloudTransport,
    SeedanceAdapter,
    VeoAdapter,
    degrade_fgpp_for_blackbox,
)
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
    ReanchorRequest,
    ScheduledWindow,
    WindowSlice,
    compute_latent_flow_operator,
    fuse_overlapping_latents,
)

# Import lora subpackage for LoRA manifest, manager and patcher.
from synapse.adapters.lora import (
    LoRAManifest,
    LoRAPatchHook,
    MultiLoRAManager,
    patch_wan_adapter_with_lora,
)

__all__ = [
    "AdapterTier",
    "CausalWindowPlan",
    "CloudApiTransport",
    "compute_latent_flow_operator",
    "ConditionScheduler",
    "create_adapter",
    "DefectInjectionConfig",
    "DegradedSegment",
    "degrade_fgpp_for_blackbox",
    "DiscontinuityDetector",
    "FoundaryHttpTransport",
    "fuse_overlapping_latents",
    "KlingAdapter",
    "list_adapters",
    "LocalCallableBackend",
    "MockCloudTransport",
    "MockVideoAdapter",
    "PreparedConditionPayload",
    "ReanchorRequest",
    "register_adapter",
    "ScheduledWindow",
    "SeedanceAdapter",
    "VeoAdapter",
    "VideoGenerationConfig",
    "VideoModelAdapter",
    "VideoResult",
    "WanAdapter",
    "WanBackendTransport",
    "WindowSlice",
    "LoRAManifest",
    "LoRAPatchHook",
    "MultiLoRAManager",
    "patch_wan_adapter_with_lora",
]
