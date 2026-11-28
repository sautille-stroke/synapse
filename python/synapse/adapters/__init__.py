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

# Import cloud module to trigger @register_adapter side-effects.
from synapse.adapters.cloud import (
    CloudApiTransport,
    DegradedSegment,
    HttpCloudTransport,
    HttpSegmentTransport,
    KlingAdapter,
    MockCloudTransport,
    SeedanceAdapter,
    VeoAdapter,
    degrade_fgpp_for_blackbox,
)

# Import lora subpackage for LoRA manifest, manager and patcher.
from synapse.adapters.lora import (
    LoRAManifest,
    LoRAPatchHook,
    MultiLoRAManager,
    patch_wan_adapter_with_lora,
)

# Import mock module to trigger @register_adapter side-effect.
from synapse.adapters.mock import DefectInjectionConfig, MockVideoAdapter
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
    ReanchorRequest,
    ScheduledWindow,
    WindowSlice,
    compute_latent_flow_operator,
    cross_dissolve_pixels,
    fuse_overlapping_latents,
)

# Import vace module to trigger @register_adapter side-effect.
from synapse.adapters.vace import (
    ConditionMode,
    MockVaceBackend,
    VaceAdapter,
    VaceBackendTransport,
    VaceTaskMode,
    VCUPayload,
)

# Import wan module to trigger @register_adapter side-effect.
from synapse.adapters.wan import (
    FoundaryHttpTransport,
    LocalCallableBackend,
    WanAdapter,
    WanBackendTransport,
)
from synapse.adapters.wan_diffusers import DiffusersWanBackend

__all__ = [
    "AdapterTier",
    "CausalWindowPlan",
    "CloudApiTransport",
    "ConditionScheduler",
    "DefectInjectionConfig",
    "DegradedSegment",
    "DiffusersWanBackend",
    "DiscontinuityDetector",
    "FoundaryHttpTransport",
    "HttpCloudTransport",
    "HttpSegmentTransport",
    "KlingAdapter",
    "LoRAManifest",
    "LoRAPatchHook",
    "LocalCallableBackend",
    "MockCloudTransport",
    "MockVideoAdapter",
    "MultiLoRAManager",
    "PreparedConditionPayload",
    "ReanchorRequest",
    "ScheduledWindow",
    "SeedanceAdapter",
    "VaceAdapter",
    "VaceBackendTransport",
    "VaceTaskMode",
    "VCUPayload",
    "ConditionMode",
    "MockVaceBackend",
    "VeoAdapter",
    "VideoGenerationConfig",
    "VideoModelAdapter",
    "VideoResult",
    "WanAdapter",
    "WanBackendTransport",
    "WindowSlice",
    "compute_latent_flow_operator",
    "create_adapter",
    "cross_dissolve_pixels",
    "degrade_fgpp_for_blackbox",
    "fuse_overlapping_latents",
    "list_adapters",
    "patch_wan_adapter_with_lora",
    "register_adapter",
]
