"""synapse.adapters.lora — LoRA manifest, manager and patcher.

Provides the ``LoRAManifest`` schema, ``MultiLoRAManager`` for
Cryptomatte-aware multi-LoRA weight merging, ``LoRAPatchHook`` for
spatial mask-gated forward hooks, and WanAdapter integration via
``patch_wan_adapter_with_lora``.
"""

from __future__ import annotations

from synapse.adapters.lora.manifest import LoRAManifest
from synapse.adapters.lora.manager import MultiLoRAManager
from synapse.adapters.lora.patcher import (
    LoRAPatchHook,
    patch_wan_adapter_with_lora,
)

__all__ = [
    "LoRAManifest",
    "LoRAPatchHook",
    "MultiLoRAManager",
    "patch_wan_adapter_with_lora",
]
