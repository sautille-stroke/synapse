"""LoRA manifest schema for Synapse adapters.

Defines ``LoRAManifest`` — a Pydantic model that validates LoRA adapter
metadata, enforcing safetensors-only format, per-category target-module
whitelists, and SHA256 checksums.

Schema version: ``synapse.lora.v1.0``
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ---------------------------------------------------------------------------
# Category -> allowed target-module patterns
# ---------------------------------------------------------------------------

_CATEGORY_WHITELIST: dict[str, list[str]] = {
    "style": [
        r"cross_attn\.to_k",
        r"cross_attn\.to_v",
        r"ffn\..*",
    ],
    "identity": [
        r"spatial_attn\..*",
        r"cross_attn\.to_k",
        r"cross_attn\.to_v",
    ],
    "control": [
        r"spatial_attn\..*",
        r"ffn\..*",
    ],
    "whitebox": [
        r"spatial_attn\..*",
        r"ffn\..*",
    ],
    "motion": [
        r"temporal_attn\..*",
    ],
}


def _matches_any_pattern(module_name: str, patterns: list[str]) -> bool:
    """Return True if *module_name* fully matches any regex in *patterns*."""
    return any(re.fullmatch(p, module_name) for p in patterns)


# ---------------------------------------------------------------------------
# LoRAManifest
# ---------------------------------------------------------------------------


class LoRAManifest(BaseModel):
    """Pydantic manifest for a single LoRA adapter.

    Attributes:
        schema_version: Fixed to ``synapse.lora.v1.0``.
        name: Human-readable adapter name.
        category: LoRA category controlling allowed target modules.
        rank: LoRA rank (typically 8-64).
        alpha: LoRA scaling factor.
        adapter_path: Path to ``.safetensors`` file (pickle rejected).
        adapter_sha256: Hex SHA256 checksum of the adapter file.
        target_modules: Layer-name patterns to patch (validated per
            category whitelist).
        training_metadata: Optional free-form training metadata.
    """

    model_config = ConfigDict(extra="forbid")

    SCHEMA_VERSION: ClassVar[str] = "synapse.lora.v1.0"

    schema_version: str = "synapse.lora.v1.0"
    name: str
    category: Literal["style", "identity", "control", "whitebox", "motion"]
    rank: int
    alpha: float
    adapter_path: str
    adapter_sha256: str
    target_modules: list[str]
    training_metadata: dict[str, Any] | None = None

    # -- validators ----------------------------------------------------------

    @field_validator("name")
    @classmethod
    def _validate_name_non_empty(cls, v: str) -> str:
        """Ensure name is non-empty."""
        if not v.strip():
            raise ValueError("name must be a non-empty string.")
        return v

    @field_validator("rank")
    @classmethod
    def _validate_rank_positive(cls, v: int) -> int:
        """Ensure rank >= 1."""
        if v < 1:
            raise ValueError(f"rank must be >= 1; got {v}.")
        return v

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure target_modules is not empty."""
        if not v:
            raise ValueError("target_modules must contain at least one entry.")
        return v

    @field_validator("adapter_path")
    @classmethod
    def _reject_unsafe_formats(cls, v: str) -> str:
        """Reject pickle-based adapter formats (.bin, .pt)."""
        if not v.endswith(".safetensors"):
            raise ValueError(
                f"Only .safetensors adapters are allowed; "
                f"got '{v}'. Pickle-based formats (.bin, .pt) "
                f"are rejected for security."
            )
        return v

    @field_validator("adapter_sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        """Ensure SHA256 is a 64-char lowercase hex string."""
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"adapter_sha256 must be a 64-char lowercase hex string; got '{v}'.")
        return v

    @model_validator(mode="after")
    def _validate_target_modules(self) -> LoRAManifest:
        """Validate target_modules against category whitelist."""
        patterns = _CATEGORY_WHITELIST.get(self.category, [])
        for module in self.target_modules:
            if not re.fullmatch(r"[A-Za-z0-9_.*]+", module):
                raise ValueError(
                    f"target_module '{module}' contains invalid characters or regex "
                    f"injection. Must only contain alphanumeric, underscore, dot, and asterisk."
                )
            if not _matches_any_pattern(module, patterns):
                allowed = ", ".join(patterns)
                raise ValueError(
                    f"target_module '{module}' is not allowed for "
                    f"category '{self.category}'. "
                    f"Allowed patterns: [{allowed}]"
                )
        return self
