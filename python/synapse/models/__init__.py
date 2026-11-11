"""synapse.models — 3D Model asset loader, normalization, and benchmark meshes."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from synapse.models.asset import ModelAsset

if TYPE_CHECKING:  # pragma: no cover
    from synapse.models.loader import list_builtin_models, load_model


def __getattr__(name: str):  # noqa: N807
    if name in ("load_model", "list_builtin_models"):
        mod = importlib.import_module("synapse.models.loader")
        return getattr(mod, name)
    raise AttributeError(f"module 'synapse.models' has no attribute {name!r}")


__all__ = [
    "ModelAsset",
    "load_model",
    "list_builtin_models",
]
