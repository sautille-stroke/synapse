"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.conditions`` instead.
"""

from __future__ import annotations

import warnings

from synapse.render_core.conditions import (  # noqa: F401
    _VALID_FACTOR_KEYS,
    _VALID_LEVELS,
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    ConditionLevel,
    Conditions,
)

warnings.warn(
    "synapse.bench.conditions is deprecated; use synapse.render_core.conditions instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CROSS_CONDITIONS",
    "ConditionLevel",
    "Conditions",
    "RENDERING_CONDITION_FACTORS",
    "SINGLE_FACTOR_CONDITIONS",
]
