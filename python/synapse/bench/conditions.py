"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.conditions`` instead.
"""

from synapse.render_core.conditions import (  # noqa: F401
    ConditionLevel,
    RENDERING_CONDITION_FACTORS,
    Conditions,
    SINGLE_FACTOR_CONDITIONS,
    CROSS_CONDITIONS,
    _VALID_LEVELS,
    _VALID_FACTOR_KEYS,
)
