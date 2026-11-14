"""Backward-compatible re-exports — canonical code moved to synapse.render_core.

.. deprecated::
    Import from ``synapse.render_core.conditions`` instead.
"""

from synapse.render_core.conditions import (  # noqa: F401
    CROSS_CONDITIONS,
    RENDERING_CONDITION_FACTORS,
    SINGLE_FACTOR_CONDITIONS,
    ConditionLevel,
    Conditions,
    _VALID_FACTOR_KEYS,
    _VALID_LEVELS,
)
