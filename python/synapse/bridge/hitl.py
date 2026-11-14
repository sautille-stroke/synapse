"""HITL (Human-In-The-Loop) circuit breaker and CommandIR records.

Implements the 15-minute engineering circuit breaker from RFC 0044 §4.4:
any single asset or shot that exceeds the maximum manual-refinement
budget triggers a :class:`~synapse.errors.CircuitBreakerError` with a
domain-specific fallback recommendation.

.. note::

   ``evaluate_hitl_circuit_breaker`` has been moved to :mod:`synapse.errors`
   (Tier-0) to break the ``refine → bridge`` circular dependency.
   This module re-exports it for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from synapse.errors import CircuitBreakerError
# Backward-compatible re-export from Tier-0 location
from synapse.errors import evaluate_hitl_circuit_breaker  # noqa: F401

# -----------------------------------------------------------------------
# Operator identity
# -----------------------------------------------------------------------


@dataclass
class OperatorIdentity:
    """Identity of a human or agent operator.

    Attributes:
        kind: Operator tier — ``"human_l1"`` (basic QA),
            ``"human_l2"`` (senior artist), or ``"agent"`` (AI).
        operator_id: Unique operator identifier.
        model_urn: URN of the AI model, if ``kind == "agent"``.
    """

    kind: Literal["human_l1", "human_l2", "agent"]
    operator_id: str
    model_urn: str | None = None


# -----------------------------------------------------------------------
# CommandIR data record
# -----------------------------------------------------------------------


@dataclass
class CommandIRRecord:
    """A single Command Interaction Record for RSI data collection.

    Captures the intent, before/after state URIs, and timing of a
    manual or agent-driven refinement action so that the RSI flywheel
    can learn from operator behaviour.

    Attributes:
        intent_group_id: Grouping key for related intents.
        intent: Free-text description of the operator's goal.
        state_before_uri: URI pointing to the scene state snapshot
            before the action.
        actions: Ordered list of atomic action dicts.
        state_after_uri: URI pointing to the scene state snapshot
            after the action.
        feedback: Operator feedback or quality label.
        elapsed_seconds: Wall-clock time spent on this action.
    """

    intent_group_id: str
    intent: str
    state_before_uri: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    state_after_uri: str = ""
    feedback: str = ""
    elapsed_seconds: float = 0.0


# -----------------------------------------------------------------------
# Fallback action literals
# -----------------------------------------------------------------------

FallbackAction = Literal[
    "FALLBACK_REBAKE_3D_FGPP",
    "FALLBACK_SINGLE_VIEW_DECAL_OR_PROCEDURAL",
]
"""Domain-specific fallback actions triggered by the circuit breaker.

- ``"FALLBACK_REBAKE_3D_FGPP"`` — for video shots: re-bake FGPP from
  updated 3D scene.
- ``"FALLBACK_SINGLE_VIEW_DECAL_OR_PROCEDURAL"`` — for 3D assets:
  fall back to single-view decal projection or procedural material.
"""


# -----------------------------------------------------------------------
# Circuit breaker — canonical implementation moved to synapse.errors
# -----------------------------------------------------------------------

_DEFAULT_MAX_SECONDS: float = 900.0  # 15 minutes (kept for backward compat)
