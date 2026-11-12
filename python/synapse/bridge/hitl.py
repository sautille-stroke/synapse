"""HITL (Human-In-The-Loop) circuit breaker and CommandIR records.

Implements the 15-minute engineering circuit breaker from RFC 0044 §4.4:
any single asset or shot that exceeds the maximum manual-refinement
budget triggers a :class:`~synapse.errors.CircuitBreakerError` with a
domain-specific fallback recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from synapse.errors import CircuitBreakerError

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
# Circuit breaker
# -----------------------------------------------------------------------

_DEFAULT_MAX_SECONDS: float = 900.0  # 15 minutes


def evaluate_hitl_circuit_breaker(
    elapsed_seconds: float,
    domain: Literal["video_shot", "asset_3d"],
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> FallbackAction | None:
    """Evaluate the HITL 15-minute circuit breaker.

    If the elapsed time exceeds *max_seconds*, a
    :class:`~synapse.errors.CircuitBreakerError` is raised with a
    domain-appropriate fallback hint.

    Args:
        elapsed_seconds: Cumulative wall-clock time spent on manual /
            agent refinement for this single asset or shot.
        domain: ``"video_shot"`` or ``"asset_3d"`` — determines the
            fallback action and fix hint.
        max_seconds: Budget ceiling in seconds (default 900 = 15 min).

    Returns:
        ``None`` if the budget has **not** been exceeded.

    Raises:
        CircuitBreakerError: If ``elapsed_seconds > max_seconds``
            (exit_code=40).
    """
    if elapsed_seconds <= max_seconds:
        return None

    msg = (
        f"HITL circuit breaker: {elapsed_seconds:.1f}s > "
        f"{max_seconds:.1f}s limit for {domain}"
    )

    if domain == "video_shot":
        raise CircuitBreakerError(
            msg,
            agent_fix_hint=(
                "Fallback to FGPP re-bake from updated 3D. "
                "Reduce complexity or accept quality trade-off."
            ),
        )
    else:  # asset_3d
        raise CircuitBreakerError(
            msg,
            agent_fix_hint=(
                "Fallback to single-view decal projection or "
                "procedural material. Skip manual refinement."
            ),
        )
