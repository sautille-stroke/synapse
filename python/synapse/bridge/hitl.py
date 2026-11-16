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

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from synapse.errors import (
    CircuitBreakerError,  # noqa: F401
    evaluate_hitl_circuit_breaker,  # noqa: F401
)

__all__ = [
    "CircuitBreakerError",
    "CommandIRRecord",
    "OperatorIdentity",
    "PersistentHitlLedger",
    "evaluate_hitl_circuit_breaker",
]

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

# -----------------------------------------------------------------------
# Persistent HITL Ledger with cross-process file locking
# -----------------------------------------------------------------------


class PersistentHitlLedger:
    """Thread-safe and process-safe persistent ledger for HITL actions and timing.

    Uses file locking (fcntl on POSIX) and atomic tempfile replacement to guarantee
    consistent cross-process tracking across CLI worker invocations.
    """

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        if ledger_path is None:
            ledger_path = os.environ.get(
                "SYNAPSE_HITL_LEDGER_PATH",
                str(Path(tempfile.gettempdir()) / "synapse_hitl_ledger.json"),
            )
        self.ledger_path = Path(ledger_path)

    def _lock_and_read(self) -> tuple[dict[str, Any], Any]:
        """Read state with exclusive or shared lock."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.ledger_path.with_suffix(".lock")
        f = open(lock_file, "a+", encoding="utf-8")
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass

        data: dict[str, Any] = {}
        if self.ledger_path.is_file():
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        return data, f

    def _unlock_and_write(self, data: dict[str, Any], lock_handle: Any) -> None:
        """Write state via atomic replace and release lock."""
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.ledger_path.parent),
                prefix="synapse_hitl_tmp_",
                suffix=".json",
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tf:
                json.dump(data, tf, indent=2)
            os.replace(tmp_path, self.ledger_path)
        finally:
            try:
                import fcntl
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            lock_handle.close()

    def record_action(
        self,
        record: CommandIRRecord,
        max_seconds: float = _DEFAULT_MAX_SECONDS,
        domain: Literal["video_shot", "asset_3d"] = "video_shot",
    ) -> float:
        """Record a CommandIR action and verify circuit breaker timeout."""
        data, handle = self._lock_and_read()
        group_id = record.intent_group_id
        entry = data.setdefault(group_id, {"elapsed_seconds": 0.0, "records": []})
        entry["elapsed_seconds"] += float(record.elapsed_seconds)
        entry["records"].append({
            "intent": record.intent,
            "elapsed_seconds": record.elapsed_seconds,
            "feedback": record.feedback,
            "actions": record.actions,
        })
        total_elapsed = float(entry["elapsed_seconds"])
        self._unlock_and_write(data, handle)

        evaluate_hitl_circuit_breaker(total_elapsed, max_seconds=max_seconds, domain=domain)
        return total_elapsed

    def get_accumulated_elapsed(self, group_id: str) -> float:
        """Return total elapsed time for a given group/asset."""
        data, handle = self._lock_and_read()
        try:
            return float(data.get(group_id, {}).get("elapsed_seconds", 0.0))
        finally:
            self._unlock_and_write(data, handle)
