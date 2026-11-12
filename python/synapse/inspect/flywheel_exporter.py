"""Refinement roundtrip orchestrator and RL flywheel data exporter.

Provides:
- ``run_targeted_refinement``: Orchestrate auto-fix from
  InspectionReport, enforce roundtrip hard gates
- ``export_rl_flywheel_dataset``: Export PRM / DPO training
  data from intent group transaction logs

Heavy dependencies are lazy-imported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from synapse.errors import (
    ContractValidationError,
    QualityGateError,
)
from synapse.metrics.profiles import (
    TRACK_A_ASSET,
    GateProfileName,
    get_profile_thresholds,
)

# -------------------------------------------------------------------
# Roundtrip gate thresholds
# -------------------------------------------------------------------


class RoundtripGate(BaseModel):
    """Stage 5 roundtrip hard gate thresholds.

    Defaults are sourced from ``synapse.metrics.profiles.TRACK_A_ASSET``
    (SSOT aligned with ``DESIGN_REVIEW_AND_ALIGNMENT.md`` §3.4).

    Attributes:
        delta_fscore_1mm: Minimum F-score improvement (must be >= 0).
        volume_breathing_pct: Maximum volume change percentage.
        max_non_manifold_edges: Must be exactly 0 for production.
    """

    delta_fscore_1mm: float = Field(default=0.0, ge=0.0)
    volume_breathing_pct: float = Field(
        default=float(TRACK_A_ASSET["volume_breathing_pct"]), ge=0.0,
    )
    max_non_manifold_edges: int = Field(default=0, ge=0)

    @classmethod
    def from_profile(
        cls,
        profile: GateProfileName = "track_a_asset",
        delta_fscore_1mm: float = 0.0,
        max_non_manifold_edges: int = 0,
    ) -> RoundtripGate:
        """Construct roundtrip gate from a canonical ``synapse.metrics.profiles`` profile."""
        table = get_profile_thresholds(profile)
        return cls(
            delta_fscore_1mm=delta_fscore_1mm,
            volume_breathing_pct=table["volume_breathing_pct"],
            max_non_manifold_edges=max_non_manifold_edges,
        )


# -------------------------------------------------------------------
# Intent log models
# -------------------------------------------------------------------


class StepAnnotation(BaseModel):
    """Fine-grained step annotation for PRM export.

    Attributes:
        step_id: Unique step identifier.
        action_type: The type of action taken.
        label: Correctness label (+/-).
        metric_delta: Change in the quality metric.
        state_uri: URI to the state snapshot at this step.
    """

    step_id: str
    action_type: str = ""
    label: Literal["+", "-"] = "+"
    metric_delta: float = 0.0
    state_uri: str = ""


class IntentGroup(BaseModel):
    """Transaction log for a single edit intent group.

    Attributes:
        intent_id: Unique identifier for this intent group.
        asset_id: Asset being edited.
        source: Whether auto-fix or manual HITL.
        steps: Ordered list of step annotations.
        undo_triggered: Whether the artist pressed Undo.
        final_passed_gate: Whether the final state passed QA.
        initial_state_uri: URI to state before edits.
        final_state_uri: URI to state after edits.
        prompt_context: Textual description of the defect/intent.
    """

    intent_id: str
    asset_id: str = ""
    source: Literal["auto", "manual"] = "auto"
    steps: list[StepAnnotation] = Field(default_factory=list)
    undo_triggered: bool = False
    final_passed_gate: bool = True
    initial_state_uri: str = ""
    final_state_uri: str = ""
    prompt_context: str = ""


class DPORecord(BaseModel):
    """A single DPO preference pair compatible with ``PreferencePair``.

    Attributes:
        pair_id: Unique pair identifier (defaults to ``source_intent_id``).
        prompt: The prompt/context describing the task.
        state_before: URI to the state before editing (also aliased as ``condition_key``).
        chosen: URI to the preferred (accepted) state (also aliased as ``chosen_uri``).
        rejected: URI to the rejected (undone) state (also aliased as ``rejected_uri``).
        asset_id: Asset identifier.
        source_intent_id: Source intent group ID.
        source: Origin of the pair (`human`, `self`, `golden`).
        category: Target LoRA category.
        is_golden: Whether this record is an L0 golden sample.
    """

    pair_id: str = ""
    prompt: str
    state_before: str
    chosen: str
    rejected: str
    asset_id: str = ""
    source_intent_id: str = ""
    source: Literal["self", "human", "golden"] = "human"
    category: str = "style"
    is_golden: bool = False

    @property
    def chosen_uri(self) -> str:
        """Canonical alias for ``PreferencePair`` compatibility."""
        return self.chosen

    @property
    def rejected_uri(self) -> str:
        """Canonical alias for ``PreferencePair`` compatibility."""
        return self.rejected

    @property
    def condition_key(self) -> str:
        """Canonical alias for ``PreferencePair`` compatibility."""
        return self.state_before

    def to_preference_pair(self) -> Any:
        """Convert this ``DPORecord`` into a ``synapse.rsi.pair_builder.PreferencePair``."""
        from synapse.rsi.pair_builder import PreferencePair

        return PreferencePair.from_dpo_record(
            self,
            source=self.source,
            category=self.category,
            is_golden=self.is_golden,
        )


class PRMRecord(BaseModel):
    """A single PRM step-level annotation record.

    Attributes:
        intent_id: Source intent group ID.
        step_id: Step identifier.
        action_type: Action type.
        label: Correctness label.
        metric_delta: Quality metric change.
        state_uri: State snapshot URI.
        prompt_context: Textual context.
    """

    intent_id: str
    step_id: str
    action_type: str = ""
    label: str = "+"
    metric_delta: float = 0.0
    state_uri: str = ""
    prompt_context: str = ""


# -------------------------------------------------------------------
# Refinement orchestrator
# -------------------------------------------------------------------


def run_targeted_refinement(
    *,
    report: dict[str, Any],
    gate: RoundtripGate | None = None,
    auto_fix_fn: Any | None = None,
    manifold_check_fn: Any | None = None,
    require_manifold_check: bool = False,
) -> dict[str, Any]:
    """Orchestrate targeted auto-fix from an InspectionReport.

    Routes each defect to the appropriate remediation action
    and enforces Stage 5 roundtrip gates.

    Args:
        report: InspectionReport as a dict (from model_dump()).
        gate: Roundtrip gate thresholds.
        auto_fix_fn: Optional callable for auto-fix operations.
            Signature: (defect_dict, action) -> result_dict.
        manifold_check_fn: Optional callable to verify manifold
            integrity.  Signature: () -> dict with
            ``non_manifold_edges`` (int), ``volume_change_pct``
            (float), ``fscore_delta`` (float).
        require_manifold_check: If True, raises QualityGateError
            when ``manifold_check_fn`` is not provided (fail-closed).

    Returns:
        Dict with ``status``, ``fixed_count``, ``hitl_count``,
        ``gate_result``, and ``defect_actions``.

    Raises:
        QualityGateError: If roundtrip gate fails after fixes.
    """
    if gate is None:
        gate = RoundtripGate()

    if require_manifold_check and manifold_check_fn is None:
        raise QualityGateError(
            "Manifold check is required (require_manifold_check=True), "
            "but no manifold_check_fn was provided.",
            agent_fix_hint=(
                "Provide a manifold_check_fn callback from vg_bridge "
                "to verify watertightness and volume breathing."
            ),
        )

    defects = report.get("defects", [])
    defect_actions: list[dict[str, Any]] = []
    fixed_count = 0
    hitl_count = 0
    unfixed_auto_count = 0

    for defect in defects:
        action = defect.get("suggested_action", "accept")
        defect_id = defect.get("defect_id", "unknown")
        category = defect.get("category", "unknown")

        result: dict[str, Any] = {
            "defect_id": defect_id,
            "category": category,
            "action": action,
            "status": "skipped",
        }

        if action == "accept":
            result["status"] = "accepted"
        elif action in ("auto_remesh", "auto_inpaint_texture"):
            if auto_fix_fn is not None:
                try:
                    fix_result = auto_fix_fn(defect, action)
                    result["status"] = "fixed"
                    result["fix_result"] = fix_result
                    fixed_count += 1
                except Exception as exc:
                    result["status"] = "fix_failed"
                    result["error"] = str(exc)
                    hitl_count += 1
            else:
                # F1 fix: do NOT increment fixed_count when no fixer is provided
                result["status"] = "no_auto_fix_available"
                unfixed_auto_count += 1
        elif action == "manual_hitl":
            result["status"] = "requires_hitl"
            hitl_count += 1

        defect_actions.append(result)

    # Run manifold check if provided
    gate_result: dict[str, Any] = {
        "passed": True,
        "waived": manifold_check_fn is None,
        "checks": {},
    }

    if manifold_check_fn is not None:
        check = manifold_check_fn()
        nm_edges = check.get("non_manifold_edges")
        vol_pct = check.get("volume_change_pct")
        fs_delta = check.get("fscore_delta")

        gate_result["checks"] = {
            "non_manifold_edges": nm_edges,
            "volume_change_pct": vol_pct,
            "fscore_delta": fs_delta,
        }

        failures: list[str] = []
        if (
            fs_delta is None
            or not isinstance(fs_delta, (int, float))
            or not math.isfinite(fs_delta)
        ):
            failures.append(
                f"fscore_delta={fs_delta!r} is missing or non-finite"
            )
        elif fs_delta < gate.delta_fscore_1mm:
            failures.append(
                f"delta_fscore_1mm={fs_delta:.4f} "
                f"< threshold={gate.delta_fscore_1mm:.4f}"
            )

        if (
            vol_pct is None
            or not isinstance(vol_pct, (int, float))
            or not math.isfinite(vol_pct)
        ):
            failures.append(
                f"volume_change_pct={vol_pct!r} is missing or non-finite"
            )
        elif vol_pct > gate.volume_breathing_pct:
            failures.append(
                f"volume_breathing_pct={vol_pct:.2f}% "
                f"> threshold="
                f"{gate.volume_breathing_pct:.2f}%"
            )

        if (
            nm_edges is None
            or not isinstance(nm_edges, int)
        ):
            failures.append(
                f"non_manifold_edges={nm_edges!r} is missing or non-int"
            )
        elif nm_edges > gate.max_non_manifold_edges:
            failures.append(
                f"non_manifold_edges={nm_edges} "
                f"> threshold="
                f"{gate.max_non_manifold_edges}"
            )

        if failures:
            gate_result["passed"] = False
            gate_result["failures"] = failures

    result_dict = {
        "status": "completed",
        "fixed_count": fixed_count,
        "unfixed_auto_count": unfixed_auto_count,
        "hitl_count": hitl_count,
        "gate_result": gate_result,
        "defect_actions": defect_actions,
    }

    if not gate_result["passed"]:
        raise QualityGateError(
            "Roundtrip gate failed after auto-fix: "
            + "; ".join(gate_result.get("failures", [])),
            agent_fix_hint=(
                "Auto-fix did not meet quality thresholds. "
                "Escalate to L2 manual HITL refinement."
            ),
            details=result_dict,
        )

    return result_dict


# -------------------------------------------------------------------
# RL flywheel data exporter
# -------------------------------------------------------------------


def _load_intent_groups(
    log_dir: Path,
) -> list[IntentGroup]:
    """Load IntentGroup records from a directory of JSON files.

    Args:
        log_dir: Directory containing ``*.json`` intent logs.

    Returns:
        List of parsed IntentGroup records.
    """
    groups: list[IntentGroup] = []
    log_path = Path(log_dir)

    if not log_path.is_dir():
        raise ContractValidationError(
            f"Intent log directory does not exist: {log_path}",
            agent_fix_hint=(
                f"Ensure '{log_path}' is a valid directory "
                "containing intent group JSON files."
            ),
        )

    for f in sorted(log_path.glob("*.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                groups.append(IntentGroup(**item))
        else:
            groups.append(IntentGroup(**raw))

    return groups


def export_rl_flywheel_dataset(
    *,
    log_dir: Path,
    output_path: Path,
    format: Literal["prm", "dpo"] = "dpo",
    intent_groups: list[IntentGroup] | None = None,
) -> dict[str, Any]:
    """Export RL flywheel training data from intent logs.

    For DPO format:
    - Identifies undo-triggered intents or auto-fix failures
      followed by manual success
    - Records the undone/failed state as ``rejected`` and
      the final passing state as ``chosen``
    - Manifold-violating samples are NEVER placed in ``chosen``

    For PRM format:
    - Extracts per-step annotations with labels and metric deltas

    Args:
        log_dir: Directory containing IntentGroup JSON files.
        output_path: Path to write the output JSONL file.
        format: Export format (``prm`` or ``dpo``).
        intent_groups: Pre-loaded groups (overrides log_dir scan).

    Returns:
        Dict with ``format``, ``record_count``, ``output_path``.
    """
    if intent_groups is None:
        intent_groups = _load_intent_groups(log_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[DPORecord] | list[PRMRecord]
    if format == "dpo":
        records = _build_dpo_records(intent_groups)
    else:
        records = _build_prm_records(intent_groups)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(rec.model_dump(), default=str)
                + "\n"
            )

    return {
        "format": format,
        "record_count": len(records),
        "output_path": str(output_path),
    }


def _build_dpo_records(
    groups: list[IntentGroup],
) -> list[DPORecord]:
    """Build DPO preference pairs from intent groups.

    Rules:
    - An intent with ``undo_triggered=True`` that was later
      fixed by a human is a valid DPO pair
    - Auto-fix that failed gate then got manually fixed is
      also a valid pair
    - Manifold-violating final states are NEVER in ``chosen``
    """
    records: list[DPORecord] = []

    # Group by asset_id to find undo -> fix pairs
    by_asset: dict[str, list[IntentGroup]] = {}
    for g in groups:
        by_asset.setdefault(g.asset_id, []).append(g)

    for asset_id, asset_groups in by_asset.items():
        # Find rejected (undo) and chosen (final pass) pairs
        rejected_intents: list[IntentGroup] = []
        chosen_intents: list[IntentGroup] = []

        for g in asset_groups:
            if g.undo_triggered or not g.final_passed_gate:
                rejected_intents.append(g)
            elif g.final_passed_gate:
                chosen_intents.append(g)

        # Pair each rejected intent with a distinct chosen intent on the same initial_state_uri
        used_chosen_ids: set[str] = set()
        for rej_idx, rej in enumerate(asset_groups):
            if not (rej.undo_triggered or not rej.final_passed_gate):
                continue

            def _matches_initial_state(
                r: IntentGroup, c: IntentGroup,
            ) -> bool:
                if not c.final_passed_gate or c.undo_triggered:
                    return False
                if c.final_state_uri == r.final_state_uri and c.final_state_uri != "":
                    return False
                # If both specify initial_state_uri, they MUST match exactly
                if r.initial_state_uri and c.initial_state_uri:
                    return r.initial_state_uri == c.initial_state_uri
                # Only allow empty initial_state_uri when both are empty
                return r.initial_state_uri == c.initial_state_uri

            # Prefer a subsequent passing intent in chronological order that hasn't been used yet
            matched_ch: IntentGroup | None = None
            for ch in asset_groups[rej_idx + 1 :]:
                if ch.intent_id not in used_chosen_ids and _matches_initial_state(rej, ch):
                    matched_ch = ch
                    break

            # Fallback 1: any unused passing intent on the same initial_state_uri
            if matched_ch is None:
                for ch in chosen_intents:
                    if ch.intent_id not in used_chosen_ids and _matches_initial_state(rej, ch):
                        matched_ch = ch
                        break

            # Fallback 2: any passing intent on the same initial_state_uri
            if matched_ch is None:
                for ch in chosen_intents:
                    if _matches_initial_state(rej, ch):
                        matched_ch = ch
                        break

            if matched_ch is not None:
                used_chosen_ids.add(matched_ch.intent_id)
                records.append(
                    DPORecord(
                        pair_id=f"{rej.intent_id}__{matched_ch.intent_id}",
                        prompt=rej.prompt_context
                        or f"Fix defect on asset {asset_id}",
                        state_before=rej.initial_state_uri,
                        chosen=matched_ch.final_state_uri,
                        rejected=rej.final_state_uri,
                        asset_id=asset_id,
                        source_intent_id=rej.intent_id,
                        source="human" if matched_ch.source == "manual" else "self",
                    )
                )

    return records


def _build_prm_records(
    groups: list[IntentGroup],
) -> list[PRMRecord]:
    """Build PRM step-level records from intent groups."""
    records: list[PRMRecord] = []

    for g in groups:
        for step in g.steps:
            records.append(
                PRMRecord(
                    intent_id=g.intent_id,
                    step_id=step.step_id,
                    action_type=step.action_type,
                    label=step.label,
                    metric_delta=step.metric_delta,
                    state_uri=step.state_uri,
                    prompt_context=g.prompt_context,
                )
            )

    return records

