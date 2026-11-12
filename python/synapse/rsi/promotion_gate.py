"""Monotonic promotion gate with quarantine manifest.

Provides:
- ``evaluate_monotonic_promotion``: Strict 4-condition conjunctive
  check ensuring model weights only promote when ALL metrics
  improve monotonically with zero per-scenario regression.

The gate prevents "robbing Peter to pay Paul" — no scenario
may regress even if overall scores improve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapse.errors import QualityGateError


# -------------------------------------------------------------------
# Report models
# -------------------------------------------------------------------


class BenchScenarioScore(BaseModel):
    """Score for a single benchmark scenario.

    Attributes:
        scenario_id: Scenario identifier.
        overall: Overall quality score for this scenario.
        geom: Geometry sub-score.
        temporal: Temporal consistency sub-score.
    """

    scenario_id: str
    overall: float = Field(default=0.0, ge=0.0)
    geom: float = Field(default=0.0, ge=0.0)
    temporal: float = Field(default=0.0, ge=0.0)


class BenchReport(BaseModel):
    """Synapse-Bench full evaluation report.

    Attributes:
        overall_score: Aggregate overall score.
        geom_score: Aggregate geometry score.
        temporal_score: Aggregate temporal score.
        scenarios: Per-scenario score breakdown.
        lora_sha256: SHA256 of the LoRA weights evaluated.
    """

    overall_score: float = Field(default=0.0, ge=0.0)
    geom_score: float = Field(default=0.0, ge=0.0)
    temporal_score: float = Field(default=0.0, ge=0.0)
    scenarios: list[BenchScenarioScore] = Field(
        default_factory=list,
    )
    lora_sha256: str = ""


class QuarantineEntry(BaseModel):
    """A single quarantine entry for a regressed scenario.

    Attributes:
        scenario_id: The regressed scenario.
        metric: Which metric regressed.
        baseline_value: Baseline value.
        candidate_value: Candidate value.
        regression: Amount of regression (positive = worse).
        diagnosis: Suggested fix.
    """

    scenario_id: str
    metric: str = "overall"
    baseline_value: float = 0.0
    candidate_value: float = 0.0
    regression: float = 0.0
    diagnosis: str = ""


class PromotionResult(BaseModel):
    """Result of monotonic promotion evaluation.

    Attributes:
        promoted: Whether the candidate was promoted.
        conditions: Status of each of the 4 conditions.
        failures: List of failure descriptions.
        quarantine: List of quarantine entries.
        lora_sha256: SHA256 of candidate LoRA weights.
    """

    promoted: bool = True
    conditions: dict[str, bool] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    quarantine: list[QuarantineEntry] = Field(
        default_factory=list,
    )
    lora_sha256: str = ""


# -------------------------------------------------------------------
# Core evaluation
# -------------------------------------------------------------------


def evaluate_monotonic_promotion(
    *,
    baseline: BenchReport,
    candidate: BenchReport,
    epsilon_min: float = 0.0,
    noise_tolerance: float = 0.0,
) -> PromotionResult:
    """Evaluate whether candidate weights should be promoted.

    Checks 4 conjunctive conditions:
    1. OverallScore(candidate) > OverallScore(baseline) + epsilon
    2. GeomScore(candidate) >= GeomScore(baseline)
    3. TemporalScore(candidate) >= TemporalScore(baseline)
    4. Zero per-scenario regression (within noise tolerance)

    Args:
        baseline: Current production baseline report.
        candidate: Candidate weights full benchmark report.
        epsilon_min: Minimum overall score improvement required.
        noise_tolerance: Maximum allowed per-scenario regression
            (0.0 = strict zero tolerance).

    Returns:
        PromotionResult with pass/fail and quarantine entries.

    Raises:
        QualityGateError: If any condition fails (exit code 40).
    """
    failures: list[str] = []
    conditions: dict[str, bool] = {}
    quarantine: list[QuarantineEntry] = []

    # Condition 1: Overall score improvement
    overall_improved = (
        candidate.overall_score
        > baseline.overall_score + epsilon_min
    )
    conditions["overall_improvement"] = overall_improved
    if not overall_improved:
        failures.append(
            f"Condition 1 failed: OverallScore "
            f"{candidate.overall_score:.4f} not > "
            f"{baseline.overall_score:.4f} + "
            f"{epsilon_min:.4f}"
        )

    # Condition 2: Geometry score non-regression
    geom_ok = candidate.geom_score >= baseline.geom_score
    conditions["geom_non_regression"] = geom_ok
    if not geom_ok:
        failures.append(
            f"Condition 2 failed: GeomScore "
            f"{candidate.geom_score:.4f} < "
            f"{baseline.geom_score:.4f}"
        )

    # Condition 3: Temporal score non-regression
    temporal_ok = (
        candidate.temporal_score >= baseline.temporal_score
    )
    conditions["temporal_non_regression"] = temporal_ok
    if not temporal_ok:
        failures.append(
            f"Condition 3 failed: TemporalScore "
            f"{candidate.temporal_score:.4f} < "
            f"{baseline.temporal_score:.4f}"
        )

    # Condition 4: Zero per-scenario regression
    baseline_by_id = {
        s.scenario_id: s for s in baseline.scenarios
    }
    per_scenario_ok = True

    for cs in candidate.scenarios:
        bs = baseline_by_id.get(cs.scenario_id)
        if bs is None:
            continue

        regression = bs.overall - cs.overall
        if regression > noise_tolerance:
            per_scenario_ok = False
            entry = QuarantineEntry(
                scenario_id=cs.scenario_id,
                metric="overall",
                baseline_value=bs.overall,
                candidate_value=cs.overall,
                regression=round(regression, 6),
                diagnosis=(
                    f"Scenario '{cs.scenario_id}' regressed by "
                    f"{regression:.4f} (baseline={bs.overall:.4f}"
                    f" -> candidate={cs.overall:.4f}). "
                    "Review LoRA training data and "
                    "hyperparameters for this scenario."
                ),
            )
            quarantine.append(entry)

    conditions["zero_per_scenario_regression"] = per_scenario_ok
    if not per_scenario_ok:
        regressed_ids = [q.scenario_id for q in quarantine]
        failures.append(
            f"Condition 4 failed: {len(quarantine)} scenario(s) "
            f"regressed: {regressed_ids}"
        )

    promoted = len(failures) == 0

    result = PromotionResult(
        promoted=promoted,
        conditions=conditions,
        failures=failures,
        quarantine=quarantine,
        lora_sha256=candidate.lora_sha256,
    )

    if not promoted:
        raise QualityGateError(
            "Monotonic promotion gate failed: "
            + "; ".join(failures),
            agent_fix_hint=(
                "Review the quarantine manifest for regressed "
                "scenarios. Retrain LoRA with adjusted data "
                "or hyperparameters to fix regressions."
            ),
            details=result.model_dump(),
        )

    return result


def write_quarantine_manifest(
    result: PromotionResult,
    output_path: Path,
) -> Path:
    """Write quarantine manifest JSON.

    Args:
        result: Promotion result with quarantine entries.
        output_path: Path to write the manifest.

    Returns:
        Absolute path to the written manifest.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "promoted": result.promoted,
        "lora_sha256": result.lora_sha256,
        "failures": result.failures,
        "quarantine": [
            q.model_dump() for q in result.quarantine
        ],
    }
    output_path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    return output_path.resolve()
