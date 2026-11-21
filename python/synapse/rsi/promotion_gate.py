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
import math
from pathlib import Path

from pydantic import BaseModel, Field

from synapse.errors import ContractValidationError, QualityGateError

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
        style: Style consistency sub-score.
        fgpp_fidelity: Physical FGPP fidelity sub-score.
    """

    scenario_id: str
    overall: float = Field(default=0.0, ge=0.0)
    geom: float = Field(default=0.0, ge=0.0)
    temporal: float = Field(default=0.0, ge=0.0)
    style: float = Field(default=0.0, ge=0.0)
    fgpp_fidelity: float = Field(default=0.0, ge=0.0)


class BenchReport(BaseModel):
    """Synapse-Bench full evaluation report.

    Attributes:
        overall_score: Aggregate overall score.
        geom_score: Aggregate geometry score.
        temporal_score: Aggregate temporal score.
        style_score: Aggregate style score.
        fgpp_fidelity_score: Aggregate physical FGPP fidelity score.
        scenarios: Per-scenario score breakdown.
        lora_sha256: SHA256 of the LoRA weights evaluated.
    """

    overall_score: float = Field(default=0.0, ge=0.0)
    geom_score: float = Field(default=0.0, ge=0.0)
    temporal_score: float = Field(default=0.0, ge=0.0)
    style_score: float = Field(default=0.0, ge=0.0)
    fgpp_fidelity_score: float = Field(default=0.0, ge=0.0)
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


def _validate_finite_report(
    report: BenchReport,
    label: str,
) -> None:
    """Ensure all aggregate and scenario scores are finite."""
    for name in ("overall_score", "geom_score", "temporal_score", "style_score", "fgpp_fidelity_score"):
        val = getattr(report, name)
        if not math.isfinite(val):
            raise ContractValidationError(
                f"Non-finite {name}={val!r} in {label} BenchReport",
                agent_fix_hint=(
                    f"Ensure {label} benchmark evaluation produced finite aggregate scores."
                ),
            )
    for sc in report.scenarios:
        for name in ("overall", "geom", "temporal", "style", "fgpp_fidelity"):
            val = getattr(sc, name)
            if not math.isfinite(val):
                raise ContractValidationError(
                    f"Non-finite {name}={val!r} for scenario "
                    f"'{sc.scenario_id}' in {label} BenchReport",
                    agent_fix_hint=(
                        f"Re-evaluate scenario '{sc.scenario_id}' to eliminate NaN/Inf scores."
                    ),
                )


def evaluate_monotonic_promotion(
    *,
    baseline: BenchReport,
    candidate: BenchReport,
    epsilon_min: float = 0.0,
    noise_tolerance: float = 0.0,
    min_scenarios: int = 30,
    raise_on_reject: bool = True,
) -> PromotionResult:
    """Evaluate whether candidate weights should be promoted.

    Checks 4 conjunctive conditions:
    1. OverallScore(candidate) > OverallScore(baseline) + epsilon
    2. GeomScore(candidate) >= GeomScore(baseline)
    3. TemporalScore(candidate) >= TemporalScore(baseline)
    4. Zero per-scenario regression (within noise tolerance) across
       identical scenario sets

    Args:
        baseline: Current production baseline report.
        candidate: Candidate weights full benchmark report.
        epsilon_min: Minimum overall score improvement required.
        noise_tolerance: Maximum allowed per-scenario regression
            (0.0 = strict zero tolerance).
        min_scenarios: Minimum required number of scenarios when scenario
            breakdowns are provided (default: 30).
        raise_on_reject: Whether to raise QualityGateError if not promoted
            (default: True).

    Returns:
        PromotionResult with pass/fail and quarantine entries.

    Raises:
        ContractValidationError: If scenario sets mismatch or contain
            non-finite scores (exit code 10).
        QualityGateError: If any condition fails and raise_on_reject is True
            (exit code 40).
    """
    _validate_finite_report(baseline, "baseline")
    _validate_finite_report(candidate, "candidate")

    # Validate parameter constraints
    if not math.isfinite(epsilon_min) or epsilon_min < 0:
        raise ContractValidationError(
            f"epsilon_min must be finite and >= 0, got {epsilon_min!r}",
            agent_fix_hint="Pass a non-negative finite value for epsilon_min.",
        )
    if not math.isfinite(noise_tolerance) or noise_tolerance < 0:
        raise ContractValidationError(
            f"noise_tolerance must be finite and >= 0, got {noise_tolerance!r}",
            agent_fix_hint="Pass a non-negative finite value for noise_tolerance.",
        )

    # Reject duplicate scenario_id entries within each report
    def _check_no_duplicate_scenarios(scenarios: list[BenchScenarioScore], label: str) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for s in scenarios:
            if s.scenario_id in seen:
                duplicates.append(s.scenario_id)
            seen.add(s.scenario_id)
        if duplicates:
            raise ContractValidationError(
                f"Duplicate scenario_id detected in {label} BenchReport: {duplicates}. "
                "Silent deduplication is FORBIDDEN.",
                agent_fix_hint="Ensure each scenario_id appears exactly once in the scenarios list.",
            )

    _check_no_duplicate_scenarios(baseline.scenarios, "baseline")
    _check_no_duplicate_scenarios(candidate.scenarios, "candidate")

    baseline_scenarios = baseline.scenarios
    candidate_scenarios = candidate.scenarios

    # Cross-verify BenchReport aggregate scores with scenario means
    def _verify_consistency(report: BenchReport, label: str) -> None:
        if not report.scenarios:
            return
        n = len(report.scenarios)
        mean_overall = sum(s.overall for s in report.scenarios) / n
        if abs(report.overall_score - mean_overall) > 1e-2:
            raise ContractValidationError(
                f"Inconsistent overall_score in {label} BenchReport: aggregate {report.overall_score:.4f} "
                f"diverges from scenario mean {mean_overall:.4f}.",
                agent_fix_hint="Ensure aggregate overall_score accurately reflects scenario mean.",
            )

    _verify_consistency(baseline, "baseline")
    _verify_consistency(candidate, "candidate")

    baseline_ids = {s.scenario_id for s in baseline_scenarios}
    candidate_ids = {s.scenario_id for s in candidate_scenarios}
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        raise ContractValidationError(
            "Candidate and baseline scenario sets do not match: "
            f"missing_in_candidate={missing}, "
            f"extra_in_candidate={extra}. "
            "Dropping scenarios to bypass promotion gate is FORBIDDEN.",
            agent_fix_hint=(
                "Evaluate candidate on the exact same scenario set as the baseline report."
            ),
        )

    failures: list[str] = []
    conditions: dict[str, bool] = {}
    quarantine: list[QuarantineEntry] = []

    # Condition 0: Minimum scenario count
    if min_scenarios > 0:
        if len(baseline_scenarios) == 0:
            raise ContractValidationError(
                "Condition 0 failed: empty scenario list. "
                f"At least {min_scenarios} scenarios are required when scenario validation is enabled.",
                agent_fix_hint="Provide a non-empty scenario list or pass min_scenarios=0.",
            )
        if len(baseline_scenarios) < min_scenarios:
            failures.append(
                f"Condition 0 failed: Scenario count {len(baseline_scenarios)} "
                f"< required minimum {min_scenarios}"
            )
            conditions["min_scenarios"] = False
        else:
            conditions["min_scenarios"] = True
    else:
        conditions["min_scenarios"] = True

    # Condition 1: Overall score improvement
    overall_improved = candidate.overall_score > baseline.overall_score + epsilon_min
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
            f"Condition 2 failed: GeomScore {candidate.geom_score:.4f} < {baseline.geom_score:.4f}"
        )

    # Condition 3: Temporal score non-regression
    temporal_ok = candidate.temporal_score >= baseline.temporal_score
    conditions["temporal_non_regression"] = temporal_ok
    if not temporal_ok:
        failures.append(
            f"Condition 3 failed: TemporalScore "
            f"{candidate.temporal_score:.4f} < "
            f"{baseline.temporal_score:.4f}"
        )

    # Condition 5: Style score non-regression (if populated)
    if baseline.style_score > 0.0 or candidate.style_score > 0.0:
        style_ok = candidate.style_score >= baseline.style_score
        conditions["style_non_regression"] = style_ok
        if not style_ok:
            failures.append(
                f"Condition 5 failed: StyleScore {candidate.style_score:.4f} < {baseline.style_score:.4f}"
            )

    # Condition 6: FGPP fidelity score non-regression (if populated)
    if baseline.fgpp_fidelity_score > 0.0 or candidate.fgpp_fidelity_score > 0.0:
        fgpp_ok = candidate.fgpp_fidelity_score >= baseline.fgpp_fidelity_score
        conditions["fgpp_fidelity_non_regression"] = fgpp_ok
        if not fgpp_ok:
            failures.append(
                f"Condition 6 failed: FgppFidelityScore {candidate.fgpp_fidelity_score:.4f} < {baseline.fgpp_fidelity_score:.4f}"
            )

    # Condition 4: Zero per-scenario regression
    baseline_by_id = {s.scenario_id: s for s in baseline_scenarios}
    per_scenario_ok = True

    for cs in candidate_scenarios:
        bs = baseline_by_id[cs.scenario_id]
        # Evaluate per-scenario regressions across all pillars
        regressions: list[tuple[str, float, float, float]] = [
            ("overall", bs.overall - cs.overall, bs.overall, cs.overall),
            ("geom", bs.geom - cs.geom, bs.geom, cs.geom),
            ("temporal", bs.temporal - cs.temporal, bs.temporal, cs.temporal),
        ]
        if bs.style > 0.0 or cs.style > 0.0:
            regressions.append(("style", bs.style - cs.style, bs.style, cs.style))
        if bs.fgpp_fidelity > 0.0 or cs.fgpp_fidelity > 0.0:
            regressions.append(("fgpp_fidelity", bs.fgpp_fidelity - cs.fgpp_fidelity, bs.fgpp_fidelity, cs.fgpp_fidelity))
        active_regs = [r for r in regressions if r[1] > noise_tolerance]
        if active_regs:
            per_scenario_ok = False
            # Dedup per scenario: record single worst regression entry in quarantine
            worst = max(active_regs, key=lambda x: x[1])
            metric_name, reg_val, b_val, c_val = worst
            entry = QuarantineEntry(
                scenario_id=cs.scenario_id,
                metric=metric_name,
                baseline_value=b_val,
                candidate_value=c_val,
                regression=round(reg_val, 6),
                diagnosis=(
                    f"Scenario '{cs.scenario_id}' {metric_name} regressed by "
                    f"{reg_val:.4f} (baseline={b_val:.4f} -> candidate={c_val:.4f}). "
                    "Review LoRA training data and hyperparameters for this scenario."
                ),
            )
            quarantine.append(entry)

    conditions["zero_per_scenario_regression"] = per_scenario_ok
    if not per_scenario_ok:
        regressed_ids = [q.scenario_id for q in quarantine]
        failures.append(
            f"Condition 4 failed: {len(quarantine)} scenario(s) regressed: {regressed_ids}"
        )

    promoted = len(failures) == 0

    result = PromotionResult(
        promoted=promoted,
        conditions=conditions,
        failures=failures,
        quarantine=quarantine,
        lora_sha256=candidate.lora_sha256,
    )

    if not promoted and raise_on_reject:
        raise QualityGateError(
            "Monotonic promotion gate failed: " + "; ".join(failures),
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
        "quarantine": [q.model_dump() for q in result.quarantine],
    }
    output_path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    return output_path.resolve()
