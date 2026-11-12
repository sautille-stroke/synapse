"""Tests for PR-25: Monotonic promotion gate + quarantine + CLI.

Validates:
- All 4 conjunctive conditions checked correctly
- 29/30 scenarios improve +5%, 1 regresses -> exit code 40
- Quarantine manifest precisely identifies regressed scenario
- All scenarios improve -> promotion passes
- CLI synapse rsi sub-commands work with --json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synapse.errors import ContractValidationError, QualityGateError
from synapse.rsi.promotion_gate import (
    BenchReport,
    BenchScenarioScore,
    PromotionResult,
    QuarantineEntry,
    evaluate_monotonic_promotion,
    write_quarantine_manifest,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


def _make_scenario_scores(
    n: int = 30,
    base_score: float = 0.7,
) -> list[BenchScenarioScore]:
    """Generate n scenario scores."""
    return [
        BenchScenarioScore(
            scenario_id=f"t{i+1:02d}_scenario_{i}",
            overall=base_score,
            geom=base_score,
            temporal=base_score,
        )
        for i in range(n)
    ]


def _make_baseline(n: int = 30) -> BenchReport:
    """Create a baseline report with 30 scenarios."""
    scores = _make_scenario_scores(n, 0.7)
    return BenchReport(
        overall_score=0.7,
        geom_score=0.7,
        temporal_score=0.7,
        scenarios=scores,
        lora_sha256="a" * 64,
    )


def _make_improved_candidate(
    n: int = 30,
    improvement: float = 0.05,
) -> BenchReport:
    """Create a candidate that improves all scenarios."""
    base = 0.7 + improvement
    scores = _make_scenario_scores(n, base)
    return BenchReport(
        overall_score=base,
        geom_score=base,
        temporal_score=base,
        scenarios=scores,
        lora_sha256="b" * 64,
    )


# -------------------------------------------------------------------
# Condition 1: Overall improvement
# -------------------------------------------------------------------


class TestOverallImprovement:
    """Tests for Condition 1: OverallScore improvement."""

    def test_improved_passes(self) -> None:
        """Candidate with higher overall passes."""
        baseline = _make_baseline()
        candidate = _make_improved_candidate()
        result = evaluate_monotonic_promotion(
            baseline=baseline, candidate=candidate,
        )
        assert result.promoted is True

    def test_same_overall_fails(self) -> None:
        """Same overall score fails (need strict improvement)."""
        baseline = _make_baseline()
        candidate = _make_improved_candidate(improvement=0.0)
        with pytest.raises(QualityGateError) as exc:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert exc.value.exit_code == 40
        assert "Condition 1" in str(exc.value)

    def test_epsilon_min_enforced(self) -> None:
        """Tiny improvement below epsilon_min fails."""
        baseline = _make_baseline()
        candidate = _make_improved_candidate(
            improvement=0.001,
        )
        with pytest.raises(QualityGateError):
            evaluate_monotonic_promotion(
                baseline=baseline,
                candidate=candidate,
                epsilon_min=0.01,
            )


# -------------------------------------------------------------------
# Condition 2 & 3: Geom and Temporal non-regression
# -------------------------------------------------------------------


class TestGeomTemporalNonRegression:
    """Tests for Conditions 2 and 3."""

    def test_geom_regression_fails(self) -> None:
        """Geom score regression fails Condition 2."""
        baseline = _make_baseline()
        candidate = _make_improved_candidate()
        candidate.geom_score = 0.6  # below baseline 0.7
        with pytest.raises(QualityGateError) as exc:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert "Condition 2" in str(exc.value)

    def test_temporal_regression_fails(self) -> None:
        """Temporal score regression fails Condition 3."""
        baseline = _make_baseline()
        candidate = _make_improved_candidate()
        candidate.temporal_score = 0.65
        with pytest.raises(QualityGateError) as exc:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert "Condition 3" in str(exc.value)

    def test_equal_geom_passes(self) -> None:
        """Equal geom score passes (non-strict)."""
        baseline = _make_baseline()
        candidate = _make_improved_candidate()
        candidate.geom_score = 0.7  # equal to baseline
        result = evaluate_monotonic_promotion(
            baseline=baseline, candidate=candidate,
        )
        assert result.conditions["geom_non_regression"] is True


# -------------------------------------------------------------------
# Condition 4: Zero per-scenario regression
# -------------------------------------------------------------------


class TestPerScenarioRegression:
    """Tests for Condition 4: per-scenario zero regression."""

    def test_one_regressed_scenario_triggers_gate(
        self,
    ) -> None:
        """29/30 improve, 1 regresses -> gate failure."""
        baseline = _make_baseline(30)
        candidate = _make_improved_candidate(30)

        # Make scenario t01 regress
        candidate.scenarios[0] = BenchScenarioScore(
            scenario_id="t01_scenario_0",
            overall=0.695,  # baseline is 0.7 -> regression
            geom=0.695,
            temporal=0.695,
        )

        with pytest.raises(QualityGateError) as exc:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert exc.value.exit_code == 40
        assert "Condition 4" in str(exc.value)
        assert "t01_scenario_0" in str(exc.value)

    def test_quarantine_identifies_regressed_scenario(
        self,
    ) -> None:
        """Quarantine manifest pinpoints exact scenario."""
        baseline = _make_baseline(30)
        candidate = _make_improved_candidate(30)

        # Regress t01
        candidate.scenarios[0] = BenchScenarioScore(
            scenario_id="t01_scenario_0",
            overall=0.695,
            geom=0.695,
            temporal=0.695,
        )

        with pytest.raises(QualityGateError) as exc:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )

        details = exc.value.details
        quarantine = details.get("quarantine", [])
        assert len(quarantine) == 1
        assert quarantine[0]["scenario_id"] == "t01_scenario_0"
        assert quarantine[0]["regression"] > 0

    def test_all_improve_passes(self) -> None:
        """All 30 scenarios improving passes gate."""
        baseline = _make_baseline(30)
        candidate = _make_improved_candidate(30)
        result = evaluate_monotonic_promotion(
            baseline=baseline, candidate=candidate,
        )
        assert result.promoted is True
        assert result.quarantine == []
        assert result.conditions[
            "zero_per_scenario_regression"
        ] is True

    def test_noise_tolerance_allows_minor_regression(
        self,
    ) -> None:
        """Regression within noise tolerance passes."""
        baseline = _make_baseline(30)
        candidate = _make_improved_candidate(30)

        # Minor regression of 0.001
        candidate.scenarios[0] = BenchScenarioScore(
            scenario_id="t01_scenario_0",
            overall=0.699,  # 0.001 regression from 0.7
            geom=0.699,
            temporal=0.699,
        )

        result = evaluate_monotonic_promotion(
            baseline=baseline,
            candidate=candidate,
            noise_tolerance=0.005,
        )
        assert result.promoted is True

    def test_depth_regression_scenario(self) -> None:
        """Spec test: t1_rigid_sphere_orbit depth regression.

        Per spec: candidate improves 29 scenarios by +5%,
        t1_rigid_sphere_orbit worsens depth by 0.005m.
        Gate must trigger exit code 40 and quarantine that
        exact scenario.
        """
        baseline = _make_baseline(30)
        # Override first scenario ID to match spec
        baseline.scenarios[0].scenario_id = (
            "t1_rigid_sphere_orbit"
        )

        candidate = _make_improved_candidate(30)
        candidate.scenarios[0] = BenchScenarioScore(
            scenario_id="t1_rigid_sphere_orbit",
            overall=0.695,  # regressed
            geom=0.695,
            temporal=0.75,
        )

        with pytest.raises(QualityGateError) as exc:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert exc.value.exit_code == 40

        details = exc.value.details
        quarantine = details["quarantine"]
        scenario_ids = [q["scenario_id"] for q in quarantine]
        assert "t1_rigid_sphere_orbit" in scenario_ids

    def test_candidate_dropping_scenario_raises_contract_error(
        self,
    ) -> None:
        """Dropping a scenario from candidate raises ContractValidationError(10)."""
        baseline = _make_baseline(30)
        candidate = _make_improved_candidate(29)
        with pytest.raises(ContractValidationError) as exc_info:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert int(exc_info.value.exit_code) == 10
        assert "missing_in_candidate" in str(exc_info.value)

    def test_nan_score_raises_contract_error(self) -> None:
        """NaN score in candidate report raises ContractValidationError(10)."""
        baseline = _make_baseline(5)
        candidate = _make_improved_candidate(5)
        candidate.scenarios[0].overall = float("nan")
        with pytest.raises(ContractValidationError) as exc_info:
            evaluate_monotonic_promotion(
                baseline=baseline, candidate=candidate,
            )
        assert int(exc_info.value.exit_code) == 10


# -------------------------------------------------------------------
# Quarantine manifest
# -------------------------------------------------------------------


class TestQuarantineManifest:
    """Tests for quarantine manifest writing."""

    def test_write_manifest(self, tmp_path: Path) -> None:
        """Quarantine manifest is written correctly."""
        result = PromotionResult(
            promoted=False,
            failures=["Condition 4 failed"],
            quarantine=[
                QuarantineEntry(
                    scenario_id="t01_test",
                    metric="overall",
                    baseline_value=0.7,
                    candidate_value=0.695,
                    regression=0.005,
                    diagnosis="Fix depth regression",
                ),
            ],
            lora_sha256="c" * 64,
        )

        out = tmp_path / "quarantine_manifest.json"
        written = write_quarantine_manifest(result, out)
        assert written.exists()

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["promoted"] is False
        assert len(data["quarantine"]) == 1
        assert data["quarantine"][0][
            "scenario_id"
        ] == "t01_test"
        assert data["lora_sha256"] == "c" * 64


# -------------------------------------------------------------------
# CLI tests
# -------------------------------------------------------------------


class TestRSICLI:
    """Tests for synapse rsi CLI sub-commands."""

    def test_rsi_help(self) -> None:
        """synapse rsi --help does not crash."""
        from synapse.cli import main_synapse
        with pytest.raises(SystemExit) as exc:
            main_synapse(["rsi", "--help"])
        assert exc.value.code == 0

    def test_rsi_sample_curriculum_json(self) -> None:
        """synapse rsi sample-curriculum --json works."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "rsi", "sample-curriculum",
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert "scenarios" in parsed

    def test_rsi_verify_json(self) -> None:
        """synapse rsi verify --json works with explicit physics metrics."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "rsi", "verify",
                "--sample-id", "test_001",
                "--physics-metrics",
                json.dumps({
                    "chamfer_mm": 1.5,
                    "normal_angular_error_deg": 4.0,
                    "silhouette_iou": 0.96,
                    "depth_rmse_m": 0.01,
                    "foot_slide_velocity_ms": 0.005,
                    "volume_breathing_pct": 1.0,
                    "border_ring_ciede2000": 1.2,
                }),
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert parsed["verdict"]["physics_passed"] is True
        assert parsed["verdict"]["verdict"] in ("ACCEPT", "HARD_NEGATIVE")

    def test_rsi_verify_missing_metrics_exits_10(self) -> None:
        """synapse rsi verify without --physics-metrics exits with code 10."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "rsi", "verify",
                "--sample-id", "test_001",
            ])
        finally:
            sys.stdout = old

        assert code == 10
        parsed = json.loads(buf.getvalue())
        assert parsed["exit_code"] == 10

    def test_rsi_gate_promote_pass_json(self) -> None:
        """synapse rsi gate-promote --json with passing data."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "rsi", "gate-promote",
                "--baseline-overall", "0.7",
                "--candidate-overall", "0.8",
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert parsed["promoted"] is True

    def test_rsi_gate_promote_fail_json(self) -> None:
        """synapse rsi gate-promote --json with failing data."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "rsi", "gate-promote",
                "--baseline-overall", "0.8",
                "--candidate-overall", "0.7",
            ])
        finally:
            sys.stdout = old

        assert code == 40
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["exit_code"] == 40
        assert "agent_fix_hint" in parsed
