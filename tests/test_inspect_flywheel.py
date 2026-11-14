"""Tests for PR-23: Refinement roundtrip + flywheel exporter.

Validates:
- run_targeted_refinement routes defects correctly
- Roundtrip gate enforcement (volume breathing, manifold, fscore)
- export_rl_flywheel_dataset DPO mode produces correct pairs
- export_rl_flywheel_dataset PRM mode extracts step annotations
- Manifold-violating samples never enter DPO chosen set
- CLI inspect refine/export-rl-data sub-commands
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synapse.errors import QualityGateError
from synapse.inspect.flywheel_exporter import (
    IntentGroup,
    RoundtripGate,
    StepAnnotation,
    export_rl_flywheel_dataset,
    run_targeted_refinement,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def clean_report() -> dict:
    """InspectionReport with no defects."""
    return {
        "schema_version": "2.0.0",
        "asset_id": "test_asset_001",
        "asset_name": "Clean Sphere",
        "inspection_timestamp": "2026-01-01T00:00:00Z",
        "overall_quality_score": 0.95,
        "defects": [],
        "requires_hitl": False,
    }


@pytest.fixture
def defective_report() -> dict:
    """InspectionReport with mixed defect actions."""
    return {
        "schema_version": "2.0.0",
        "asset_id": "test_asset_002",
        "asset_name": "Defective Model",
        "inspection_timestamp": "2026-01-01T00:00:00Z",
        "overall_quality_score": 0.3,
        "defects": [
            {
                "defect_id": "d001",
                "category": "mesh_crumple",
                "severity": 0.5,
                "confidence": 0.6,
                "center_3d": [0.0, 0.0, 0.0],
                "bounding_radius_m": 0.05,
                "affected_vertices_count": 120,
                "suggested_action": "auto_remesh",
            },
            {
                "defect_id": "d002",
                "category": "texture_stretch",
                "severity": 0.4,
                "confidence": 0.5,
                "center_3d": [1.0, 0.0, 0.0],
                "bounding_radius_m": 0.03,
                "affected_vertices_count": 50,
                "suggested_action": "auto_inpaint_texture",
            },
            {
                "defect_id": "d003",
                "category": "anatomical_distortion",
                "severity": 0.85,
                "confidence": 0.9,
                "center_3d": [0.0, 1.0, 0.0],
                "bounding_radius_m": 0.1,
                "affected_vertices_count": 300,
                "suggested_action": "manual_hitl",
            },
        ],
        "requires_hitl": True,
    }


@pytest.fixture
def intent_log_dir(tmp_path: Path) -> Path:
    """Create a directory with intent group JSON logs."""
    log_dir = tmp_path / "intent_logs"
    log_dir.mkdir()

    # Auto-fix succeeded (3 intents)
    for i in range(3):
        intent = {
            "intent_id": f"auto_{i:03d}",
            "asset_id": "asset_A",
            "source": "auto",
            "steps": [
                {
                    "step_id": f"s{i}_0",
                    "action_type": "remesh",
                    "label": "+",
                    "metric_delta": 0.1,
                    "state_uri": f"file:///state_auto_{i}.glb",
                },
            ],
            "undo_triggered": False,
            "final_passed_gate": True,
            "initial_state_uri": "file:///initial_A.glb",
            "final_state_uri": f"file:///final_auto_{i}.glb",
            "prompt_context": "Fix mesh crumple defect",
        }
        (log_dir / f"auto_{i:03d}.json").write_text(
            json.dumps(intent), encoding="utf-8",
        )

    # Undo-triggered intents (2)
    for i in range(2):
        intent = {
            "intent_id": f"undo_{i:03d}",
            "asset_id": "asset_A",
            "source": "auto",
            "steps": [
                {
                    "step_id": f"u{i}_0",
                    "action_type": "auto_remesh",
                    "label": "-",
                    "metric_delta": -0.05,
                    "state_uri": f"file:///state_undo_{i}.glb",
                },
            ],
            "undo_triggered": True,
            "final_passed_gate": False,
            "initial_state_uri": "file:///initial_A.glb",
            "final_state_uri": f"file:///bad_auto_{i}.glb",
            "prompt_context": "Fix texture stretch defect",
        }
        (log_dir / f"undo_{i:03d}.json").write_text(
            json.dumps(intent), encoding="utf-8",
        )

    return log_dir


# -------------------------------------------------------------------
# RoundtripGate tests
# -------------------------------------------------------------------


class TestRoundtripGate:
    """Tests for RoundtripGate model."""

    def test_defaults(self) -> None:
        g = RoundtripGate()
        assert g.delta_fscore_1mm == 0.0
        assert g.volume_breathing_pct == 3.0
        assert g.max_non_manifold_edges == 0

    def test_roundtrip(self) -> None:
        g = RoundtripGate(volume_breathing_pct=2.0)
        d = g.model_dump()
        g2 = RoundtripGate(**d)
        assert g2.volume_breathing_pct == 2.0


# -------------------------------------------------------------------
# run_targeted_refinement tests
# -------------------------------------------------------------------


class TestTargetedRefinement:
    """Tests for run_targeted_refinement."""

    def test_clean_report_no_fixes(
        self, clean_report: dict,
    ) -> None:
        """Clean report produces no fixes and passes gate."""
        result = run_targeted_refinement(report=clean_report)
        assert result["status"] == "completed"
        assert result["fixed_count"] == 0
        assert result["hitl_count"] == 0
        assert result["gate_result"]["passed"] is True

    def test_defective_report_routes_actions(
        self, defective_report: dict,
    ) -> None:
        """Defects are correctly routed to actions."""
        result = run_targeted_refinement(
            report=defective_report,
        )
        assert result["status"] == "completed"
        actions = result["defect_actions"]
        assert len(actions) == 3

        # auto_remesh -> no_auto_fix_available (no fn provided)
        assert actions[0]["action"] == "auto_remesh"
        assert actions[0]["status"] == "no_auto_fix_available"
        # F1 regression check: fixed_count must be 0 when auto_fix_fn is None
        assert result["fixed_count"] == 0
        assert result["unfixed_auto_count"] == 2

        # auto_inpaint_texture
        assert actions[1]["action"] == "auto_inpaint_texture"

        # manual_hitl
        assert actions[2]["action"] == "manual_hitl"
        assert actions[2]["status"] == "requires_hitl"

    def test_require_manifold_check_without_fn_raises_40(
        self, clean_report: dict,
    ) -> None:
        """require_manifold_check=True without manifold_check_fn raises 40."""
        with pytest.raises(QualityGateError) as exc_info:
            run_targeted_refinement(
                report=clean_report,
                require_manifold_check=True,
            )
        assert int(exc_info.value.exit_code) == 40

    def test_auto_fix_callback(
        self, defective_report: dict,
    ) -> None:
        """Auto-fix callback is invoked for fixable defects."""
        fix_calls: list[tuple] = []

        def mock_fix(defect: dict, action: str) -> dict:
            fix_calls.append((defect["defect_id"], action))
            return {"ok": True}

        result = run_targeted_refinement(
            report=defective_report,
            auto_fix_fn=mock_fix,
        )
        assert len(fix_calls) == 2
        assert fix_calls[0] == ("d001", "auto_remesh")
        assert fix_calls[1] == ("d002", "auto_inpaint_texture")
        assert result["fixed_count"] == 2

    def test_auto_fix_failure_escalates_to_hitl(
        self, defective_report: dict,
    ) -> None:
        """Failed auto-fix increments hitl count."""
        def failing_fix(defect: dict, action: str) -> dict:
            raise RuntimeError("Fix engine crashed")

        result = run_targeted_refinement(
            report=defective_report,
            auto_fix_fn=failing_fix,
        )
        # 2 auto-fix failures + 1 manual_hitl = 3
        assert result["hitl_count"] == 3

    def test_manifold_check_passes(
        self, clean_report: dict,
    ) -> None:
        """Passing manifold check keeps gate green."""
        result = run_targeted_refinement(
            report=clean_report,
            manifold_check_fn=lambda: {
                "non_manifold_edges": 0,
                "volume_change_pct": 1.5,
                "fscore_delta": 0.02,
            },
        )
        assert result["gate_result"]["passed"] is True

    def test_manifold_check_fails_volume(
        self, clean_report: dict,
    ) -> None:
        """Volume breathing exceeding threshold raises error."""
        with pytest.raises(QualityGateError) as exc_info:
            run_targeted_refinement(
                report=clean_report,
                manifold_check_fn=lambda: {
                    "non_manifold_edges": 0,
                    "volume_change_pct": 5.0,
                    "fscore_delta": 0.01,
                },
            )
        assert exc_info.value.exit_code == 40
        assert "volume_breathing_pct" in str(exc_info.value)

    def test_manifold_check_fails_non_manifold(
        self, clean_report: dict,
    ) -> None:
        """Non-manifold edges trigger gate failure."""
        with pytest.raises(QualityGateError):
            run_targeted_refinement(
                report=clean_report,
                manifold_check_fn=lambda: {
                    "non_manifold_edges": 5,
                    "volume_change_pct": 1.0,
                    "fscore_delta": 0.01,
                },
            )


# -------------------------------------------------------------------
# IntentGroup / StepAnnotation model tests
# -------------------------------------------------------------------


class TestIntentModels:
    """Tests for IntentGroup and related models."""

    def test_step_annotation_defaults(self) -> None:
        s = StepAnnotation(step_id="s001")
        assert s.label == "+"
        assert s.metric_delta == 0.0

    def test_intent_group_roundtrip(self) -> None:
        g = IntentGroup(
            intent_id="ig001",
            asset_id="a001",
            source="manual",
            undo_triggered=True,
            final_passed_gate=False,
        )
        d = g.model_dump()
        g2 = IntentGroup(**d)
        assert g2.undo_triggered is True
        assert g2.final_passed_gate is False
        assert g2.source == "manual"


# -------------------------------------------------------------------
# DPO export tests
# -------------------------------------------------------------------


class TestDPOExport:
    """Tests for DPO format export."""

    def test_dpo_export_correct_pair_count(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """DPO export produces exactly 2 pairs for 2 undos."""
        out = tmp_path / "dpo.jsonl"
        result = export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="dpo",
        )
        assert result["format"] == "dpo"
        assert result["record_count"] == 2
        assert out.exists()

    def test_dpo_records_structure(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """Each DPO record has required fields."""
        out = tmp_path / "dpo.jsonl"
        export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="dpo",
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            rec = json.loads(line)
            assert "prompt" in rec
            assert "state_before" in rec
            assert "chosen" in rec
            assert "rejected" in rec
            assert rec["asset_id"] == "asset_A"

    def test_dpo_rejected_is_undone_state(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """DPO rejected points to the undone state URI."""
        out = tmp_path / "dpo.jsonl"
        export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="dpo",
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            rec = json.loads(line)
            assert "bad_auto_" in rec["rejected"]

    def test_dpo_chosen_is_passing_state(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """DPO chosen points to a state that passed gate."""
        out = tmp_path / "dpo.jsonl"
        export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="dpo",
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            rec = json.loads(line)
            assert "final_auto_" in rec["chosen"]

    def test_dpo_no_undo_no_pairs(
        self, tmp_path: Path,
    ) -> None:
        """No undo-triggered intents means zero DPO pairs."""
        log_dir = tmp_path / "no_undo_logs"
        log_dir.mkdir()
        intent = {
            "intent_id": "clean_001",
            "asset_id": "asset_B",
            "source": "auto",
            "steps": [],
            "undo_triggered": False,
            "final_passed_gate": True,
            "initial_state_uri": "file:///init.glb",
            "final_state_uri": "file:///final.glb",
            "prompt_context": "All good",
        }
        (log_dir / "clean.json").write_text(
            json.dumps(intent), encoding="utf-8",
        )

        out = tmp_path / "dpo.jsonl"
        result = export_rl_flywheel_dataset(
            log_dir=log_dir,
            output_path=out,
            format="dpo",
        )
        assert result["record_count"] == 0


# -------------------------------------------------------------------
# PRM export tests
# -------------------------------------------------------------------


class TestPRMExport:
    """Tests for PRM format export."""

    def test_prm_export_step_count(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """PRM export produces one record per step."""
        out = tmp_path / "prm.jsonl"
        result = export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="prm",
        )
        assert result["format"] == "prm"
        # 3 auto (1 step each) + 2 undo (1 step each) = 5
        assert result["record_count"] == 5

    def test_prm_record_fields(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """Each PRM record has required fields."""
        out = tmp_path / "prm.jsonl"
        export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="prm",
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            rec = json.loads(line)
            assert "intent_id" in rec
            assert "step_id" in rec
            assert "label" in rec
            assert rec["label"] in ("+", "-")

    def test_prm_negative_labels(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """Undo-triggered steps have negative labels."""
        out = tmp_path / "prm.jsonl"
        export_rl_flywheel_dataset(
            log_dir=intent_log_dir,
            output_path=out,
            format="prm",
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        neg_count = sum(
            1 for line in lines
            if json.loads(line)["label"] == "-"
        )
        assert neg_count == 2  # 2 undo-triggered intents


# -------------------------------------------------------------------
# Pre-loaded intent_groups tests
# -------------------------------------------------------------------


class TestPreloadedGroups:
    """Tests with pre-loaded IntentGroup objects."""

    def test_dpo_with_preloaded_groups(
        self, tmp_path: Path,
    ) -> None:
        """DPO export works with pre-loaded groups."""
        groups = [
            IntentGroup(
                intent_id="rej_001",
                asset_id="ax",
                undo_triggered=True,
                final_passed_gate=False,
                initial_state_uri="init",
                final_state_uri="bad",
                prompt_context="fix it",
            ),
            IntentGroup(
                intent_id="good_001",
                asset_id="ax",
                undo_triggered=False,
                final_passed_gate=True,
                initial_state_uri="init",
                final_state_uri="good",
                prompt_context="fixed",
            ),
        ]

        out = tmp_path / "dpo.jsonl"
        # Use a dummy log dir but override with intent_groups
        log_dir = tmp_path / "dummy_logs"
        log_dir.mkdir()

        result = export_rl_flywheel_dataset(
            log_dir=log_dir,
            output_path=out,
            format="dpo",
            intent_groups=groups,
        )
        assert result["record_count"] == 1

        line = out.read_text(encoding="utf-8").strip()
        rec = json.loads(line)
        assert rec["chosen"] == "good"
        assert rec["rejected"] == "bad"

    def test_manifold_violating_never_chosen(
        self, tmp_path: Path,
    ) -> None:
        """Final state that failed gate is never in chosen."""
        groups = [
            IntentGroup(
                intent_id="bad_001",
                asset_id="ax",
                undo_triggered=True,
                final_passed_gate=False,
                final_state_uri="bad_manifold",
            ),
            IntentGroup(
                intent_id="bad_002",
                asset_id="ax",
                undo_triggered=False,
                final_passed_gate=False,
                final_state_uri="also_bad",
            ),
        ]

        out = tmp_path / "dpo.jsonl"
        log_dir = tmp_path / "dummy_logs"
        log_dir.mkdir()

        result = export_rl_flywheel_dataset(
            log_dir=log_dir,
            output_path=out,
            format="dpo",
            intent_groups=groups,
        )
        # No valid chosen state -> no pairs
        assert result["record_count"] == 0

    def test_dpo_does_not_cross_pair_different_initial_states(
        self, tmp_path: Path,
    ) -> None:
        """Intents with different initial_state_uri on the same asset do not cross-pair."""
        groups = [
            IntentGroup(
                intent_id="rej_state1",
                asset_id="ax",
                undo_triggered=True,
                final_passed_gate=False,
                initial_state_uri="file:///state_1.glb",
                final_state_uri="file:///bad_1.glb",
            ),
            IntentGroup(
                intent_id="good_state2",
                asset_id="ax",
                undo_triggered=False,
                final_passed_gate=True,
                initial_state_uri="file:///state_2.glb",
                final_state_uri="file:///good_2.glb",
            ),
        ]
        out = tmp_path / "dpo_mismatch.jsonl"
        log_dir = tmp_path / "dummy_logs"
        log_dir.mkdir()
        result = export_rl_flywheel_dataset(
            log_dir=log_dir,
            output_path=out,
            format="dpo",
            intent_groups=groups,
        )
        assert result["record_count"] == 0


# -------------------------------------------------------------------
# CLI tests
# -------------------------------------------------------------------


class TestInspectCLI:
    """Tests for synapse inspect CLI sub-commands."""

    def test_inspect_help(self) -> None:
        """synapse inspect --help does not crash."""
        from synapse.cli import main_synapse
        with pytest.raises(SystemExit) as exc:
            main_synapse(["inspect", "--help"])
        assert exc.value.code == 0

    def test_inspect_eval_json(self, tmp_path: Path) -> None:
        """synapse inspect eval --json produces JSON output."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "inspect", "eval",
                "--asset-id", "test_001",
                "--asset-name", "Test Asset",
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"

    def test_inspect_export_rl_data_json(
        self, intent_log_dir: Path, tmp_path: Path,
    ) -> None:
        """synapse inspect export-rl-data --json works."""
        import io
        import sys

        from synapse.cli import main_synapse

        out = tmp_path / "export.jsonl"
        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "inspect", "export-rl-data",
                "--log-dir", str(intent_log_dir),
                "--output", str(out),
                "--format", "dpo",
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert parsed["record_count"] == 2

    def test_inspect_align_not_implemented_exits_50(self) -> None:
        """synapse inspect align --json exits 50."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = main_synapse([
                "--json", "inspect", "align",
                "--mesh", "m.glb",
                "--image", "i.png",
            ])
        finally:
            sys.stdout = old

        assert code == 50

    def test_inspect_refine_missing_report_exits_20(self) -> None:
        """synapse inspect refine without valid --report exits 20."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = main_synapse([
                "--json", "inspect", "refine",
            ])
        finally:
            sys.stdout = old

        assert code == 20
