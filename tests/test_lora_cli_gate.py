"""Tests for PR-19: LoRA packager + CLI + benchmark gate.

Validates:
- package_and_gate_lora_bundle creates bundle with manifest + report
- Quality gate passes for clean metrics
- Quality gate rejects for threshold violations (exit_code=40)
- CLI lora sub-commands produce correct JSON output
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from synapse.training.packager import (
    LoRAEvaluationReport,
    LoRAGateThresholds,
    package_and_gate_lora_bundle,
)
from synapse.errors import QualityGateError


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def tmp_bundle_dir(tmp_path: Path) -> Path:
    """Create a temp directory for LoRA bundle output."""
    return tmp_path / "lora_bundle"


@pytest.fixture
def fake_weights(tmp_path: Path) -> Path:
    """Create a fake safetensors file."""
    p = tmp_path / "weights.safetensors"
    p.write_bytes(b"\x00" * 128)
    return p


# -------------------------------------------------------------------
# LoRAGateThresholds tests
# -------------------------------------------------------------------


class TestLoRAGateThresholds:
    """Tests for LoRAGateThresholds model."""

    def test_defaults(self) -> None:
        t = LoRAGateThresholds()
        assert t.chamfer_mm == 4.0
        assert t.normal_angular_error_deg == 12.0
        assert t.border_ring_ciede2000 == 2.0
        assert t.negative_regression_pct == 8.0

    def test_custom_values(self) -> None:
        t = LoRAGateThresholds(
            chamfer_mm=3.0,
            normal_angular_error_deg=8.0,
        )
        assert t.chamfer_mm == 3.0
        assert t.normal_angular_error_deg == 8.0

    def test_roundtrip_json(self) -> None:
        t = LoRAGateThresholds(chamfer_mm=2.5)
        d = t.model_dump()
        t2 = LoRAGateThresholds(**d)
        assert t2.chamfer_mm == 2.5


# -------------------------------------------------------------------
# package_and_gate_lora_bundle tests
# -------------------------------------------------------------------


class TestPackager:
    """Tests for package_and_gate_lora_bundle function."""

    def test_creates_bundle_structure(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Bundle dir contains manifest + evaluation report."""
        result = package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            category="style",
            rank=8,
            alpha=1.0,
            skip_bench=True,
        )
        assert result["passed"] is True
        assert (tmp_bundle_dir / "lora_manifest.json").exists()
        assert (
            tmp_bundle_dir / "evaluation_report.json"
        ).exists()
        assert (
            tmp_bundle_dir / "adapter_model.safetensors"
        ).exists()

    def test_manifest_content(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Manifest JSON has correct fields."""
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            category="identity",
            rank=16,
            alpha=2.0,
            training_steps=500,
            final_loss=0.05,
            model_tag="gemini4",
            skip_bench=True,
        )
        manifest = json.loads(
            (tmp_bundle_dir / "lora_manifest.json").read_text(
                encoding="utf-8",
            )
        )
        assert manifest["schema_version"] == "synapse.lora.v1.0"
        assert manifest["category"] == "identity"
        assert manifest["rank"] == 16
        assert manifest["alpha"] == 2.0
        meta = manifest["training_metadata"]
        assert meta["steps"] == 500
        assert meta["final_loss"] == 0.05
        assert meta["model_tag"] == "gemini4"

    def test_sha256_computed(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Manifest contains correct SHA256 of adapter file."""
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            skip_bench=True,
        )
        manifest = json.loads(
            (tmp_bundle_dir / "lora_manifest.json").read_text(
                encoding="utf-8",
            )
        )
        sha = manifest["adapter_sha256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_clean_metrics_pass(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Bundle with metrics within thresholds passes gate."""
        result = package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            bench_metrics={
                "chamfer_mm": 3.0,
                "normal_angular_error_deg": 8.0,
                "border_ring_ciede2000": 1.5,
            },
        )
        assert result["passed"] is True

    def test_chamfer_exceeds_gate(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Chamfer distance exceeding threshold raises error."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={
                    "chamfer_mm": 10.0,
                },
            )
        assert exc_info.value.exit_code == 40
        assert "chamfer_mm" in str(exc_info.value)
        assert exc_info.value.agent_fix_hint != ""

    def test_normal_error_exceeds_gate(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Normal angular error exceeding threshold raises error."""
        with pytest.raises(QualityGateError):
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={
                    "normal_angular_error_deg": 15.0,
                },
            )

    def test_ciede2000_exceeds_gate(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Border CIE dE2000 exceeding threshold raises error."""
        with pytest.raises(QualityGateError):
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={
                    "border_ring_ciede2000": 5.0,
                },
            )

    def test_negative_regression_exceeds_gate(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Negative scenario regression exceeding threshold."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                negative_scenarios=["color_shift_test"],
                bench_metrics={
                    "regression_color_shift_test_pct": 12.0,
                },
            )
        assert "regression_color_shift_test_pct" in str(
            exc_info.value,
        )

    def test_custom_thresholds(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Custom thresholds are respected."""
        # Should pass with relaxed thresholds
        result = package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            bench_metrics={"chamfer_mm": 8.0},
            thresholds=LoRAGateThresholds(chamfer_mm=10.0),
        )
        assert result["passed"] is True

    def test_missing_weights_creates_placeholder(
        self, tmp_path: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Non-existent weights file creates placeholder."""
        fake = tmp_path / "nonexistent.safetensors"
        result = package_and_gate_lora_bundle(
            weights_path=fake,
            output_dir=tmp_bundle_dir,
            skip_bench=True,
        )
        assert result["passed"] is True
        assert (
            tmp_bundle_dir / "adapter_model.safetensors"
        ).exists()

    def test_evaluation_report_content(
        self, fake_weights: Path, tmp_bundle_dir: Path,
    ) -> None:
        """Evaluation report JSON has correct structure."""
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            bench_metrics={
                "chamfer_mm": 2.0,
            },
        )
        report = json.loads(
            (
                tmp_bundle_dir / "evaluation_report.json"
            ).read_text(encoding="utf-8")
        )
        assert report["passed"] is True
        assert "thresholds" in report
        assert "measured" in report
        assert report["failures"] == []


# -------------------------------------------------------------------
# CLI tests
# -------------------------------------------------------------------


class TestLoRACLI:
    """Tests for synapse lora CLI sub-commands."""

    def test_lora_help(self) -> None:
        """synapse lora --help does not crash."""
        from synapse.cli import main_synapse
        with pytest.raises(SystemExit) as exc:
            main_synapse(["lora", "--help"])
        assert exc.value.code == 0

    def test_lora_prepare_data_json(self) -> None:
        """synapse lora prepare-data --json produces JSON output."""
        import io
        import sys
        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "lora", "prepare-data",
                "--input", ".",
                "--output", ".",
            ])
        finally:
            sys.stdout = old

        output = buf.getvalue()
        assert code == 0
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert "input_dir" in parsed

    def test_lora_eval_pass_json(
        self, fake_weights: Path, tmp_path: Path,
    ) -> None:
        """synapse lora eval --json with passing bundle."""
        import io
        import sys
        from synapse.cli import main_synapse

        # Create a passing bundle first
        bundle_dir = tmp_path / "bundle"
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=bundle_dir,
            skip_bench=True,
        )

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "lora", "eval",
                "--bundle", str(bundle_dir),
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert parsed["passed"] is True

    def test_lora_eval_fail_json(
        self, fake_weights: Path, tmp_path: Path,
    ) -> None:
        """synapse lora eval --json with failing metrics."""
        import io
        import sys
        from synapse.cli import main_synapse

        # Create bundle dir with bad evaluation
        bundle_dir = tmp_path / "bad_bundle"
        bundle_dir.mkdir(parents=True)

        # Create a minimal bundle
        adapter = bundle_dir / "adapter_model.safetensors"
        adapter.write_bytes(b"\x00" * 64)

        manifest = {
            "schema_version": "synapse.lora.v1.0",
            "name": "test_lora",
            "category": "style",
            "rank": 8,
            "alpha": 1.0,
            "adapter_path": "adapter_model.safetensors",
            "adapter_sha256": "a" * 64,
            "target_modules": [],
        }
        (bundle_dir / "lora_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        # Write bad evaluation report
        report = {
            "passed": False,
            "thresholds": {"chamfer_mm": 5.0},
            "measured": {"chamfer_mm": 10.0},
            "failures": ["chamfer_mm=10.00 > threshold=5.00"],
            "agent_fix_hint": "Review training.",
        }
        (bundle_dir / "evaluation_report.json").write_text(
            json.dumps(report), encoding="utf-8",
        )

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "lora", "eval",
                "--bundle", str(bundle_dir),
            ])
        finally:
            sys.stdout = old

        assert code == 40
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["exit_code"] == 40
        assert "agent_fix_hint" in parsed

    def test_lora_train_json(self) -> None:
        """synapse lora train --json produces JSON output."""
        import io
        import sys
        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "lora", "train",
                "--config", "{}",
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert "config" in parsed

    def test_lora_merge_json(self) -> None:
        """synapse lora merge --json produces JSON output."""
        import io
        import sys
        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse([
                "--json", "lora", "merge",
                "--bundle", ".",
                "--output", ".",
            ])
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert "bundle_dir" in parsed
