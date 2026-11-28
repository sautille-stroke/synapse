"""Tests for PR-19: LoRA packager + CLI + benchmark gate.

Validates:
- package_and_gate_lora_bundle creates bundle with manifest + report
- Quality gate passes for clean metrics
- Quality gate rejects for threshold violations (exit_code=40)
- CLI lora sub-commands produce correct JSON output
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from synapse.errors import AssetNotFoundError, ContractValidationError, QualityGateError
from synapse.training.packager import (
    LoRAGateThresholds,
    package_and_gate_lora_bundle,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def tmp_bundle_dir(tmp_path: Path) -> Path:
    """Create a temp directory for LoRA bundle output."""
    return tmp_path / "lora_bundle"


@pytest.fixture
def fake_weights(tmp_path: Path) -> Path:
    """Create a real safetensors file with valid LoRA weights."""
    from safetensors.numpy import save_file

    p = tmp_path / "weights.safetensors"
    weights = {
        "model.cross_attn.to_k.lora_A.weight": np.zeros((4, 16), dtype=np.float32),
        "model.cross_attn.to_k.lora_B.weight": np.zeros((16, 4), dtype=np.float32),
        "model.cross_attn.to_v.lora_A.weight": np.zeros((4, 16), dtype=np.float32),
        "model.cross_attn.to_v.lora_B.weight": np.zeros((16, 4), dtype=np.float32),
    }
    save_file(weights, str(p))
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
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
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
        assert (tmp_bundle_dir / "evaluation_report.json").exists()
        assert (tmp_bundle_dir / "adapter_model.safetensors").exists()

    def test_manifest_content(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
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
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
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
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
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
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Chamfer distance exceeding threshold raises error."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={
                    "chamfer_mm": 10.0,
                    "normal_angular_error_deg": 8.0,
                    "border_ring_ciede2000": 1.5,
                },
            )
        assert exc_info.value.exit_code == 40
        assert "chamfer_mm" in str(exc_info.value)
        assert exc_info.value.agent_fix_hint != ""

    def test_normal_error_exceeds_gate(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Normal angular error exceeding threshold raises error."""
        with pytest.raises(QualityGateError):
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={
                    "chamfer_mm": 2.0,
                    "normal_angular_error_deg": 15.0,
                    "border_ring_ciede2000": 1.5,
                },
            )

    def test_ciede2000_exceeds_gate(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Border CIE dE2000 exceeding threshold raises error."""
        with pytest.raises(QualityGateError):
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={
                    "chamfer_mm": 2.0,
                    "normal_angular_error_deg": 8.0,
                    "border_ring_ciede2000": 5.0,
                },
            )

    def test_negative_regression_exceeds_gate(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Negative scenario regression exceeding threshold."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                negative_scenarios=["color_shift_test"],
                bench_metrics={
                    "chamfer_mm": 2.0,
                    "normal_angular_error_deg": 8.0,
                    "border_ring_ciede2000": 1.5,
                    "regression_color_shift_test_pct": 12.0,
                },
            )
        assert "regression_color_shift_test_pct" in str(
            exc_info.value,
        )

    def test_custom_thresholds(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Custom thresholds are respected."""
        # Should pass with relaxed thresholds
        result = package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            bench_metrics={
                "chamfer_mm": 8.0,
                "normal_angular_error_deg": 8.0,
                "border_ring_ciede2000": 1.5,
            },
            thresholds=LoRAGateThresholds(chamfer_mm=10.0),
        )
        assert result["passed"] is True

    def test_missing_weights_raises_asset_not_found(
        self,
        tmp_path: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Non-existent weights file raises AssetNotFoundError(20)."""
        fake = tmp_path / "nonexistent.safetensors"
        with pytest.raises(AssetNotFoundError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake,
                output_dir=tmp_bundle_dir,
                skip_bench=True,
            )
        assert int(exc_info.value.exit_code) == 20
        assert not (tmp_bundle_dir / "adapter_model.safetensors").exists()

    def test_empty_metrics_without_skip_bench_fails_gate(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Missing or empty bench_metrics fails gate when skip_bench=False."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                skip_bench=False,
            )
        assert int(exc_info.value.exit_code) == 40

        with pytest.raises(QualityGateError):
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={},
                skip_bench=False,
            )

    def test_nan_metric_fails_gate(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """NaN metric is caught by fail-closed gate."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                bench_metrics={"chamfer_mm": float("nan")},
            )
        assert int(exc_info.value.exit_code) == 40
        assert "non-finite" in str(exc_info.value)

    def test_missing_negative_scenario_metric_fails_gate(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Missing required negative scenario metric fails gate."""
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=tmp_bundle_dir,
                negative_scenarios=["color_shift_test"],
                bench_metrics={
                    "chamfer_mm": 1.0,
                    "normal_angular_error_deg": 8.0,
                    "border_ring_ciede2000": 1.5,
                },
            )
        assert "Missing required negative scenario" in str(
            exc_info.value,
        )

    def test_evaluation_report_content(
        self,
        fake_weights: Path,
        tmp_bundle_dir: Path,
    ) -> None:
        """Evaluation report JSON has correct structure."""
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=tmp_bundle_dir,
            bench_metrics={
                "chamfer_mm": 2.0,
                "normal_angular_error_deg": 8.0,
                "border_ring_ciede2000": 1.5,
            },
        )
        report = json.loads((tmp_bundle_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        assert report["passed"] is True
        assert "thresholds" in report
        assert "measured" in report
        assert report["failures"] == []


# -------------------------------------------------------------------
# CLI tests
# -------------------------------------------------------------------


@pytest.mark.slow
class TestLoRACLI:
    """Tests for synapse lora CLI sub-commands."""

    def test_lora_help(self) -> None:
        """synapse lora --help does not crash."""
        from synapse.cli import main_synapse

        with pytest.raises(SystemExit) as exc:
            main_synapse(["lora", "--help"])
        assert exc.value.code == 0

    def test_lora_prepare_data_json(self) -> None:
        """synapse lora prepare-data --json fails honestly with exit code 50 until wired."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "prepare-data",
                    "--input",
                    ".",
                    "--output",
                    ".",
                ]
            )
        finally:
            sys.stdout = old

        output = buf.getvalue()
        assert code == 20  # AssetNotFoundError: no .npy files in '.'
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert parsed["exit_code"] == 20
        assert "agent_fix_hint" in parsed

    def test_lora_eval_pass_json(
        self,
        fake_weights: Path,
        tmp_path: Path,
    ) -> None:
        """synapse lora eval --json with passing bundle."""
        import io
        import sys

        from synapse.cli import main_synapse

        # Create a passing bundle first (with real bench_metrics, not waived)
        bundle_dir = tmp_path / "bundle"
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=bundle_dir,
            bench_metrics={
                "chamfer_mm": 2.0,
                "normal_angular_error_deg": 8.0,
                "border_ring_ciede2000": 1.5,
            },
            skip_bench=False,
        )

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "eval",
                    "--bundle",
                    str(bundle_dir),
                ]
            )
        finally:
            sys.stdout = old

        assert code == 0
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "ok"
        assert parsed["passed"] is True

    def test_lora_eval_waived_rejected_json(
        self,
        fake_weights: Path,
        tmp_path: Path,
    ) -> None:
        """synapse lora eval rejects waived bundle (skip_bench=True)."""
        import io
        import sys

        from synapse.cli import main_synapse

        bundle_dir = tmp_path / "waived_bundle"
        package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=bundle_dir,
            skip_bench=True,
        )

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "eval",
                    "--bundle",
                    str(bundle_dir),
                ]
            )
        finally:
            sys.stdout = old

        assert code == 40
        parsed = json.loads(buf.getvalue())
        assert parsed["exit_code"] == 40
        assert "waived" in parsed["message"]

    def test_lora_eval_fail_json(
        self,
        fake_weights: Path,
        tmp_path: Path,
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
            json.dumps(manifest),
            encoding="utf-8",
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
            json.dumps(report),
            encoding="utf-8",
        )

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "eval",
                    "--bundle",
                    str(bundle_dir),
                ]
            )
        finally:
            sys.stdout = old

        assert code == 40
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["exit_code"] == 40
        assert "agent_fix_hint" in parsed

    def test_lora_train_json(self, tmp_path: Path) -> None:
        """PR-187: default lora train fails closed (exit 50); --mock-backbone runs & trains."""
        import io
        import sys

        from synapse.cli import main_synapse

        # 1. Production fail-closed: without --base-model or --mock-backbone -> exit 50
        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "train",
                    "--config",
                    '{"category": "style"}',
                ]
            )
        finally:
            sys.stdout = old

        assert code == 50
        parsed_err = json.loads(buf.getvalue())
        assert parsed_err["exit_code"] == 50

        # 2. Explicit --mock-backbone succeeds with real non-zero updates (including motion!)
        out_dir = tmp_path / "motion_weights"
        old = sys.stdout
        sys.stdout = buf2 = io.StringIO()
        try:
            code2 = main_synapse(
                [
                    "--json",
                    "lora",
                    "train",
                    "--mock-backbone",
                    "--output",
                    str(out_dir),
                    "--config",
                    '{"category": "motion", "max_steps": 3, "rank": 4}',
                ]
            )
        finally:
            sys.stdout = old

        assert code2 == 0
        parsed = json.loads(buf2.getvalue())
        assert parsed["status"] == "ok"
        assert parsed["steps_completed"] == 3
        assert parsed["backbone_kind"] == "mock"
        assert parsed["mock_backbone"] is True

        # 3. Reject passing --base-model and --mock-backbone simultaneously (exit 10)
        old = sys.stdout
        sys.stdout = buf3 = io.StringIO()
        try:
            code3 = main_synapse(
                [
                    "--json",
                    "lora",
                    "train",
                    "--base-model",
                    str(tmp_path),
                    "--mock-backbone",
                    "--config",
                    '{"category": "style"}',
                ]
            )
        finally:
            sys.stdout = old
        assert code3 == 10
        parsed3 = json.loads(buf3.getvalue())
        assert parsed3["exit_code"] == 10
        from safetensors.numpy import load_file

        weights = load_file(str(out_dir / "adapter_model.safetensors"))
        lora_b_keys = [k for k in weights if "lora_B" in k]
        assert lora_b_keys
        assert any(float(np.abs(weights[k]).max()) > 0.0 for k in lora_b_keys), (
            "motion LoRA temporal_attn must receive real non-zero gradients"
        )

    def test_lora_train_invalid_json_exits_10(self) -> None:
        """synapse lora train --json with malformed JSON exits with code 10."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "train",
                    "--config",
                    "{not-valid-json",
                ]
            )
        finally:
            sys.stdout = old

        assert code == 10
        parsed = json.loads(buf.getvalue())
        assert parsed["exit_code"] == 10

    def test_lora_train_missing_category_exits_10(self) -> None:
        """synapse lora train --json without explicit category exits with code 10."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "train",
                    "--config",
                    "{}",
                ]
            )
        finally:
            sys.stdout = old

        assert code == 10
        parsed = json.loads(buf.getvalue())
        assert parsed["exit_code"] == 10

    def test_lora_merge_json(self) -> None:
        """synapse lora merge --json fails honestly with exit code 50."""
        import io
        import sys

        from synapse.cli import main_synapse

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "merge",
                    "--bundle",
                    ".",
                    "--output",
                    ".",
                ]
            )
        finally:
            sys.stdout = old

        assert code == 20  # AssetNotFoundError: no adapter_model.safetensors in '.'
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["status"] == "error"
        assert parsed["exit_code"] == 20

    def test_no_fake_stub_statuses_in_cli_or_trainer(self) -> None:
        """Ensure fake success stub strings are purged from cli and trainer."""
        repo_root = Path(__file__).resolve().parent.parent
        cli_dir = repo_root / "python" / "synapse" / "cli"
        trainer_dir = repo_root / "python" / "synapse" / "training"
        forbidden_strings = (
            "configured_for_execution",
            "align_configured",
            '"status": "configured"',
        )
        violations: list[str] = []
        for search_dir in (cli_dir, trainer_dir):
            for py_file in sorted(search_dir.rglob("*.py")):
                src = py_file.read_text(encoding="utf-8")
                for forbidden in forbidden_strings:
                    if forbidden in src:
                        violations.append(f"{py_file.name}: contains '{forbidden}'")
        assert violations == [], f"Fake stub statuses found: {violations}"


# -------------------------------------------------------------------
# Packager safetensors validation tests (PR-77)
# -------------------------------------------------------------------


class TestPackagerSafetensorsValidation:
    """PR-77: Validate .safetensors header and weight key whitelist."""

    def test_valid_safetensors_passes(
        self,
        tmp_path: Path,
    ) -> None:
        """Valid safetensors with matching target_modules passes."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        weights = {
            "ffn.0.lora_A.weight": np.zeros((8, 64), dtype=np.float32),
            "ffn.0.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
            "ffn.2.lora_A.weight": np.zeros((8, 64), dtype=np.float32),
            "ffn.2.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
        }
        p = tmp_path / "valid.safetensors"
        save_file(weights, str(p))

        bundle = tmp_path / "bundle"
        result = package_and_gate_lora_bundle(
            weights_path=p,
            output_dir=bundle,
            target_modules=["ffn.0", "ffn.2"],
            skip_bench=True,
        )
        assert result["passed"] is True

    def test_unauthorized_weight_key_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """Weight key not matching target_modules is rejected."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        weights = {
            "ffn.0.lora_A.weight": np.zeros((8, 64), dtype=np.float32),
            "UNAUTHORIZED.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
        }
        p = tmp_path / "bad_keys.safetensors"
        save_file(weights, str(p))

        bundle = tmp_path / "bundle"
        with pytest.raises(ContractValidationError, match="Unauthorized"):
            package_and_gate_lora_bundle(
                weights_path=p,
                output_dir=bundle,
                target_modules=["ffn.0", "ffn.2"],
                skip_bench=True,
            )

    def test_empty_safetensors_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """Safetensors file with no tensors is rejected."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        p = tmp_path / "empty.safetensors"
        save_file({}, str(p))

        bundle = tmp_path / "bundle"
        with pytest.raises(ContractValidationError, match="no weight"):
            package_and_gate_lora_bundle(
                weights_path=p,
                output_dir=bundle,
                skip_bench=True,
            )


# -------------------------------------------------------------------
# PR-117: Required gate metrics and fnmatch validation
# -------------------------------------------------------------------


class TestPR117GateKeyEnforcement:
    """PR-117: Fail-closed enforcement of required gate metrics."""

    def test_missing_required_key_fails_gate(
        self,
        fake_weights: Path,
        tmp_path: Path,
    ) -> None:
        """Missing chamfer_mm, normal_angular_error_deg, or border_ring_ciede2000 fails gate."""
        bundle = tmp_path / "missing_key_bundle"
        # Provide only chamfer_mm — missing the other two required keys
        with pytest.raises(QualityGateError) as exc_info:
            package_and_gate_lora_bundle(
                weights_path=fake_weights,
                output_dir=bundle,
                bench_metrics={
                    "chamfer_mm": 2.0,
                },
                skip_bench=False,
            )
        assert exc_info.value.exit_code == 40
        assert "Missing required gate metric" in str(exc_info.value)

    def test_all_required_keys_present_passes(
        self,
        fake_weights: Path,
        tmp_path: Path,
    ) -> None:
        """When all 3 required gate metrics are present and within thresholds, gate passes."""
        bundle = tmp_path / "complete_bundle"
        result = package_and_gate_lora_bundle(
            weights_path=fake_weights,
            output_dir=bundle,
            bench_metrics={
                "chamfer_mm": 2.0,
                "normal_angular_error_deg": 8.0,
                "border_ring_ciede2000": 1.5,
            },
            skip_bench=False,
        )
        assert result["passed"] is True

    def test_fnmatch_safetensors_validation(
        self,
        tmp_path: Path,
    ) -> None:
        """PR-117: fnmatch glob matching works in safetensors weight validation."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        weights = {
            "model.cross_attn.to_k.lora_A.weight": np.zeros((8, 64), dtype=np.float32),
            "model.cross_attn.to_k.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
            "model.cross_attn.to_v.lora_A.weight": np.zeros((8, 64), dtype=np.float32),
            "model.cross_attn.to_v.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
        }
        p = tmp_path / "fnmatch.safetensors"
        save_file(weights, str(p))

        bundle = tmp_path / "fnmatch_bundle"
        # Using 'cross_attn' as pattern — with fnmatch it should match weight keys
        result = package_and_gate_lora_bundle(
            weights_path=p,
            output_dir=bundle,
            target_modules=["cross_attn.to_k", "cross_attn.to_v"],
            skip_bench=True,
        )
        assert result["passed"] is True

    def test_pr187_lora_merge_two_bundles_orthogonal(self, tmp_path: Path) -> None:
        """PR-187: synapse lora merge executes real MultiLoRAManager orthogonal merge."""
        import io
        import sys

        pytest.importorskip("safetensors")
        from safetensors.numpy import load_file, save_file
        from synapse.cli import main_synapse

        rng = np.random.default_rng(187)
        b1_dir = tmp_path / "b1"
        b2_dir = tmp_path / "b2"
        out_dir = tmp_path / "merged"

        w1_path = tmp_path / "w1.safetensors"
        w2_path = tmp_path / "w2.safetensors"
        a1 = rng.standard_normal((4, 16)).astype(np.float32) * 0.2
        b1 = rng.standard_normal((16, 4)).astype(np.float32) * 0.2
        # Make bundle 2 partially aligned with bundle 1 to trigger orthogonalization
        a2 = a1 * 0.9 + rng.standard_normal((4, 16)).astype(np.float32) * 0.05
        b2 = b1 * 0.9 + rng.standard_normal((16, 4)).astype(np.float32) * 0.05

        save_file({"blocks.0.cross_attn.to_k.lora_A.weight": a1, "blocks.0.cross_attn.to_k.lora_B.weight": b1}, str(w1_path))
        save_file({"blocks.0.cross_attn.to_k.lora_A.weight": a2, "blocks.0.cross_attn.to_k.lora_B.weight": b2}, str(w2_path))

        package_and_gate_lora_bundle(
            weights_path=w1_path,
            output_dir=b1_dir,
            category="style",
            rank=4,
            alpha=4.0,
            target_modules=["blocks.0.cross_attn.to_k"],
            skip_bench=True,
        )
        package_and_gate_lora_bundle(
            weights_path=w2_path,
            output_dir=b2_dir,
            category="style",
            rank=4,
            alpha=4.0,
            target_modules=["blocks.0.cross_attn.to_k"],
            skip_bench=True,
        )

        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            code = main_synapse(
                [
                    "--json",
                    "lora",
                    "merge",
                    "--bundle",
                    str(b1_dir),
                    str(b2_dir),
                    "--output",
                    str(out_dir),
                ]
            )
        finally:
            sys.stdout = old

        assert code == 0
        parsed = json.loads(buf.getvalue())
        assert parsed["bundles_merged"] == 2
        assert parsed["orthogonalized_pairs"] == 1
        merged = load_file(str(out_dir / "adapter_model.safetensors"))
        assert "blocks.0.cross_attn.to_k.lora_A.weight" in merged
        assert float(np.abs(merged["blocks.0.cross_attn.to_k.lora_B.weight"]).max()) > 0.0

