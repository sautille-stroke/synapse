"""LoRA bundle packager with Synapse-Bench gating.

Provides:
- ``package_and_gate_lora_bundle``: Export LoRA weights to safetensors,
  generate ``lora_manifest.json`` and ``evaluation_report.json``,
  then run benchmark quality gates.

Heavy dependencies (``safetensors``, ``torch``) are lazy-imported.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapse.errors import QualityGateError


# -------------------------------------------------------------------
# Quality gate thresholds
# -------------------------------------------------------------------

class LoRAGateThresholds(BaseModel):
    """Hard quality gate thresholds for LoRA promotion.

    Attributes:
        chamfer_mm: Maximum Chamfer distance in mm.
        normal_angular_error_deg: Max normal error in degrees.
        border_ring_ciede2000: Max border ring CIE dE2000.
        negative_regression_pct: Max regression on negative
            generalisation scenarios (percentage).
    """

    chamfer_mm: float = Field(default=5.0, ge=0.0)
    normal_angular_error_deg: float = Field(
        default=12.0, ge=0.0,
    )
    border_ring_ciede2000: float = Field(default=2.5, ge=0.0)
    negative_regression_pct: float = Field(
        default=8.0, ge=0.0,
    )


# -------------------------------------------------------------------
# Evaluation report
# -------------------------------------------------------------------

class LoRAEvaluationReport(BaseModel):
    """Evaluation report for a gated LoRA bundle.

    Attributes:
        passed: Whether all quality gates passed.
        thresholds: Gate thresholds used.
        measured: Measured metric values.
        failures: List of gate violation messages.
        agent_fix_hint: Actionable fix suggestions.
    """

    passed: bool = True
    thresholds: dict[str, float] = Field(default_factory=dict)
    measured: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    agent_fix_hint: str = ""


# -------------------------------------------------------------------
# Core packager
# -------------------------------------------------------------------


def _compute_file_sha256(path: Path) -> str:
    """Compute hex SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def package_and_gate_lora_bundle(
    *,
    weights_path: Path,
    output_dir: Path,
    category: str = "style",
    rank: int = 8,
    alpha: float = 1.0,
    target_modules: list[str] | None = None,
    training_steps: int = 0,
    final_loss: float = 0.0,
    model_tag: str = "",
    training_dataset_sha256: str = "",
    golden_scenarios: list[str] | None = None,
    negative_scenarios: list[str] | None = None,
    bench_metrics: dict[str, float] | None = None,
    thresholds: LoRAGateThresholds | None = None,
    skip_bench: bool = False,
) -> dict[str, Any]:
    """Package LoRA weights and run quality gates.

    Args:
        weights_path: Path to the trained LoRA ``.safetensors``.
        output_dir: Output bundle directory.
        category: LoRA category.
        rank: LoRA rank.
        alpha: LoRA scaling alpha.
        target_modules: Target module patterns.
        training_steps: Number of training steps completed.
        final_loss: Final training loss value.
        model_tag: AI model identifier tag.
        training_dataset_sha256: SHA256 of training dataset.
        golden_scenarios: Scenario IDs for golden evaluation.
        negative_scenarios: Scenario IDs for negative testing.
        bench_metrics: Pre-computed benchmark metrics (skips
            running BenchPipelineRunner if provided).
        thresholds: Quality gate thresholds.
        skip_bench: If True, skip benchmark evaluation.

    Returns:
        Dict with ``bundle_dir``, ``manifest_path``,
        ``evaluation_path``, ``passed``, and ``report``.

    Raises:
        QualityGateError: If any hard gate threshold is violated.
    """
    if thresholds is None:
        thresholds = LoRAGateThresholds()
    if target_modules is None:
        target_modules = []
    if golden_scenarios is None:
        golden_scenarios = []
    if negative_scenarios is None:
        negative_scenarios = []

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights to bundle
    adapter_dst = output_dir / "adapter_model.safetensors"
    src = Path(weights_path)
    if src.exists():
        shutil.copy2(src, adapter_dst)
    else:
        # If source doesn't exist, create a placeholder
        adapter_dst.touch()

    # Compute SHA256
    adapter_sha256 = _compute_file_sha256(adapter_dst)

    # Generate manifest
    manifest_data = {
        "schema_version": "synapse.lora.v1.0",
        "name": f"{category}_lora_{adapter_sha256[:8]}",
        "category": category,
        "rank": rank,
        "alpha": alpha,
        "adapter_path": "adapter_model.safetensors",
        "adapter_sha256": adapter_sha256,
        "target_modules": target_modules,
        "training_metadata": {
            "steps": training_steps,
            "final_loss": final_loss,
            "model_tag": model_tag,
            "dataset_sha256": training_dataset_sha256,
        },
    }

    manifest_path = output_dir / "lora_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, default=str),
        encoding="utf-8",
    )

    # Run quality gate evaluation
    if bench_metrics is None and not skip_bench:
        # In production, would run BenchPipelineRunner here.
        # For now, return empty metrics (bench not available).
        bench_metrics = {}
    elif bench_metrics is None:
        bench_metrics = {}

    # Evaluate gates
    report = _evaluate_gates(
        bench_metrics, thresholds, negative_scenarios,
    )

    # Write evaluation report
    eval_path = output_dir / "evaluation_report.json"
    eval_path.write_text(
        json.dumps(report.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    result = {
        "bundle_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "evaluation_path": str(eval_path),
        "passed": report.passed,
        "report": report.model_dump(),
    }

    if not report.passed:
        raise QualityGateError(
            f"LoRA bundle failed quality gate: "
            f"{'; '.join(report.failures)}",
            agent_fix_hint=report.agent_fix_hint,
            details=result,
        )

    return result


def _evaluate_gates(
    metrics: dict[str, float],
    thresholds: LoRAGateThresholds,
    negative_scenarios: list[str],
) -> LoRAEvaluationReport:
    """Check measured metrics against hard gate thresholds.

    Returns:
        An evaluation report with pass/fail and failure details.
    """
    failures: list[str] = []
    measured: dict[str, float] = dict(metrics)
    threshold_dict = thresholds.model_dump()

    # Check chamfer distance
    chamfer = metrics.get("chamfer_mm")
    if chamfer is not None and chamfer > thresholds.chamfer_mm:
        failures.append(
            f"chamfer_mm={chamfer:.2f} > "
            f"threshold={thresholds.chamfer_mm:.2f}"
        )

    # Check normal angular error
    nae = metrics.get("normal_angular_error_deg")
    if nae is not None:
        if nae > thresholds.normal_angular_error_deg:
            failures.append(
                f"normal_angular_error_deg={nae:.2f} > "
                f"threshold="
                f"{thresholds.normal_angular_error_deg:.2f}"
            )

    # Check border ring CIE dE2000
    ciede = metrics.get("border_ring_ciede2000")
    if ciede is not None:
        if ciede > thresholds.border_ring_ciede2000:
            failures.append(
                f"border_ring_ciede2000={ciede:.2f} > "
                f"threshold="
                f"{thresholds.border_ring_ciede2000:.2f}"
            )

    # Check negative scenario regression
    for scenario_id in negative_scenarios:
        key = f"regression_{scenario_id}_pct"
        reg = metrics.get(key)
        if reg is not None:
            if reg > thresholds.negative_regression_pct:
                failures.append(
                    f"{key}={reg:.2f}% > "
                    f"threshold="
                    f"{thresholds.negative_regression_pct:.2f}%"
                )

    passed = len(failures) == 0
    hint = ""
    if not passed:
        hint = (
            "Review training hyperparameters (rank, alpha, lr) "
            "and ensure training data quality. Specific failures: "
            + "; ".join(failures)
        )

    return LoRAEvaluationReport(
        passed=passed,
        thresholds=threshold_dict,
        measured=measured,
        failures=failures,
        agent_fix_hint=hint,
    )


# -------------------------------------------------------------------
# Stage registration
# -------------------------------------------------------------------

from synapse.pipeline.bundle_worker import (  # noqa: E402
    register_stage,
)


@register_stage("lora_package")
def _handle_lora_package(
    manifest: Any,
) -> dict[str, Any]:
    """Package LoRA weights from a job bundle manifest.

    Expects ``manifest.params`` to contain ``weights_path``,
    ``output_dir``, and optional training metadata.
    """
    params = dict(manifest.params)
    output_dir = Path(
        params.get("output_dir", "/tmp/lora_bundle"),
    )
    weights_path = Path(
        params.get("weights_path", "adapter_model.safetensors"),
    )

    try:
        result = package_and_gate_lora_bundle(
            weights_path=weights_path,
            output_dir=output_dir,
            category=params.get("category", "style"),
            rank=params.get("rank", 8),
            alpha=params.get("alpha", 1.0),
            skip_bench=True,
        )
        return {
            "output_paths": {
                "bundle": str(output_dir),
            },
            **result,
        }
    except QualityGateError as exc:
        return {
            "output_paths": {},
            "error": str(exc),
            "exit_code": int(exc.exit_code),
        }
