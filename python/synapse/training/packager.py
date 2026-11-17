"""LoRA bundle packager with Synapse-Bench gating.

Provides:
- ``package_and_gate_lora_bundle``: Export LoRA weights to safetensors,
  generate ``lora_manifest.json`` and ``evaluation_report.json``,
  then run benchmark quality gates.

Heavy dependencies (``safetensors``, ``torch``) are lazy-imported.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapse.adapters.lora.manifest import LoRAManifest
from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    ContractValidationError,
    QualityGateError,
)
from synapse.metrics.profiles import (
    RSI_PROMOTION,
    GateProfileName,
    get_profile_thresholds,
)

# -------------------------------------------------------------------
# Quality gate thresholds
# -------------------------------------------------------------------


class LoRAGateThresholds(BaseModel):
    """Hard quality gate thresholds for LoRA promotion.

    Defaults are sourced directly from ``synapse.metrics.profiles.RSI_PROMOTION``
    (SSOT aligned with ``DESIGN_REVIEW_AND_ALIGNMENT.md`` §3.4).

    Attributes:
        chamfer_mm: Maximum Chamfer distance in mm.
        normal_angular_error_deg: Max normal error in degrees.
        border_ring_ciede2000: Max border ring CIE dE2000.
        negative_regression_pct: Max regression on negative
            generalisation scenarios (percentage).
    """

    chamfer_mm: float = Field(
        default=float(RSI_PROMOTION["chamfer_mm"]),
        ge=0.0,
    )
    normal_angular_error_deg: float = Field(
        default=float(RSI_PROMOTION["normal_angular_error_deg"]),
        ge=0.0,
    )
    border_ring_ciede2000: float = Field(
        default=float(RSI_PROMOTION["border_ring_ciede2000"]),
        ge=0.0,
    )
    negative_regression_pct: float = Field(
        default=8.0,
        ge=0.0,
    )

    @classmethod
    def from_profile(
        cls,
        profile: GateProfileName = "rsi_promotion",
        negative_regression_pct: float = 8.0,
    ) -> LoRAGateThresholds:
        """Construct thresholds from a canonical ``synapse.metrics.profiles`` profile."""
        table = get_profile_thresholds(profile)
        return cls(
            chamfer_mm=table["chamfer_mm"],
            normal_angular_error_deg=table["normal_angular_error_deg"],
            border_ring_ciede2000=table["border_ring_ciede2000"],
            negative_regression_pct=negative_regression_pct,
        )


# -------------------------------------------------------------------
# Evaluation report
# -------------------------------------------------------------------


class LoRAEvaluationReport(BaseModel):
    """Evaluation report for a gated LoRA bundle.

    Attributes:
        passed: Whether all quality gates passed.
        waived: Whether benchmark evaluation was explicitly skipped.
        thresholds: Gate thresholds used.
        measured: Measured metric values.
        failures: List of gate violation messages.
        agent_fix_hint: Actionable fix suggestions.
    """

    passed: bool = True
    waived: bool = False
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
        skip_bench: If True, skip benchmark evaluation and mark
            report as waived.

    Returns:
        Dict with ``bundle_dir``, ``manifest_path``,
        ``evaluation_path``, ``passed``, and ``report``.

    Raises:
        AssetNotFoundError: If *weights_path* does not exist or is
            empty (exit code 20).
        QualityGateError: If any hard gate threshold is violated or
            required metrics are missing/non-finite (exit code 40).
    """
    _DEFAULT_TARGET_MODULES_BY_CATEGORY: dict[str, list[str]] = {
        "style": ["cross_attn.to_k", "cross_attn.to_v"],
        "identity": ["cross_attn.to_k", "cross_attn.to_v"],
        "control": ["spatial_attn.to_q"],
        "whitebox": ["spatial_attn.to_q"],
        "motion": ["temporal_attn.to_q"],
    }

    if thresholds is None:
        thresholds = LoRAGateThresholds()
    if target_modules is None:
        target_modules = _DEFAULT_TARGET_MODULES_BY_CATEGORY.get(
            category, ["cross_attn.to_k", "cross_attn.to_v"]
        )
    elif len(target_modules) == 0:
        raise ContractValidationError(
            "target_modules must not be empty when packaging a LoRA bundle",
            agent_fix_hint=(
                "Provide at least one target module name (e.g. ['cross_attn.to_k', 'cross_attn.to_v'])."
            ),
        )
    if golden_scenarios is None:
        golden_scenarios = []
    if negative_scenarios is None:
        negative_scenarios = []

    src = Path(weights_path)
    if not src.is_file():
        raise AssetNotFoundError(
            f"LoRA weights file not found: {src}",
            agent_fix_hint=(f"Ensure trained LoRA weights exist at '{src}' before packaging."),
        )
    if src.stat().st_size == 0:
        raise AssetNotFoundError(
            f"LoRA weights file is empty (0 bytes): {src}",
            agent_fix_hint=("Re-export non-empty LoRA safetensors checkpoint."),
        )

    # ── Validate .safetensors header and weight keys ────────────
    _validate_safetensors_weights(src, target_modules)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights to bundle
    adapter_dst = output_dir / "adapter_model.safetensors"
    shutil.copy2(src, adapter_dst)

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

    # Validate manifest schema and category whitelist
    try:
        LoRAManifest.model_validate(manifest_data)
    except Exception as e:
        raise ContractValidationError(
            f"LoRA manifest validation failed: {e}",
            agent_fix_hint=(
                "Ensure category, rank, alpha, and target_modules match LoRAManifest schema."
            ),
        ) from e

    manifest_path = output_dir / "lora_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, default=str),
        encoding="utf-8",
    )

    # Run quality gate evaluation
    if bench_metrics is None and not skip_bench:
        raise QualityGateError(
            "LoRA quality gate requires 'bench_metrics' when "
            "skip_bench=False, but none were provided.",
            agent_fix_hint=(
                "Provide measured benchmark metrics or pass skip_bench=True for un-gated packaging."
            ),
        )
    elif bench_metrics is None:
        bench_metrics = {}

    report = _evaluate_gates(
        bench_metrics,
        thresholds,
        negative_scenarios,
        skip_bench=skip_bench,
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
        "waived": report.waived,
        "report": report.model_dump(),
    }

    if not report.passed:
        raise QualityGateError(
            f"LoRA bundle failed quality gate: {'; '.join(report.failures)}",
            agent_fix_hint=report.agent_fix_hint,
            details=result,
        )

    return result


def _validate_safetensors_weights(
    weights_path: Path,
    target_modules: list[str],
) -> None:
    """Validate ``.safetensors`` header, weight keys, and non-emptiness fail-closed.

    Checks:
    1. The safetensors library is installed (BackendUnavailableError 30 if missing).
    2. The file is parseable as safetensors format (ContractValidationError 10 on corruption).
    3. At least one weight tensor is present (no empty checkpoints).
    4. Every weight key name references one of the ``target_modules`` patterns.

    Args:
        weights_path: Path to the ``.safetensors`` file.
        target_modules: Allowed module name patterns from ``LoRAManifest.target_modules``.

    Raises:
        BackendUnavailableError: If safetensors library is not installed (exit code 30).
        ContractValidationError: If file is corrupted, empty, or contains unauthorized keys (exit code 10).
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise BackendUnavailableError(
            "safetensors library is required to validate and package LoRA weights",
            agent_fix_hint="Install safetensors via 'pip install safetensors'.",
        ) from e

    try:
        with safe_open(str(weights_path), framework="numpy") as f:
            keys = list(f.keys())
    except Exception as e:
        raise ContractValidationError(
            f"Failed to parse safetensors weights file '{weights_path}': {e}",
            agent_fix_hint="Ensure the weights file is a valid, uncorrupted safetensors container.",
        ) from e

    if not keys:
        raise ContractValidationError(
            f"Safetensors file contains no weight tensors: {weights_path}",
            agent_fix_hint="Re-export checkpoint with at least one LoRA weight.",
        )

    # Whitelist enforcement: every key must reference a target_module
    if target_modules:
        unauthorized = []
        for key in keys:
            if not any(fnmatch.fnmatch(key, f"*{pat}*") or pat in key for pat in target_modules):
                unauthorized.append(key)
        if unauthorized:
            raise ContractValidationError(
                f"Unauthorized weight keys not matching target_modules whitelist: "
                f"{unauthorized[:5]}{'...' if len(unauthorized) > 5 else ''}",
                agent_fix_hint=(
                    "Ensure all weight keys reference target_modules: "
                    f"{target_modules}. Remove or rename unauthorized keys."
                ),
            )


def _evaluate_gates(
    metrics: dict[str, float],
    thresholds: LoRAGateThresholds,
    negative_scenarios: list[str],
    *,
    skip_bench: bool = False,
) -> LoRAEvaluationReport:
    """Check measured metrics against hard gate thresholds.

    Enforces fail-closed validation: any non-finite value (NaN/Inf) or
    empty metric set (when ``skip_bench=False``) fails the gate.
    """
    failures: list[str] = []
    measured: dict[str, float] = dict(metrics)
    threshold_dict = thresholds.model_dump()

    if not skip_bench and not metrics:
        failures.append("No benchmark metrics provided (fail-closed gate)")

    # Fail-closed: required gate metrics must be present when not skipping bench
    _REQUIRED_GATE_KEYS = ("chamfer_mm", "normal_angular_error_deg", "border_ring_ciede2000")
    if not skip_bench and metrics:
        for req_key in _REQUIRED_GATE_KEYS:
            if req_key not in metrics:
                failures.append(f"Missing required gate metric '{req_key}' (fail-closed)")

    for key, val in metrics.items():
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            failures.append(f"{key}={val!r} is non-finite (NaN/Inf)")

    # Check chamfer distance
    chamfer = metrics.get("chamfer_mm")
    if chamfer is not None and math.isfinite(chamfer) and chamfer > thresholds.chamfer_mm:
        failures.append(f"chamfer_mm={chamfer:.2f} > threshold={thresholds.chamfer_mm:.2f}")

    # Check normal angular error
    nae = metrics.get("normal_angular_error_deg")
    if nae is not None and math.isfinite(nae) and nae > thresholds.normal_angular_error_deg:
        failures.append(
            f"normal_angular_error_deg={nae:.2f} > "
            f"threshold="
            f"{thresholds.normal_angular_error_deg:.2f}"
        )

    # Check border ring CIE dE2000
    ciede = metrics.get("border_ring_ciede2000")
    if ciede is not None and math.isfinite(ciede) and ciede > thresholds.border_ring_ciede2000:
        failures.append(
            f"border_ring_ciede2000={ciede:.2f} > threshold={thresholds.border_ring_ciede2000:.2f}"
        )

    # Check negative scenario regression
    for scenario_id in negative_scenarios:
        key = f"regression_{scenario_id}_pct"
        reg = metrics.get(key)
        if reg is None:
            failures.append(f"Missing required negative scenario metric '{key}'")
        elif math.isfinite(reg) and reg > thresholds.negative_regression_pct:
            failures.append(
                f"{key}={reg:.2f}% > threshold={thresholds.negative_regression_pct:.2f}%"
            )

    passed = len(failures) == 0
    hint = ""
    if not passed:
        hint = (
            "Review training hyperparameters (rank, alpha, lr) "
            "and ensure training data quality. Specific failures: " + "; ".join(failures)
        )

    return LoRAEvaluationReport(
        passed=passed,
        waived=skip_bench,
        thresholds=threshold_dict,
        measured=measured,
        failures=failures,
        agent_fix_hint=hint,
    )
