# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Three-tier refinement quality gates with HITL 900 s circuit breaker.

Implements the three-tier gate evaluation pipeline from RFC 0061 §5:

  - **Tier 1 (Automatic)**: Per-pixel / per-patch metrics against thresholds.
  - **Tier 2 (Neural)**: Placeholder for VLM-based perceptual assessment.
  - **Tier 3 (HITL)**: Delegates to ``bridge.hitl.evaluate_hitl_circuit_breaker``
    for the 900 s budget enforcement.

Hard negatives that fail gating are exported to ``hard_negatives.jsonl``
for RSI flywheel ingestion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------


@dataclass
class GateThresholds:
    """Quality gate thresholds for local refinement.

    Attributes:
        min_psnr_db: Minimum PSNR at the stitch boundary (dB).
        max_ciede2000: Maximum CIE ΔE₀₀ at the border ring.
        max_temporal_warp_error: Maximum normalised temporal warp error.
        max_elapsed_seconds: HITL circuit breaker budget (seconds).
    """

    min_psnr_db: float = 38.0
    max_ciede2000: float = 2.0
    max_temporal_warp_error: float = 0.020
    max_elapsed_seconds: float = 900.0


# ---------------------------------------------------------------------------
# Gate verdict
# ---------------------------------------------------------------------------


@dataclass
class GateVerdict:
    """Result of evaluating the three-tier quality gates.

    Attributes:
        passed: ``True`` if all tiers passed.
        tier_results: Per-tier pass/fail status.
        metrics: Computed metric values.
        failure_reasons: List of human-readable failure descriptions.
    """

    passed: bool = True
    tier_results: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_refinement_gates(
    border_psnr_db: float,
    border_ciede2000: float,
    temporal_warp_error: float = 0.0,
    elapsed_seconds: float = 0.0,
    domain: Literal["video_shot", "asset_3d"] = "video_shot",
    thresholds: GateThresholds | None = None,
) -> GateVerdict:
    """Evaluate the three-tier refinement quality gates.

    Args:
        border_psnr_db: Measured PSNR at the 16 px stitch boundary.
        border_ciede2000: Measured CIE ΔE₀₀ at the border ring.
        temporal_warp_error: Normalised temporal warp error in ``[0, 1]``.
        elapsed_seconds: Cumulative wall-clock time for this refinement.
        domain: ``"video_shot"`` or ``"asset_3d"`` for HITL fallback.
        thresholds: Custom thresholds (defaults to production values).

    Returns:
        :class:`GateVerdict` with pass/fail status and metrics.

    Raises:
        CircuitBreakerError: If ``elapsed_seconds`` exceeds the HITL
            budget (exit_code=40).
    """
    th = thresholds or GateThresholds()

    verdict = GateVerdict(
        metrics={
            "border_psnr_db": border_psnr_db,
            "border_ciede2000": border_ciede2000,
            "temporal_warp_error": temporal_warp_error,
            "elapsed_seconds": elapsed_seconds,
        },
    )

    # -- Tier 1: Automatic metric gates --
    tier1_pass = True

    if border_psnr_db < th.min_psnr_db:
        tier1_pass = False
        verdict.failure_reasons.append(
            f"border_psnr_db {border_psnr_db:.1f} < {th.min_psnr_db:.1f}"
        )

    if border_ciede2000 > th.max_ciede2000:
        tier1_pass = False
        verdict.failure_reasons.append(
            f"border_ciede2000 {border_ciede2000:.3f} > {th.max_ciede2000:.3f}"
        )

    if temporal_warp_error > th.max_temporal_warp_error:
        tier1_pass = False
        verdict.failure_reasons.append(
            f"temporal_warp_error {temporal_warp_error:.4f} "
            f"> {th.max_temporal_warp_error:.4f}"
        )

    verdict.tier_results["tier1_automatic"] = tier1_pass

    # -- Tier 2: Neural perceptual (placeholder — always pass) --
    verdict.tier_results["tier2_neural"] = True

    # -- Tier 3: HITL circuit breaker --
    from synapse.bridge.hitl import evaluate_hitl_circuit_breaker

    # This raises CircuitBreakerError if elapsed_seconds > max
    evaluate_hitl_circuit_breaker(
        elapsed_seconds=elapsed_seconds,
        domain=domain,
        max_seconds=th.max_elapsed_seconds,
    )
    verdict.tier_results["tier3_hitl"] = True

    # Final verdict
    verdict.passed = all(verdict.tier_results.values())
    return verdict


# ---------------------------------------------------------------------------
# Hard-negative exporter
# ---------------------------------------------------------------------------


def export_hard_negatives(
    negatives: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Export hard-negative samples to a JSONL file for RSI ingestion.

    Each entry in *negatives* should contain at minimum:

    - ``"frame_index"``: int
    - ``"region"``: str (anatomical region name)
    - ``"failure_reasons"``: list[str]
    - ``"metrics"``: dict[str, float]

    Args:
        negatives: List of hard-negative dicts.
        output_path: Path to write ``hard_negatives.jsonl``.

    Returns:
        Resolved output path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for entry in negatives:
            f.write(json.dumps(entry, default=str) + "\n")

    return out
