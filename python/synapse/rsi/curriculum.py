"""RSI curriculum sampler for adaptive difficulty scheduling.

Provides:
- ``CurriculumSampler``: Adaptive difficulty distribution based
  on Synapse-Bench weakness scores, with factor-level upsampling

The sampler analyses the previous round's benchmark report to
identify weak dimensions across all 11 canonical Synapse-Bench
condition factors (imported from ``synapse.bench.conditions``)
and increases sampling probability for harder condition levels
(L3/L4) on those factors.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapse.render_core.conditions import RENDERING_CONDITION_FACTORS

# -------------------------------------------------------------------
# Factor definitions (SSOT from synapse.bench.conditions)
# -------------------------------------------------------------------

_CONDITION_FACTORS: list[str] = list(RENDERING_CONDITION_FACTORS.keys())

_DIFFICULTY_LEVELS: list[str] = ["L1", "L2", "L3", "L4"]


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------


class CurriculumConfig(BaseModel):
    """Configuration for curriculum sampling.

    Attributes:
        weakness_threshold: Factor score below this triggers
            upsampling (default 0.7 = 70%).
        boost_factor: Multiplier for weak factor sampling weight.
        num_scenarios: Number of challenge scenarios to generate.
        seed: Random seed for reproducibility.
        max_level: Maximum difficulty level index (0-3).
    """

    weakness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    boost_factor: float = Field(default=2.0, ge=1.0)
    num_scenarios: int = Field(default=10, ge=1)
    seed: int = 42
    max_level: int = Field(default=3, ge=0, le=3)


# -------------------------------------------------------------------
# Challenge scenario output
# -------------------------------------------------------------------


class ChallengeScenario(BaseModel):
    """A generated challenge scenario for RSI training.

    Attributes:
        scenario_id: Unique ID for this challenge.
        factor: The condition factor being tested.
        level: Difficulty level (L1-L4).
        weight: Sampling weight that selected this scenario.
        description: Human-readable description.
    """

    scenario_id: str
    factor: str
    level: str
    weight: float = 1.0
    description: str = ""

    def to_bench_scenario(self) -> Any:
        """Convert this ChallengeScenario into a Synapse-Bench Scenario instance."""
        from synapse.bench.scenario import Scenario, ScenarioTier

        return Scenario(
            id=self.scenario_id,
            tier=ScenarioTier.T1_RIGID,
            description=self.description or f"RSI challenge for {self.factor} at {self.level}",
            tags=["rsi_challenge", self.factor, self.level],
            conditions={self.factor: self.level},
        )


# -------------------------------------------------------------------
# CurriculumSampler
# -------------------------------------------------------------------


class CurriculumSampler:
    """Adaptive curriculum sampler for RSI self-play.

    Analyses previous Synapse-Bench scores to identify weak
    dimensions and upsamples harder conditions on those factors.

    Args:
        config: Curriculum sampling configuration.
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
    ) -> None:
        self._config = config or CurriculumConfig()

    @property
    def config(self) -> CurriculumConfig:
        """Return the curriculum config."""
        return self._config

    def analyse_weaknesses(
        self,
        bench_scores: dict[str, float],
    ) -> dict[str, float]:
        """Identify weak factors from benchmark scores.

        Uses a continuous, bounded weight formula::

            weight = 1.0 + (boost - 1.0) * ((threshold - score) / max(threshold, 0.01))

        so that ``score == threshold`` gives weight ``1.0`` (no cliff jump),
        and ``score == 0.0`` gives weight ``boost``. Factors not present in
        *bench_scores* default to neutral weight ``1.0`` unless *bench_scores*
        is completely empty.

        Args:
            bench_scores: Mapping of factor name to aggregate
                score in [0, 1].

        Returns:
            Mapping of factor name to sampling weight.
        """
        weights: dict[str, float] = {}
        threshold = self._config.weakness_threshold
        boost = self._config.boost_factor

        all_factors = list(_CONDITION_FACTORS)
        for extra_k in bench_scores:
            if extra_k not in all_factors:
                all_factors.append(extra_k)

        for factor in all_factors:
            if factor not in bench_scores:
                weights[factor] = 1.0
                continue
            raw_score = float(bench_scores[factor])
            if not math.isfinite(raw_score):
                # NaN / Inf: treat as maximum weakness / failure (score = 0.0)
                score = 0.0
            else:
                score = max(0.0, min(1.0, raw_score))
            if score < threshold:
                deficit = (threshold - score) / max(threshold, 0.01)
                weights[factor] = 1.0 + (boost - 1.0) * deficit
            else:
                weights[factor] = 1.0

        return weights

    def sample_scenarios(
        self,
        bench_scores: dict[str, float],
        round_idx: int = 0,
    ) -> list[ChallengeScenario]:
        """Generate a set of challenge scenarios.

        Samples from the weighted factor distribution, preferring
        higher difficulty levels for weaker factors. Mixes in *round_idx*
        to prevent duplicate challenges across successive training rounds.

        Args:
            bench_scores: Previous round's per-factor scores.
            round_idx: Current training round index (0-based).

        Returns:
            List of ``ChallengeScenario`` objects.
        """
        rng = random.Random(self._config.seed + round_idx * 10007)
        weights = self.analyse_weaknesses(bench_scores)

        # Build weighted pool of (factor, level) pairs
        pool: list[tuple[str, str, float]] = []
        for factor, w in weights.items():
            raw_val = float(bench_scores[factor]) if factor in bench_scores else 1.0
            is_weak = factor in bench_scores and (
                not math.isfinite(raw_val) or raw_val < self._config.weakness_threshold
            )
            for i, level in enumerate(_DIFFICULTY_LEVELS):
                if i > self._config.max_level:
                    break
                if is_weak:
                    level_w = w * (1.0 + i * 0.5)
                else:
                    level_w = w
                pool.append((factor, level, level_w))

        if not pool:
            return []

        scenarios: list[ChallengeScenario] = []
        total_weight = sum(p[2] for p in pool)

        for idx in range(self._config.num_scenarios):
            r = rng.random() * total_weight
            cumulative = 0.0
            selected = pool[0]
            for item in pool:
                cumulative += item[2]
                if cumulative >= r:
                    selected = item
                    break

            factor, level, weight = selected
            sc_id = (
                f"rsi_challenge_r{round_idx:02d}_{idx:04d}"
                if round_idx > 0
                else f"rsi_challenge_{idx:04d}"
            )
            scenarios.append(
                ChallengeScenario(
                    scenario_id=sc_id,
                    factor=factor,
                    level=level,
                    weight=round(weight, 4),
                    description=(f"Challenge {factor} at {level} (weight={weight:.2f})"),
                )
            )

        return scenarios


# -------------------------------------------------------------------
# CurriculumManager — closed-loop auto-pool orchestrator
# -------------------------------------------------------------------


class CurriculumManager:
    """End-to-end curriculum lifecycle manager with quarantine isolation.

    Wires :class:`CurriculumSampler` into the main evaluation workflow
    auto-pool closed loop:

    1. **Sample** challenge scenarios based on previous Synapse-Bench scores
    2. **Evaluate** candidate LoRA against sampled challenges
    3. **Gate** via monotonic promotion (delegates to ``promotion_gate``)
    4. **Quarantine** regressed scenarios into an isolation directory
    5. **Update** curriculum weights for the next training round

    Args:
        config: Curriculum sampling configuration.
        quarantine_dir: Directory to write quarantine manifests.
            Created on demand.
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        quarantine_dir: Path | None = None,
    ) -> None:
        self._sampler = CurriculumSampler(config)
        self._quarantine_dir = Path(quarantine_dir) if quarantine_dir else None
        self._round: int = 0
        self._history: list[dict[str, float]] = []

    @property
    def sampler(self) -> CurriculumSampler:
        """Access the underlying :class:`CurriculumSampler`."""
        return self._sampler

    @property
    def round(self) -> int:
        """Current training round (0-based)."""
        return self._round

    @property
    def history(self) -> list[dict[str, float]]:
        """Per-round score history."""
        return list(self._history)

    def plan_round(
        self,
        bench_scores: dict[str, float],
    ) -> list[ChallengeScenario]:
        """Plan the next training round's challenge pool.

        Args:
            bench_scores: Previous round's per-factor scores.

        Returns:
            List of challenge scenarios for the next round.
        """
        scenarios = self._sampler.sample_scenarios(bench_scores, round_idx=self._round)
        self._history.append(dict(bench_scores))
        return scenarios

    def record_quarantine(
        self,
        quarantine_entries: list[dict[str, object]],
        lora_sha256: str = "",
    ) -> Path | None:
        """Write quarantine manifest for regressed scenarios.

        Args:
            quarantine_entries: List of quarantine entry dicts
                (from ``PromotionResult.quarantine``).
            lora_sha256: SHA256 of the candidate LoRA that failed gate.

        Returns:
            Path to the written manifest, or None if quarantine_dir is unset.
        """
        if self._quarantine_dir is None:
            return None

        import json

        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._quarantine_dir / f"quarantine_round_{self._round:04d}.json"
        manifest = {
            "round": self._round,
            "lora_sha256": lora_sha256,
            "quarantine": quarantine_entries,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )
        return manifest_path

    def advance_round(self) -> int:
        """Advance to the next training round.

        Returns:
            The new round number.
        """
        self._round += 1
        return self._round


# -------------------------------------------------------------------
# Scenario run reports to BenchReport bridge
# -------------------------------------------------------------------


def scenario_reports_to_bench_report(
    scenario_reports: list[Any],
    lora_sha256: str = "",
) -> Any:
    """Convert a list of ScenarioRunReport results into a unified BenchReport.

    Connects Synapse-Bench pipeline execution directly into monotonic promotion gating.
    """
    from synapse.rsi.promotion_gate import BenchReport, BenchScenarioScore

    scenario_scores: list[BenchScenarioScore] = []
    for rep in scenario_reports:
        # Detect crashed / errored reports to prevent 0.0 wash-out.
        # A report with an error field should NOT be silently scored as 0.0
        # and averaged in — that would wash out the crash signal.
        rep_error = None
        if hasattr(rep, "error"):
            rep_error = rep.error
        elif isinstance(rep, dict):
            rep_error = rep.get("error")
        if rep_error is not None and rep_error:
            # Skip errored scenarios entirely — they must not pollute
            # the aggregate with false 0.0 scores.
            continue

        if hasattr(rep, "scenario_name"):
            sc_id = str(rep.scenario_name)
            metrics = dict(rep.metrics) if hasattr(rep, "metrics") and rep.metrics else {}
            verdict = getattr(rep, "gate_verdict", None)
        elif isinstance(rep, dict):
            sc_id = str(rep.get("scenario_name", rep.get("scenario_id", "unknown")))
            metrics = dict(rep.get("metrics", {}))
            verdict = rep.get("gate_verdict")
        else:
            sc_id = getattr(rep, "scenario_id", "unknown")
            metrics = getattr(rep, "metrics", {})
            verdict = None

        # Extract pillar sub-scores fail-closed: guard math.isfinite() so NaN never
        # washes out via min(1.0, NaN) == 1.0, and never fabricate passing scores
        # (80.0 / 0.85 / 0.9) when metrics are missing.
        def _clamp_unit(val: Any, allow_pct: bool = False) -> tuple[float, bool]:
            if val is None:
                return 0.0, False
            fval = float(val)
            if not math.isfinite(fval):
                return 0.0, False
            if allow_pct and fval > 1.0:
                fval = fval / 100.0
            return max(0.0, min(1.0, fval)), True

        raw_geom = metrics.get("f_score_1mm")
        if raw_geom is None:
            raw_geom = metrics.get("f_score")
        if raw_geom is None:
            raw_geom = metrics.get("watertight_pct")
        geom_score, geom_present = _clamp_unit(raw_geom, allow_pct=True)

        raw_temp = metrics.get("temporal_consistency")
        if raw_temp is None:
            raw_temp = metrics.get("ssim")
        temp_score, temp_present = _clamp_unit(raw_temp)

        raw_style = metrics.get("style_consistency")
        if raw_style is None:
            raw_style = metrics.get("histogram_intersection")
        style_score, style_present = _clamp_unit(raw_style)

        raw_fgpp = metrics.get("silhouette_iou")
        if raw_fgpp is None:
            raw_fgpp = metrics.get("fgpp_fidelity")
        fgpp_score, fgpp_present = _clamp_unit(raw_fgpp)

        if verdict is not None and hasattr(verdict, "overall_score"):
            v_overall = float(verdict.overall_score)
            overall_score = max(0.0, min(1.0, v_overall)) if math.isfinite(v_overall) else 0.0
        elif "overall" in metrics:
            m_overall = float(metrics["overall"])
            overall_score = max(0.0, min(1.0, m_overall)) if math.isfinite(m_overall) else 0.0
        else:
            present_pillars = [
                score
                for score, ok in (
                    (geom_score, geom_present),
                    (temp_score, temp_present),
                    (style_score, style_present),
                    (fgpp_score, fgpp_present),
                )
                if ok
            ]
            overall_score = (
                sum(present_pillars) / len(present_pillars) if present_pillars else 0.0
            )

        scenario_scores.append(
            BenchScenarioScore(
                scenario_id=sc_id,
                overall=round(overall_score, 4),
                geom=round(geom_score, 4),
                temporal=round(temp_score, 4),
                style=round(style_score, 4),
                fgpp_fidelity=round(fgpp_score, 4),
            )
        )

    n = len(scenario_scores)
    if n == 0:
        return BenchReport(
            overall_score=0.0,
            geom_score=0.0,
            temporal_score=0.0,
            style_score=0.0,
            fgpp_fidelity_score=0.0,
            scenarios=[],
            lora_sha256=lora_sha256,
        )

    avg_overall = sum(s.overall for s in scenario_scores) / n
    avg_geom = sum(s.geom for s in scenario_scores) / n
    avg_temp = sum(s.temporal for s in scenario_scores) / n
    avg_style = sum(s.style for s in scenario_scores) / n
    avg_fgpp = sum(s.fgpp_fidelity for s in scenario_scores) / n

    return BenchReport(
        overall_score=round(avg_overall, 4),
        geom_score=round(avg_geom, 4),
        temporal_score=round(avg_temp, 4),
        style_score=round(avg_style, 4),
        fgpp_fidelity_score=round(avg_fgpp, 4),
        scenarios=scenario_scores,
        lora_sha256=lora_sha256,
    )
