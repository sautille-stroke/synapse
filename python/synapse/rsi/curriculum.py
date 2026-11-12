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

import random

from pydantic import BaseModel, Field

from synapse.bench.conditions import RENDERING_CONDITION_FACTORS

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
            score = max(0.0, min(1.0, float(bench_scores[factor])))
            if score < threshold:
                deficit = (threshold - score) / max(threshold, 0.01)
                weights[factor] = 1.0 + (boost - 1.0) * deficit
            else:
                weights[factor] = 1.0

        return weights

    def sample_scenarios(
        self,
        bench_scores: dict[str, float],
    ) -> list[ChallengeScenario]:
        """Generate a set of challenge scenarios.

        Samples from the weighted factor distribution, preferring
        higher difficulty levels for weaker factors.

        Args:
            bench_scores: Previous round's per-factor scores.

        Returns:
            List of ``ChallengeScenario`` objects.
        """
        rng = random.Random(self._config.seed)
        weights = self.analyse_weaknesses(bench_scores)

        # Build weighted pool of (factor, level) pairs
        pool: list[tuple[str, str, float]] = []
        for factor, w in weights.items():
            is_weak = (
                factor in bench_scores
                and float(bench_scores[factor])
                < self._config.weakness_threshold
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
            scenarios.append(
                ChallengeScenario(
                    scenario_id=f"rsi_challenge_{idx:04d}",
                    factor=factor,
                    level=level,
                    weight=round(weight, 4),
                    description=(
                        f"Challenge {factor} at {level} "
                        f"(weight={weight:.2f})"
                    ),
                )
            )

        return scenarios
