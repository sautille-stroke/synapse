"""RSI curriculum sampler for adaptive difficulty scheduling.

Provides:
- ``CurriculumSampler``: Adaptive difficulty distribution based
  on Synapse-Bench weakness scores, with factor-level upsampling

The sampler analyses the previous round's benchmark report to
identify weak dimensions and increases sampling probability for
harder condition levels (L3/L4) on those factors.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Factor definitions (matching Synapse-Bench condition factors)
# -------------------------------------------------------------------

_CONDITION_FACTORS: list[str] = [
    "F01_viewpoint",
    "F02_distance",
    "F03_lighting",
    "F04_material",
    "F05_animation",
    "F06_occlusion",
    "F07_background",
    "F08_resolution",
]

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
        self, config: CurriculumConfig | None = None,
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

        Args:
            bench_scores: Mapping of factor name to aggregate
                score in [0, 1].

        Returns:
            Mapping of factor name to sampling weight.
            Weak factors get boosted weights.
        """
        weights: dict[str, float] = {}
        threshold = self._config.weakness_threshold
        boost = self._config.boost_factor

        for factor in _CONDITION_FACTORS:
            score = bench_scores.get(factor, 0.5)
            if score < threshold:
                # Weak factor: boost proportional to weakness
                weakness = 1.0 - score
                weights[factor] = 1.0 + (boost - 1.0) * (
                    weakness / max(1.0 - threshold, 0.01)
                )
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
            factor_score = bench_scores.get(factor, 0.5)
            for i, level in enumerate(_DIFFICULTY_LEVELS):
                if i > self._config.max_level:
                    break
                # Higher levels get more weight for weak factors
                if factor_score < self._config.weakness_threshold:
                    level_w = w * (1.0 + i * 0.5)
                else:
                    level_w = w
                pool.append((factor, level, level_w))

        if not pool:
            return []

        # Weighted sampling
        scenarios: list[ChallengeScenario] = []
        total_weight = sum(p[2] for p in pool)

        for idx in range(self._config.num_scenarios):
            # Weighted random selection
            r = rng.random() * total_weight
            cumulative = 0.0
            selected = pool[0]
            for item in pool:
                cumulative += item[2]
                if cumulative >= r:
                    selected = item
                    break

            factor, level, weight = selected
            scenarios.append(ChallengeScenario(
                scenario_id=f"rsi_challenge_{idx:04d}",
                factor=factor,
                level=level,
                weight=round(weight, 4),
                description=(
                    f"Challenge {factor} at {level} "
                    f"(weight={weight:.2f})"
                ),
            ))

        return scenarios
