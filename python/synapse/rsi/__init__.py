"""RSI --- Recursive Self-Improvement flywheel for neural video generation.

Submodules:
    curriculum      Adaptive difficulty curriculum sampler
    dual_verifier   Physics + PRM dual verification engine
    pair_builder    Preference dataset builder with 20% golden enforcement
    promotion_gate  Monotonic promotion gate with quarantine manifest
"""

from __future__ import annotations

from synapse.rsi.curriculum import (
    ChallengeScenario,
    CurriculumConfig,
    CurriculumSampler,
)
from synapse.rsi.dual_verifier import (
    DualVerifierEngine,
    PhysicsCheckResult,
    PhysicsThresholds,
    VerificationVerdict,
)
from synapse.rsi.pair_builder import (
    PreferenceDatasetBuilder,
    PreferencePair,
)
from synapse.rsi.promotion_gate import (
    BenchReport,
    BenchScenarioScore,
    PromotionResult,
    QuarantineEntry,
    evaluate_monotonic_promotion,
    write_quarantine_manifest,
)

__all__ = [
    "ChallengeScenario",
    "CurriculumConfig",
    "CurriculumSampler",
    "DualVerifierEngine",
    "PhysicsCheckResult",
    "PhysicsThresholds",
    "VerificationVerdict",
    "PreferenceDatasetBuilder",
    "PreferencePair",
    "BenchReport",
    "BenchScenarioScore",
    "PromotionResult",
    "QuarantineEntry",
    "evaluate_monotonic_promotion",
    "write_quarantine_manifest",
]
