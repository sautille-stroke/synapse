"""Tests for PR-24: RSI curriculum + dual verifier + pair builder.

Validates:
- CurriculumSampler weakness analysis and scenario generation
- DualVerifierEngine physics-first enforcement
- Anti echo-chamber redline: missing physics metrics -> error
- Physics failure routes to correct LoRA bucket
- Jev high score + foot_sliding -> HARD_NEGATIVE in motion bucket
- PreferenceDatasetBuilder 20% golden enforcement
- Empty L0 golden set -> ContractValidationError(10)
- Auto-oversample meets golden ratio
"""

from __future__ import annotations

from pathlib import Path

import pytest
from synapse.errors import ContractValidationError
from synapse.rsi.curriculum import (
    CurriculumConfig,
    CurriculumSampler,
)
from synapse.rsi.dual_verifier import (
    DualVerifierEngine,
    PhysicsThresholds,
)
from synapse.rsi.pair_builder import (
    PreferenceDatasetBuilder,
    PreferencePair,
)

# -------------------------------------------------------------------
# CurriculumSampler tests
# -------------------------------------------------------------------


class TestCurriculumSampler:
    """Tests for CurriculumSampler."""

    def test_default_config(self) -> None:
        cfg = CurriculumConfig()
        assert cfg.weakness_threshold == 0.7
        assert cfg.num_scenarios == 10

    def test_analyse_weaknesses_identifies_low_scores(
        self,
    ) -> None:
        """Low-scoring factors get boosted weights."""
        sampler = CurriculumSampler()
        scores = {
            "F01_viewpoint": 0.4,  # weak
            "F03_lighting": 0.9,  # strong
            "F06_occlusion": 0.3,  # very weak
        }
        weights = sampler.analyse_weaknesses(scores)
        assert weights["F01_viewpoint"] > 1.0
        assert weights["F03_lighting"] == 1.0
        assert weights["F06_occlusion"] > weights["F01_viewpoint"]

    def test_analyse_weaknesses_all_strong(self) -> None:
        """All strong factors get weight 1.0."""
        sampler = CurriculumSampler()
        # Provide scores for ALL factors to avoid default 0.5
        scores = {
            "F01_viewpoint": 0.9,
            "F02_distance": 0.9,
            "F03_lighting": 0.9,
            "F04_material": 0.9,
            "F05_animation": 0.9,
            "F06_occlusion": 0.9,
            "F07_background": 0.9,
            "F08_resolution": 0.9,
        }
        weights = sampler.analyse_weaknesses(scores)
        for f in scores:
            assert weights[f] == 1.0

    def test_sample_scenarios_count(self) -> None:
        """Sampler produces requested number of scenarios."""
        cfg = CurriculumConfig(num_scenarios=5, seed=42)
        sampler = CurriculumSampler(cfg)
        scenarios = sampler.sample_scenarios(
            {"F01_viewpoint": 0.5, "F03_lighting": 0.8},
        )
        assert len(scenarios) == 5

    def test_sample_scenarios_ids_unique(self) -> None:
        """Scenario IDs are unique."""
        cfg = CurriculumConfig(num_scenarios=10, seed=42)
        sampler = CurriculumSampler(cfg)
        scenarios = sampler.sample_scenarios(
            {"F01_viewpoint": 0.4},
        )
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_weak_factor_sampled_more(self) -> None:
        """Weak factors appear more frequently in output."""
        cfg = CurriculumConfig(num_scenarios=100, seed=42)
        sampler = CurriculumSampler(cfg)
        scores = {
            "F01_viewpoint": 0.3,  # very weak
            "F03_lighting": 0.95,  # very strong
        }
        scenarios = sampler.sample_scenarios(scores)
        viewpoint_count = sum(
            1 for s in scenarios
            if s.factor == "F01_viewpoint"
        )
        lighting_count = sum(
            1 for s in scenarios
            if s.factor == "F03_lighting"
        )
        # Viewpoint should appear more often
        assert viewpoint_count > lighting_count

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same scenarios."""
        cfg = CurriculumConfig(num_scenarios=5, seed=123)
        s1 = CurriculumSampler(cfg).sample_scenarios(
            {"F01_viewpoint": 0.5},
        )
        s2 = CurriculumSampler(cfg).sample_scenarios(
            {"F01_viewpoint": 0.5},
        )
        for a, b in zip(s1, s2):
            assert a.scenario_id == b.scenario_id
            assert a.factor == b.factor
            assert a.level == b.level


# -------------------------------------------------------------------
# DualVerifierEngine tests
# -------------------------------------------------------------------


class TestDualVerifierEngine:
    """Tests for DualVerifierEngine."""

    def _good_metrics(self) -> dict[str, float]:
        """Return metrics that pass all physics checks."""
        return {
            "silhouette_iou": 0.95,
            "depth_rmse_m": 0.01,
            "normal_angular_error_deg": 5.0,
            "interpenetration_volume_ratio": 0.0001,
            "foot_slide_velocity_ms": 0.005,
            "is_manifold_watertight": 1.0,
        }

    def test_all_physics_pass(self) -> None:
        """Good metrics pass all physics checks."""
        engine = DualVerifierEngine()
        result = engine.verify_physics(
            "s001", self._good_metrics(),
        )
        assert result.passed is True
        assert result.failures == []
        assert result.lora_buckets == []

    def test_missing_metrics_raises_contract_error(
        self,
    ) -> None:
        """Missing required physics metrics -> exit code 10."""
        engine = DualVerifierEngine()
        with pytest.raises(ContractValidationError) as exc:
            engine.verify_physics("s001", {})
        assert exc.value.exit_code == 10
        assert "physics metrics" in str(exc.value).lower()

    def test_missing_partial_metrics_raises(self) -> None:
        """Partial metrics (missing depth_rmse_m) raise error."""
        engine = DualVerifierEngine()
        with pytest.raises(ContractValidationError):
            engine.verify_physics("s001", {
                "silhouette_iou": 0.95,
                "normal_angular_error_deg": 5.0,
                # missing depth_rmse_m
            })

    def test_silhouette_drift_detected(self) -> None:
        """Low silhouette IoU causes SILHOUETTE_DRIFT failure."""
        engine = DualVerifierEngine()
        m = self._good_metrics()
        m["silhouette_iou"] = 0.5
        result = engine.verify_physics("s001", m)
        assert result.passed is False
        assert "SILHOUETTE_DRIFT" in result.failures
        assert "control" in result.lora_buckets

    def test_depth_inconsistency_detected(self) -> None:
        """High depth RMSE causes DEPTH_INCONSISTENCY failure."""
        engine = DualVerifierEngine()
        m = self._good_metrics()
        m["depth_rmse_m"] = 0.1
        result = engine.verify_physics("s001", m)
        assert result.passed is False
        assert "DEPTH_INCONSISTENCY" in result.failures
        assert "control" in result.lora_buckets

    def test_foot_sliding_detected(self) -> None:
        """Foot sliding exceeding threshold detected."""
        engine = DualVerifierEngine()
        m = self._good_metrics()
        m["foot_slide_velocity_ms"] = 0.1
        result = engine.verify_physics("s001", m)
        assert result.passed is False
        assert "FOOT_SLIPPING" in result.failures
        assert "motion" in result.lora_buckets

    def test_jev_high_score_but_foot_sliding_is_hard_negative(
        self,
    ) -> None:
        """Jev high PRM score + foot sliding = HARD_NEGATIVE.

        This validates the anti echo-chamber redline:
        even with a perfect PRM score, physics failure means
        HARD_NEGATIVE, routed to 'motion' LoRA bucket.
        """
        engine = DualVerifierEngine()
        m = self._good_metrics()
        m["foot_slide_velocity_ms"] = 0.1  # exceeds 0.02

        verdict = engine.verify(
            "s001", m, prm_score=0.99,
        )

        assert verdict.verdict == "HARD_NEGATIVE"
        assert verdict.physics_passed is False
        assert "FOOT_SLIPPING" in verdict.failure_attributions
        assert "motion" in verdict.target_lora_buckets

    def test_physics_pass_prm_high_accept(self) -> None:
        """Good physics + high PRM score = ACCEPT."""
        engine = DualVerifierEngine()
        verdict = engine.verify(
            "s001", self._good_metrics(), prm_score=0.9,
        )
        assert verdict.verdict == "ACCEPT"
        assert verdict.physics_passed is True

    def test_physics_pass_prm_low_reject(self) -> None:
        """Good physics + low PRM score = REJECT."""
        engine = DualVerifierEngine()
        verdict = engine.verify(
            "s001", self._good_metrics(), prm_score=0.1,
        )
        assert verdict.verdict == "REJECT"

    def test_physics_pass_prm_mid_hard_negative(self) -> None:
        """Good physics + medium PRM score = HARD_NEGATIVE."""
        engine = DualVerifierEngine()
        verdict = engine.verify(
            "s001", self._good_metrics(), prm_score=0.5,
        )
        assert verdict.verdict == "HARD_NEGATIVE"

    def test_custom_thresholds(self) -> None:
        """Custom thresholds are respected."""
        t = PhysicsThresholds(
            silhouette_iou=0.5,
            depth_rmse_m=0.1,
        )
        engine = DualVerifierEngine(thresholds=t)
        m = {
            "silhouette_iou": 0.6,
            "depth_rmse_m": 0.05,
            "normal_angular_error_deg": 5.0,
        }
        result = engine.verify_physics("s001", m)
        assert result.passed is True

    def test_multiple_failures_multiple_buckets(self) -> None:
        """Multiple failures route to multiple LoRA buckets."""
        engine = DualVerifierEngine()
        m = self._good_metrics()
        m["silhouette_iou"] = 0.5
        m["foot_slide_velocity_ms"] = 0.1
        result = engine.verify_physics("s001", m)
        assert not result.passed
        assert "control" in result.lora_buckets
        assert "motion" in result.lora_buckets

    def test_nan_physics_metric_raises_contract_error(
        self,
    ) -> None:
        """NaN or Inf physics metric raises ContractValidationError(10)."""
        engine = DualVerifierEngine()
        m = self._good_metrics()
        m["silhouette_iou"] = float("nan")
        with pytest.raises(ContractValidationError) as exc_info:
            engine.verify_physics("s_nan", m)
        assert int(exc_info.value.exit_code) == 10
        assert "non-finite" in str(exc_info.value).lower()

    def test_nan_prm_score_raises_contract_error(self) -> None:
        """NaN PRM score raises ContractValidationError(10)."""
        engine = DualVerifierEngine()
        with pytest.raises(ContractValidationError) as exc_info:
            engine.verify(
                "s_nan_prm",
                self._good_metrics(),
                prm_score=float("nan"),
            )
        assert int(exc_info.value.exit_code) == 10


# -------------------------------------------------------------------
# PreferenceDatasetBuilder tests
# -------------------------------------------------------------------


class TestPreferenceDatasetBuilder:
    """Tests for PreferenceDatasetBuilder."""

    def _golden_pairs(self, n: int = 3) -> list[PreferencePair]:
        return [
            PreferencePair(
                pair_id=f"golden_{i:03d}",
                prompt=f"Golden fix {i}",
                chosen_uri=f"file:///golden_chosen_{i}.glb",
                rejected_uri=f"file:///golden_rejected_{i}.glb",
                source="golden",
                is_golden=True,
            )
            for i in range(n)
        ]

    def _self_pairs(self, n: int = 10) -> list[PreferencePair]:
        return [
            PreferencePair(
                pair_id=f"self_{i:03d}",
                prompt=f"Auto-fix {i}",
                chosen_uri=f"file:///self_chosen_{i}.glb",
                rejected_uri=f"file:///self_rejected_{i}.glb",
                source="self",
            )
            for i in range(n)
        ]

    def test_empty_golden_raises_error(self) -> None:
        """Empty L0 golden set raises ContractValidationError."""
        builder = PreferenceDatasetBuilder()
        builder.add_self_pairs(self._self_pairs(5))
        with pytest.raises(ContractValidationError) as exc:
            builder.build_pairs()
        assert exc.value.exit_code == 10
        assert "golden" in str(exc.value).lower()

    def test_golden_ratio_met_naturally(self) -> None:
        """When golden ratio is already sufficient, no oversampling."""
        builder = PreferenceDatasetBuilder()
        # 3 golden + 8 self = 11 total, 3/11 = 27% > 20%
        builder.add_golden_pairs(self._golden_pairs(3))
        builder.add_self_pairs(self._self_pairs(8))
        pairs = builder.build_pairs()
        golden_count = sum(1 for p in pairs if p.is_golden)
        total = len(pairs)
        ratio = golden_count / total
        assert ratio >= 0.20

    def test_auto_oversample_meets_ratio(self) -> None:
        """Auto-oversample increases golden to meet 20%."""
        builder = PreferenceDatasetBuilder(auto_oversample=True)
        # 1 golden + 100 self = very low ratio
        builder.add_golden_pairs(self._golden_pairs(1))
        builder.add_self_pairs(self._self_pairs(100))
        pairs = builder.build_pairs()
        golden_count = sum(1 for p in pairs if p.is_golden)
        total = len(pairs)
        ratio = golden_count / total
        assert ratio >= 0.20

    def test_no_oversample_raises_when_insufficient(
        self,
    ) -> None:
        """Disabled oversample + low ratio raises error."""
        builder = PreferenceDatasetBuilder(
            auto_oversample=False,
        )
        builder.add_golden_pairs(self._golden_pairs(1))
        builder.add_self_pairs(self._self_pairs(100))
        with pytest.raises(ContractValidationError) as exc:
            builder.build_pairs()
        assert exc.value.exit_code == 10
        assert "golden" in str(exc.value).lower()

    def test_export_jsonl(self, tmp_path: Path) -> None:
        """Export produces JSONL file with correct counts."""
        builder = PreferenceDatasetBuilder()
        builder.add_golden_pairs(self._golden_pairs(3))
        builder.add_self_pairs(self._self_pairs(8))
        out = tmp_path / "prefs.jsonl"
        result = builder.export_jsonl(out)
        assert result["total"] == 11
        assert result["golden_count"] == 3
        assert result["golden_ratio"] >= 0.20
        assert out.exists()

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 11

    def test_golden_pairs_have_correct_flags(self) -> None:
        """Golden pairs are marked with is_golden=True."""
        builder = PreferenceDatasetBuilder()
        builder.add_golden_pairs(self._golden_pairs(2))
        builder.add_self_pairs(self._self_pairs(5))
        pairs = builder.build_pairs()
        golden = [p for p in pairs if p.is_golden]
        assert len(golden) >= 2
        for g in golden:
            assert g.source == "golden"

    def test_oversampled_pairs_have_unique_ids(self) -> None:
        """Oversampled golden pairs get unique suffixed IDs."""
        builder = PreferenceDatasetBuilder()
        builder.add_golden_pairs(self._golden_pairs(1))
        builder.add_self_pairs(self._self_pairs(50))
        pairs = builder.build_pairs()
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids))

    def test_human_pairs_included(self) -> None:
        """Human expert pairs are included in output."""
        builder = PreferenceDatasetBuilder()
        human = [
            PreferencePair(
                pair_id="human_001",
                prompt="Human fix",
                source="human",
            ),
        ]
        builder.add_golden_pairs(self._golden_pairs(3))
        builder.add_human_pairs(human)
        pairs = builder.build_pairs()
        human_found = any(
            p.pair_id == "human_001" for p in pairs
        )
        assert human_found


# -------------------------------------------------------------------
# Stage registration test
# -------------------------------------------------------------------


class TestStageRegistration:
    """Test that rsi_dual_verify stage is registered."""

    def test_stage_registered(self) -> None:
        # Importing rsi.pair_builder triggers registration
        import synapse.rsi.pair_builder  # noqa: F401
        from synapse.pipeline.bundle_worker import (
            _STAGE_HANDLERS,
        )
        assert "rsi_dual_verify" in _STAGE_HANDLERS
