"""Tests for Traceboard instrumentation in rsi, recon, and inspect modules.

Verifies that Traceboard probes are called with correct tags/values
when active, and that modules behave identically when Traceboard is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared mock probe fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_probe() -> MagicMock:
    """Return a mock Traceboard probe with active=True."""
    probe = MagicMock()
    probe.active = True
    return probe


# ---------------------------------------------------------------------------
# 1. PromotionGate instrumentation
# ---------------------------------------------------------------------------


class TestPromotionGateTraceboard:
    """Verify Traceboard logging in evaluate_monotonic_promotion."""

    @staticmethod
    def _make_bench_report(
        overall: float = 0.8,
        geom: float = 0.75,
        temporal: float = 0.7,
        scenarios: list[dict[str, Any]] | None = None,
    ) -> Any:
        from synapse.rsi.promotion_gate import BenchReport, BenchScenarioScore

        if scenarios is None:
            scenarios = [
                {"scenario_id": f"s{i}", "overall": overall, "geom": geom, "temporal": temporal}
                for i in range(30)
            ]
        return BenchReport(
            overall_score=overall,
            geom_score=geom,
            temporal_score=temporal,
            scenarios=[BenchScenarioScore(**s) for s in scenarios],
        )

    def test_promotion_logs_scalars_on_promote(self, mock_probe: MagicMock) -> None:
        """When promotion succeeds, scalars are logged for scores and conditions."""
        from synapse.rsi.promotion_gate import evaluate_monotonic_promotion

        baseline = self._make_bench_report(overall=0.5, geom=0.5, temporal=0.5)
        candidate = self._make_bench_report(overall=0.8, geom=0.7, temporal=0.7)

        with patch("synapse.rsi.promotion_gate.get_probe", return_value=mock_probe):
            result = evaluate_monotonic_promotion(
                baseline=baseline,
                candidate=candidate,
                epsilon_min=0.0,
                min_scenarios=0,
            )

        assert result.promoted
        # Check that log_scalar was called with promoted=1.0
        promoted_calls = [
            c for c in mock_probe.log_scalar.call_args_list
            if c.args[0] == "rsi/promotion/promoted"
        ]
        assert len(promoted_calls) == 1
        assert promoted_calls[0].args[1] == 1.0

    def test_promotion_logs_quarantine_table_on_reject(self, mock_probe: MagicMock) -> None:
        """When promotion fails with quarantine, log_table is called."""
        from synapse.rsi.promotion_gate import (
            BenchScenarioScore,
            evaluate_monotonic_promotion,
        )

        baseline = self._make_bench_report(overall=0.8, geom=0.8, temporal=0.8)
        # Candidate is worse in scenario s0
        cand_scenarios = [
            BenchScenarioScore(scenario_id="s0", overall=0.5, geom=0.5, temporal=0.5),
            *[
                BenchScenarioScore(scenario_id=f"s{i}", overall=0.9, geom=0.9, temporal=0.9)
                for i in range(1, 30)
            ],
        ]
        candidate = self._make_bench_report(overall=0.88, geom=0.87, temporal=0.87)
        candidate.scenarios = cand_scenarios
        candidate.overall_score = sum(s.overall for s in cand_scenarios) / len(cand_scenarios)

        with patch("synapse.rsi.promotion_gate.get_probe", return_value=mock_probe):
            result = evaluate_monotonic_promotion(
                baseline=baseline,
                candidate=candidate,
                epsilon_min=0.0,
                min_scenarios=0,
                raise_on_reject=False,
            )

        assert not result.promoted
        # Quarantine table should be logged with (tag, columns, rows) matching Probe.log_table
        table_calls = [
            c for c in mock_probe.log_table.call_args_list
            if "quarantine" in c.args[0]
        ]
        assert len(table_calls) >= 1
        tag, cols, rows = table_calls[0].args
        assert isinstance(tag, str)
        assert isinstance(cols, list) and all("name" in col for col in cols)
        assert isinstance(rows, list) and len(rows) >= 1

    def test_promotion_silent_without_probe(self) -> None:
        """When probe is None, promotion still works without errors."""
        from synapse.rsi.promotion_gate import evaluate_monotonic_promotion

        baseline = self._make_bench_report(overall=0.5, geom=0.5, temporal=0.5)
        candidate = self._make_bench_report(overall=0.8, geom=0.7, temporal=0.7)

        with patch("synapse.rsi.promotion_gate.get_probe", return_value=None):
            result = evaluate_monotonic_promotion(
                baseline=baseline,
                candidate=candidate,
                epsilon_min=0.0,
                min_scenarios=0,
            )

        assert result.promoted


# ---------------------------------------------------------------------------
# 2. CurriculumSampler instrumentation
# ---------------------------------------------------------------------------


class TestCurriculumTraceboard:
    """Verify Traceboard logging in CurriculumSampler."""

    def test_analyse_weaknesses_logs_weights(self, mock_probe: MagicMock) -> None:
        """analyse_weaknesses logs per-factor weights and weak factor text."""
        from synapse.rsi.curriculum import CurriculumConfig, CurriculumSampler

        sampler = CurriculumSampler(CurriculumConfig(weakness_threshold=0.8))
        scores = {"lighting": 0.5, "motion_speed": 0.9}

        with patch("synapse.rsi.curriculum.get_probe", return_value=mock_probe):
            weights = sampler.analyse_weaknesses(scores)

        assert "lighting" in weights
        # Check weight scalars were logged
        weight_calls = [
            c for c in mock_probe.log_scalar.call_args_list
            if "curriculum/weight" in c.args[0]
        ]
        assert len(weight_calls) > 0
        # Check weak factors text was logged
        text_calls = [
            c for c in mock_probe.log_text.call_args_list
            if "weak_factors" in c.args[0]
        ]
        assert len(text_calls) >= 1

    def test_sample_scenarios_logs_distribution(self, mock_probe: MagicMock) -> None:
        """sample_scenarios logs round index and scenario count."""
        from synapse.rsi.curriculum import CurriculumConfig, CurriculumSampler

        sampler = CurriculumSampler(CurriculumConfig(num_scenarios=5))
        scores = {"lighting": 0.5}

        with patch("synapse.rsi.curriculum.get_probe", return_value=mock_probe):
            scenarios = sampler.sample_scenarios(scores, round_idx=2)

        assert len(scenarios) == 5
        round_calls = [
            c for c in mock_probe.log_scalar.call_args_list
            if c.args[0] == "rsi/curriculum/round_idx"
        ]
        assert len(round_calls) == 1
        assert round_calls[0].args[1] == 2.0


# ---------------------------------------------------------------------------
# 3. DualVerifier instrumentation
# ---------------------------------------------------------------------------


class TestDualVerifierTraceboard:
    """Verify Traceboard logging in DualVerifierEngine."""

    @staticmethod
    def _good_metrics() -> dict[str, float]:
        return {
            "silhouette_iou": 0.95,
            "depth_rmse_m": 0.01,
            "normal_angular_error_deg": 5.0,
        }

    def test_verify_physics_logs_metrics(self, mock_probe: MagicMock) -> None:
        """Physics verification logs metric values and pass status."""
        from synapse.rsi.dual_verifier import DualVerifierEngine

        engine = DualVerifierEngine()
        metrics = self._good_metrics()

        with patch("synapse.rsi.dual_verifier.get_probe", return_value=mock_probe):
            result = engine.verify_physics("sample_001", metrics)

        assert result.passed
        # Check physics_passed scalar
        passed_calls = [
            c for c in mock_probe.log_scalar.call_args_list
            if "physics_passed" in c.args[0]
        ]
        assert len(passed_calls) == 1
        assert passed_calls[0].args[1] == 1.0

    def test_verify_logs_verdict(self, mock_probe: MagicMock) -> None:
        """Full verify logs verdict text."""
        from synapse.rsi.dual_verifier import DualVerifierEngine

        engine = DualVerifierEngine()
        metrics = self._good_metrics()

        with patch("synapse.rsi.dual_verifier.get_probe", return_value=mock_probe):
            verdict = engine.verify("sample_002", metrics, prm_score=0.9)

        assert verdict.verdict == "ACCEPT"
        text_calls = [
            c for c in mock_probe.log_text.call_args_list
            if "verdict" in c.args[0]
        ]
        assert len(text_calls) >= 1
        assert text_calls[0].args[1] == "ACCEPT"

    def test_verify_silent_without_probe(self) -> None:
        """When probe is None, verify still works."""
        from synapse.rsi.dual_verifier import DualVerifierEngine

        engine = DualVerifierEngine()
        with patch("synapse.rsi.dual_verifier.get_probe", return_value=None):
            verdict = engine.verify("sample_003", self._good_metrics(), prm_score=0.8)
        assert verdict.verdict == "ACCEPT"


# ---------------------------------------------------------------------------
# 4. ReconPipeline instrumentation
# ---------------------------------------------------------------------------


class TestReconPipelineTraceboard:
    """Verify Traceboard logging in ReconPipeline.run()."""

    def test_recon_pipeline_logs_telemetry(self, mock_probe: MagicMock) -> None:
        """Pipeline run logs vertex count, manifold status, etc."""
        import numpy as np
        from synapse.recon.pipeline import ReconPipeline
        from synapse.recon.types import MeshGeometry

        pipeline = ReconPipeline(target_quad_faces=100, geometry_backend="python")
        # Minimal synthetic video: 4 frames of 8x8 pixels
        frames = np.random.default_rng(42).integers(0, 255, (4, 8, 8, 3), dtype=np.uint8)

        # Provide initial_mesh to skip multi-view camera validation
        # (this test validates traceboard logging, not reconstruction)
        rng = np.random.default_rng(99)
        initial_mesh = MeshGeometry(
            vertices=rng.random((50, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6]], dtype=np.int32),
        )

        with patch("synapse.recon.pipeline.get_probe", return_value=mock_probe):
            pipeline.run(frames, fps=10.0, initial_mesh=initial_mesh, has_gen3d_highpoly=True)

        # Verify scalars were logged
        scalar_tags = [c.args[0] for c in mock_probe.log_scalar.call_args_list]
        assert "recon/pipeline/elapsed_s" in scalar_tags
        assert "recon/pipeline/num_frames" in scalar_tags
        assert "recon/pipeline/mesh_vertex_count" in scalar_tags
        assert "recon/pipeline/is_manifold" in scalar_tags


# ---------------------------------------------------------------------------
# 5. VisualJevCritic instrumentation
# ---------------------------------------------------------------------------


class TestJevCriticTraceboard:
    """Verify Traceboard logging in VisualJevCritic.evaluate()."""

    def test_evaluate_logs_quality_and_defects(self, mock_probe: MagicMock) -> None:
        """Evaluation logs overall quality, defect count, and defect table."""
        from synapse.inspect.jev_critic import MockJevTransport, VisualJevCritic

        transport = MockJevTransport(
            defect_overrides={"mesh_crumple": 0.6, "texture_stretch": 0.4},
        )
        critic = VisualJevCritic(transport=transport)

        with patch("synapse.inspect.jev_critic.get_probe", return_value=mock_probe):
            critic.evaluate(
                asset_id="test_asset",
                asset_name="TestAsset",
                image_uris=["view_0.png"],
            )

        # Overall quality should be logged
        quality_calls = [
            c for c in mock_probe.log_scalar.call_args_list
            if "overall_quality" in c.args[0]
        ]
        assert len(quality_calls) == 1
        # Defect count should be logged
        defect_count_calls = [
            c for c in mock_probe.log_scalar.call_args_list
            if "defect_count" in c.args[0]
        ]
        assert len(defect_count_calls) == 1

    def test_evaluate_logs_defect_table(self, mock_probe: MagicMock) -> None:
        """When defects are found, a table is logged."""
        from synapse.inspect.jev_critic import MockJevTransport, VisualJevCritic

        # Set high severity for mesh_crumple
        transport = MockJevTransport(
            defect_overrides={"mesh_crumple": 0.9},
        )
        critic = VisualJevCritic(transport=transport)

        with patch("synapse.inspect.jev_critic.get_probe", return_value=mock_probe):
            report = critic.evaluate(
                asset_id="defective_asset",
                asset_name="DefectiveAsset",
                image_uris=["view_0.png"],
            )

        assert len(report.defects) > 0
        table_calls = [
            c for c in mock_probe.log_table.call_args_list
            if "defects" in c.args[0]
        ]
        assert len(table_calls) >= 1
        tag, cols, rows = table_calls[0].args
        assert isinstance(tag, str)
        assert isinstance(cols, list) and all("name" in col for col in cols)
        assert isinstance(rows, list) and len(rows) >= 1

    def test_jev_critic_real_probe_e2e(self, tmp_path: Path) -> None:
        """Real Probe writes defect TABLE events without signature mismatch."""
        try:
            import aether_traceboard  # noqa: F401
        except ImportError:
            pytest.skip("aether-traceboard not installed")

        import synapse.observability as obs
        from aether_traceboard._types import EventType
        from aether_traceboard.reader import read_events
        from synapse.inspect.jev_critic import MockJevTransport, VisualJevCritic

        obs._probe = None
        obs._initialized = False
        probe = obs.init_traceboard(tmp_path, "jev_table_e2e")
        assert probe is not None
        try:
            critic = VisualJevCritic(
                transport=MockJevTransport(defect_overrides={"mesh_crumple": 0.9}),
            )
            critic.evaluate(
                asset_id="defective_asset",
                asset_name="DefectiveAsset",
                image_uris=["view_0.png"],
            )
        finally:
            obs.shutdown_traceboard()

        records = read_events(next(tmp_path.glob("*.tbevt")))
        assert any(r.event_type == EventType.TABLE for r in records)

    def test_evaluate_silent_without_probe(self) -> None:
        """When probe is None, evaluate still works."""
        from synapse.inspect.jev_critic import MockJevTransport, VisualJevCritic

        transport = MockJevTransport()
        critic = VisualJevCritic(transport=transport)

        with patch("synapse.inspect.jev_critic.get_probe", return_value=None):
            report = critic.evaluate(
                asset_id="quiet_asset",
                asset_name="QuietAsset",
                image_uris=["view_0.png"],
            )

        assert report.overall_quality_score >= 0.0
