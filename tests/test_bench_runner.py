"""Tests for synapse.pipeline.bench_runner — BenchPipelineRunner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from synapse.adapters.base import (
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoModelAdapter,
    VideoResult,
)
from synapse.adapters.mock import MockVideoAdapter
from synapse.pipeline.bench_runner import (
    BenchPipelineRunner,
    ScenarioRunReport,
    evaluate_bundle,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


class _BrokenAdapter(VideoModelAdapter):
    """Adapter that always raises for error-handling tests."""

    @property
    def adapter_id(self) -> str:
        return "broken"

    @property
    def tier(self) -> str:  # type: ignore[override]
        return "mock"

    def prepare_conditions(
        self,
        fgpp_sequence: Any,
        config: VideoGenerationConfig,
    ) -> PreparedConditionPayload:
        raise RuntimeError("Intentional adapter failure")

    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> VideoResult:
        raise RuntimeError("Intentional adapter failure")


def _available_scenario_name() -> str:
    """Return the first available scenario name, or skip."""
    from synapse.bench.scenario import list_scenarios
    scenarios = list_scenarios()
    if not scenarios:
        pytest.skip("No built-in scenarios available")
    return scenarios[0].id


# -------------------------------------------------------------------
# TestBenchPipelineRunner
# -------------------------------------------------------------------


class TestBenchPipelineRunner:
    """Tests for the BenchPipelineRunner class."""

    def test_run_single_scenario(self) -> None:
        """Run with MockVideoAdapter, verify valid metrics."""
        name = _available_scenario_name()
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter)

        report = runner.run_scenario(
            scenario_name=name,
            seed=42,
            resolution=(64, 64),
        )

        assert isinstance(report, ScenarioRunReport)
        assert report.scenario_name == name
        assert report.adapter_id == "mock"
        assert report.error is None
        assert report.num_frames > 0
        assert report.seed == 42

        # Metrics should be present and non-NaN
        assert len(report.metrics) > 0
        for key, val in report.metrics.items():
            assert isinstance(val, float), (
                f"Metric {key} is not float: {type(val)}"
            )
            assert not np.isnan(val), (
                f"Metric {key} is NaN"
            )

    def test_run_scenario_with_gate(self) -> None:
        """Run with gate_profile, verify gate_verdict is present."""
        name = _available_scenario_name()
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(
            adapter, gate_profile="track_b_video",
        )

        report = runner.run_scenario(
            scenario_name=name,
            seed=42,
            resolution=(64, 64),
        )

        assert report.error is None
        assert report.gate_verdict is not None
        assert hasattr(report.gate_verdict, "passed")
        assert hasattr(report.gate_verdict, "profile")
        assert report.gate_verdict.profile == "track_b_video"

    def test_run_all_scenarios(self) -> None:
        """Run all built-in scenarios, verify report completeness."""
        from synapse.bench.scenario import list_scenarios

        all_scenarios = list_scenarios()
        if not all_scenarios:
            pytest.skip("No built-in scenarios available")

        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter)

        reports = runner.run_all(
            seed=42, resolution=(64, 64),
        )

        assert len(reports) == len(all_scenarios)
        expected_names = {s.id for s in all_scenarios}
        actual_names = {r.scenario_name for r in reports}
        assert actual_names == expected_names

    def test_run_specific_scenarios(self) -> None:
        """Run a specific subset of scenarios."""
        name = _available_scenario_name()
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter)

        reports = runner.run_all(
            scenarios=[name],
            seed=42,
            resolution=(64, 64),
        )

        assert len(reports) == 1
        assert reports[0].scenario_name == name

    def test_timing_recorded(self) -> None:
        """Verify all timing keys are present and > 0."""
        name = _available_scenario_name()
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter)

        report = runner.run_scenario(
            scenario_name=name,
            seed=42,
            resolution=(64, 64),
        )

        assert report.error is None
        expected_keys = {
            "scenario_load_ms",
            "fgpp_load_ms",
            "generate_ms",
            "metrics_ms",
            "total_ms",
        }
        for key in expected_keys:
            assert key in report.timings, (
                f"Missing timing key: {key}"
            )
            assert report.timings[key] > 0, (
                f"Timing {key} should be > 0, got {report.timings[key]}"
            )

    def test_error_handling(self) -> None:
        """Broken adapter: verify error is captured, not raised."""
        name = _available_scenario_name()
        broken = _BrokenAdapter()
        runner = BenchPipelineRunner(broken)

        report = runner.run_scenario(
            scenario_name=name,
            seed=42,
            resolution=(64, 64),
        )

        assert report.error is not None
        assert "Intentional adapter failure" in report.error
        assert report.timings.get("total_ms", 0) > 0

    def test_report_to_dict(self) -> None:
        """Verify ScenarioRunReport serialises to JSON-safe dict."""
        name = _available_scenario_name()
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(
            adapter, gate_profile="track_b_video",
        )

        report = runner.run_scenario(
            scenario_name=name,
            seed=42,
            resolution=(64, 64),
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["scenario_name"] == name
        assert d["adapter_id"] == "mock"
        # Ensure JSON-serialisable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_resolution_propagated(self) -> None:
        """Verify the output resolution is captured in the report."""
        name = _available_scenario_name()
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter)

        report = runner.run_scenario(
            scenario_name=name,
            seed=42,
            resolution=(64, 64),
        )

        assert report.error is None
        assert report.resolution[0] > 0
        assert report.resolution[1] > 0


# -------------------------------------------------------------------
# TestEvaluateBundle
# -------------------------------------------------------------------


class TestEvaluateBundle:
    """Tests for the evaluate_bundle function."""

    def _create_bundle(self, tmp_path: Path) -> Path:
        """Create a minimal InspectionBundle directory."""
        bundle = tmp_path / "test_bundle"
        frames_dir = bundle / "frames"
        fgpp_dir = bundle / "fgpp"
        frames_dir.mkdir(parents=True)
        fgpp_dir.mkdir(parents=True)

        # Create synthetic generated frames [5, 32, 32, 3] uint8
        rng = np.random.Generator(np.random.PCG64(42))
        for i in range(5):
            frame = rng.integers(
                50, 200, size=(32, 32, 3), dtype=np.uint8,
            )
            np.save(str(frames_dir / f"frame_{i:04d}.npy"), frame)

        # Create GT depth [32, 32] float32
        depth = rng.uniform(
            0.5, 5.0, size=(32, 32),
        ).astype(np.float32)
        np.save(str(fgpp_dir / "depth.npy"), depth)

        # Create GT normals [32, 32, 3] float32
        normals = rng.standard_normal(
            size=(32, 32, 3),
        ).astype(np.float32)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)
        np.save(str(fgpp_dir / "normals.npy"), normals)

        # Create GT mask [32, 32] uint32
        mask = np.ones((32, 32), dtype=np.uint32)
        mask[:5, :] = 0  # some background
        np.save(str(fgpp_dir / "mask.npy"), mask)

        # Create metadata.json
        meta = {"scene_id": "test", "num_frames": 5}
        (bundle / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        return bundle

    def test_evaluate_creates_comparison_json(
        self, tmp_path: Path,
    ) -> None:
        """Run evaluate_bundle and check comparison.json exists."""
        bundle = self._create_bundle(tmp_path)
        result = evaluate_bundle(bundle, profile="track_b_video")

        assert isinstance(result, dict)
        comparison = bundle / "metrics" / "comparison.json"
        assert comparison.is_file()

        loaded = json.loads(comparison.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)

    def test_evaluate_metrics_valid(self, tmp_path: Path) -> None:
        """Verify output metrics are valid floats."""
        bundle = self._create_bundle(tmp_path)
        result = evaluate_bundle(bundle, profile="track_b_video")

        assert len(result) > 0
        for key, val in result.items():
            assert isinstance(val, float), (
                f"Metric {key} is not float: {type(val)}"
            )
            assert not np.isnan(val), (
                f"Metric {key} is NaN"
            )

    def test_evaluate_empty_bundle(self, tmp_path: Path) -> None:
        """Evaluate an empty bundle without crashing."""
        bundle = tmp_path / "empty_bundle"
        bundle.mkdir(parents=True)
        (bundle / "metadata.json").write_text(
            json.dumps({"scene_id": "empty"}), encoding="utf-8",
        )

        result = evaluate_bundle(bundle, profile="track_b_video")
        assert isinstance(result, dict)
        # gate_passed should still be present
        assert "gate_passed" in result

    def test_evaluate_with_different_profiles(
        self, tmp_path: Path,
    ) -> None:
        """Evaluate with different gate profiles."""
        bundle = self._create_bundle(tmp_path)

        for profile in [
            "track_b_video", "track_a_asset", "rsi_promotion",
        ]:
            result = evaluate_bundle(bundle, profile=profile)
            assert isinstance(result, dict)
            assert "gate_passed" in result
