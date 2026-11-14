"""Full Synapse-Bench scenario suite tests with 0 asset resolution errors (AUD-P0-04)."""

from __future__ import annotations

import pytest
from synapse.adapters.mock import MockVideoAdapter
from synapse.bench.scenario import list_scenarios
from synapse.pipeline.bench_runner import BenchPipelineRunner

pytestmark = pytest.mark.slow


class TestBenchFullSuite:
    """Validate that every built-in scenario executes cleanly end-to-end."""

    def test_all_scenarios_asset_resolution_and_execution(self) -> None:
        """Every built-in scenario must run without error (0 asset resolution failures)."""
        scenarios = list_scenarios()
        assert len(scenarios) >= 16, f"Expected at least 16 scenarios, found {len(scenarios)}"

        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter, gate_profile="track_b_video")

        reports = runner.run_all(
            seed=42,
            resolution=(32, 32),
        )
        assert len(reports) == len(scenarios)

        failed_scenarios: list[tuple[str, str | None]] = []
        for r in reports:
            if r.error is not None:
                failed_scenarios.append((r.scenario_name, r.error))

        assert not failed_scenarios, (
            f"Expected 0 scenario failures, but {len(failed_scenarios)} failed:\n"
            + "\n".join(f"  - {name}: {err}" for name, err in failed_scenarios)
        )

        for r in reports:
            assert r.num_frames > 0
            d = r.to_dict()
            assert d["metrics"]["depth_rmse_m"] == "N/A (not_produced)"
            assert d["metrics"]["normal_angular_error_deg"] == "N/A (not_produced)"
            assert "flicker_score" in d["metrics"]
