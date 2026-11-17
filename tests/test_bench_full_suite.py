"""Full Synapse-Bench scenario suite tests with 0 asset resolution errors (AUD-P0-04)."""

from __future__ import annotations

from typing import Any

import pytest
from synapse.adapters.mock import MockVideoAdapter
from synapse.bench.scenario import list_scenarios
from synapse.pipeline.bench_runner import BenchPipelineRunner

pytestmark = pytest.mark.slow

_SCENARIO_NAMES = [s.id for s in list_scenarios()]


class TestBenchFullSuite:
    """Validate that every built-in scenario executes cleanly end-to-end."""

    def test_scenario_count(self) -> None:
        """Suite must contain at least 16 scenarios (30 total in standard suite)."""
        assert len(_SCENARIO_NAMES) >= 16, (
            f"Expected at least 16 scenarios, found {len(_SCENARIO_NAMES)}"
        )

    @pytest.mark.parametrize("scenario_name", _SCENARIO_NAMES)
    def test_scenario_asset_resolution_and_execution(
        self,
        scenario_name: str,
        shared_rasterizer_cache: dict[str, Any],
    ) -> None:
        """Every built-in scenario must run without error (0 asset resolution failures)."""
        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(
            adapter,
            gate_profile="track_b_video",
            rasterizer_cache=shared_rasterizer_cache,
        )

        report = runner.run_scenario(
            scenario_name,
            seed=42,
            resolution=(32, 32),
            num_frames=4,
        )
        assert report.error is None, f"Scenario '{scenario_name}' failed: {report.error}"
        assert report.num_frames > 0
        d = report.to_dict()
        assert d["metrics"]["depth_rmse_m"] == "N/A (not_produced)"
        assert d["metrics"]["normal_angular_error_deg"] == "N/A (not_produced)"
        assert "flicker_score" in d["metrics"]
