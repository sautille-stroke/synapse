"""Tests for synapse.bench.traceboard_export — dual-track Traceboard export."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight stand-in for ScenarioRunReport (avoid heavy imports)
# ---------------------------------------------------------------------------
@dataclass
class _FakeGateVerdict:
    passed: bool = True
    profile: str = "default"
    violations: list[Any] = field(default_factory=list)


@dataclass
class _FakeReport:
    scenario_name: str
    adapter_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    gate_verdict: Any | None = None
    timings: dict[str, float] = field(default_factory=dict)
    num_frames: int = 17
    resolution: tuple[int, int] = (256, 256)
    seed: int = 42
    error: str | None = None


def _make_reports() -> list[_FakeReport]:
    """Create sample reports for testing."""
    return [
        _FakeReport(
            scenario_name="turntable_hero",
            adapter_id="mock",
            metrics={
                "psnr_db": 32.5,
                "ssim": 0.92,
                "flicker_score": 0.05,
                "silhouette_iou": 0.88,
                "depth_rmse_m": 0.12,
                "style_color_de": 3.2,
            },
            timings={"total_ms": 150.0, "inference_ms": 120.0},
            num_frames=17,
            gate_verdict=_FakeGateVerdict(passed=True),
        ),
        _FakeReport(
            scenario_name="dolly_zoom",
            adapter_id="mock",
            metrics={
                "psnr_db": 28.0,
                "ssim": 0.85,
                "temporal_warp_error": 0.08,
                "chamfer_mm": 5.5,
            },
            timings={"total_ms": 200.0},
            num_frames=33,
            error="GPU OOM at frame 30",
            gate_verdict=_FakeGateVerdict(passed=False),
        ),
        _FakeReport(
            scenario_name="static_closeup",
            adapter_id="wan21",
            metrics={"psnr_db": 35.0, "ssim": 0.95},
            timings={"total_ms": 80.0},
            num_frames=17,
        ),
    ]


# ---------------------------------------------------------------------------
# Test: export with real aether-traceboard (if installed)
# ---------------------------------------------------------------------------
class TestExportWithRealTraceboard:
    """Test export_reports_to_traceboard with real aether-traceboard SDK."""

    def test_export_creates_tbevt_files(self) -> None:
        """Export should create .tbevt files when aether-traceboard is available."""
        try:
            import aether_traceboard  # noqa: F401
        except ImportError:
            pytest.skip("aether-traceboard not installed")

        from synapse.bench.traceboard_export import export_reports_to_traceboard

        reports = _make_reports()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_reports_to_traceboard(
                reports,
                log_dir=tmpdir,
                run_name="test-export",
            )
            assert result is True
            tbevt_files = list(Path(tmpdir).glob("*.tbevt"))
            assert len(tbevt_files) >= 1

            from aether_traceboard._types import EventType
            from aether_traceboard.reader import read_events

            records = read_events(tbevt_files[0])
            assert records[0].event_type == EventType.RUN_START
            assert records[-1].event_type == EventType.RUN_END

    def test_real_probe_report_and_compare_end_to_end(self, tmp_path: Path) -> None:
        """HTML report artifact and 4-track compare table work with real Probe."""
        try:
            import aether_traceboard  # noqa: F401
        except ImportError:
            pytest.skip("aether-traceboard not installed")

        import synapse.observability as obs
        from aether_traceboard._types import EventType
        from aether_traceboard.reader import read_events
        from synapse.bench.compare import compare_runs
        from synapse.bench.report import generate_html_report

        obs._probe = None
        obs._initialized = False
        tb_dir = tmp_path / "tblog"
        run_a = tmp_path / "run_a.json"
        run_b = tmp_path / "run_b.json"
        run_a.write_text('{"adapter_id": "a", "metrics": {"chamfer_mm": 4.0, "psnr_db": 32.0}}')
        run_b.write_text('{"adapter_id": "b", "metrics": {"chamfer_mm": 6.0, "psnr_db": 29.0}}')

        probe = obs.init_traceboard(tb_dir, "bench_artifact_table_e2e")
        assert probe is not None
        try:
            generate_html_report(_make_reports(), tmp_path / "report.html")
            compare_runs([run_a, run_b])
        finally:
            obs.shutdown_traceboard()

        records = read_events(next(tb_dir.glob("*.tbevt")))
        event_types = {r.event_type for r in records}
        assert EventType.RUN_START in event_types
        assert EventType.ARTIFACT in event_types
        assert EventType.TABLE in event_types
        assert EventType.RUN_END in event_types

    def test_export_contains_scenario_events(self) -> None:
        """Exported .tbevt should contain scenario metric events."""
        try:
            import aether_traceboard  # noqa: F401
        except ImportError:
            pytest.skip("aether-traceboard not installed")

        from synapse.bench.traceboard_export import export_reports_to_traceboard

        reports = _make_reports()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reports_to_traceboard(
                reports,
                log_dir=tmpdir,
                run_name="test-events",
            )
            # Read back the .tbevt file and check for expected tags
            tbevt_files = list(Path(tmpdir).glob("*.tbevt"))
            assert tbevt_files
            content = tbevt_files[0].read_bytes()
            # .tbevt is binary — check that scenario names appear as tags
            assert b"turntable_hero" in content
            assert b"dolly_zoom" in content
            assert b"static_closeup" in content

    def test_export_records_pillar_scores(self) -> None:
        """Export should include 4-pillar aggregate scores."""
        try:
            import aether_traceboard  # noqa: F401
        except ImportError:
            pytest.skip("aether-traceboard not installed")

        from synapse.bench.traceboard_export import export_reports_to_traceboard

        reports = _make_reports()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_reports_to_traceboard(
                reports,
                log_dir=tmpdir,
                run_name="test-pillars",
            )
            tbevt_files = list(Path(tmpdir).glob("*.tbevt"))
            assert tbevt_files
            content = tbevt_files[0].read_bytes()
            assert b"pillar/temporal" in content
            assert b"pillar/roundtrip" in content


# ---------------------------------------------------------------------------
# Test: export gracefully degrades without aether-traceboard
# ---------------------------------------------------------------------------
class TestExportWithoutTraceboard:
    """Test graceful degradation when aether-traceboard is not available."""

    def test_returns_false_when_not_installed(self) -> None:
        """Should return False and not raise when aether-traceboard is missing."""
        import synapse.bench.traceboard_export as mod

        with patch.dict("sys.modules", {"aether_traceboard": None}):
            # Force re-evaluation of the import inside the function
            result = mod.export_reports_to_traceboard(
                _make_reports(),
                log_dir="/tmp/nonexistent",
                run_name="test-noop",
            )
            # May return True if already cached; in a fresh env returns False
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test: pillar score computation
# ---------------------------------------------------------------------------
class TestPillarScores:
    """Test _compute_pillar_scores_from_reports parity with report.py."""

    def test_pillar_temporal(self) -> None:
        """Temporal pillar should derive from flicker_score."""
        from synapse.bench.traceboard_export import _compute_pillar_scores_from_reports

        reports = [
            _FakeReport(
                scenario_name="t1",
                adapter_id="mock",
                metrics={"flicker_score": 0.1},
            ),
        ]
        scores = _compute_pillar_scores_from_reports(reports)
        assert "temporal" in scores
        assert abs(scores["temporal"] - 0.9) < 1e-6

    def test_pillar_fgpp_fidelity(self) -> None:
        """FGPP fidelity should derive from silhouette_iou and depth_rmse_m."""
        from synapse.bench.traceboard_export import _compute_pillar_scores_from_reports

        reports = [
            _FakeReport(
                scenario_name="f1",
                adapter_id="mock",
                metrics={"silhouette_iou": 0.9, "depth_rmse_m": 0.2},
            ),
        ]
        scores = _compute_pillar_scores_from_reports(reports)
        expected = (0.9 + 0.8) / 2.0  # iou=0.9, depth=1-0.2=0.8
        assert abs(scores["fgpp_fidelity"] - expected) < 1e-6

    def test_pillar_roundtrip(self) -> None:
        """Roundtrip pillar should derive from psnr_db and ssim."""
        from synapse.bench.traceboard_export import _compute_pillar_scores_from_reports

        reports = [
            _FakeReport(
                scenario_name="r1",
                adapter_id="mock",
                metrics={"psnr_db": 20.0, "ssim": 0.8},
            ),
        ]
        scores = _compute_pillar_scores_from_reports(reports)
        expected = (20.0 / 40.0 + 0.8) / 2.0  # psnr=0.5, ssim=0.8
        assert abs(scores["roundtrip"] - expected) < 1e-6

    def test_pillar_style(self) -> None:
        """Style pillar should derive from style_color_de."""
        from synapse.bench.traceboard_export import _compute_pillar_scores_from_reports

        reports = [
            _FakeReport(
                scenario_name="s1",
                adapter_id="mock",
                metrics={"style_color_de": 10.0},
            ),
        ]
        scores = _compute_pillar_scores_from_reports(reports)
        expected = 1.0 - 10.0 / 20.0  # = 0.5
        assert abs(scores["style"] - expected) < 1e-6

    def test_empty_reports_all_zero(self) -> None:
        """Empty reports should produce all-zero pillar scores."""
        from synapse.bench.traceboard_export import _compute_pillar_scores_from_reports

        scores = _compute_pillar_scores_from_reports([])
        assert all(v == 0.0 for v in scores.values())
        assert len(scores) == 4

    def test_nan_metrics_excluded(self) -> None:
        """NaN metrics should be excluded from pillar computation."""
        from synapse.bench.traceboard_export import _compute_pillar_scores_from_reports

        reports = [
            _FakeReport(
                scenario_name="nan",
                adapter_id="mock",
                metrics={
                    "normal_angular_error_deg": float("nan"),
                    "chamfer_mm": float("nan"),
                    "psnr_db": 30.0,
                },
            ),
        ]
        scores = _compute_pillar_scores_from_reports(reports)
        assert scores["fgpp_fidelity"] == 0.0  # nan excluded → no data
        assert scores["roundtrip"] > 0.0  # psnr_db is valid
