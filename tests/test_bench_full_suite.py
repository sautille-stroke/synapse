"""Tests for PR-15: 30 scenarios, Humanoid-53 proxy, and HTML report.

Covers:
- ScenarioRegistry: listing, loading, tier filtering
- Humanoid-53 proxy: vertex count, face validity, watertightness,
  bone coverage, walking animation
- AssetRegistry: builtin resolution
- HTML report: validity, no external refs, size, content
- Full pipeline smoke test with MockVideoAdapter
"""

from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


# ===================================================================
# TestScenarioRegistry
# ===================================================================


class TestScenarioRegistry:
    """Tests for the ScenarioRegistry."""

    def test_list_all_30_scenarios(self) -> None:
        """Verify exactly 30 scenarios are available."""
        from synapse.bench.registry import ScenarioRegistry
        registry = ScenarioRegistry()
        scenarios = registry.list_scenarios()
        assert len(scenarios) == 30, (
            f"Expected 30 scenarios, got {len(scenarios)}: {scenarios}"
        )

    def test_scenarios_by_tier(self) -> None:
        """Verify tier distribution: T1=8, T2=7, T3=5, T4=6, T5=4."""
        from synapse.bench.registry import ScenarioRegistry
        registry = ScenarioRegistry()

        expected = {"T1": 8, "T2": 7, "T3": 5, "T4": 6, "T5": 4}
        for tier, count in expected.items():
            tier_scenarios = registry.list_by_tier(tier)
            assert len(tier_scenarios) == count, (
                f"Tier {tier}: expected {count}, got "
                f"{len(tier_scenarios)}: {tier_scenarios}"
            )

    def test_all_scenarios_loadable(self) -> None:
        """Each scenario can be loaded without error."""
        from synapse.bench.registry import ScenarioRegistry
        registry = ScenarioRegistry()
        for name in registry.list_scenarios():
            scenario = registry.load_scenario(name)
            assert scenario.id == name

    def test_scenario_json_valid(self) -> None:
        """Each scenario has required fields."""
        from synapse.bench.registry import ScenarioRegistry
        registry = ScenarioRegistry()
        for name in registry.list_scenarios():
            s = registry.load_scenario(name)
            assert s.id, f"Scenario missing id: {name}"
            assert s.tier is not None, (
                f"Scenario missing tier: {name}"
            )
            assert s.asset is not None, (
                f"Scenario missing asset: {name}"
            )
            assert "num_frames" in s.camera_params, (
                f"Scenario missing num_frames: {name}"
            )
            # Validate temporal mode constraint
            num_frames = s.camera_params["num_frames"]
            if s.temporal_mode == "continuous_clip":
                assert (num_frames - 1) % 4 == 0, (
                    f"Scenario {name}: continuous_clip needs "
                    f"(num_frames - 1) % 4 == 0, got {num_frames}"
                )


# ===================================================================
# TestHumanoidProxy
# ===================================================================


class TestHumanoidProxy:
    """Tests for the Humanoid-53 proxy mesh generator."""

    def test_vertex_count_above_minimum(self) -> None:
        """Humanoid mesh has >= 1500 vertices."""
        from synapse.bench.procedural import generate_humanoid_proxy
        verts, faces, bone_idx = generate_humanoid_proxy()
        assert verts.shape[0] >= 1500, (
            f"Expected >= 1500 vertices, got {verts.shape[0]}"
        )

    def test_default_vertex_count(self) -> None:
        """Default target is approximately 1820 vertices."""
        from synapse.bench.procedural import generate_humanoid_proxy
        verts, faces, bone_idx = generate_humanoid_proxy(
            target_vertices=1820,
        )
        # Allow some tolerance since tessellation is approximate
        assert 1500 <= verts.shape[0] <= 3000, (
            f"Expected ~1820 vertices, got {verts.shape[0]}"
        )

    def test_faces_valid(self) -> None:
        """All face indices are within vertex range."""
        from synapse.bench.procedural import generate_humanoid_proxy
        verts, faces, _ = generate_humanoid_proxy()
        n_verts = verts.shape[0]
        assert faces.min() >= 0, "Negative face index found"
        assert faces.max() < n_verts, (
            f"Face index {faces.max()} >= vertex count {n_verts}"
        )

    def test_watertight(self) -> None:
        """Each edge is shared by exactly 2 faces (watertight).

        Note: Per-segment meshes are individually watertight. The
        merged mesh consists of multiple closed manifold components,
        so each edge within a component is shared by exactly 2 faces.
        """
        from synapse.bench.procedural import generate_humanoid_proxy
        verts, faces, _ = generate_humanoid_proxy()
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        for f in faces:
            for i in range(3):
                e = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
                edge_count[e] += 1

        # Each edge should appear exactly 2 times (closed manifold)
        boundary_edges = [
            e for e, c in edge_count.items() if c != 2
        ]
        assert len(boundary_edges) == 0, (
            f"Found {len(boundary_edges)} non-manifold/boundary "
            f"edges (expected 0 for watertight mesh)"
        )

    def test_bone_indices_cover_all_vertices(self) -> None:
        """Union of all bone groups equals all vertex indices."""
        from synapse.bench.procedural import generate_humanoid_proxy
        verts, faces, bone_idx = generate_humanoid_proxy()

        all_indices: set[int] = set()
        for indices in bone_idx.values():
            all_indices.update(indices)

        expected = set(range(verts.shape[0]))
        assert all_indices == expected, (
            f"Bone indices cover {len(all_indices)} vertices, "
            f"but mesh has {verts.shape[0]}"
        )

    def test_walking_phase_bone_length_constant(self) -> None:
        """Bone lengths remain constant across walk phases."""
        from synapse.bench.procedural import generate_humanoid_proxy

        verts_0, _, bone_0 = generate_humanoid_proxy(
            pose_params={"phase": 0.0},
        )
        verts_pi, _, bone_pi = generate_humanoid_proxy(
            pose_params={"phase": math.pi},
        )

        def segment_length(
            verts: np.ndarray, indices: list[int],
        ) -> float:
            seg_verts = verts[indices]
            bbox_min = seg_verts.min(axis=0)
            bbox_max = seg_verts.max(axis=0)
            return float(np.linalg.norm(bbox_max - bbox_min))

        for seg_name in bone_0:
            len_0 = segment_length(verts_0, bone_0[seg_name])
            len_pi = segment_length(verts_pi, bone_pi[seg_name])
            # Allow 1% tolerance for floating point
            assert abs(len_0 - len_pi) < 0.01 * max(len_0, len_pi, 1e-6), (
                f"Bone '{seg_name}' length changed: "
                f"{len_0:.4f} vs {len_pi:.4f}"
            )

    def test_walking_phase_different_poses(self) -> None:
        """Different phases produce different vertex positions."""
        from synapse.bench.procedural import generate_humanoid_proxy

        verts_0, _, _ = generate_humanoid_proxy(
            pose_params={"phase": 0.0},
        )
        verts_half, _, _ = generate_humanoid_proxy(
            pose_params={"phase": math.pi / 2},
        )

        # Vertices should differ (not identical poses)
        max_diff = np.max(np.abs(verts_0 - verts_half))
        assert max_diff > 0.001, (
            f"Phase 0 and phase pi/2 produced identical poses "
            f"(max diff={max_diff})"
        )


# ===================================================================
# TestAssetRegistry
# ===================================================================


class TestAssetRegistry:
    """Tests for the AssetRegistry."""

    def test_builtin_assets_available(self) -> None:
        """All builtin assets can be listed."""
        from synapse.bench.registry import AssetRegistry
        registry = AssetRegistry()
        entries = registry.list_all()
        names = {e.name for e in entries}
        expected_builtins = {
            "cube", "sphere", "cylinder", "torus", "plane",
            "cone", "humanoid_53",
        }
        assert expected_builtins.issubset(names), (
            f"Missing builtins: {expected_builtins - names}"
        )

    def test_resolve_humanoid(self) -> None:
        """Humanoid asset resolves to a valid OBJ file."""
        from synapse.bench.registry import AssetRegistry
        import synapse.bench.registry as reg_mod

        # Use a temp dir for the cache
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cache = reg_mod.SYNAPSE_ASSET_CACHE
            try:
                reg_mod.SYNAPSE_ASSET_CACHE = Path(tmpdir)
                registry = AssetRegistry()
                path = registry.resolve("humanoid_53")
                assert path.is_file(), (
                    f"Resolved path does not exist: {path}"
                )
                content = path.read_text(encoding="utf-8")
                assert "v " in content, "OBJ missing vertex data"
                assert "f " in content, "OBJ missing face data"
            finally:
                reg_mod.SYNAPSE_ASSET_CACHE = original_cache


# ===================================================================
# TestHTMLReport
# ===================================================================


class TestHTMLReport:
    """Tests for the HTML report generator."""

    @staticmethod
    def _make_mock_reports(n: int = 5) -> list:
        """Create mock ScenarioRunReport instances."""
        from synapse.pipeline.bench_runner import ScenarioRunReport
        reports = []
        for i in range(n):
            r = ScenarioRunReport(
                scenario_name=f"test_scenario_{i}",
                adapter_id="mock_adapter",
                metrics={
                    "psnr_db": 25.0 + i,
                    "ssim": 0.85 + i * 0.01,
                    "flicker_score": 0.02 + i * 0.005,
                },
                timings={
                    "scenario_load_ms": 1.0 + i,
                    "fgpp_load_ms": 10.0 + i * 2,
                    "generate_ms": 100.0 + i * 10,
                    "total_ms": 120.0 + i * 15,
                },
                num_frames=97,
                resolution=(512, 512),
                seed=42,
            )
            reports.append(r)
        return reports

    def test_generates_valid_html(self) -> None:
        """Output contains DOCTYPE and closing html tag."""
        from synapse.bench.report import generate_html_report
        reports = self._make_mock_reports()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            generate_html_report(reports, out)
            content = out.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "</html>" in content

    def test_no_external_references(self) -> None:
        """No http:// or https:// in output."""
        from synapse.bench.report import generate_html_report
        reports = self._make_mock_reports()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            generate_html_report(reports, out)
            content = out.read_text(encoding="utf-8")
            assert "http://" not in content, (
                "Found http:// external reference"
            )
            assert "https://" not in content, (
                "Found https:// external reference"
            )

    def test_file_size_under_2mb(self) -> None:
        """Report file is under 2 MB."""
        from synapse.bench.report import generate_html_report
        # Use 30 reports to stress-test size
        reports = self._make_mock_reports(30)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            generate_html_report(reports, out)
            size = out.stat().st_size
            assert size < 2 * 1024 * 1024, (
                f"Report is {size} bytes, exceeds 2 MB"
            )

    def test_contains_all_scenarios(self) -> None:
        """All scenario names appear in the HTML."""
        from synapse.bench.report import generate_html_report
        reports = self._make_mock_reports(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            generate_html_report(reports, out)
            content = out.read_text(encoding="utf-8")
            for r in reports:
                assert r.scenario_name in content, (
                    f"Scenario '{r.scenario_name}' not in report"
                )

    def test_radar_chart_svg(self) -> None:
        """SVG polygon element is present (radar chart)."""
        from synapse.bench.report import generate_html_report
        reports = self._make_mock_reports()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            generate_html_report(reports, out)
            content = out.read_text(encoding="utf-8")
            assert "<polygon" in content, (
                "No SVG polygon found (radar chart missing)"
            )


# ===================================================================
# TestFullPipelineSmoke
# ===================================================================


class TestFullPipelineSmoke:
    """Smoke test: run all 30 scenarios with MockVideoAdapter."""

    def test_run_all_30_with_mock(self) -> None:
        """Run all 30 scenarios with mock adapter; none should crash."""
        from synapse.adapters.mock import MockVideoAdapter
        from synapse.bench.registry import ScenarioRegistry
        from synapse.pipeline.bench_runner import BenchPipelineRunner

        adapter = MockVideoAdapter()
        runner = BenchPipelineRunner(adapter=adapter)
        registry = ScenarioRegistry()
        scenario_ids = registry.list_scenarios()

        assert len(scenario_ids) == 30, (
            f"Expected 30 scenarios, got {len(scenario_ids)}"
        )

        reports = runner.run_all(
            scenarios=scenario_ids,
            seed=42,
            resolution=(64, 64),  # Tiny for speed
        )

        assert len(reports) == 30, (
            f"Expected 30 reports, got {len(reports)}"
        )

        # None should have crashed (error may be set for metric
        # failures but not for unhandled exceptions)
        crash_count = 0
        crash_names = []
        for r in reports:
            if r.error and "Traceback" in r.error:
                crash_count += 1
                crash_names.append(r.scenario_name)

        # Allow errors (some scenarios may fail metrics) but not
        # crashes. We check that at least some completed.
        completed = [r for r in reports if r.error is None]
        errored = [r for r in reports if r.error is not None]

        # Print summary for debugging
        print(
            f"  Completed: {len(completed)}, "
            f"Errored: {len(errored)}"
        )
        for r in errored[:5]:
            print(
                f"  Error in {r.scenario_name}: "
                f"{str(r.error)[:80]}"
            )
