"""PR-192 coverage tests for aether_bridge, bench/registry, bench/report,
iter_fgpp_frames, telemetry, and SSOT doc anti-drift invariants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synapse.bench.registry import AssetRegistry, ScenarioRegistry
from synapse.bench.report import generate_html_report
from synapse.bridge import aether_bridge
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.io import export_fgpp, iter_fgpp_frames
from synapse.pipeline.bench_runner import ScenarioRunReport
from synapse.telemetry import FallbackLevel, MemoryForecaster, TraceContext

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_aether_bridge_probe_consistency() -> None:
    """aether_bridge probe booleans match module getters without raising."""
    assert aether_bridge.has_aether_geo() == (aether_bridge.get_aether_geo() is not None)
    assert aether_bridge.has_aether_metrics() == (aether_bridge.get_aether_metrics() is not None)
    assert isinstance(aether_bridge.has_aether_color(), bool)


def test_bench_registries_enumerate_builtins() -> None:
    """AssetRegistry and ScenarioRegistry resolve built-in assets and all 30 scenarios."""
    asset_reg = AssetRegistry()
    assert len(asset_reg.list_all()) >= 4

    sc_reg = ScenarioRegistry()
    all_sc = sc_reg.list_scenarios()
    assert len(all_sc) >= 30


def test_bench_report_generates_self_contained_html(tmp_path: Path) -> None:
    """generate_html_report renders SVG radar/bar charts and scenario metrics cleanly."""
    reports = [
        ScenarioRunReport(
            scenario_name="t1_rigid_cube_orbit",
            adapter_id="mock",
            metrics={
                "psnr_db": 32.5,
                "ssim": 0.94,
                "silhouette_iou": 0.96,
                "temporal_warp_error": 0.004,
                "flicker_score": 0.001,
                "border_ring_ciede2000": 1.1,
            },
            num_frames=8,
            resolution=(64, 64),
        )
    ]
    out_html = tmp_path / "bench_report.html"
    written = generate_html_report(reports, out_html)
    assert written == out_html
    text = out_html.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "t1_rigid_cube_orbit" in text


def test_iter_fgpp_frames_streams_exported_sequence(tmp_path: Path) -> None:
    """iter_fgpp_frames yields each exported FGPP frame lazily in order."""
    pytest.importorskip("OpenEXR")
    h, w = 16, 16
    frames = [
        FGPPFrame(
            camera=CameraParams(
                fx=16.0,
                fy=16.0,
                cx=8.0,
                cy=8.0,
                width=w,
                height=h,
                view_matrix=np.eye(4, dtype=np.float64),
            ),
            depth=np.full((h, w), 2.0 + i, dtype=np.float32),
            world_normal=np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=np.float16), (h, w, 3)).copy(),
            direct_radiance=np.full((h, w, 3), 0.25, dtype=np.float16),
            motion_vector=np.zeros((h, w, 2), dtype=np.float16),
            base_color=np.full((h, w, 3), 0.5, dtype=np.float16),
            roughness_metallic=np.full((h, w, 2), 0.5, dtype=np.float16),
            cryptomatte_id=np.ones((h, w), dtype=np.uint32),
            frame_index=i,
        )
        for i in range(3)
    ]
    seq = FGPPSequence(frames=frames, fps=24.0, scene_id="stream_test")
    export_fgpp(seq, tmp_path)

    streamed = list(iter_fgpp_frames(tmp_path))
    assert len(streamed) == 3
    for i, fr in enumerate(streamed):
        assert fr.depth is not None
        assert float(np.median(fr.depth)) == pytest.approx(2.0 + i, rel=1e-3)


def test_telemetry_trace_and_memory_forecaster() -> None:
    """TraceContext and MemoryForecaster produce finite structured diagnostics."""
    ctx = TraceContext.current()
    assert len(ctx.trace_id) == 32
    assert "traceparent" in ctx.to_headers()

    forecaster = MemoryForecaster()
    fc = forecaster.estimate_dit_peak_gb(num_frames=16, width=256, height=256)
    assert fc.peak_gb > 0.0
    assert isinstance(fc.fallback_level, FallbackLevel)


def test_ssot_docs_and_skills_contain_zero_csrc_geometry_references() -> None:
    """AGENTS.md, README.md, and .agents/skills/*/SKILL.md must not cite deleted csrc/geometry."""
    targets = [
        _REPO_ROOT / "AGENTS.md",
        _REPO_ROOT / "README.md",
        *sorted((_REPO_ROOT / ".agents" / "skills").glob("*/SKILL.md")),
    ]
    assert len(targets) >= 4
    offenders: list[str] = []
    for path in targets:
        content = path.read_text(encoding="utf-8")
        if "csrc/geometry" in content:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, f"Stale csrc/geometry reference found in: {offenders}"
