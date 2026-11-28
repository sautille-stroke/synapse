"""Tests for M52 (PR-213 ~ PR-215): consolidation, RFC-0061/0062 hardening, 4-track compare, v5.0.0."""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synapse
from synapse.bench import (
    TRACK_DEFINITIONS,
    DatasetConfig,
    DatasetGenerator,
    aggregate_four_track_scores,
    compare_runs,
    list_builtin_models,
    load_model,
    render_comparison_html,
)
from synapse.bench.dataset_visualizer import colorize_mask
from synapse.cli import main as cli_main
from synapse.errors import ContractValidationError
from synapse.recon.loop import ClosedLoopRunner, LoopConfig
from synapse.refine.pipeline import LocalRefinementPipeline
from synapse.refine.types import AnatomicalRegion, RefinementROI


def test_pr213_consolidation_zero_deprecation_warning() -> None:
    """PR-213: consolidated bench modules work cleanly and legacy shims emit zero DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        gen_pkg = importlib.import_module("synapse.generators")
        gen_ds = importlib.import_module("synapse.generators.dataset")
        gen_vis = importlib.import_module("synapse.generators.visualizer")

        assert gen_pkg.DatasetConfig is DatasetConfig
        assert gen_pkg.DatasetGenerator is DatasetGenerator
        assert gen_ds.DatasetConfig is DatasetConfig
        assert gen_ds.DatasetGenerator is DatasetGenerator
        assert gen_vis.colorize_mask is colorize_mask

    assert "cube" in list_builtin_models()
    model = load_model("cube", normalize=True)
    assert model.num_vertices > 0


def test_pr214_rfc0061_foundary_k2_and_rfc0062_kappa_hard_cut() -> None:
    """PR-214: LocalRefinementPipeline dispatches Foundary K=2; ClosedLoopRunner tracks kappa boundary."""
    frames = np.full((2, 64, 64, 3), 0.3, dtype=np.float32)
    frames[:, 16:48, 16:48, :] = 0.7
    roi = RefinementROI(
        bbox_xyxy=np.array([16.0, 16.0, 48.0, 48.0], dtype=np.float32),
        crop_acescg=np.full((32, 32, 3), 0.4, dtype=np.float32),
        region=AnatomicalRegion.HAND,
        confidence=0.9,
        anatomical_quality_score=0.4,
        frame_index=0,
    )

    calls: list[str] = []

    class _MockFoundaryK2:
        @staticmethod
        def infer(service: str, inputs: dict[str, Any] | None = None) -> Any:
            _ = inputs
            calls.append(service)
            delta = 0.01 if service == "flux" else 0.02
            patch = np.full((32, 32, 3), 0.65 + delta, dtype=np.float32)
            return type("Resp", (), {"outputs": {"refined_patch": patch}})()

    pipe = LocalRefinementPipeline(strict=False)
    ref_res = pipe.run(
        frames,
        [[roi], []],
        foundary_client=_MockFoundaryK2(),
    )
    assert calls == ["flux", "sdxl"]
    assert ref_res.num_patches_fused == 1

    # RFC 0062 ClosedLoopRunner kappa hard-cut telemetry
    rng = np.random.default_rng(214)
    src = rng.uniform(0.2, 0.8, size=(5, 24, 24, 3)).astype(np.float32)
    nrms = np.zeros((5, 24, 24, 3), dtype=np.float32)
    nrms[..., 2] = 1.0
    uvs = np.zeros((5, 24, 24, 2), dtype=np.float32)
    u = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    v = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    uvs[..., 0] = u[None, None, :]
    uvs[..., 1] = v[None, :, None]

    runner = ClosedLoopRunner(loop_config=LoopConfig(n_preroll=4, kappa=1.0, max_iterations=3))
    cl_res = runner.run(src, nrms, uvs)
    assert cl_res.kappa_applied == pytest.approx(1.0)
    assert cl_res.to_dict()["kappa_applied"] == pytest.approx(1.0)
    assert cl_res.preroll_bleed_delta >= 0.0


def test_pr215_four_track_compare_html_cli_and_v5_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PR-215: 4-track leaderboard aggregator, XSS-safe HTML, bench compare CLI, and v5.0.0."""
    assert len(TRACK_DEFINITIONS) == 4
    major, minor, patch = (int(x) for x in synapse.__version__.split(".")[:3])
    assert (major, minor, patch) >= (5, 0, 0)

    run_a = {
        "adapter_id": "wan21_synapse",
        "reports": [
            {
                "scenario_name": "s1",
                "metrics": {
                    "chamfer_mm": 1.2,
                    "f_score_1mm": 0.94,
                    "uv_stretch": 0.12,
                    "is_watertight": 1.0,
                    "fvd": 18.0,
                    "temporal_warp_error": 0.004,
                    "foot_sliding_cm": 0.15,
                    "bone_length_drift_pct": 0.4,
                    "volume_breathing_pct": 0.8,
                    "silhouette_iou": 0.96,
                    "depth_rmse_m": 0.015,
                    "border_ring_ciede2000": 1.1,
                    "albedo_delta_e00": 1.4,
                    "light_dir_error_deg": 4.2,
                },
            }
        ],
    }
    run_b = {
        "adapter_id": "baseline<script>alert(1)</script>",
        "reports": [
            {
                "scenario_name": "s1",
                "metrics": {
                    "chamfer_mm": 8.5,
                    "f_score_1mm": 0.72,
                    "uv_stretch": 0.65,
                    "is_watertight": 0.0,
                    "fvd": 95.0,
                    "temporal_warp_error": 0.025,
                    "foot_sliding_cm": 0.62,
                    "bone_length_drift_pct": 2.4,
                    "volume_breathing_pct": 4.5,
                    "silhouette_iou": 0.81,
                    "depth_rmse_m": 0.06,
                    "border_ring_ciede2000": 3.4,
                    "albedo_delta_e00": 3.8,
                    "light_dir_error_deg": 18.0,
                },
            }
        ],
    }

    report = aggregate_four_track_scores({"wan21_synapse": run_a, "baseline": run_b})
    assert report.overall_winner == "wan21_synapse"
    assert all(winner == "wan21_synapse" for winner in report.track_winners.values())

    file_a = tmp_path / "run_a.json"
    file_b = tmp_path / "run_b.json"
    file_a.write_text(json.dumps(run_a), encoding="utf-8")
    file_b.write_text(json.dumps(run_b), encoding="utf-8")

    # Reject < 2 runs fail-closed with ContractValidationError (exit 10)
    with pytest.raises(ContractValidationError) as e10:
        compare_runs([file_a])
    assert e10.value.exit_code == 10

    loaded_rep = compare_runs([file_a, file_b])
    html_out = tmp_path / "leaderboard.html"
    json_out = tmp_path / "leaderboard.json"
    html_doc = render_comparison_html(loaded_rep, output_path=html_out)
    assert "<script>alert(1)</script>" not in html_doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_doc
    assert "<svg" in html_doc

    capsys.readouterr()
    rc = cli_main(
        [
            "bench",
            "compare",
            "--inputs",
            str(file_a),
            str(file_b),
            "--output-html",
            str(html_out),
            "--output-json",
            str(json_out),
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "ok"
    assert cli_payload["overall_winner"] == "wan21_synapse"
    assert html_out.is_file()
    assert json_out.is_file()
