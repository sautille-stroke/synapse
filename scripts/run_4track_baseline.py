#!/usr/bin/env python3
"""Reproducible 4-track Synapse-Bench baseline runner & leaderboard generator (PR-TD05).

Runs all built-in Synapse-Bench scenarios across two deterministic baseline
configurations (clean ``MockVideoAdapter`` vs. defect-injected
``MockVideoAdapter``) plus real procedural Track-1 geometric probes, then
produces individual run JSONs and aggregated 4-track ``leaderboard.json`` /
``leaderboard.html`` via :func:`synapse.bench.compare.compare_runs`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from synapse.adapters.mock import DefectInjectionConfig, MockVideoAdapter
from synapse.bench.compare import compare_runs
from synapse.bench.model_loader import load_model
from synapse.bench.report import render_comparison_html
from synapse.errors import ExitCode, SynapseError
from synapse.metrics.roundtrip import chamfer_mm, f_score_1mm, sample_mesh_surface
from synapse.pipeline.bench_runner import BenchPipelineRunner
from synapse.recon.geometry import check_watertight


def _evaluate_track1_geometry(perturbation_m: float, seed: int) -> dict[str, float]:
    """Measure real Track-1 Chamfer / F-score / watertight metrics on built-in meshes."""
    rng = np.random.default_rng(seed)
    chamfer_vals: list[float] = []
    fscore_vals: list[float] = []
    watertight_vals: list[float] = []

    for model_name in ("cube", "sphere", "torus"):
        model = load_model(model_name, normalize=True)
        topo_v, topo_f = model.get_topology_mesh()
        verts = np.asarray(topo_v, dtype=np.float64)
        faces = np.asarray(topo_f, dtype=np.int64)
        gt_pts = sample_mesh_surface(verts, faces, num_samples=1024, seed=seed)

        if perturbation_m > 0.0:
            offset = rng.normal(0.0, perturbation_m, size=verts.shape)
            pred_verts = verts + offset
            # Drop one face on large perturbation to exercise non-watertight penalty
            pred_faces = faces[:-1] if perturbation_m >= 0.01 else faces
            pred_seed = seed + 1
        else:
            pred_verts = verts
            pred_faces = faces
            pred_seed = seed

        pred_pts = sample_mesh_surface(pred_verts, pred_faces, num_samples=1024, seed=pred_seed)
        chamfer_vals.append(float(chamfer_mm(gt_pts, pred_pts)))
        fscore_vals.append(float(f_score_1mm(gt_pts, pred_pts)))
        wt = check_watertight(pred_verts, pred_faces)
        watertight_vals.append(1.0 if wt else 0.0)

    return {
        "chamfer_mm": round(float(np.mean(chamfer_vals)), 6),
        "f_score_1mm": round(float(np.mean(fscore_vals)), 6),
        "is_watertight": round(float(np.mean(watertight_vals)), 6),
    }


def generate_four_track_baseline(
    *,
    output_dir: Path,
    seed: int = 42,
    resolution: int = 32,
    num_frames: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute Synapse-Bench across 2 baseline adapters and emit 4-track comparison artifacts."""
    t0 = time.monotonic()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, MockVideoAdapter, float]] = [
        ("mock_clean", MockVideoAdapter(), 0.0),
        (
            "mock_perturbed_baseline",
            MockVideoAdapter(
                defects=[
                    DefectInjectionConfig(
                        silhouette_shift_px=2,
                        flicker_amplitude=0.12,
                        kelvin_shift=900.0,
                    )
                ]
            ),
            0.012,
        ),
    ]

    run_files: list[Path] = []
    run_labels: list[str] = []
    scenario_count = 0

    for label, adapter, geom_noise_m in configs:
        runner = BenchPipelineRunner(adapter, gate_profile="track_b_video")
        if limit is not None:
            from synapse.bench.scenario import list_scenarios

            selected = list_scenarios()[:limit]
            reports = [
                runner.run_scenario(
                    s.id,
                    seed=seed,
                    resolution=(resolution, resolution),
                    num_frames=num_frames,
                )
                for s in selected
            ]
        else:
            reports = runner.run_all(
                seed=seed,
                resolution=(resolution, resolution),
                num_frames=num_frames,
            )

        scenario_count = len(reports)
        geom_metrics = _evaluate_track1_geometry(geom_noise_m, seed=seed)

        payload: dict[str, Any] = {
            "adapter_id": label,
            "seed": int(seed),
            "resolution": [int(resolution), int(resolution)],
            "num_frames": int(num_frames),
            "scenario_count": scenario_count,
            "metrics": geom_metrics,
            "reports": [r.to_dict() for r in reports],
        }
        run_path = out_dir / f"{label}.json"
        run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run_files.append(run_path)
        run_labels.append(label)

    comparison = compare_runs(run_files, labels=run_labels)
    comp_dict = comparison.to_dict()
    comp_dict["scenario_count"] = scenario_count
    comp_dict["elapsed_sec"] = round(time.monotonic() - t0, 4)

    leaderboard_json = out_dir / "leaderboard.json"
    leaderboard_html = out_dir / "leaderboard.html"
    leaderboard_json.write_text(json.dumps(comp_dict, indent=2), encoding="utf-8")
    render_comparison_html(comparison, output_path=leaderboard_html)

    return {
        "status": "ok",
        "exit_code": int(ExitCode.OK),
        "scenario_count": scenario_count,
        "overall_winner": comparison.overall_winner,
        "track_winners": comparison.track_winners,
        "run_files": [str(p) for p in run_files],
        "leaderboard_json": str(leaderboard_json),
        "leaderboard_html": str(leaderboard_html),
        "elapsed_sec": comp_dict["elapsed_sec"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for generating reproducible 4-track Synapse-Bench baseline artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "baselines" / "synapse_bench_4track",
        help="Directory to write per-model run JSONs and leaderboard artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Square frame resolution for baseline evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Frames per scenario for baseline evaluation (default: 4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional scenario count cap (default: all 50 built-in scenarios).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary JSON only to stdout.",
    )
    args = parser.parse_args(argv)

    try:
        res = generate_four_track_baseline(
            output_dir=args.output_dir,
            seed=args.seed,
            resolution=args.resolution,
            num_frames=args.num_frames,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(
                f"[4-Track Baseline] scenarios={res['scenario_count']} "
                f"winner={res['overall_winner']} out={res['leaderboard_json']}"
            )
        return int(ExitCode.OK)
    except SynapseError as exc:
        if args.json:
            print(json.dumps(exc.to_json_dict(), indent=2))
        else:
            print(f"Error [{exc.exit_code}]: {exc}", file=sys.stderr)
        return int(exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
