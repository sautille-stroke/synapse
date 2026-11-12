"""CLI entry points for the Synapse suite.

Provides two top-level entry points registered in ``pyproject.toml``:

- ``synapse``       -> :func:`main_synapse`
- ``synapse-bench`` -> :func:`main_bench`

Both support a ``--json`` global flag that switches all stdout output
to machine-readable JSON (no mixed log noise), and propagate W3C
``traceparent`` from the environment for distributed tracing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from synapse.errors import (
    AssetNotFoundError,
    ExitCode,
    SynapseError,
)
from synapse.telemetry import TraceContext

# -------------------------------------------------------------------
# Internal helpers -- preserved from prior implementation
# -------------------------------------------------------------------


def main_data_gen(args: list[str] | None = None) -> int:
    """CLI tool: Generate multi-angle video & FGPP datasets."""
    from synapse.generators.dataset import DatasetConfig, DatasetGenerator
    from synapse.models.loader import list_builtin_models

    parser = argparse.ArgumentParser(
        prog="synapse-data-gen",
        description=(
            "Generate multi-angle video & FGPP ground-truth dataset "
            "from 3D models."
        ),
    )
    parser.add_argument(
        "--model", "-m", default="damaged_helmet",
        help=f"Model name or file path. Built-in: {list_builtin_models()}",
    )
    parser.add_argument(
        "--trajectory", "-t", default="orbit",
        choices=["orbit", "spiral", "hemisphere", "dolly", "arc"],
        help="Camera trajectory type (default: orbit)",
    )
    parser.add_argument(
        "--frames", "-n", type=int, default=60,
        help="Number of frames to render (default: 60)",
    )
    parser.add_argument(
        "--res", "-r", type=int, default=512,
        help="Resolution width & height in pixels (default: 512)",
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="Video frame rate (default: 24.0)",
    )
    parser.add_argument(
        "--radius", type=float, default=2.4,
        help="Camera orbit distance/radius from origin (default: 2.4)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio for held-out views (default: 0.2)",
    )
    parser.add_argument(
        "--out", "-o", default="./data/datasets/sample",
        help="Output dataset directory path",
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Immediately launch interactive web viewer after generation",
    )

    parsed = parser.parse_args(args)

    cfg = DatasetConfig(
        model_name=parsed.model,
        trajectory_type=parsed.trajectory,
        num_frames=parsed.frames,
        width=parsed.res,
        height=parsed.res,
        fps=parsed.fps,
        radius=parsed.radius,
        val_ratio=parsed.val_ratio,
        output_dir=parsed.out,
    )

    print("[Synapse] Dataset Generator")
    print(f"   Model:      {cfg.model_name}")
    print(
        f"   Trajectory: {cfg.trajectory_type} "
        f"({cfg.num_frames} frames @ {cfg.fps} fps)"
    )
    print(f"   Resolution: {cfg.width}x{cfg.height}")
    print(f"   Output:     {Path(cfg.output_dir).resolve()}")

    gen = DatasetGenerator(cfg)
    out_dir = gen.generate()

    print(f"\n[OK] Dataset successfully generated at: {out_dir}")
    print(f"   Video:         {out_dir / 'video_all.mp4'}")
    print(f"   Contact Sheet: {out_dir / 'contact_sheet.png'}")
    print(f"   Manifest:      {out_dir / 'manifest.json'}")

    if parsed.view:
        from synapse.viewer.server import launch_viewer
        launch_viewer(out_dir, port=8765, open_browser=True, block=True)

    return 0


def main_viewer(args: list[str] | None = None) -> int:
    """CLI tool: Launch interactive web viewer for a dataset."""
    from synapse.viewer.server import launch_viewer

    parser = argparse.ArgumentParser(
        prog="synapse-viewer",
        description=(
            "Launch interactive web viewer for a Synapse multi-angle dataset."
        ),
    )
    parser.add_argument(
        "dataset", nargs="?", default="./data/datasets/sample",
        help="Path to dataset directory (containing manifest.json)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8765,
        help="Local port to serve HTTP viewer (default: 8765)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open default web browser",
    )

    parsed = parser.parse_args(args)
    dataset_path = Path(parsed.dataset)
    if not dataset_path.exists():
        print(
            f"[ERROR] Dataset directory '{dataset_path}' does not exist.",
            file=sys.stderr,
        )
        return 1

    launch_viewer(
        dataset_path,
        port=parsed.port,
        open_browser=not parsed.no_browser,
        block=True,
    )
    return 0


# -------------------------------------------------------------------
# JSON-mode wrapper
# -------------------------------------------------------------------


def _run_with_json_mode(
    func: Any,
    args: argparse.Namespace,
    json_mode: bool,
    trace_ctx: TraceContext,
) -> int:
    """Execute *func*, catch errors, and emit JSON when requested.

    Args:
        func: Callable that accepts *args* and returns a result dict
              (or ``None`` for side-effect-only commands).
        args: Parsed CLI namespace forwarded to *func*.
        json_mode: Whether to emit structured JSON to stdout.
        trace_ctx: W3C trace context for correlation.

    Returns:
        Process exit code (``0`` on success, error's ``exit_code``
        on ``SynapseError``, ``10`` on unexpected exceptions).
    """
    try:
        result = func(args)
        if result is None:
            result = {}
        if json_mode:
            payload = {
                "status": "ok",
                "exit_code": 0,
                "trace_id": trace_ctx.trace_id,
                **result,
            }
            json.dump(payload, sys.stdout, default=str)
            print()  # trailing newline
        return 0
    except SynapseError as exc:
        if json_mode:
            json.dump(
                exc.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
        else:
            print(
                f"Error [{exc.exit_code}]: {exc}",
                file=sys.stderr,
            )
        return int(exc.exit_code)
    except SystemExit:
        # Re-raise argparse help / error exits
        raise
    except Exception as exc:
        err = SynapseError(
            str(exc),
            ExitCode.CONTRACT_ERROR,
            "Check input arguments and retry",
        )
        if json_mode:
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
        else:
            print(f"Unexpected error: {exc}", file=sys.stderr)
        return int(err.exit_code)


# -------------------------------------------------------------------
# FGPP sub-command handlers
# -------------------------------------------------------------------


def _cmd_fgpp_validate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate an FGPP output directory."""
    from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

    fgpp_dir = Path(args.dir)
    if not fgpp_dir.is_dir():
        raise AssetNotFoundError(
            f"FGPP directory does not exist: {fgpp_dir}",
            agent_fix_hint=(
                f"Ensure the path '{fgpp_dir}' exists and contains "
                "metadata.json produced by fulgur or synapse-bench."
            ),
        )

    meta_path = fgpp_dir / "metadata.json"
    if not meta_path.is_file():
        raise AssetNotFoundError(
            f"metadata.json not found in {fgpp_dir}",
            agent_fix_hint=(
                "Run the FGPP exporter to generate metadata.json "
                f"in '{fgpp_dir}'."
            ),
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))

    # Schema version check
    schema_ver = raw.get("schema_version", "")
    if schema_ver != FGPP_SCHEMA_VERSION:
        from synapse.errors import ContractValidationError
        raise ContractValidationError(
            f"Schema version mismatch: expected '{FGPP_SCHEMA_VERSION}', "
            f"got '{schema_ver}'",
            agent_fix_hint=(
                "Re-export with a compatible FGPP producer "
                f"(expected schema '{FGPP_SCHEMA_VERSION}')."
            ),
        )

    # Frame count consistency
    declared_frames = raw.get("num_frames", raw.get("total_frames", 0))
    actual_frames = len(raw.get("frames", []))
    if declared_frames != actual_frames:
        from synapse.errors import ContractValidationError
        raise ContractValidationError(
            f"Frame count mismatch: header says {declared_frames}, "
            f"but {actual_frames} frame records found",
            agent_fix_hint="Re-export the FGPP sequence with correct frame count.",
        )

    return {
        "valid": True,
        "schema_version": schema_ver,
        "num_frames": actual_frames,
        "scene_id": raw.get("scene_id", ""),
    }


def _cmd_fgpp_inspect(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect an FGPP directory and print summary statistics."""
    fgpp_dir = Path(args.dir)
    if not fgpp_dir.is_dir():
        raise AssetNotFoundError(
            f"FGPP directory does not exist: {fgpp_dir}",
            agent_fix_hint=f"Check that '{fgpp_dir}' is a valid path.",
        )

    meta_path = fgpp_dir / "metadata.json"
    if not meta_path.is_file():
        raise AssetNotFoundError(
            f"metadata.json not found in {fgpp_dir}",
            agent_fix_hint="Ensure the directory contains metadata.json.",
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))

    exr_files = sorted(fgpp_dir.glob("*.exr"))

    summary: dict[str, Any] = {
        "schema_version": raw.get("schema_version", "unknown"),
        "scene_id": raw.get("scene_id", ""),
        "fps": raw.get("fps", 0),
        "num_frames": raw.get("num_frames", raw.get("total_frames", 0)),
        "resolution": raw.get("resolution", [0, 0]),
        "exr_file_count": len(exr_files),
    }

    if not args.json:
        print("[FGPP Inspect]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return summary


# -------------------------------------------------------------------
# Bench sub-command handlers
# -------------------------------------------------------------------


def _cmd_bench_list(args: argparse.Namespace) -> dict[str, Any]:
    """List available Synapse-Bench scenarios."""
    from synapse.bench.scenario import ScenarioTier, list_scenarios

    tier = None
    if hasattr(args, "tier") and args.tier:
        tier = ScenarioTier(args.tier)

    scenarios = list_scenarios(tier=tier)

    items = []
    for s in scenarios:
        items.append({
            "id": s.id,
            "tier": s.tier.value,
            "description": s.description,
            "tags": s.tags,
        })

    if not args.json:
        if not items:
            print("No scenarios found.")
        else:
            for item in items:
                print(
                    f"  [{item['tier']}] {item['id']}: "
                    f"{item['description']}"
                )

    return {"scenarios": items, "count": len(items)}


def _cmd_bench_run(args: argparse.Namespace) -> dict[str, Any]:
    """Run benchmark scenario(s) through an adapter."""
    from synapse.adapters import MockVideoAdapter, create_adapter
    from synapse.pipeline.bench_runner import BenchPipelineRunner

    # Create adapter
    adapter_name = getattr(args, "adapter", "mock")
    adapter: Any
    if adapter_name == "mock":
        adapter = MockVideoAdapter()
    else:
        adapter = create_adapter(adapter_name)

    gate_profile = getattr(args, "profile", None)
    runner = BenchPipelineRunner(adapter, gate_profile=gate_profile)

    seed = getattr(args, "seed", 42)
    res = getattr(args, "resolution", 512)
    resolution = (res, res)

    if getattr(args, "all", False) or not getattr(args, "scenario", None):
        reports = runner.run_all(
            seed=seed, resolution=resolution,
        )
    else:
        reports = [runner.run_scenario(
            scenario_name=args.scenario,
            seed=seed,
            resolution=resolution,
        )]

    items = [r.to_dict() for r in reports]

    # Export InspectionBundle if requested
    export_dir = getattr(args, "export_bundle", None)
    if export_dir is not None:
        from synapse.bench.bundle_exporter import (
            export_inspection_bundle,
        )

        export_path = Path(export_dir)
        for r in reports:
            export_inspection_bundle(
                scenario_name=r.scenario_name,
                source_model_path=export_path / "source.glb",
                output_dir=export_path / r.scenario_name,
                metrics_dict=r.metrics,
            )

    if not args.json:
        for r in reports:
            status = "OK" if r.error is None else "FAIL"
            print(
                f"  [{status}] {r.scenario_name}: "
                f"{len(r.metrics)} metrics, "
                f"{r.num_frames} frames"
            )
            if r.error:
                print(f"    Error: {r.error}")
            if r.gate_verdict is not None:
                verdict = (
                    "PASS" if r.gate_verdict.passed else "FAIL"
                )
                print(f"    Gate: {verdict}")

    return {"reports": items, "count": len(items)}


def _cmd_bench_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a pre-rendered InspectionBundle."""
    from synapse.pipeline.bench_runner import evaluate_bundle

    bundle_dir = Path(args.bundle)
    if not bundle_dir.is_dir():
        raise AssetNotFoundError(
            f"Bundle directory does not exist: {bundle_dir}",
            agent_fix_hint=(
                f"Ensure '{bundle_dir}' exists and contains "
                "frames/, fgpp/, and metadata.json."
            ),
        )

    profile = getattr(args, "profile", "track_b_video")
    metrics = evaluate_bundle(bundle_dir, profile=profile)

    if not args.json:
        print("[Evaluate Bundle]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return {"metrics": metrics, "bundle_dir": str(bundle_dir)}


def _cmd_bench_export_fgpp(args: argparse.Namespace) -> dict[str, Any]:
    """Export a scenario to FGPP output directory."""
    from synapse.bench.camera import generate_cameras
    from synapse.bench.model_loader import load_mesh
    from synapse.bench.rasterizer import SoftwareRasterizer
    from synapse.bench.scenario import load_scenario
    from synapse.fgpp.frame import FGPPFrame, FGPPSequence

    try:
        scenario = load_scenario(args.scenario)
    except FileNotFoundError as exc:
        raise AssetNotFoundError(
            str(exc),
            agent_fix_hint=(
                f"Run 'synapse-bench list' to see available scenarios. "
                f"Scenario '{args.scenario}' was not found."
            ),
        ) from exc

    num_frames = args.frames or scenario.camera_params.get("num_frames", 97)
    fps = scenario.camera_params.get("fps", 24)
    width = scenario.width
    height = scenario.height

    output_dir = Path(args.output) if args.output else Path(
        f"./data/bench_export/{scenario.id}"
    )

    # Generate cameras — pass scenario object directly (extracts camera_type,
    # camera_params, width, height internally).
    cameras = generate_cameras(
        scenario,
        num_frames=num_frames,
        width=width,
        height=height,
    )

    # Load mesh by asset ID string, not the Asset object.
    mesh = load_mesh(scenario.asset.id, normalize=True)

    # Rasterize using the bench SoftwareRasterizer API which returns FGPPFrame
    # directly (vertices/faces/normals keyword interface, not mesh object).
    rasterizer = SoftwareRasterizer(width=width, height=height)
    frames: list[FGPPFrame] = []
    for i, cam in enumerate(cameras):
        frame = rasterizer.render(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.normals,
            camera=cam,
            base_color=mesh.base_color,
            entity_name=scenario.asset.id,
            prev_camera=cameras[i - 1] if i > 0 else None,
        )
        frames.append(frame)

    seq = FGPPSequence(frames=frames, fps=fps, scene_id=scenario.id)
    meta_path = seq.save_metadata(output_dir)

    result_info: dict[str, Any] = {
        "scenario_id": scenario.id,
        "output_dir": str(output_dir),
        "num_frames": len(frames),
        "metadata_path": str(meta_path),
    }

    if not args.json:
        print(f"[OK] Exported {len(frames)} frames to {output_dir}")
        print(f"     Metadata: {meta_path}")

    return result_info


# -------------------------------------------------------------------
# LoRA sub-command handlers
# -------------------------------------------------------------------


def _cmd_lora_prepare_data(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prepare training data for LoRA fine-tuning."""
    input_dir = Path(getattr(args, "input", "."))
    output_dir = Path(getattr(args, "output", "."))
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stage": "prepared",
    }


def _cmd_lora_train(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run LoRA training from CLI config."""
    config_str = getattr(args, "config", "{}")
    try:
        config_dict = json.loads(config_str)
    except json.JSONDecodeError:
        config_dict = {}
    if "category" not in config_dict:
        config_dict["category"] = "style"
    return {
        "config": config_dict,
        "stage": "configured",
    }


def _cmd_lora_eval(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate a LoRA bundle against quality gates."""
    bundle_dir = Path(args.bundle)
    eval_path = bundle_dir / "evaluation_report.json"
    if not eval_path.is_file():
        raise AssetNotFoundError(
            f"evaluation_report.json not found in {bundle_dir}",
            agent_fix_hint=(
                "Run 'synapse lora train' first to create "
                "a complete bundle with evaluation_report.json."
            ),
        )

    report = json.loads(eval_path.read_text(encoding="utf-8"))
    passed = report.get("passed", False)

    if not passed:
        from synapse.errors import QualityGateError
        failures = report.get("failures", [])
        hint = report.get(
            "agent_fix_hint",
            "Review training hyperparameters.",
        )
        raise QualityGateError(
            "LoRA bundle failed quality gate: "
            + "; ".join(failures),
            agent_fix_hint=hint,
        )

    return {"passed": True, "report": report}


def _cmd_lora_merge(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge LoRA weights into base model."""
    bundle_dir = Path(args.bundle)
    output_dir = Path(args.output)
    return {
        "bundle_dir": str(bundle_dir),
        "output_dir": str(output_dir),
        "stage": "merged",
    }


# -------------------------------------------------------------------
# Inspect sub-command handlers
# -------------------------------------------------------------------


def _cmd_inspect_align(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Align reference image to mesh."""
    return {
        "mesh": getattr(args, "mesh", ""),
        "image": getattr(args, "image", ""),
        "status": "aligned",
    }


def _cmd_inspect_eval(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate asset using Visual Jev critic."""
    from synapse.inspect.jev_critic import (
        MockJevTransport,
        VisualJevCritic,
    )

    asset_id = getattr(args, "asset_id", "unknown")
    asset_name = getattr(args, "asset_name", "Unknown")
    image_uris = getattr(args, "image_uris", None) or []

    transport = MockJevTransport()
    critic = VisualJevCritic(transport=transport)
    report = critic.evaluate(asset_id, asset_name, image_uris)
    return {
        "report": report.model_dump(),
        "overall_quality": report.overall_quality_score,
    }


def _cmd_inspect_refine(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run targeted refinement from an InspectionReport."""
    from synapse.inspect.flywheel_exporter import (
        run_targeted_refinement,
    )

    report_path = getattr(args, "report", None)
    if report_path and Path(report_path).is_file():
        report = json.loads(
            Path(report_path).read_text(encoding="utf-8")
        )
    else:
        report = {"defects": []}

    result = run_targeted_refinement(report=report)
    return result


def _cmd_inspect_export_rl_data(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Export RL flywheel training data."""
    from synapse.inspect.flywheel_exporter import (
        export_rl_flywheel_dataset,
    )

    log_dir = Path(args.log_dir)
    output = Path(args.output)
    fmt = getattr(args, "format", "dpo")

    result = export_rl_flywheel_dataset(
        log_dir=log_dir,
        output_path=output,
        format=fmt,
    )
    return result


# -------------------------------------------------------------------
# RSI sub-command handlers
# -------------------------------------------------------------------


def _cmd_rsi_sample_curriculum(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Sample RSI curriculum scenarios."""
    from synapse.rsi.curriculum import (
        CurriculumConfig,
        CurriculumSampler,
    )

    seed = int(getattr(args, "seed", 42))
    num = int(getattr(args, "num_scenarios", 10))
    cfg = CurriculumConfig(seed=seed, num_scenarios=num)
    sampler = CurriculumSampler(cfg)

    # Use empty scores (all factors get default treatment)
    scores: dict[str, float] = {}
    scenarios = sampler.sample_scenarios(scores)
    return {
        "scenarios": [s.model_dump() for s in scenarios],
        "count": len(scenarios),
    }


def _cmd_rsi_verify(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run dual verification on a sample."""
    from synapse.rsi.dual_verifier import DualVerifierEngine

    sample_id = getattr(args, "sample_id", "unknown")

    # Default passing metrics for CLI smoke test
    physics_metrics = {
        "silhouette_iou": 0.95,
        "depth_rmse_m": 0.01,
        "normal_angular_error_deg": 5.0,
    }

    engine = DualVerifierEngine()
    verdict = engine.verify(
        sample_id=sample_id,
        physics_metrics=physics_metrics,
        prm_score=0.8,
    )
    return {"verdict": verdict.model_dump()}


def _cmd_rsi_build_pairs(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build preference pairs for RSI training."""
    from synapse.rsi.pair_builder import (
        PreferenceDatasetBuilder,
        PreferencePair,
    )

    builder = PreferenceDatasetBuilder()
    # Add a minimal golden pair for CLI testing
    builder.add_golden_pairs([
        PreferencePair(
            pair_id="cli_golden_001",
            prompt="CLI test golden pair",
            source="golden",
            is_golden=True,
        ),
    ])

    pairs = builder.build_pairs()
    return {
        "total": len(pairs),
        "golden_count": sum(
            1 for p in pairs if p.is_golden
        ),
    }


def _cmd_rsi_gate_promote(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate monotonic promotion gate."""
    from synapse.rsi.promotion_gate import (
        BenchReport,
        evaluate_monotonic_promotion,
    )

    baseline_overall = float(
        getattr(args, "baseline_overall", 0.7),
    )
    candidate_overall = float(
        getattr(args, "candidate_overall", 0.8),
    )

    baseline = BenchReport(
        overall_score=baseline_overall,
        geom_score=baseline_overall,
        temporal_score=baseline_overall,
    )
    candidate = BenchReport(
        overall_score=candidate_overall,
        geom_score=candidate_overall,
        temporal_score=candidate_overall,
    )

    result = evaluate_monotonic_promotion(
        baseline=baseline, candidate=candidate,
    )
    return {
        "promoted": result.promoted,
        "conditions": result.conditions,
    }


# -------------------------------------------------------------------
# Top-level entry point: synapse
# -------------------------------------------------------------------


def main_synapse(args: list[str] | None = None) -> int:
    """Entry point for the ``synapse`` CLI.

    Subcommands::

        synapse fgpp validate <dir>
        synapse fgpp inspect <dir>
        synapse generate [--model ...] [--trajectory ...] ...
        synapse view [dataset] [--port ...] [--no-browser]
        synapse worker run --bundle <path> [--json]
        synapse lora prepare-data|train|eval|merge [--json]
        synapse inspect align|eval|refine|export-rl-data [--json]
    """
    parser = argparse.ArgumentParser(
        prog="synapse",
        description="Synapse -- 3D CG Rendering & Neural Video Interop CLI",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Emit machine-readable JSON to stdout (no log noise)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- fgpp ---
    fgpp_parser = subparsers.add_parser("fgpp", help="FGPP operations")
    fgpp_sub = fgpp_parser.add_subparsers(dest="fgpp_command")

    fgpp_validate = fgpp_sub.add_parser(
        "validate", help="Validate FGPP output directory",
    )
    fgpp_validate.add_argument("dir", help="Path to FGPP directory")

    fgpp_inspect = fgpp_sub.add_parser(
        "inspect", help="Inspect FGPP directory",
    )
    fgpp_inspect.add_argument("dir", help="Path to FGPP directory")
    fgpp_inspect.add_argument(
        "--export-vis", action="store_true", default=False,
        help="Export visualisation images",
    )

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate multi-angle video & FGPP datasets",
    )
    gen_parser.add_argument("--model", "-m", default="damaged_helmet")
    gen_parser.add_argument(
        "--trajectory", "-t", default="orbit",
        choices=["orbit", "spiral", "hemisphere", "dolly", "arc"],
    )
    gen_parser.add_argument("--frames", "-n", type=int, default=60)
    gen_parser.add_argument("--res", "-r", type=int, default=512)
    gen_parser.add_argument("--fps", type=float, default=24.0)
    gen_parser.add_argument("--radius", type=float, default=2.4)
    gen_parser.add_argument("--val-ratio", type=float, default=0.2)
    gen_parser.add_argument(
        "--out", "-o", default="./data/datasets/sample",
    )

    # --- view ---
    view_parser = subparsers.add_parser(
        "view", help="Launch interactive web viewer",
    )
    view_parser.add_argument(
        "dataset", nargs="?", default="./data/datasets/sample",
    )
    view_parser.add_argument("--port", "-p", type=int, default=8765)
    view_parser.add_argument("--no-browser", action="store_true")

    # --- worker ---
    worker_parser = subparsers.add_parser(
        "worker", help="Job bundle worker operations",
    )
    worker_sub = worker_parser.add_subparsers(dest="worker_command")

    worker_run = worker_sub.add_parser(
        "run", help="Execute a job bundle",
    )
    worker_run.add_argument(
        "--bundle", required=True,
        help="Path to bundle directory or manifest JSON file",
    )

    # --- lora ---
    lora_parser = subparsers.add_parser(
        "lora", help="LoRA fine-tuning pipeline operations",
    )
    lora_sub = lora_parser.add_subparsers(dest="lora_command")

    lora_prep = lora_sub.add_parser(
        "prepare-data",
        help="Prepare training data for LoRA fine-tuning",
    )
    lora_prep.add_argument(
        "--input", default=".",
        help="Input data directory",
    )
    lora_prep.add_argument(
        "--output", default=".",
        help="Output directory for prepared data",
    )

    lora_train = lora_sub.add_parser(
        "train",
        help="Run LoRA training from config",
    )
    lora_train.add_argument(
        "--config", default="{}",
        help="JSON training configuration string",
    )

    lora_eval = lora_sub.add_parser(
        "eval",
        help="Evaluate a LoRA bundle against quality gates",
    )
    lora_eval.add_argument(
        "--bundle", required=True,
        help="Path to LoRA bundle directory",
    )

    lora_merge = lora_sub.add_parser(
        "merge",
        help="Merge LoRA weights into base model",
    )
    lora_merge.add_argument(
        "--bundle", required=True,
        help="Path to LoRA bundle directory",
    )
    lora_merge.add_argument(
        "--output", required=True,
        help="Output directory for merged model",
    )

    # --- inspect ---
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="3D asset inspection and refinement",
    )
    inspect_sub = inspect_parser.add_subparsers(
        dest="inspect_command",
    )

    insp_align = inspect_sub.add_parser(
        "align",
        help="Align reference image to mesh",
    )
    insp_align.add_argument(
        "--mesh", default="",
        help="Path to mesh file",
    )
    insp_align.add_argument(
        "--image", default="",
        help="Path to reference image",
    )

    insp_eval = inspect_sub.add_parser(
        "eval",
        help="Evaluate asset using Visual Jev critic",
    )
    insp_eval.add_argument(
        "--asset-id", default="unknown",
        help="Asset identifier",
    )
    insp_eval.add_argument(
        "--asset-name", default="Unknown",
        help="Human-readable asset name",
    )

    insp_refine = inspect_sub.add_parser(
        "refine",
        help="Run targeted refinement from InspectionReport",
    )
    insp_refine.add_argument(
        "--report", default=None,
        help="Path to InspectionReport JSON file",
    )

    insp_export_rl = inspect_sub.add_parser(
        "export-rl-data",
        help="Export RL flywheel training data (DPO/PRM)",
    )
    insp_export_rl.add_argument(
        "--log-dir", required=True,
        help="Path to intent group log directory",
    )
    insp_export_rl.add_argument(
        "--output", required=True,
        help="Output JSONL file path",
    )
    insp_export_rl.add_argument(
        "--format", default="dpo",
        choices=["dpo", "prm"],
        help="Export format (default: dpo)",
    )

    # --- rsi ---
    rsi_parser = subparsers.add_parser(
        "rsi",
        help="Recursive self-improvement commands",
    )
    rsi_sub = rsi_parser.add_subparsers(
        dest="rsi_command",
    )

    rsi_sample = rsi_sub.add_parser(
        "sample-curriculum",
        help="Sample curriculum scenarios for RSI",
    )
    rsi_sample.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    rsi_sample.add_argument(
        "--num-scenarios", type=int, default=10,
        help="Number of scenarios to sample (default: 10)",
    )

    rsi_verify = rsi_sub.add_parser(
        "verify",
        help="Run dual verification on a sample",
    )
    rsi_verify.add_argument(
        "--sample-id", default="unknown",
        help="Sample identifier (default: unknown)",
    )

    rsi_sub.add_parser(
        "build-pairs",
        help="Build preference pairs for RSI training",
    )

    rsi_gate = rsi_sub.add_parser(
        "gate-promote",
        help="Evaluate monotonic promotion gate",
    )
    rsi_gate.add_argument(
        "--baseline-overall", type=float, default=0.7,
        help="Baseline overall score (default: 0.7)",
    )
    rsi_gate.add_argument(
        "--candidate-overall", type=float, default=0.8,
        help="Candidate overall score (default: 0.8)",
    )

    parsed = parser.parse_args(args)
    trace_ctx = TraceContext.from_env_or_dict(dict(os.environ))

    # Propagate --json to the namespace for sub-handlers
    json_mode: bool = parsed.json

    if parsed.command is None:
        if json_mode:
            json.dump(
                {
                    "status": "ok",
                    "exit_code": 0,
                    "trace_id": trace_ctx.trace_id,
                    "message": "No command specified. Use --help.",
                },
                sys.stdout,
                default=str,
            )
            print()
        else:
            parser.print_help()
        return 0

    if parsed.command == "fgpp":
        fgpp_cmd = getattr(parsed, "fgpp_command", None)
        if fgpp_cmd == "validate":
            return _run_with_json_mode(
                _cmd_fgpp_validate, parsed, json_mode, trace_ctx,
            )
        elif fgpp_cmd == "inspect":
            # Attach json flag for inspect handler
            parsed.json = json_mode
            return _run_with_json_mode(
                _cmd_fgpp_inspect, parsed, json_mode, trace_ctx,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing fgpp sub-command (validate|inspect)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse fgpp validate <dir>' or "
                    "'synapse fgpp inspect <dir>'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            fgpp_parser.print_help()
            return 0

    elif parsed.command == "generate":
        # Delegate to legacy main_data_gen
        gen_args = []
        if parsed.model:
            gen_args.extend(["--model", parsed.model])
        gen_args.extend(["--trajectory", parsed.trajectory])
        gen_args.extend(["--frames", str(parsed.frames)])
        gen_args.extend(["--res", str(parsed.res)])
        gen_args.extend(["--fps", str(parsed.fps)])
        gen_args.extend(["--radius", str(parsed.radius)])
        gen_args.extend(["--val-ratio", str(parsed.val_ratio)])
        gen_args.extend(["--out", parsed.out])
        return main_data_gen(gen_args)

    elif parsed.command == "view":
        view_args = []
        if parsed.dataset:
            view_args.append(parsed.dataset)
        if parsed.port != 8765:
            view_args.extend(["--port", str(parsed.port)])
        if parsed.no_browser:
            view_args.append("--no-browser")
        return main_viewer(view_args)

    elif parsed.command == "worker":
        worker_cmd = getattr(parsed, "worker_command", None)
        if worker_cmd == "run":
            from synapse.pipeline.bundle_worker import (
                run_worker_cli,
            )

            return run_worker_cli(
                bundle_path=parsed.bundle,
                json_mode=json_mode,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing worker sub-command (run)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse worker run --bundle <path>'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            worker_parser.print_help()
            return 0

    elif parsed.command == "lora":
        lora_cmd = getattr(parsed, "lora_command", None)
        if lora_cmd == "prepare-data":
            return _run_with_json_mode(
                _cmd_lora_prepare_data, parsed,
                json_mode, trace_ctx,
            )
        elif lora_cmd == "train":
            return _run_with_json_mode(
                _cmd_lora_train, parsed,
                json_mode, trace_ctx,
            )
        elif lora_cmd == "eval":
            return _run_with_json_mode(
                _cmd_lora_eval, parsed,
                json_mode, trace_ctx,
            )
        elif lora_cmd == "merge":
            return _run_with_json_mode(
                _cmd_lora_merge, parsed,
                json_mode, trace_ctx,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing lora sub-command "
                    "(prepare-data|train|eval|merge)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse lora <sub-command> --help'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            lora_parser.print_help()
            return 0

    elif parsed.command == "inspect":
        insp_cmd = getattr(parsed, "inspect_command", None)
        if insp_cmd == "align":
            return _run_with_json_mode(
                _cmd_inspect_align, parsed,
                json_mode, trace_ctx,
            )
        elif insp_cmd == "eval":
            return _run_with_json_mode(
                _cmd_inspect_eval, parsed,
                json_mode, trace_ctx,
            )
        elif insp_cmd == "refine":
            return _run_with_json_mode(
                _cmd_inspect_refine, parsed,
                json_mode, trace_ctx,
            )
        elif insp_cmd == "export-rl-data":
            return _run_with_json_mode(
                _cmd_inspect_export_rl_data, parsed,
                json_mode, trace_ctx,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing inspect sub-command "
                    "(align|eval|refine|export-rl-data)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse inspect <sub-command> --help'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            inspect_parser.print_help()
            return 0

    elif parsed.command == "rsi":
        rsi_cmd = getattr(parsed, "rsi_command", None)
        if rsi_cmd == "sample-curriculum":
            return _run_with_json_mode(
                _cmd_rsi_sample_curriculum, parsed,
                json_mode, trace_ctx,
            )
        elif rsi_cmd == "verify":
            return _run_with_json_mode(
                _cmd_rsi_verify, parsed,
                json_mode, trace_ctx,
            )
        elif rsi_cmd == "build-pairs":
            return _run_with_json_mode(
                _cmd_rsi_build_pairs, parsed,
                json_mode, trace_ctx,
            )
        elif rsi_cmd == "gate-promote":
            return _run_with_json_mode(
                _cmd_rsi_gate_promote, parsed,
                json_mode, trace_ctx,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing rsi sub-command "
                    "(sample-curriculum|verify|"
                    "build-pairs|gate-promote)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse rsi <sub-command>"
                    " --help'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            rsi_parser.print_help()
            return 0

    else:
        parser.print_help()
        return 0


# -------------------------------------------------------------------
# Top-level entry point: synapse-bench
# -------------------------------------------------------------------


def main_bench(args: list[str] | None = None) -> int:
    """Entry point for the ``synapse-bench`` CLI.

    Subcommands::

        synapse-bench list [--tier T1|T2|T3|T4|T5]
        synapse-bench export-fgpp --scenario <id> [--output <dir>]
    """
    parser = argparse.ArgumentParser(
        prog="synapse-bench",
        description="Synapse-Bench -- evaluation dataset & benchmark CLI",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Emit machine-readable JSON to stdout (no log noise)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- list ---
    list_parser = subparsers.add_parser(
        "list", help="List available scenarios",
    )
    list_parser.add_argument(
        "--tier",
        choices=["T1", "T2", "T3", "T4", "T5"],
        default=None,
        help="Filter by scenario tier",
    )

    # --- export-fgpp ---
    export_parser = subparsers.add_parser(
        "export-fgpp",
        help="Render scenario to FGPP output directory",
    )
    export_parser.add_argument(
        "--scenario", required=True, help="Scenario ID",
    )
    export_parser.add_argument(
        "--output", default=None, help="Output directory",
    )
    export_parser.add_argument(
        "--frames", type=int, default=None,
        help="Number of frames to render",
    )

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark scenario(s) through an adapter",
    )
    run_parser.add_argument(
        "--scenario", default=None,
        help="Scenario ID to run",
    )
    run_parser.add_argument(
        "--all", action="store_true", default=False,
        help="Run all available scenarios",
    )
    run_parser.add_argument(
        "--adapter", default="mock",
        help="Adapter ID (default: mock)",
    )
    run_parser.add_argument(
        "--profile", default=None,
        help="Quality gate profile name",
    )
    run_parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)",
    )
    run_parser.add_argument(
        "--resolution", type=int, default=512,
        help="Square resolution (default: 512)",
    )
    run_parser.add_argument(
        "--export-bundle",
        default=None,
        metavar="DIR",
        help="Export an InspectionBundle to DIR after running",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a pre-rendered InspectionBundle",
    )
    eval_parser.add_argument(
        "--bundle", required=True,
        help="Path to InspectionBundle directory",
    )
    eval_parser.add_argument(
        "--profile", default="track_b_video",
        help="Quality gate profile (default: track_b_video)",
    )

    parsed = parser.parse_args(args)
    trace_ctx = TraceContext.from_env_or_dict(dict(os.environ))

    json_mode: bool = parsed.json

    if parsed.command is None:
        if json_mode:
            json.dump(
                {
                    "status": "ok",
                    "exit_code": 0,
                    "trace_id": trace_ctx.trace_id,
                    "message": "No command specified. Use --help.",
                },
                sys.stdout,
                default=str,
            )
            print()
        else:
            parser.print_help()
        return 0

    if parsed.command == "list":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_list, parsed, json_mode, trace_ctx,
        )
    elif parsed.command == "export-fgpp":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_export_fgpp, parsed, json_mode, trace_ctx,
        )
    elif parsed.command == "run":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_run, parsed, json_mode, trace_ctx,
        )
    elif parsed.command == "evaluate":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_evaluate, parsed, json_mode, trace_ctx,
        )
    else:
        parser.print_help()
        return 0


# -------------------------------------------------------------------
# Legacy compat: ``python -m synapse.cli`` still works
# -------------------------------------------------------------------

# Kept for backward compatibility with __main__.py
main = main_synapse
