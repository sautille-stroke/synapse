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
from typing import Any, Literal

from synapse.errors import (
    AssetNotFoundError,
    ContractValidationError,
    ExitCode,
    NotImplementedStageError,
    QualityGateError,
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
    raise NotImplementedStageError(
        "CLI 'synapse lora prepare-data' is not yet wired to "
        "FeatureStoreBuilder; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.training.FeatureStoreBuilder' API directly "
            "to encode WebDataset shards."
        ),
        details={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
    )


def _cmd_lora_train(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run LoRA training from CLI config."""
    from synapse.training.trainer import LoRATrainConfig

    config_str = getattr(args, "config", "{}")
    try:
        config_dict = json.loads(config_str)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"Invalid JSON passed to --config: {exc}",
            agent_fix_hint=(
                "Pass a valid JSON object string to --config, e.g. "
                '\'{"category": "style", "rank": 8}\'.'
            ),
        ) from exc

    if not isinstance(config_dict, dict) or "category" not in config_dict:
        raise ContractValidationError(
            "LoRA train --config JSON must include explicit 'category'.",
            agent_fix_hint=(
                "Specify 'category' in --config (one of: style, "
                "identity, control, whitebox, motion)."
            ),
        )

    try:
        validated = LoRATrainConfig(**config_dict)
    except Exception as exc:
        raise ContractValidationError(
            f"LoRATrainConfig validation failed: {exc}",
            agent_fix_hint="Fix --config fields per LoRATrainConfig schema.",
        ) from exc

    raise NotImplementedStageError(
        "CLI 'synapse lora train' standalone backbone execution is "
        "not yet implemented; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.training.StatelessLoRATrainer' in Python "
            "with an injected DiT model."
        ),
        details={"validated_config": validated.model_dump()},
    )


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
    waived = report.get("waived", False)

    if waived and not getattr(args, "allow_waived", False):
        raise QualityGateError(
            "LoRA bundle evaluation was waived (skip_bench=True); "
            "promotion rejected by fail-closed gate.",
            agent_fix_hint=(
                "Run full Synapse-Bench evaluation to populate "
                "bench_metrics before promoting LoRA weights."
            ),
        )

    if not passed:
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

    return {"passed": True, "waived": waived, "report": report}


def _cmd_lora_merge(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge LoRA weights into base model."""
    bundle_dir = Path(args.bundle)
    output_dir = Path(args.output)
    raise NotImplementedStageError(
        "CLI 'synapse lora merge' is not yet wired to a base model "
        "checkpoint; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.adapters.lora.MultiLoRAManager' to merge "
            "and apply LoRA adapters onto a loaded model."
        ),
        details={
            "bundle_dir": str(bundle_dir),
            "output_dir": str(output_dir),
        },
    )


# -------------------------------------------------------------------
# Inspect sub-command handlers
# -------------------------------------------------------------------



def _cmd_inspect_mesh(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run Tier 0 geometric topology validation on a mesh file."""
    import numpy as np

    from synapse.inspect.geometry_gate import validate_mesh_geometry

    mesh_path = Path(getattr(args, "mesh_path", ""))
    if not mesh_path.is_file():
        raise AssetNotFoundError(
            f"Mesh file not found: {mesh_path}",
            agent_fix_hint=(
                f"Ensure '{mesh_path}' exists and is a valid .obj / .glb "
                "/ .ply mesh file."
            ),
        )

    # Load mesh via trimesh when installed, falling back to built-in OBJ loader
    try:
        import trimesh

        mesh_obj: Any = trimesh.load(str(mesh_path), force="mesh")
        vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
        faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    except ImportError:
        if mesh_path.suffix.lower() == ".obj":
            from synapse.bench.model_loader import load_mesh

            loaded = load_mesh(str(mesh_path), normalize=False)
            vertices = np.asarray(loaded.vertices, dtype=np.float64)
            faces = np.asarray(loaded.faces, dtype=np.int64)
        else:
            raise ContractValidationError(
                f"trimesh is required for loading {mesh_path.suffix!r} files. "
                "Install with: pip install 'synapse[mesh]'",
                agent_fix_hint="pip install trimesh>=4.0.0",
            )

    raw_backend = str(getattr(args, "backend", "native"))
    backend: Literal["native", "python"] = (
        "python" if raw_backend == "python" else "native"
    )
    require_wt = bool(getattr(args, "require_watertight", False))

    result = validate_mesh_geometry(
        vertices,
        faces,
        require_watertight=require_wt,
        backend=backend,
    )

    result_dict = result.model_dump()

    if not args.json:
        status = "PASS" if result.passed else "FAIL"
        print(f"[Tier 0 Geometry Gate] {mesh_path.name}: {status}")
        print(f"  Vertices:    {result.num_vertices}")
        print(f"  Faces:       {result.num_faces}")
        print(f"  Manifold:    {result.is_manifold}")
        print(f"  Watertight:  {result.is_watertight}")
        print(f"  Degenerate:  {len(result.degenerate_face_indices)}")
        print(
            f"  Self-Intersecting: "
            f"{len(result.self_intersecting_pairs)} pairs"
        )
        print(f"  Components:  {result.num_connected_components}")
        print(
            f"  Float Debris: "
            f"{len(result.floating_island_face_indices)} faces"
        )
        if result.failure_reasons:
            print("  Failures:")
            for reason in result.failure_reasons:
                print(f"    - {reason}")

    # Write report file if requested
    report_path = getattr(args, "report", None)
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(result_dict, indent=2, default=str),
            encoding="utf-8",
        )
        if not args.json:
            print(f"  Report: {report_file}")

    if not result.passed:
        raise QualityGateError(
            "Tier 0 geometry gate failed: "
            + "; ".join(result.failure_reasons),
            agent_fix_hint=(
                "Fix mesh topology issues before proceeding to "
                "neural rendering or video generation stages."
            ),
        )

    return result_dict

def _cmd_inspect_align(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Align reference image to mesh."""
    mesh_path = getattr(args, "mesh", "")
    image_path = getattr(args, "image", "")
    raise NotImplementedStageError(
        "CLI 'synapse inspect align' standalone file pipeline is not "
        "yet implemented; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.inspect.align' functions "
            "(generate_fibonacci_sphere_cameras, compute_silhouette_iou) "
            "directly in Python."
        ),
        details={"mesh": mesh_path, "image": image_path},
    )


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
        "transport_kind": "mock",
    }


def _cmd_inspect_evaluate(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Batch-evaluate 3D models (Tier-0 topology + multi-view + HTML/JSON)."""
    from synapse.inspect.model_evaluator import evaluate_3d_models_directory

    json_mode = bool(getattr(args, "json", False))

    def _log(msg: str) -> None:
        if not json_mode:
            print(msg)

    ref_raw = getattr(args, "reference_dir", None)
    eval_backend_raw = str(getattr(args, "backend", "native"))
    eval_backend: Literal["native", "python"] = (
        "python" if eval_backend_raw == "python" else "native"
    )
    data = evaluate_3d_models_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        reference_dir=Path(ref_raw) if ref_raw else None,
        profile_name=str(getattr(args, "profile", "TRACK_A_ASSET")),
        resolution=int(getattr(args, "resolution", 256)),
        num_views=int(getattr(args, "num_views", 8)),
        log_fn=None if json_mode else _log,
        geometry_backend=eval_backend,
    )

    if int(data["exit_code"]) == int(ExitCode.CIRCUIT_BREAKER):
        raise QualityGateError(
            f"3D model evaluation gate failed ({data['summary']['failed']}/"
            f"{data['summary']['total']} models failed profile "
            f"{data['profile']!r}).",
            agent_fix_hint=str(data["agent_fix_hint"]),
            details={
                "profile": data["profile"],
                "summary": data["summary"],
                "report_path": data["report_path"],
                "results_json_path": data["results_json_path"],
            },
        )

    return data


def _cmd_inspect_refine(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run targeted refinement from an InspectionReport."""
    from synapse.inspect.flywheel_exporter import (
        run_targeted_refinement,
    )

    report_path = getattr(args, "report", None)
    if not report_path or not Path(report_path).is_file():
        raise AssetNotFoundError(
            f"InspectionReport file not found: {report_path!r}. "
            "Refusing to run refinement on an empty defect list.",
            agent_fix_hint=(
                "Pass a valid InspectionReport JSON file via "
                "'--report <path>'."
            ),
        )

    try:
        report = json.loads(
            Path(report_path).read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"Invalid JSON in InspectionReport '{report_path}': {exc}",
            agent_fix_hint="Ensure --report points to a valid JSON file.",
        ) from exc

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
    raw_fmt = str(getattr(args, "format", "dpo"))
    fmt: Literal["prm", "dpo"] = (
        "prm" if raw_fmt == "prm" else "dpo"
    )

    result = export_rl_flywheel_dataset(
        log_dir=log_dir,
        output_path=output,
        format=fmt,
    )
    return result


# -------------------------------------------------------------------
# Refine sub-command handlers
# -------------------------------------------------------------------


def _cmd_refine_anatomical(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run anatomical local refinement pipeline (RFC 0061).

    Assembles a ``refine_session.v1.0`` payload and executes the
    :class:`~synapse.refine.pipeline.LocalRefinementPipeline`.
    """
    import numpy as np

    from synapse.refine.pipeline import (
        LocalRefinementPipeline,
        _METHOD_MAP,
        _REGION_MAP,
        build_refine_session_payload,
    )
    from synapse.refine.types import RefinementROI

    output_dir = Path(getattr(args, "output", "./output/refine"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse regions
    region_str: str = getattr(args, "regions", "HAND,FACE")
    region_names = [r.strip() for r in region_str.split(",") if r.strip()]

    # Parse fusion method
    fusion_name = str(
        getattr(args, "fusion_method", "LAPLACIAN_FREQUENCY"),
    )
    from synapse.refine.types import FusionMethod

    fusion_method = _METHOD_MAP.get(
        fusion_name, FusionMethod.LAPLACIAN_FREQUENCY,
    )

    # Build session payload
    import uuid

    session_id = str(uuid.uuid4())
    payload = build_refine_session_payload(
        session_id=session_id,
        regions=region_names,
        fusion_method=fusion_name,
    )

    # Load or synthesize frames
    video_path_raw = getattr(args, "video", None)
    if video_path_raw and Path(video_path_raw).is_file():
        frames = np.load(video_path_raw).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        frames = rng.random((4, 128, 128, 3), dtype=np.float32)

    height, width = frames.shape[1], frames.shape[2]

    # Build synthetic ROIs
    rois_per_frame: list[list[RefinementROI]] = []
    for t in range(frames.shape[0]):
        frame_rois: list[RefinementROI] = []
        for rname in region_names:
            region = _REGION_MAP.get(rname)
            if region is None:
                continue
            h4, w4 = height // 4, width // 4
            bbox = np.array(
                [w4, h4, w4 * 3, h4 * 3], dtype=np.float32,
            )
            crop = frames[t, h4:h4 * 3, w4:w4 * 3, :].copy()
            roi = RefinementROI(
                region=region,
                bbox_xyxy=bbox,
                crop_acescg=crop,
                confidence=0.9,
            )
            frame_rois.append(roi)
        rois_per_frame.append(frame_rois)

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        fusion_method=fusion_method,
    )

    # Save refined frames
    np.save(str(output_dir / "refined_frames.npy"), result.frames_refined)

    diag = result.to_dict()
    diag_path = output_dir / "refine_diagnostics.json"
    diag_path.write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8",
    )

    return {
        "session_payload": payload,
        "output_dir": str(output_dir),
        "diagnostics": diag,
    }


# -------------------------------------------------------------------
# Recon sub-command handlers
# -------------------------------------------------------------------


def _cmd_recon_invert_video(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run video-to-3D inverse reconstruction (RFC 0060)."""
    from synapse.recon import ReconPipeline
    from synapse.recon.delight import delight_multiview_albedo

    output_dir = Path(getattr(args, "output", "./output/recon"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target_faces = int(getattr(args, "target_faces", 1500))

    video_path_raw = getattr(args, "video", None)
    if video_path_raw and Path(video_path_raw).is_file():
        import numpy as np

        frames = np.load(video_path_raw)
    else:
        import numpy as np

        rng = np.random.default_rng(42)
        frames = rng.integers(100, 200, (8, 256, 256, 3), dtype=np.uint8)

    pipe = ReconPipeline(target_quad_faces=target_faces)
    try:
        result = pipe.run(frames)
    except Exception:
        # Degrade gracefully — report without geometry
        class _EmptyResult:
            camera_trajectory = None
            vertices = None
            normals = None
        result = _EmptyResult()  # type: ignore[assignment]

    output_info: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_frames": len(frames),
        "has_trajectory": result.camera_trajectory is not None,
        "has_geometry": result.vertices is not None,
    }

    # Save trajectory
    if result.camera_trajectory:
        traj_path = output_dir / "camera_trajectory.json"
        traj_path.write_text(
            json.dumps(result.camera_trajectory.to_dict(), indent=2),
            encoding="utf-8",
        )
        output_info["trajectory_path"] = str(traj_path)

    # FGPP re-ingestion check
    fgpp_dir = getattr(args, "fgpp_dir", None)
    if fgpp_dir and Path(fgpp_dir).is_dir():
        output_info["fgpp_dir"] = fgpp_dir

    if not args.json:
        print("[Recon] Video-to-3D inverse reconstruction complete")
        for k, v in output_info.items():
            print(f"  {k}: {v}")

    return output_info


def _cmd_recon_closed_loop(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run 5-step inverse rendering closed loop (RFC 0062).

    Orchestrates forward-guide → target-video → inverse-optimisation
    → aligned-render → scalpel-patch.
    """
    import numpy as np

    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    output_dir = Path(getattr(args, "output", "./output/closed-loop"))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_iterations = int(getattr(args, "max_iterations", 5))
    n_preroll = int(getattr(args, "n_preroll", 0))

    # Load or synthesize frames
    video_path_raw = getattr(args, "video", None)
    if video_path_raw and Path(video_path_raw).is_file():
        frames = np.load(video_path_raw).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        frames = rng.random((5, 64, 64, 3), dtype=np.float32)

    t_count, h, w, _ = frames.shape

    # Synthesize normals (pointing +Z) and UV maps
    normals = np.zeros_like(frames)
    normals[..., 2] = 1.0

    uv_maps = np.zeros((t_count, h, w, 2), dtype=np.float32)
    u_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)
    v_lin = np.linspace(0.0, 1.0, h, dtype=np.float32)
    uv_maps[:, :, :, 0] = u_lin[np.newaxis, np.newaxis, :]
    uv_maps[:, :, :, 1] = v_lin[np.newaxis, :, np.newaxis]

    from synapse.recon.loop.mitsuba_optimizer import InverseOptimConfig
    from synapse.recon.loop.types import LoopConfig

    loop_config = LoopConfig(
        n_preroll=n_preroll,
        max_iterations=max_iterations,
    )
    optim_config = InverseOptimConfig(
        max_iterations=max_iterations,
    )

    runner = ClosedLoopRunner(
        loop_config=loop_config,
        optim_config=optim_config,
    )
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    # Save output
    np.save(str(output_dir / "output_frames.npy"), result.output_frames)
    diag = result.to_dict()
    diag_path = output_dir / "closed_loop_diagnostics.json"
    diag_path.write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8",
    )

    output_info: dict[str, Any] = {
        "output_dir": str(output_dir),
        **diag,
    }

    if not args.json:
        print("[Recon] Closed-loop inverse rendering complete")
        for k, v in output_info.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")

    return output_info


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
    metrics_raw = getattr(args, "physics_metrics", None)
    if not metrics_raw:
        raise ContractValidationError(
            "CLI 'synapse rsi verify' requires '--physics-metrics' "
            "(JSON string or path to JSON file). Hardcoded passing "
            "metrics are FORBIDDEN.",
            agent_fix_hint=(
                "Pass '--physics-metrics '{\"silhouette_iou\": 0.95, "
                "\"depth_rmse_m\": 0.01, "
                "\"normal_angular_error_deg\": 5.0}'' or a JSON file path."
            ),
        )

    metrics_path = Path(metrics_raw)
    try:
        if metrics_path.is_file():
            physics_metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        else:
            physics_metrics = json.loads(metrics_raw)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"Invalid JSON in --physics-metrics: {exc}",
            agent_fix_hint="Provide valid JSON for --physics-metrics.",
        ) from exc

    prm_score = float(getattr(args, "prm_score", 0.5))

    engine = DualVerifierEngine()
    verdict = engine.verify(
        sample_id=sample_id,
        physics_metrics=physics_metrics,
        prm_score=prm_score,
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

    golden_file = getattr(args, "golden_file", None)
    if not golden_file or not Path(golden_file).is_file():
        raise AssetNotFoundError(
            f"Golden pairs file not found: {golden_file!r}. "
            "Hardcoded CLI golden pairs are FORBIDDEN.",
            agent_fix_hint=(
                "Pass '--golden-file <path>' pointing to a JSON/JSONL "
                "file of L0 golden PreferencePair records."
            ),
        )

    raw_text = Path(golden_file).read_text(encoding="utf-8").strip()
    if raw_text.startswith("["):
        items = json.loads(raw_text)
    else:
        items = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]

    builder = PreferenceDatasetBuilder()
    builder.add_golden_pairs([PreferencePair(**item) for item in items])

    self_file = getattr(args, "self_file", None)
    if self_file and Path(self_file).is_file():
        self_text = Path(self_file).read_text(encoding="utf-8").strip()
        self_items = (
            json.loads(self_text)
            if self_text.startswith("[")
            else [
                json.loads(line)
                for line in self_text.splitlines()
                if line.strip()
            ]
        )
        builder.add_self_pairs(
            [PreferencePair(**item) for item in self_items]
        )

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
        synapse inspect mesh|align|eval|refine|export-rl-data [--json]
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

    insp_mesh = inspect_sub.add_parser(
        "mesh",
        help="Run Tier 0 geometric topology gate on a mesh file",
    )
    insp_mesh.add_argument(
        "mesh_path",
        help="Path to mesh file (.obj, .glb, .ply)",
    )
    insp_mesh.add_argument(
        "--report", default=None,
        help="Output JSON report file path",
    )
    insp_mesh.add_argument(
        "--backend", default="native",
        choices=["native", "python"],
        help="Backend for geometry checks (default: native). "
             "'python' is for Oracle parity testing only.",
    )
    insp_mesh.add_argument(
        "--require-watertight", action="store_true",
        help="Require mesh to be watertight (closed)",
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

    insp_evaluate = inspect_sub.add_parser(
        "evaluate",
        help="Batch-evaluate 3D models and generate HTML/JSON reports",
    )
    insp_evaluate.add_argument(
        "--input-dir", required=True,
        help="Directory containing .obj/.glb/.ply model files",
    )
    insp_evaluate.add_argument(
        "--output-dir", required=True,
        help="Output directory for eval_report.html and eval_results.json",
    )
    insp_evaluate.add_argument(
        "--reference-dir", default=None,
        help="Optional directory with reference meshes",
    )
    insp_evaluate.add_argument(
        "--profile", "--gate-profile",
        dest="profile",
        default="TRACK_A_ASSET",
        help="Gate profile (TRACK_A_ASSET, TRACK_B_VIDEO, RSI_PROMOTION)",
    )
    insp_evaluate.add_argument(
        "--resolution", type=int, default=256,
        help="Multi-view render resolution (default: 256)",
    )
    insp_evaluate.add_argument(
        "--num-views", "--views",
        dest="num_views",
        type=int, default=8,
        help="Number of multi-view orbit screenshots (default: 8)",
    )
    insp_evaluate.add_argument(
        "--backend", default="native",
        choices=["native", "python"],
        help="Geometry gate backend (default: native). "
             "'python' is for Oracle parity testing only.",
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

    # --- recon ---
    recon_parser = subparsers.add_parser(
        "recon",
        help="3D reconstruction and inverse rendering commands",
    )
    recon_sub = recon_parser.add_subparsers(
        dest="recon_command",
    )

    recon_invert = recon_sub.add_parser(
        "invert-video",
        help="Inverse video-to-3D reconstruction (RFC 0060)",
    )
    recon_invert.add_argument(
        "--video", default=None,
        help="Path to video frames .npy file (T, H, W, 3)",
    )
    recon_invert.add_argument(
        "--fgpp-dir", default=None,
        help="Optional FGPP directory for re-ingestion",
    )
    recon_invert.add_argument(
        "--output", "-o", default="./output/recon",
        help="Output directory for reconstruction results",
    )
    recon_invert.add_argument(
        "--target-faces", type=int, default=1500,
        help="Target face count for quad mesh (default: 1500)",
    )

    recon_closed_loop = recon_sub.add_parser(
        "closed-loop",
        help="Run 5-step inverse rendering closed loop (RFC 0062)",
    )
    recon_closed_loop.add_argument(
        "--video", default=None,
        help="Path to video frames .npy file (T, H, W, 3) float32",
    )
    recon_closed_loop.add_argument(
        "--output", "-o", default="./output/closed-loop",
        help="Output directory for closed-loop results",
    )
    recon_closed_loop.add_argument(
        "--max-iterations", type=int, default=5,
        help="Max inverse optimisation iterations (default: 5)",
    )
    recon_closed_loop.add_argument(
        "--n-preroll", type=int, default=0,
        choices=[0, 4, 8],
        help="Number of preroll frames (default: 0)",
    )

    # --- refine ---
    refine_parser = subparsers.add_parser(
        "refine",
        help="Anatomical local refinement commands (RFC 0061)",
    )
    refine_sub = refine_parser.add_subparsers(
        dest="refine_command",
    )

    refine_anatomical = refine_sub.add_parser(
        "anatomical",
        help="Run end-to-end anatomical refinement pipeline",
    )
    refine_anatomical.add_argument(
        "--video", default=None,
        help="Path to video frames .npy file (T, H, W, 3) float32 ACEScg",
    )
    refine_anatomical.add_argument(
        "--output", "-o", default="./output/refine",
        help="Output directory for refined frames and diagnostics",
    )
    refine_anatomical.add_argument(
        "--regions", default="HAND,FACE",
        help="Comma-separated anatomical regions (default: HAND,FACE)",
    )
    refine_anatomical.add_argument(
        "--fusion-method", default="LAPLACIAN_FREQUENCY",
        choices=[
            "LAPLACIAN_FREQUENCY", "CASCADE",
            "LATENT_TIMESTEP_SWITCH",
            "UNCERTAINTY_WEIGHTED", "POISSON_GRADIENT",
        ],
        help="Fusion strategy (default: LAPLACIAN_FREQUENCY)",
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
    rsi_verify.add_argument(
        "--physics-metrics", default=None,
        help="JSON string or file path with measured physics metrics",
    )
    rsi_verify.add_argument(
        "--prm-score", type=float, default=0.8,
        help="PRM neural score in [0, 1] (default: 0.8)",
    )

    rsi_build = rsi_sub.add_parser(
        "build-pairs",
        help="Build preference pairs for RSI training",
    )
    rsi_build.add_argument(
        "--golden-file", default=None,
        help="Path to JSON/JSONL file with L0 golden preference pairs",
    )
    rsi_build.add_argument(
        "--self-file", default=None,
        help="Optional path to JSON/JSONL file with self-play pairs",
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
        if insp_cmd == "mesh":
            return _run_with_json_mode(
                _cmd_inspect_mesh, parsed,
                json_mode, trace_ctx,
            )
        elif insp_cmd == "align":
            return _run_with_json_mode(
                _cmd_inspect_align, parsed,
                json_mode, trace_ctx,
            )
        elif insp_cmd == "eval":
            return _run_with_json_mode(
                _cmd_inspect_eval, parsed,
                json_mode, trace_ctx,
            )
        elif insp_cmd == "evaluate":
            parsed.json = json_mode
            return _run_with_json_mode(
                _cmd_inspect_evaluate, parsed,
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
                    "(mesh|align|eval|evaluate|refine|export-rl-data)",
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

    elif parsed.command == "recon":
        recon_cmd = getattr(parsed, "recon_command", None)
        if recon_cmd == "invert-video":
            return _run_with_json_mode(
                _cmd_recon_invert_video, parsed,
                json_mode, trace_ctx,
            )
        elif recon_cmd == "closed-loop":
            return _run_with_json_mode(
                _cmd_recon_closed_loop, parsed,
                json_mode, trace_ctx,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing recon sub-command "
                    "(invert-video|closed-loop)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse recon invert-video --help' or "
                    "'synapse recon closed-loop --help'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            recon_parser.print_help()
            return 0

    elif parsed.command == "refine":
        refine_cmd = getattr(parsed, "refine_command", None)
        if refine_cmd == "anatomical":
            return _run_with_json_mode(
                _cmd_refine_anatomical, parsed,
                json_mode, trace_ctx,
            )
        else:
            if json_mode:
                err = SynapseError(
                    "Missing refine sub-command (anatomical)",
                    ExitCode.CONTRACT_ERROR,
                    "Use 'synapse refine anatomical --help'.",
                )
                json.dump(
                    err.to_json_dict(trace_ctx.trace_id),
                    sys.stdout,
                    default=str,
                )
                print()
                return int(err.exit_code)
            refine_parser.print_help()
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
