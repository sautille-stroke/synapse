"""Bench subcommand handlers: list, run, evaluate, export-fgpp."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    Path,
    QualityGateError,
    _run_with_json_mode,
    argparse,
)


def _cmd_bench_list(args: argparse.Namespace) -> dict[str, Any]:
    """List available Synapse-Bench scenarios."""
    from synapse.bench.scenario import ScenarioTier, list_scenarios

    tier = None
    if hasattr(args, "tier") and args.tier:
        tier = ScenarioTier(args.tier)

    scenarios = list_scenarios(tier=tier)

    items = []
    for s in scenarios:
        items.append(
            {
                "id": s.id,
                "tier": s.tier.value,
                "description": s.description,
                "tags": s.tags,
            }
        )

    if not args.json:
        if not items:
            print("No scenarios found.")
        else:
            for item in items:
                print(f"  [{item['tier']}] {item['id']}: {item['description']}")

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
            seed=seed,
            resolution=resolution,
        )
    else:
        reports = [
            runner.run_scenario(
                scenario_name=args.scenario,
                seed=seed,
                resolution=resolution,
            )
        ]

    items = [r.to_dict() for r in reports]

    # Export InspectionBundle if requested
    export_dir = getattr(args, "export_bundle", None)
    if export_dir is not None:
        from synapse.bench.bundle_exporter import (
            export_inspection_bundle,
        )

        export_path = Path(export_dir).resolve()
        export_path.mkdir(parents=True, exist_ok=True)
        if not export_path.is_dir():
            raise AssetNotFoundError(
                f"Cannot create export bundle directory: {export_path}",
                agent_fix_hint=(f"Ensure '{export_dir}' is a valid writable directory path."),
            )
        for r in reports:
            # Resolve source model from scenario asset via AssetRegistry
            source_model: Path | None = None
            try:
                from synapse.bench.registry import AssetRegistry
                from synapse.bench.scenario import load_scenario

                sc = load_scenario(r.scenario_name)
                if hasattr(sc, "asset") and hasattr(sc.asset, "id"):
                    source_model = AssetRegistry().resolve(sc.asset.id)
            except Exception:
                pass
            if source_model is None or not source_model.is_file():
                from synapse.bench.registry import AssetRegistry

                source_model = AssetRegistry().resolve("cube")
            export_inspection_bundle(
                scenario_name=r.scenario_name,
                source_model_path=source_model,
                output_dir=export_path / r.scenario_name,
                metrics_dict=r.metrics,
            )

    if not args.json:
        for r in reports:
            status = "OK" if r.error is None else "FAIL"
            print(
                f"  [{status}] {r.scenario_name}: {len(r.metrics)} metrics, {r.num_frames} frames"
            )
            if r.error:
                print(f"    Error: {r.error}")
            if r.gate_verdict is not None:
                verdict = "PASS" if r.gate_verdict.passed else "FAIL"
                print(f"    Gate: {verdict}")

    if not gate_profile:
        errored = [r for r in reports if r.error]
        if errored:
            first_err = str(errored[0].error)
            if "not found" in first_err.lower() or "no such file" in first_err.lower():
                raise AssetNotFoundError(
                    f"Benchmark scenario failed: {errored[0].scenario_name}: {first_err}",
                    agent_fix_hint="Run 'synapse bench list' to inspect valid scenario names.",
                )
            from synapse.errors import ContractValidationError

            raise ContractValidationError(
                f"Benchmark scenario execution failed for {len(errored)} scenario(s): "
                + "; ".join(f"{r.scenario_name}: {r.error}" for r in errored),
                agent_fix_hint="Inspect scenario configuration and adapter inputs, then retry.",
            )

    if gate_profile:
        failed_scenarios: list[str] = []
        for r in reports:
            if r.error:
                failed_scenarios.append(f"{r.scenario_name}: error {r.error}")
            elif r.gate_verdict is not None and not r.gate_verdict.passed:
                failed_scenarios.append(
                    f"{r.scenario_name}: gate not passed ("
                    f"{', '.join(v.metric_name for v in r.gate_verdict.violations)})"
                )
        if failed_scenarios:
            raise QualityGateError(
                f"Quality gate '{gate_profile}' failed for {len(failed_scenarios)} scenario(s): "
                + "; ".join(failed_scenarios),
                agent_fix_hint=(
                    f"Adjust video generation or refinement output to satisfy "
                    f"'{gate_profile}' quality gate thresholds."
                ),
                details={"gate_profile": gate_profile, "failures": failed_scenarios},
            )

    return {"reports": items, "count": len(items)}


def _cmd_bench_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a pre-rendered InspectionBundle."""
    from synapse.pipeline.bench_runner import evaluate_bundle

    bundle_dir = Path(args.bundle)
    if not bundle_dir.is_dir():
        raise AssetNotFoundError(
            f"Bundle directory does not exist: {bundle_dir}",
            agent_fix_hint=(
                f"Ensure '{bundle_dir}' exists and contains frames/, fgpp/, and metadata.json."
            ),
        )

    profile = getattr(args, "profile", "track_b_video")
    metrics = evaluate_bundle(bundle_dir, profile=profile)

    if not args.json:
        print("[Evaluate Bundle]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if getattr(args, "strict", False) and metrics.get("gate_passed", 1.0) == 0.0:
        raise QualityGateError(
            f"InspectionBundle failed quality gate for profile '{profile}'",
            agent_fix_hint=(
                f"Review metrics against '{profile}' profile thresholds in comparison.json."
            ),
            details={"metrics": metrics, "profile": profile},
        )

    return {"metrics": metrics, "bundle_dir": str(bundle_dir)}


def _cmd_bench_export_fgpp(args: argparse.Namespace) -> dict[str, Any]:
    """Export a scenario to FGPP output directory."""
    from synapse.bench.model_loader import load_mesh
    from synapse.bench.scenario import load_scenario
    from synapse.fgpp.frame import FGPPFrame, FGPPSequence
    from synapse.fgpp.io import export_fgpp
    from synapse.render_core.camera import generate_cameras
    from synapse.render_core.rasterizer import SoftwareRasterizer

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

    output_arg = getattr(args, "output_dir", None) or getattr(args, "output", None)
    output_dir = Path(output_arg) if output_arg else Path(f"./data/bench_export/{scenario.id}")

    cameras = generate_cameras(
        scenario,
        num_frames=num_frames,
        width=width,
        height=height,
    )

    mesh = load_mesh(scenario.asset.id, normalize=True)

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
        frame.frame_index = i
        frames.append(frame)

    seq = FGPPSequence(frames=frames, fps=fps, scene_id=scenario.id)
    export_fgpp(seq, output_dir)
    meta_path = output_dir / "metadata.json"

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


def register_bench_subparsers(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    json_parent: argparse.ArgumentParser | None = None,
) -> None:
    """Register subcommands for the ``synapse-bench`` CLI."""
    parents = [json_parent] if json_parent is not None else []

    list_parser = subparsers.add_parser(
        "list",
        help="List available scenarios",
        parents=parents,
    )
    list_parser.add_argument(
        "--tier",
        choices=["T1", "T2", "T3", "T4", "T5"],
        default=None,
        help="Filter by scenario tier",
    )

    export_parser = subparsers.add_parser(
        "export-fgpp",
        help="Render scenario to FGPP output directory",
        parents=parents,
    )
    export_parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario ID",
    )
    export_parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        default=None,
        help="Output directory",
    )
    export_parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to render",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmark scenario(s) through an adapter",
        parents=parents,
    )
    run_parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario ID to run",
    )
    run_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all available scenarios",
    )
    run_parser.add_argument(
        "--adapter",
        default="mock",
        help="Adapter ID (default: mock)",
    )
    run_parser.add_argument(
        "--profile",
        default=None,
        help="Quality gate profile name",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    run_parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square resolution (default: 512)",
    )
    run_parser.add_argument(
        "--export-bundle",
        default=None,
        metavar="DIR",
        help="Export an InspectionBundle to DIR after running",
    )

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a pre-rendered InspectionBundle",
        parents=parents,
    )
    eval_parser.add_argument(
        "--bundle",
        required=True,
        help="Path to InspectionBundle directory",
    )
    eval_parser.add_argument(
        "--profile",
        default="track_b_video",
        help="Quality gate profile (default: track_b_video)",
    )
    eval_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail closed with exit code 40 if quality gate fails",
    )


def register_bench_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    json_parent: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Register the ``bench`` subcommand tree under ``synapse bench``."""
    from typing import cast

    parents = [json_parent] if json_parent is not None else []
    bench_parser = subparsers.add_parser("bench", help="Benchmark operations", parents=parents)
    bench_sub = bench_parser.add_subparsers(dest="bench_command")
    register_bench_subparsers(bench_sub, json_parent=json_parent)
    return cast(argparse.ArgumentParser, bench_parser)


def dispatch_bench(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    bench_parser: argparse.ArgumentParser | None = None,
) -> int:
    """Dispatch ``synapse-bench`` or ``synapse bench`` subcommands."""
    cmd = getattr(parsed, "bench_command", None) or getattr(parsed, "command", None)
    if cmd == "list":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_list,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif cmd == "export-fgpp":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_export_fgpp,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif cmd == "run":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_run,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif cmd == "evaluate":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_bench_evaluate,
            parsed,
            json_mode,
            trace_ctx,
        )
    else:
        from synapse.errors import ExitCode, SynapseError

        err = SynapseError(
            "Missing bench sub-command (list|export-fgpp|run|evaluate)",
            ExitCode.CONTRACT_ERROR,
            "Use 'synapse-bench list', 'synapse-bench export-fgpp', etc.",
        )
        if json_mode:
            import json
            import sys

            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
            return int(err.exit_code)
        if bench_parser is not None:
            bench_parser.print_help()
        return int(ExitCode.CONTRACT_ERROR)
