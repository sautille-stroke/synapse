"""Bench subcommand handlers: list, run, evaluate, export-fgpp."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    Path,
    QualityGateError,
    _run_with_json_mode,
    argparse,
    json,
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


def register_bench_subparsers(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    """Register subcommands for the ``synapse-bench`` CLI."""
    list_parser = subparsers.add_parser(
        "list", help="List available scenarios",
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


def dispatch_bench(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
) -> int:
    """Dispatch ``synapse-bench`` subcommands."""
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
    return 0
