"""Generate / view subcommand handlers."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    ContractValidationError,
    Path,
    _run_with_json_mode,
    argparse,
    sys,
)

# -------------------------------------------------------------------
# Legacy standalone entry points (kept for backward compatibility)
# -------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute dataset generation and return structured summary dict."""
    from synapse.generators.dataset import DatasetConfig, DatasetGenerator

    json_mode = bool(getattr(args, "json", False))
    cfg = DatasetConfig(
        model_name=args.model,
        trajectory_type=args.trajectory,
        num_frames=args.frames,
        width=args.res,
        height=args.res,
        fps=args.fps,
        radius=args.radius,
        val_ratio=args.val_ratio,
        output_dir=args.out,
    )

    if not json_mode:
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

    if not json_mode:
        print(f"\n[OK] Dataset successfully generated at: {out_dir}")
        print(f"   Video:         {out_dir / 'video_all.mp4'}")
        print(f"   Contact Sheet: {out_dir / 'contact_sheet.png'}")
        print(f"   Manifest:      {out_dir / 'manifest.json'}")

    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "frames_count": cfg.num_frames,
        "model": cfg.model_name,
        "trajectory": cfg.trajectory_type,
        "resolution": [cfg.width, cfg.height],
        "fps": cfg.fps,
        "video_path": str(out_dir / "video_all.mp4"),
        "manifest_path": str(out_dir / "manifest.json"),
    }


def _get_dev_launch_viewer() -> Any:
    """Helper to lazily load dev preview launch_viewer without importing TCPServer into core SDK."""
    try:
        from scripts.dev_preview_viewer import launch_viewer
        return launch_viewer
    except ImportError:
        import importlib.util
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "dev_preview_viewer.py"
        if script_path.is_file():
            spec = importlib.util.spec_from_file_location("dev_preview_viewer", script_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.launch_viewer
        from synapse.errors import BackendUnavailableError
        raise BackendUnavailableError(
            "Local HTTP preview server is not available (dev_preview_viewer script missing).",
            agent_fix_hint="Use static HTML export (default) or ensure scripts/dev_preview_viewer.py exists.",
        ) from None


def main_data_gen(args: list[str] | None = None) -> int:
    """CLI tool: Generate multi-angle video & FGPP datasets."""
    from synapse.models.loader import list_builtin_models
    from synapse.telemetry import TraceContext

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
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Emit structured JSON output to stdout",
    )

    parsed = parser.parse_args(args)
    trace_ctx = TraceContext.from_env_or_dict()
    code = _run_with_json_mode(_cmd_generate, parsed, parsed.json, trace_ctx)
    if code == 0 and parsed.view and not parsed.json:
        launch_viewer = _get_dev_launch_viewer()
        launch_viewer(Path(parsed.out), port=8765, open_browser=True, block=True)
    return code


def main_viewer(args: list[str] | None = None) -> int:
    """CLI tool: Launch interactive web viewer or export static HTML."""
    from synapse.viewer.server import export_static_html

    parser = argparse.ArgumentParser(
        prog="synapse-viewer",
        description=(
            "View a Synapse multi-angle dataset (default: static HTML export; "
            "--serve-local: loopback preview)."
        ),
    )
    parser.add_argument(
        "dataset", nargs="?", default="./data/datasets/sample",
        help="Path to dataset directory (containing manifest.json)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output HTML path for static export (default: <dataset>/viewer.html)",
    )
    parser.add_argument(
        "--serve-local", action="store_true", default=False,
        help="Start local loopback HTTP preview server instead of static HTML export",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8765,
        help="Local port to serve HTTP viewer (default: 8765)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Loopback host address (default: 127.0.0.1)",
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

    if parsed.serve_local:
        launch_viewer = _get_dev_launch_viewer()
        launch_viewer(
            dataset_path,
            port=parsed.port,
            host=parsed.host,
            open_browser=not parsed.no_browser,
            block=True,
        )
        return 0

    out_path = Path(parsed.output) if parsed.output else dataset_path / "viewer.html"
    result_path = export_static_html(dataset_path, output_path=out_path)
    print(f"[Synapse Viewer] Exported self-contained HTML viewer to: {result_path}")
    return 0


def _cmd_view(args: argparse.Namespace) -> dict[str, Any]:
    """Handle 'synapse view' sub-command.

    Default: Export static self-contained HTML file (zero persistent process).
    --serve-local: Launch local loopback HTTP preview server.
    --serve-local with --json: Rejected with ContractValidationError (exit_code=10).
    """
    from synapse.viewer.server import export_static_html

    json_mode = bool(getattr(args, "json", False))
    serve_local = bool(getattr(args, "serve_local", False))

    if json_mode and serve_local:
        raise ContractValidationError(
            "Combining '--serve-local' with '--json' is forbidden. "
            "Background preview servers conflict with headless agent JSON execution.",
            agent_fix_hint="Omit '--serve-local' when running with '--json' to export static HTML.",
        )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise AssetNotFoundError(
            f"Dataset directory '{dataset_path}' does not exist.",
            agent_fix_hint=f"Ensure '{dataset_path}' exists and contains manifest.json.",
        )

    if serve_local:
        host = getattr(args, "host", "127.0.0.1")
        port = int(getattr(args, "port", 8765))
        no_browser = bool(getattr(args, "no_browser", False))
        launch_viewer = _get_dev_launch_viewer()
        launch_viewer(
            dataset_path,
            port=port,
            host=host,
            open_browser=not no_browser,
            block=True,
        )
        return {
            "status": "served",
            "dataset": str(dataset_path),
            "host": host,
            "port": port,
        }

    # Default mode: export self-contained static HTML
    out_path = getattr(args, "output", None)
    output_path = Path(out_path) if out_path else dataset_path / "viewer.html"
    result_path = export_static_html(dataset_path, output_path=output_path)

    if not json_mode:
        print(f"[Synapse Viewer] Exported self-contained HTML viewer to: {result_path}")

    return {
        "output_path": str(result_path),
        "dataset": str(dataset_path),
    }


def register_generate_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    json_parent: argparse.ArgumentParser | None = None,
) -> None:
    """Register ``generate`` and ``view`` subcommands."""
    parents = [json_parent] if json_parent is not None else []
    gen_parser = subparsers.add_parser(
        "generate", help="Generate multi-angle video & FGPP datasets",
        parents=parents,
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

    view_parser = subparsers.add_parser(
        "view",
        help="View dataset (default: static HTML export; "
             "--serve-local: loopback preview)",
        parents=parents,
    )
    view_parser.add_argument(
        "dataset", nargs="?", default="./data/datasets/sample",
        help="Path to dataset directory (containing manifest.json)",
    )
    view_parser.add_argument(
        "--output", "-o", default=None,
        help="Output HTML path for static export "
             "(default: <dataset>/viewer.html)",
    )
    view_parser.add_argument(
        "--serve-local", action="store_true", default=False,
        help="Start local loopback HTTP preview server "
             "instead of static HTML export",
    )
    view_parser.add_argument(
        "--port", "-p", type=int, default=8765,
        help="Port for preview server (default: 8765)",
    )
    view_parser.add_argument(
        "--host", default="127.0.0.1",
        help="Loopback host address (default: 127.0.0.1)",
    )
    view_parser.add_argument(
        "--no-browser", action="store_true", default=False,
        help="Do not auto-open web browser in --serve-local mode",
    )


def dispatch_generate(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
) -> int:
    """Dispatch ``synapse generate``."""
    parsed.json = json_mode
    return _run_with_json_mode(
        _cmd_generate, parsed, json_mode, trace_ctx,
    )


def dispatch_view(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
) -> int:
    """Dispatch ``synapse view``."""
    parsed.json = json_mode
    return _run_with_json_mode(
        _cmd_view, parsed, json_mode, trace_ctx,
    )
