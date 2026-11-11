"""CLI entry points for Synapse suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main_data_gen(args: list[str] | None = None) -> int:
    """CLI tool: Generate multi-angle video & FGPP datasets from fine 3D models."""
    from synapse.generators.dataset import DatasetConfig, DatasetGenerator
    from synapse.models.loader import list_builtin_models

    parser = argparse.ArgumentParser(
        prog="synapse-data-gen",
        description="Generate multi-angle video & FGPP ground-truth dataset from 3D models.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="damaged_helmet",
        help=f"Model name or file path (.glb, .gltf, .obj). Built-in: {list_builtin_models()}",
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        default="orbit",
        choices=["orbit", "spiral", "hemisphere", "dolly", "arc"],
        help="Camera trajectory type (default: orbit)",
    )
    parser.add_argument(
        "--frames",
        "-n",
        type=int,
        default=60,
        help="Number of frames to render (default: 60)",
    )
    parser.add_argument(
        "--res",
        "-r",
        type=int,
        default=512,
        help="Resolution width & height in pixels (default: 512)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Video frame rate (default: 24.0)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.4,
        help="Camera orbit distance/radius from origin (default: 2.4)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for held-out views (default: 0.2)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="./data/datasets/sample",
        help="Output dataset directory path",
    )
    parser.add_argument(
        "--view",
        action="store_true",
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

    print(f"[Synapse] Dataset Generator")
    print(f"   Model:      {cfg.model_name}")
    print(f"   Trajectory: {cfg.trajectory_type} ({cfg.num_frames} frames @ {cfg.fps} fps)")
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
        description="Launch interactive web viewer for a Synapse multi-angle dataset.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="./data/datasets/sample",
        help="Path to dataset directory (containing manifest.json)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8765,
        help="Local port to serve HTTP viewer (default: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open default web browser",
    )

    parsed = parser.parse_args(args)
    dataset_path = Path(parsed.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory '{dataset_path}' does not exist.")
        return 1

    launch_viewer(
        dataset_path,
        port=parsed.port,
        open_browser=not parsed.no_browser,
        block=True,
    )
    return 0


def main() -> int:
    """Main CLI router for synapse-bench and subcommands."""
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        return main_data_gen(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "view":
        return main_viewer(sys.argv[2:])
    else:
        print("Synapse CLI Tools:")
        print("  synapse-data-gen  - Generate multi-angle video & FGPP datasets")
        print("  synapse-viewer    - View and inspect datasets in interactive browser")
        return 0
