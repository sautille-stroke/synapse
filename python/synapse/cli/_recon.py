"""Recon subcommand handlers: invert-video, closed-loop."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    ExitCode,
    Path,
    SynapseError,
    _run_with_json_mode,
    argparse,
    json,
    sys,
)


def _cmd_recon_invert_video(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run video-to-3D inverse reconstruction (RFC 0060)."""
    from synapse.recon import ReconPipeline

    output_dir = Path(getattr(args, "output", "./output/recon"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target_faces = int(getattr(args, "target_faces", 1500))

    video_path_raw = getattr(args, "video", None)
    if not video_path_raw or not Path(video_path_raw).is_file():
        raise AssetNotFoundError(
            f"Video frames file not found: {video_path_raw!r}",
            agent_fix_hint=(
                "Provide '--video <path>' pointing to a valid .npy "
                "file of shape (T, H, W, 3)."
            ),
        )

    import numpy as np

    frames = np.load(video_path_raw)

    pipe = ReconPipeline(target_quad_faces=target_faces)
    result = pipe.run(frames)

    output_info: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_frames": len(frames),
        "has_trajectory": result.camera_trajectory is not None,
        "has_geometry": result.mesh is not None,
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

    Orchestrates forward-guide -> target-video -> inverse-optimisation
    -> aligned-render -> scalpel-patch.
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


def register_recon_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``recon`` subcommand tree."""
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
    return recon_parser


def dispatch_recon(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    recon_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse recon <sub-command>``."""
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
