"""Refine subcommand handlers: anatomical."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    ExitCode,
    Path,
    SynapseError,
    _run_with_json_mode,
    argparse,
    json,
    sys,
)


def _cmd_refine_anatomical(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run anatomical local refinement pipeline (RFC 0061).

    Assembles a ``refine_session.v1.0`` payload and executes the
    :class:`~synapse.refine.pipeline.LocalRefinementPipeline`.
    """
    import numpy as np

    from synapse.refine.pipeline import (
        _METHOD_MAP,
        _REGION_MAP,
        LocalRefinementPipeline,
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


def register_refine_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``refine`` subcommand tree."""
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
    return refine_parser


def dispatch_refine(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    refine_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse refine <sub-command>``."""
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
