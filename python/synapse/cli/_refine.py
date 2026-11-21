"""Refine subcommand handlers: anatomical."""

from __future__ import annotations

from typing import cast

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    ContractValidationError,
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
        fusion_name,
        FusionMethod.LAPLACIAN_FREQUENCY,
    )

    # Build session payload
    import uuid

    session_id = str(uuid.uuid4())

    # --- PR-165: --shot-id HITL pre-validation ---
    shot_id = getattr(args, "shot_id", None)
    if shot_id:
        # Validate shot-id format early to avoid orphan HITL tracking
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise ContractValidationError(
                f"Invalid --shot-id value: {shot_id!r}. Must be a non-empty string.",
                agent_fix_hint="Provide a valid shot identifier, e.g. '--shot-id shot_001'.",
            )

    payload = build_refine_session_payload(
        session_id=session_id,
        regions=region_names,
        fusion_method=fusion_name,
    )

    # Load frames — require real input (AUD-P0-01: no fake synthesis)
    video_path_raw = getattr(args, "video", None)
    if not video_path_raw:
        raise ContractValidationError(
            "Missing required '--video' argument for anatomical refinement",
            agent_fix_hint=(
                "Provide '--video <path>' pointing to a valid .npy "
                "file of shape (T, H, W, 3) float32 ACEScg."
            ),
        )
    if not Path(video_path_raw).is_file():
        raise AssetNotFoundError(
            f"Video frames file not found: {video_path_raw!r}",
            agent_fix_hint=(
                "Provide '--video <path>' pointing to an existing .npy "
                "file of shape (T, H, W, 3) float32 ACEScg."
            ),
        )
    frames = np.load(video_path_raw).astype(np.float32)

    height, width = frames.shape[1], frames.shape[2]

    # Real ROI input required (H6-P1-05: no fake ROI synthesis)
    roi_meta_path = getattr(args, "roi_meta", None)
    if not roi_meta_path:
        raise AssetNotFoundError(
            "Missing required '--roi-meta' argument for anatomical refinement",
            agent_fix_hint=(
                "Provide '--roi-meta <path>' pointing to a valid ROI metadata file (JSON or NPY)."
            ),
        )
    if not Path(roi_meta_path).is_file():
        raise AssetNotFoundError(
            f"ROI metadata file not found: {roi_meta_path!r}",
            agent_fix_hint=(
                "Provide '--roi-meta <path>' pointing to an existing ROI metadata file."
            ),
        )

    meta_path = Path(roi_meta_path)
    if meta_path.suffix.lower() not in (".json", ".npy"):
        raise ContractValidationError(
            f"Unsupported ROI file extension: {meta_path.suffix!r}. Expected .json or .npy",
            agent_fix_hint="Provide a valid .json or .npy file for --roi-meta.",
        )

    rois_per_frame: list[list[RefinementROI]] = [[] for _ in range(frames.shape[0])]
    if meta_path.suffix.lower() == ".json":
        from synapse.refine.roi_tracker import parse_rois_json

        meta_content = json.loads(meta_path.read_text(encoding="utf-8"))
        rois_per_frame = parse_rois_json(
            meta_content,
            num_frames=frames.shape[0],
            height=height,
            width=width,
            frames=frames,
        )
    elif meta_path.suffix.lower() == ".npy":
        arr = np.load(str(meta_path))
        if arr.ndim == 2 and arr.shape[1] == 4:
            for t_idx in range(min(frames.shape[0], arr.shape[0])):
                bbox = arr[t_idx].astype(np.float32)
                x1, y1 = max(0, int(round(bbox[0]))), max(0, int(round(bbox[1])))
                x2, y2 = min(width, int(round(bbox[2]))), min(height, int(round(bbox[3])))
                if x2 > x1 and y2 > y1:
                    crop = frames[t_idx, y1:y2, x1:x2, :].copy()
                    region = _REGION_MAP.get(region_names[0], _REGION_MAP.get("HAND"))
                    if region:
                        rois_per_frame[t_idx].append(
                            RefinementROI(
                                region=region,
                                bbox_xyxy=bbox,
                                crop_acescg=crop,
                                confidence=0.9,
                                frame_index=t_idx,
                            )
                        )
        elif arr.ndim == 3 and arr.shape[2] == 4:
            for t_idx in range(min(frames.shape[0], arr.shape[0])):
                for k in range(arr.shape[1]):
                    bbox = arr[t_idx, k].astype(np.float32)
                    x1, y1 = max(0, int(round(bbox[0]))), max(0, int(round(bbox[1])))
                    x2, y2 = min(width, int(round(bbox[2]))), min(height, int(round(bbox[3])))
                    if x2 > x1 and y2 > y1:
                        crop = frames[t_idx, y1:y2, x1:x2, :].copy()
                        rname = region_names[k % len(region_names)]
                        region = _REGION_MAP.get(rname, _REGION_MAP.get("HAND"))
                        if region:
                            rois_per_frame[t_idx].append(
                                RefinementROI(
                                    region=region,
                                    bbox_xyxy=bbox,
                                    crop_acescg=crop,
                                    confidence=0.9,
                                    frame_index=t_idx,
                                )
                            )

    # Pre-check HITL 15-minute circuit breaker BEFORE running heavy refinement
    from synapse.bridge.hitl import CommandIRRecord, PersistentHitlLedger
    from synapse.errors import evaluate_hitl_circuit_breaker

    ledger = PersistentHitlLedger()
    group_id = str(
        getattr(args, "shot_id", None)
        or getattr(args, "asset_id", None)
        or output_dir.name
        or "default_refine"
    )
    evaluate_hitl_circuit_breaker(
        ledger.get_accumulated_elapsed(group_id),
        max_seconds=900.0,
        domain="video_shot",
    )

    pipeline = LocalRefinementPipeline()
    result = pipeline.run(
        frames_acescg=frames,
        rois_per_frame=rois_per_frame,
        fusion_method=fusion_method,
    )

    # Record action to PersistentHitlLedger to track cumulative budget
    elapsed = float(result.elapsed_seconds)
    record = CommandIRRecord(
        intent_group_id=group_id,
        intent="anatomical_refine_cli",
        elapsed_seconds=elapsed,
    )
    ledger.record_action(record, max_seconds=900.0)

    # Save refined frames
    np.save(str(output_dir / "refined_frames.npy"), result.frames_refined)

    diag = result.to_dict()
    diag_path = output_dir / "refine_diagnostics.json"
    diag_path.write_text(
        json.dumps(diag, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "session_payload": payload,
        "output_dir": str(output_dir),
        "diagnostics": diag,
    }


def register_refine_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    json_parent: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Register the ``refine`` subcommand tree."""
    parents = [json_parent] if json_parent is not None else []
    refine_parser = subparsers.add_parser(
        "refine",
        help="Anatomical local refinement commands (RFC 0061)",
        parents=parents,
    )
    refine_sub = refine_parser.add_subparsers(
        dest="refine_command",
    )

    refine_anatomical = refine_sub.add_parser(
        "anatomical",
        help="Run end-to-end anatomical refinement pipeline",
        parents=parents,
    )
    refine_anatomical.add_argument(
        "--video",
        default=None,
        help="Path to video frames .npy file (T, H, W, 3) float32 ACEScg",
    )
    refine_anatomical.add_argument(
        "--roi-meta",
        default=None,
        help="Path to ROI metadata JSON or .npy file",
    )
    refine_anatomical.add_argument(
        "--output",
        "-o",
        default="./output/refine",
        help="Output directory for refined frames and diagnostics",
    )
    refine_anatomical.add_argument(
        "--regions",
        default="HAND,FACE",
        help="Comma-separated anatomical regions (default: HAND,FACE)",
    )
    refine_anatomical.add_argument(
        "--fusion-method",
        default="LAPLACIAN_FREQUENCY",
        choices=[
            "LAPLACIAN_FREQUENCY",
            "CASCADE",
            "LATENT_TIMESTEP_SWITCH",
            "UNCERTAINTY_WEIGHTED",
            "POISSON_GRADIENT",
        ],
        help="Fusion strategy (default: LAPLACIAN_FREQUENCY)",
    )
    refine_anatomical.add_argument(
        "--shot-id",
        default=None,
        help=(
            "Shot identifier for HITL budget tracking.  Required for "
            "production use; omit only for quick local tests."
        ),
    )
    return cast(argparse.ArgumentParser, refine_parser)


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
            _cmd_refine_anatomical,
            parsed,
            json_mode,
            trace_ctx,
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
        return int(ExitCode.CONTRACT_ERROR)
