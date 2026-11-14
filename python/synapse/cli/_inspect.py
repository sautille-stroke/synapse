"""Inspect subcommand handlers: mesh, align, eval, evaluate, refine, export-rl-data."""

from __future__ import annotations

from synapse.cli.common import (
    Any, AssetNotFoundError, ContractValidationError, ExitCode,
    NotImplementedStageError, Path, QualityGateError,
    SynapseError, _run_with_json_mode, argparse, json, sys,
)


def _cmd_inspect_mesh(args: argparse.Namespace) -> dict[str, Any]:
    """Run Tier 0 geometric topology validation on a mesh file."""
    import numpy as np
    from synapse.inspect.geometry_gate import validate_mesh_geometry

    mesh_path = Path(getattr(args, "mesh_path", ""))
    if not mesh_path.is_file():
        raise AssetNotFoundError(
            f"Mesh file not found: {mesh_path}",
            agent_fix_hint=(
                f"Ensure '{mesh_path}' exists and is a valid "
                ".obj / .glb / .ply mesh file."
            ),
        )
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
                f"trimesh is required for loading {mesh_path.suffix!r} "
                "files. Install with: pip install 'synapse[mesh]'",
                agent_fix_hint="pip install trimesh>=4.0.0",
            )

    raw_backend = str(getattr(args, "backend", "native"))
    backend = "python" if raw_backend == "python" else "native"
    result = validate_mesh_geometry(
        vertices, faces,
        require_watertight=bool(getattr(args, "require_watertight", False)),
        backend=backend,
    )
    result_dict = result.model_dump()

    if not args.json:
        status = "PASS" if result.passed else "FAIL"
        print(f"[Tier 0 Geometry Gate] {mesh_path.name}: {status}")
        for lbl, val in [
            ("Vertices", result.num_vertices),
            ("Faces", result.num_faces),
            ("Manifold", result.is_manifold),
            ("Watertight", result.is_watertight),
            ("Degenerate", len(result.degenerate_face_indices)),
            ("Self-Intersecting", f"{len(result.self_intersecting_pairs)} pairs"),
            ("Components", result.num_connected_components),
            ("Float Debris", f"{len(result.floating_island_face_indices)} faces"),
        ]:
            print(f"  {lbl:20s} {val}")
        if result.failure_reasons:
            print("  Failures:")
            for reason in result.failure_reasons:
                print(f"    - {reason}")

    report_path = getattr(args, "report", None)
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(result_dict, indent=2, default=str), encoding="utf-8",
        )
        if not args.json:
            print(f"  Report: {report_file}")

    if not result.passed:
        raise QualityGateError(
            "Tier 0 geometry gate failed: " + "; ".join(result.failure_reasons),
            agent_fix_hint="Fix mesh topology issues before proceeding.",
        )
    return result_dict


def _cmd_inspect_align(args: argparse.Namespace) -> dict[str, Any]:
    """Align reference image to mesh."""
    raise NotImplementedStageError(
        "CLI 'synapse inspect align' standalone file pipeline is not "
        "yet implemented; refusing to emit silent success.",
        agent_fix_hint=(
            "Use 'synapse.inspect.align' functions "
            "(generate_fibonacci_sphere_cameras, compute_silhouette_iou) "
            "directly in Python."
        ),
        details={
            "mesh": getattr(args, "mesh", ""),
            "image": getattr(args, "image", ""),
        },
    )


def _cmd_inspect_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate asset using Visual Jev critic."""
    from synapse.inspect.jev_critic import MockJevTransport, VisualJevCritic

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


def _cmd_inspect_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Batch-evaluate 3D models (Tier-0 topology + multi-view + HTML/JSON)."""
    from synapse.inspect.model_evaluator import evaluate_3d_models_directory

    json_mode = bool(getattr(args, "json", False))
    ref_raw = getattr(args, "reference_dir", None)
    eval_backend_raw = str(getattr(args, "backend", "native"))
    eval_backend = "python" if eval_backend_raw == "python" else "native"
    data = evaluate_3d_models_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        reference_dir=Path(ref_raw) if ref_raw else None,
        profile_name=str(getattr(args, "profile", "TRACK_A_ASSET")),
        resolution=int(getattr(args, "resolution", 256)),
        num_views=int(getattr(args, "num_views", 8)),
        log_fn=None if json_mode else (lambda m: print(m)),
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


def _cmd_inspect_refine(args: argparse.Namespace) -> dict[str, Any]:
    """Run targeted refinement from an InspectionReport."""
    from synapse.inspect.flywheel_exporter import run_targeted_refinement

    report_path = getattr(args, "report", None)
    if not report_path or not Path(report_path).is_file():
        raise AssetNotFoundError(
            f"InspectionReport file not found: {report_path!r}. "
            "Refusing to run refinement on an empty defect list.",
            agent_fix_hint="Pass a valid InspectionReport JSON via '--report <path>'.",
        )
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"Invalid JSON in InspectionReport '{report_path}': {exc}",
            agent_fix_hint="Ensure --report points to a valid JSON file.",
        ) from exc
    return run_targeted_refinement(report=report)


def _cmd_inspect_export_rl_data(args: argparse.Namespace) -> dict[str, Any]:
    """Export RL flywheel training data."""
    from synapse.inspect.flywheel_exporter import export_rl_flywheel_dataset

    raw_fmt = str(getattr(args, "format", "dpo"))
    fmt = "prm" if raw_fmt == "prm" else "dpo"
    return export_rl_flywheel_dataset(
        log_dir=Path(args.log_dir),
        output_path=Path(args.output),
        format=fmt,
    )


# -------------------------------------------------------------------
# Subparser registration
# -------------------------------------------------------------------

def register_inspect_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``inspect`` subcommand tree."""
    p = subparsers.add_parser("inspect", help="3D asset inspection and refinement")
    sub = p.add_subparsers(dest="inspect_command")

    m = sub.add_parser("mesh", help="Run Tier 0 geometric topology gate on a mesh file")
    m.add_argument("mesh_path", help="Path to mesh file (.obj, .glb, .ply)")
    m.add_argument("--report", default=None, help="Output JSON report file path")
    m.add_argument("--backend", default="native", choices=["native", "python"],
                    help="Backend for geometry checks (default: native)")
    m.add_argument("--require-watertight", action="store_true",
                    help="Require mesh to be watertight (closed)")

    a = sub.add_parser("align", help="Align reference image to mesh")
    a.add_argument("--mesh", default="", help="Path to mesh file")
    a.add_argument("--image", default="", help="Path to reference image")

    ev = sub.add_parser("eval", help="Evaluate asset using Visual Jev critic")
    ev.add_argument("--asset-id", default="unknown", help="Asset identifier")
    ev.add_argument("--asset-name", default="Unknown", help="Human-readable asset name")

    evl = sub.add_parser("evaluate",
                          help="Batch-evaluate 3D models and generate HTML/JSON reports")
    evl.add_argument("--input-dir", required=True,
                      help="Directory containing .obj/.glb/.ply model files")
    evl.add_argument("--output-dir", required=True,
                      help="Output directory for eval_report.html and eval_results.json")
    evl.add_argument("--reference-dir", default=None,
                      help="Optional directory with reference meshes")
    evl.add_argument("--profile", "--gate-profile", dest="profile",
                      default="TRACK_A_ASSET",
                      help="Gate profile (TRACK_A_ASSET, TRACK_B_VIDEO, RSI_PROMOTION)")
    evl.add_argument("--resolution", type=int, default=256,
                      help="Multi-view render resolution (default: 256)")
    evl.add_argument("--num-views", "--views", dest="num_views", type=int, default=8,
                      help="Number of multi-view orbit screenshots (default: 8)")
    evl.add_argument("--backend", default="native", choices=["native", "python"],
                      help="Geometry gate backend (default: native)")

    r = sub.add_parser("refine", help="Run targeted refinement from InspectionReport")
    r.add_argument("--report", default=None, help="Path to InspectionReport JSON file")

    rl = sub.add_parser("export-rl-data", help="Export RL flywheel training data (DPO/PRM)")
    rl.add_argument("--log-dir", required=True, help="Path to intent group log directory")
    rl.add_argument("--output", required=True, help="Output JSONL file path")
    rl.add_argument("--format", default="dpo", choices=["dpo", "prm"],
                     help="Export format (default: dpo)")
    return p


# -------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------

_INSPECT_HANDLERS = {
    "mesh": _cmd_inspect_mesh,
    "align": _cmd_inspect_align,
    "eval": _cmd_inspect_eval,
    "refine": _cmd_inspect_refine,
    "export-rl-data": _cmd_inspect_export_rl_data,
}


def dispatch_inspect(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    inspect_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse inspect <sub-command>``."""
    insp_cmd = getattr(parsed, "inspect_command", None)

    if insp_cmd == "evaluate":
        parsed.json = json_mode
        return _run_with_json_mode(
            _cmd_inspect_evaluate, parsed, json_mode, trace_ctx,
        )

    handler = _INSPECT_HANDLERS.get(insp_cmd)  # type: ignore[arg-type]
    if handler is not None:
        return _run_with_json_mode(handler, parsed, json_mode, trace_ctx)

    if json_mode:
        err = SynapseError(
            "Missing inspect sub-command "
            "(mesh|align|eval|evaluate|refine|export-rl-data)",
            ExitCode.CONTRACT_ERROR,
            "Use 'synapse inspect <sub-command> --help'.",
        )
        json.dump(err.to_json_dict(trace_ctx.trace_id), sys.stdout, default=str)
        print()
        return int(err.exit_code)
    inspect_parser.print_help()
    return 0
