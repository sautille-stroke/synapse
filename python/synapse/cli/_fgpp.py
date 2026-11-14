"""FGPP subcommand handlers: validate, inspect."""

from __future__ import annotations

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


def register_fgpp_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``fgpp`` subcommand tree."""
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
    return fgpp_parser


def dispatch_fgpp(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    fgpp_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse fgpp <sub-command>``."""
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
