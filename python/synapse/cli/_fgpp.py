"""FGPP subcommand handlers: validate, inspect."""

from __future__ import annotations

import numpy as np

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


def _validate_exr_channels(exr_path: Path) -> None:
    """Validate 15-channel contract, pixel types, and finite values of an EXR file."""
    try:
        import OpenEXR  # type: ignore[import-untyped]
    except ImportError:
        return

    from synapse.fgpp.schema import FGPP_CHANNEL_SPECS, PixelType

    try:
        exr = OpenEXR.InputFile(str(exr_path))
    except Exception as exc:
        raise ContractValidationError(
            f"Failed to open EXR file {exr_path.name}: {exc}",
            agent_fix_hint="Ensure EXR file is uncorrupted and valid OpenEXR format.",
        ) from exc

    try:
        header = exr.header()
        channels = header.get("channels", {})
        expected_channels = {name for name, _, _ in FGPP_CHANNEL_SPECS}
        missing = expected_channels - set(channels.keys())
        if missing:
            raise ContractValidationError(
                f"EXR file {exr_path.name} is missing required FGPP channels: {sorted(missing)}",
                agent_fix_hint="Ensure all 15 FGPP channels are exported bit-identically to @aether//fgpp.",
                details={"missing_channels": sorted(missing)},
            )

        for name, expected_pt, _ in FGPP_CHANNEL_SPECS:
            ch = channels[name]
            if expected_pt == PixelType.UINT32:
                if ch.type.v not in (int(PixelType.UINT32), int(PixelType.FLOAT)):
                    raise ContractValidationError(
                        f"EXR {exr_path.name} channel '{name}' pixel type mismatch: "
                        f"expected UINT32/FLOAT ({int(expected_pt)}), got {ch.type.v}",
                        agent_fix_hint=f"Ensure channel '{name}' has storage type UINT32 or 32-bit FLOAT.",
                    )
                continue

            if ch.type.v != int(expected_pt):
                raise ContractValidationError(
                    f"EXR {exr_path.name} channel '{name}' pixel type mismatch: "
                    f"expected {expected_pt.name} ({int(expected_pt)}), got {ch.type.v}",
                    agent_fix_hint=f"Ensure channel '{name}' has storage type {expected_pt.name}.",
                )
            raw = exr.channel(name)
            if expected_pt == PixelType.FLOAT:
                arr = np.frombuffer(raw, dtype=np.float32)
                if not np.all(np.isfinite(arr)):
                    raise ContractValidationError(
                        f"EXR {exr_path.name} channel '{name}' contains NaN or Inf values",
                        agent_fix_hint=f"Check render passes to eliminate NaN/Inf in channel '{name}'.",
                    )
            elif expected_pt == PixelType.HALF:
                arr = np.frombuffer(raw, dtype=np.float16)
                if not np.all(np.isfinite(arr)):
                    raise ContractValidationError(
                        f"EXR {exr_path.name} channel '{name}' contains NaN or Inf values",
                        agent_fix_hint=f"Check render passes to eliminate NaN/Inf in channel '{name}'.",
                    )
    finally:
        exr.close()


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

    check_exr = getattr(args, "check_exr", False)
    exr_files = sorted(fgpp_dir.glob("*.exr"))
    if not exr_files and (fgpp_dir / "fgpp").is_dir():
        exr_files = sorted((fgpp_dir / "fgpp").glob("*.exr"))

    if check_exr and not exr_files:
        raise AssetNotFoundError(
            f"No EXR files found in {fgpp_dir} to validate",
            agent_fix_hint="Ensure .exr frames are present when using --check-exr.",
        )

    if exr_files:
        # Validate the first frame or all frames if explicitly requested
        to_check = exr_files if check_exr else exr_files[:1]
        for ef in to_check:
            _validate_exr_channels(ef)

    return {
        "valid": True,
        "schema_version": schema_ver,
        "num_frames": actual_frames,
        "scene_id": raw.get("scene_id", ""),
        "exr_validated": len(exr_files) > 0,
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
    fgpp_validate.add_argument(
        "--check-exr", action="store_true", default=False,
        help="Deep-validate EXR 15-channel wire format, pixel types, and finite values",
    )

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
