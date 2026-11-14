"""Worker subcommand handlers: run."""

from __future__ import annotations

from synapse.cli.common import (
    Any,
    ExitCode,
    SynapseError,
    argparse,
    json,
    sys,
)


def register_worker_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``worker`` subcommand tree."""
    worker_parser = subparsers.add_parser(
        "worker", help="Job bundle worker operations",
    )
    worker_sub = worker_parser.add_subparsers(dest="worker_command")

    worker_run = worker_sub.add_parser(
        "run", help="Execute a job bundle",
    )
    worker_run.add_argument(
        "--bundle", required=True,
        help="Path to bundle directory or manifest JSON file",
    )
    return worker_parser


def dispatch_worker(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    worker_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse worker <sub-command>``."""
    worker_cmd = getattr(parsed, "worker_command", None)
    if worker_cmd == "run":
        from synapse.pipeline.bundle_worker import (
            run_worker_cli,
        )

        return run_worker_cli(
            bundle_path=parsed.bundle,
            json_mode=json_mode,
        )
    else:
        if json_mode:
            err = SynapseError(
                "Missing worker sub-command (run)",
                ExitCode.CONTRACT_ERROR,
                "Use 'synapse worker run --bundle <path>'.",
            )
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
            return int(err.exit_code)
        worker_parser.print_help()
        return 0
