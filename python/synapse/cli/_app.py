"""Thin entry-point assembly for ``synapse`` and ``synapse-bench`` CLIs.

Imports ``register_*`` and ``dispatch_*`` from each domain module,
builds the argparse tree, and dispatches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from synapse.cli._bench import (
    dispatch_bench,
    register_bench_subparsers,
)
from synapse.cli._fgpp import dispatch_fgpp, register_fgpp_subparser
from synapse.cli._generate import (
    dispatch_generate,
    dispatch_view,
    register_generate_subparser,
)
from synapse.cli._inspect import (
    dispatch_inspect,
    register_inspect_subparser,
)
from synapse.cli._lora import dispatch_lora, register_lora_subparser
from synapse.cli._recon import dispatch_recon, register_recon_subparser
from synapse.cli._refine import (
    dispatch_refine,
    register_refine_subparser,
)
from synapse.cli._rsi import dispatch_rsi, register_rsi_subparser
from synapse.cli._worker import (
    dispatch_worker,
    register_worker_subparser,
)
from synapse.cli.common import TraceContext


def main_synapse(args: list[str] | None = None) -> int:
    """Entry point for the ``synapse`` CLI.

    Subcommands::

        synapse fgpp validate <dir>
        synapse fgpp inspect <dir>
        synapse generate [--model ...] [--trajectory ...] ...
        synapse view [dataset] [--port ...] [--no-browser]
        synapse worker run --bundle <path> [--json]
        synapse lora prepare-data|train|eval|merge [--json]
        synapse inspect mesh|align|eval|refine|export-rl-data [--json]
    """
    parser = argparse.ArgumentParser(
        prog="synapse",
        description="Synapse -- 3D CG Rendering & Neural Video Interop CLI",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Emit machine-readable JSON to stdout (no log noise)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Register all domain subparsers
    fgpp_parser = register_fgpp_subparser(subparsers)
    register_generate_subparser(subparsers)
    worker_parser = register_worker_subparser(subparsers)
    lora_parser = register_lora_subparser(subparsers)
    inspect_parser = register_inspect_subparser(subparsers)
    recon_parser = register_recon_subparser(subparsers)
    refine_parser = register_refine_subparser(subparsers)
    rsi_parser = register_rsi_subparser(subparsers)

    parsed = parser.parse_args(args)
    trace_ctx = TraceContext.from_env_or_dict(dict(os.environ))

    # Propagate --json to the namespace for sub-handlers
    json_mode: bool = parsed.json

    if parsed.command is None:
        if json_mode:
            json.dump(
                {
                    "status": "ok",
                    "exit_code": 0,
                    "trace_id": trace_ctx.trace_id,
                    "message": "No command specified. Use --help.",
                },
                sys.stdout,
                default=str,
            )
            print()
        else:
            parser.print_help()
        return 0

    if parsed.command == "fgpp":
        return dispatch_fgpp(parsed, json_mode, trace_ctx, fgpp_parser)
    elif parsed.command == "generate":
        return dispatch_generate(parsed, json_mode, trace_ctx)
    elif parsed.command == "view":
        return dispatch_view(parsed, json_mode, trace_ctx)
    elif parsed.command == "worker":
        return dispatch_worker(
            parsed, json_mode, trace_ctx, worker_parser,
        )
    elif parsed.command == "lora":
        return dispatch_lora(parsed, json_mode, trace_ctx, lora_parser)
    elif parsed.command == "inspect":
        return dispatch_inspect(
            parsed, json_mode, trace_ctx, inspect_parser,
        )
    elif parsed.command == "recon":
        return dispatch_recon(
            parsed, json_mode, trace_ctx, recon_parser,
        )
    elif parsed.command == "refine":
        return dispatch_refine(
            parsed, json_mode, trace_ctx, refine_parser,
        )
    elif parsed.command == "rsi":
        return dispatch_rsi(parsed, json_mode, trace_ctx, rsi_parser)
    else:
        parser.print_help()
        return 0


def main_bench(args: list[str] | None = None) -> int:
    """Entry point for the ``synapse-bench`` CLI.

    Subcommands::

        synapse-bench list [--tier T1|T2|T3|T4|T5]
        synapse-bench export-fgpp --scenario <id> [--output <dir>]
    """
    parser = argparse.ArgumentParser(
        prog="synapse-bench",
        description="Synapse-Bench -- evaluation dataset & benchmark CLI",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Emit machine-readable JSON to stdout (no log noise)",
    )
    subparsers = parser.add_subparsers(dest="command")

    register_bench_subparsers(subparsers)

    parsed = parser.parse_args(args)
    trace_ctx = TraceContext.from_env_or_dict(dict(os.environ))

    json_mode: bool = parsed.json

    if parsed.command is None:
        if json_mode:
            json.dump(
                {
                    "status": "ok",
                    "exit_code": 0,
                    "trace_id": trace_ctx.trace_id,
                    "message": "No command specified. Use --help.",
                },
                sys.stdout,
                default=str,
            )
            print()
        else:
            parser.print_help()
        return 0

    return dispatch_bench(parsed, json_mode, trace_ctx)
