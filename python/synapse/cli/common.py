"""Shared CLI utilities: JSON-mode wrapper and common imports.

Every domain submodule imports ``_run_with_json_mode`` from here so that
error handling and structured-JSON output are consistent across all
``synapse`` and ``synapse-bench`` subcommands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from synapse.errors import (
    AssetNotFoundError,
    ContractValidationError,
    ExitCode,
    NotImplementedStageError,
    QualityGateError,
    SynapseError,
)
from synapse.telemetry import TraceContext

__all__ = [
    "argparse",
    "json",
    "os",
    "sys",
    "Path",
    "Any",
    "Literal",
    "AssetNotFoundError",
    "ContractValidationError",
    "ExitCode",
    "NotImplementedStageError",
    "QualityGateError",
    "SynapseError",
    "TraceContext",
    "_run_with_json_mode",
]


def _run_with_json_mode(
    func: Any,
    args: argparse.Namespace,
    json_mode: bool,
    trace_ctx: TraceContext,
) -> int:
    """Execute *func*, catch errors, and emit JSON when requested.

    Args:
        func: Callable that accepts *args* and returns a result dict
              (or ``None`` for side-effect-only commands).
        args: Parsed CLI namespace forwarded to *func*.
        json_mode: Whether to emit structured JSON to stdout.
        trace_ctx: W3C trace context for correlation.

    Returns:
        Process exit code (``0`` on success, error's ``exit_code``
        on ``SynapseError``, ``10`` on unexpected exceptions).
    """
    try:
        result = func(args)
        if result is None:
            result = {}
        if json_mode:
            payload = {
                "status": "ok",
                "exit_code": 0,
                "trace_id": trace_ctx.trace_id,
                **result,
            }
            json.dump(payload, sys.stdout, default=str)
            print()  # trailing newline
        return 0
    except SynapseError as exc:
        if json_mode:
            json.dump(
                exc.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
        else:
            print(
                f"Error [{exc.exit_code}]: {exc}",
                file=sys.stderr,
            )
        return int(exc.exit_code)
    except SystemExit:
        # Re-raise argparse help / error exits
        raise
    except Exception as exc:
        err = SynapseError(
            str(exc),
            ExitCode.CONTRACT_ERROR,
            "Check input arguments and retry",
        )
        if json_mode:
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
        else:
            print(f"Unexpected error: {exc}", file=sys.stderr)
        return int(err.exit_code)
