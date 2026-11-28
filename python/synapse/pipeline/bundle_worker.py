"""Stateless RFC 0039 JobBundleWorker with proactive OOM fallback.

Implements the ``synapse worker run --bundle <path>`` execution model:
a zero-daemon, single-execution worker that reads a ``BundleManifest``,
dispatches to a registered stage handler, and writes a ``ResultManifest``
with SHA256-checksummed output files.

GPU memory forecasting for the ``neural_video_generate`` stage uses
:class:`~synapse.telemetry.MemoryForecaster` to proactively select a
fallback level (spatial tiling or short causal window) before any CUDA
allocation, avoiding hard OOM crashes.

Exit codes follow AGENTS.md section 3::

    0  -- success
    10 -- contract validation failure
    20 -- missing asset
    30 -- GPU OOM / backend unreachable
    40 -- circuit breaker / quality gate
    50 -- not implemented
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from synapse.pipeline.manifest import (
    _TRACEPARENT_RE,
    CLOSURE_TYPE_MAP,
    BundleManifest,
    ClosureType,
    EngineSpec,
    HardwareBudget,
    OutputFileEntry,
    OutputSink,
    PresignedEndpoint,
    ResultManifest,
    SceneSpec,
)
from synapse.pipeline.sandbox import (
    SandboxViolation,
    _check_ancestor_symlinks,
    _compute_sha256,
    _extract_manifest_traceboard_param,
    _validate_manifest_traceboard_dir,
    validate_deferred_asset_manifest,
    validate_io_mappings,
    validate_params_paths,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage handler registry
# ---------------------------------------------------------------------------

_STAGE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {}
_STAGE_ADAPTERS_LOADED: bool = False


def _ensure_stage_adapters_loaded() -> None:
    """Import ``synapse.pipeline.stages`` once to register all stage adapters."""
    global _STAGE_ADAPTERS_LOADED
    if not _STAGE_ADAPTERS_LOADED:
        _STAGE_ADAPTERS_LOADED = True
        import synapse.pipeline.stages  # noqa: F401


def register_stage(
    name: str,
) -> Callable[
    [Callable[..., dict[str, Any]]],
    Callable[..., dict[str, Any]],
]:
    """Decorator that registers a stage handler function.

    Usage::

        @register_stage("my_stage")
        def _handle_my_stage(
            manifest: BundleManifest,
        ) -> dict[str, Any]:
            ...

    Args:
        name: Unique stage name used in ``BundleManifest.stage``.

    Returns:
        Decorator that registers the handler and returns it unchanged.
    """

    def _decorator(
        fn: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:
        if name in _STAGE_HANDLERS and _STAGE_HANDLERS[name] is not fn:
            existing = _STAGE_HANDLERS[name]
            raise RuntimeError(
                f"Duplicate stage registration: '{name}' is already "
                f"registered to {existing.__module__}.{existing.__qualname__}; "
                f"cannot re-register to {fn.__module__}.{fn.__qualname__}."
            )
        _STAGE_HANDLERS[name] = fn
        return fn

    return _decorator


def get_registered_stages() -> list[str]:
    """Return sorted list of registered stage names."""
    _ensure_stage_adapters_loaded()
    return sorted(_STAGE_HANDLERS.keys())


def _emit_cli_error_payload(err: dict[str, Any], json_mode: bool) -> None:
    """Write a structured CLI error payload to stdout (JSON) or stderr."""
    if json_mode:
        json.dump(err, sys.stdout, default=str)
        print()
    else:
        print(f"Error: {err['message']}", file=sys.stderr)


# ---------------------------------------------------------------------------
# JobBundleWorker
# ---------------------------------------------------------------------------


def _violation_to_result(
    manifest: BundleManifest,
    violation: SandboxViolation,
    t_start: float,
) -> ResultManifest:
    """Convert a sandbox violation into an error :class:`ResultManifest`."""
    return ResultManifest(
        job_id=manifest.job_id,
        stage=manifest.stage,
        status="error",
        exit_code=violation.exit_code,
        error_type=violation.error_type,
        message=violation.message,
        timings={"total_seconds": time.monotonic() - t_start},
        agent_fix_hint=violation.agent_fix_hint,
    )


class JobBundleWorker:
    """Stateless, single-execution RFC 0039 job bundle worker.

    Reads a :class:`BundleManifest`, dispatches to the registered stage
    handler, computes SHA256 checksums for output files, and returns a
    :class:`ResultManifest`.

    Args:
        worker_id: Identifier for this worker instance.
        workspace_root: Optional workspace root for path sandboxing.
            When set, all input/output paths must be ``is_relative_to``
            this directory.
    """

    def __init__(
        self,
        worker_id: str = "local",
        workspace_root: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.workspace_root: Path | None = (
            Path(workspace_root).resolve() if workspace_root else None
        )

    def execute(
        self,
        manifest: BundleManifest,
    ) -> ResultManifest:
        """Execute a single job bundle stage with Traceboard stage telemetry.

        Args:
            manifest: The job bundle manifest to execute.

        Returns:
            A :class:`ResultManifest` with status, checksums, and
            timing information.
        """
        from synapse.observability import safe_log_scalar, safe_log_text, traceboard_stage

        trace_id = (
            manifest.traceparent.split("-")[1]
            if manifest.traceparent and "-" in manifest.traceparent
            else ""
        ) or "local"

        with traceboard_stage(
            f"worker/{manifest.stage}",
            job_id=manifest.job_id,
            trace_id=trace_id,
        ) as stage:
            result = self._execute_inner(manifest)
            elapsed_s = float(result.timings.get("total_seconds", 0.0))
            safe_log_scalar("worker/exit_code", float(result.exit_code))
            safe_log_scalar("worker/elapsed_s", elapsed_s)
            safe_log_scalar("worker/output_file_count", float(len(result.output_files)))
            safe_log_scalar("worker/succeeded", 1.0 if result.exit_code == 0 else 0.0)
            safe_log_text("worker/trace_id", trace_id)
            if result.fallback_level:
                safe_log_text("worker/fallback_level", str(result.fallback_level))
            if stage is not None:
                if result.exit_code == 0:
                    stage.set_pass()
                else:
                    stage.set_fail(f"exit_{result.exit_code}:{result.error_type or 'error'}")
            return result

    def _execute_inner(
        self,
        manifest: BundleManifest,
    ) -> ResultManifest:
        t_start = time.monotonic()
        _ensure_stage_adapters_loaded()

        manifest = manifest.model_copy(deep=True)

        violation = validate_io_mappings(
            manifest.inputs,
            manifest.outputs,
            self.workspace_root,
        )
        if violation is not None:
            return _violation_to_result(manifest, violation, t_start)

        violation = validate_params_paths(manifest.params, self.workspace_root)
        if violation is not None:
            return _violation_to_result(manifest, violation, t_start)

        if self.workspace_root is not None:
            manifest.params["_workspace_root"] = str(self.workspace_root)

        deferred_manifest = manifest.params.get("_asset_manifest", [])
        violation = validate_deferred_asset_manifest(
            deferred_manifest,
            self.workspace_root,
        )
        if violation is not None:
            return _violation_to_result(manifest, violation, t_start)

        handler = _STAGE_HANDLERS.get(manifest.stage)
        if handler is None:
            return ResultManifest(
                job_id=manifest.job_id,
                stage=manifest.stage,
                status="error",
                exit_code=10,
                error_type="ContractValidationError",
                message=(
                    f"Unknown stage '{manifest.stage}'. "
                    f"Registered stages: {get_registered_stages()}"
                ),
                timings={"total_seconds": time.monotonic() - t_start},
                agent_fix_hint=(
                    f"Unknown stage '{manifest.stage}'. "
                    f"Registered stages: {get_registered_stages()}"
                ),
            )

        try:
            result_data = handler(manifest)
        except Exception as exc:
            t_end = time.monotonic()
            exit_code = 10
            from synapse.errors import SynapseError

            if isinstance(exc, SynapseError):
                exit_code = int(exc.exit_code)

            return ResultManifest(
                job_id=manifest.job_id,
                stage=manifest.stage,
                status="error",
                exit_code=exit_code,
                error_type=type(exc).__name__,
                message=str(exc),
                timings={"total_seconds": t_end - t_start},
                agent_fix_hint=(getattr(exc, "agent_fix_hint", "") or str(exc)),
            )

        t_end = time.monotonic()

        output_paths: dict[str, str] = result_data.get("output_paths", {})
        output_files: dict[str, OutputFileEntry] = {}
        for key, fpath in output_paths.items():
            p = Path(fpath)
            if p.is_file():
                output_files[key] = OutputFileEntry(
                    path=str(p),
                    sha256=_compute_sha256(p),
                    size_bytes=p.stat().st_size,
                )
            elif p.is_dir():
                subfiles = sorted(f for f in p.rglob("*") if f.is_file())
                hasher = hashlib.sha256()
                total_bytes = 0
                for sf in subfiles:
                    rel_p = str(sf.relative_to(p)).replace("\\", "/")
                    hasher.update(rel_p.encode("utf-8"))
                    with open(sf, "rb") as fh:
                        while True:
                            chunk = fh.read(65536)
                            if not chunk:
                                break
                            hasher.update(chunk)
                    total_bytes += sf.stat().st_size
                output_files[key] = OutputFileEntry(
                    path=str(p),
                    sha256=hasher.hexdigest(),
                    size_bytes=total_bytes,
                )

        fallback_level = result_data.get("fallback_level")
        status: Literal["success", "fallback", "error"] = (
            "fallback" if fallback_level else "success"
        )

        return ResultManifest(
            job_id=manifest.job_id,
            stage=manifest.stage,
            status=status,
            exit_code=0,
            output_files=output_files,
            reanchor_requests=result_data.get("reanchor_requests"),
            timings={"total_seconds": t_end - t_start},
            fallback_level=fallback_level,
        )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def run_worker_cli(
    bundle_path: str,
    json_mode: bool = False,
    trace_ctx: Any | None = None,
    workspace_root: str | None = None,
) -> int:
    """Execute a job bundle from a JSON manifest file.

    Reads ``bundle_manifest.json`` from *bundle_path*, runs the worker,
    and writes ``result_manifest.json`` alongside the input.

    Automatically detects RFC 0039 upstream ``JobBundle`` wire format
    (by presence of ``closure_type`` key) and routes through
    :meth:`BundleManifest.from_rfc0039_job_bundle` to normalise it.

    Args:
        bundle_path: Path to the bundle directory or manifest file.
        json_mode: If ``True``, emit result JSON to stdout.
        trace_ctx: Optional :class:`~synapse.telemetry.TraceContext`
            for correlation.  Falls back to env ``TRACEPARENT`` or
            manifest ``traceparent``.
        workspace_root: Optional workspace root path for sandboxing.

    Returns:
        Process exit code from the result manifest.
    """
    from synapse.observability import (
        get_probe,
        init_traceboard,
        safe_log_scalar,
        shutdown_traceboard,
    )
    from synapse.telemetry import TraceContext

    if trace_ctx is None:
        trace_ctx = TraceContext.from_env_or_dict(dict(os.environ))

    bp = Path(bundle_path)
    if bp.is_dir():
        manifest_file = bp / "bundle_manifest.json"
    else:
        manifest_file = bp
        bp = bp.parent

    tb_started_here = False
    tb_exit_code = 1
    env_tb_dir = os.environ.get("SYNAPSE_TRACEBOARD_DIR")
    if env_tb_dir and get_probe() is None:
        probe_inst = init_traceboard(
            str(env_tb_dir),
            run_name=f"worker_{bp.name}",
            engine="synapse-worker",
            trace_id=trace_ctx.trace_id or None,
        )
        tb_started_here = probe_inst is not None

    try:
        if not manifest_file.is_file():
            tb_exit_code = 20
            safe_log_scalar("worker/exit_code", 20.0)
            err = {
                "status": "error",
                "exit_code": 20,
                "error_type": "AssetNotFoundError",
                "message": f"Bundle manifest not found: {manifest_file}",
                "trace_id": trace_ctx.trace_id,
                "agent_fix_hint": f"Ensure '{manifest_file}' exists and is a valid JSON file.",
            }
            _emit_cli_error_payload(err, json_mode)
            return 20

        try:
            raw = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            tb_exit_code = 10
            safe_log_scalar("worker/exit_code", 10.0)
            err = {
                "status": "error",
                "exit_code": 10,
                "error_type": "ContractValidationError",
                "message": f"Invalid JSON in bundle manifest: {exc}",
                "trace_id": trace_ctx.trace_id,
                "agent_fix_hint": (
                    f"Fix the JSON syntax in the bundle manifest file '{manifest_file}' and retry."
                ),
            }
            _emit_cli_error_payload(err, json_mode)
            return 10

        try:
            if "closure_type" in raw:
                manifest = BundleManifest.from_rfc0039_job_bundle(
                    raw, workspace_root=workspace_root
                )
            else:
                manifest = BundleManifest(**raw)
        except Exception as exc:
            exit_code = 10
            from synapse.errors import SynapseError

            if isinstance(exc, SynapseError):
                exit_code = int(exc.exit_code)
            tb_exit_code = exit_code
            safe_log_scalar("worker/exit_code", float(exit_code))
            err = {
                "status": "error",
                "exit_code": exit_code,
                "error_type": type(exc).__name__,
                "message": f"Bundle manifest validation failed: {exc}",
                "trace_id": trace_ctx.trace_id,
                "agent_fix_hint": (
                    getattr(exc, "agent_fix_hint", "")
                    or "Ensure the bundle manifest conforms to the "
                    "synapse.job_bundle.v1.0 schema or is a valid "
                    "RFC 0039 JobBundle with 'closure_type'. Check required "
                    "fields: job_id, stage/closure_type, schema_version."
                ),
            }
            _emit_cli_error_payload(err, json_mode)
            return exit_code

        if manifest.traceparent and not trace_ctx.trace_id:
            trace_ctx = TraceContext.from_env_or_dict(
                {"TRACEPARENT": manifest.traceparent},
            )

        if get_probe() is None:
            tb_param = _extract_manifest_traceboard_param(manifest.params)
            if tb_param is not None:
                try:
                    tb_dir = _validate_manifest_traceboard_dir(tb_param, workspace_root)
                except Exception as exc:
                    from synapse.errors import SynapseError

                    exit_code = int(exc.exit_code) if isinstance(exc, SynapseError) else 10
                    tb_exit_code = exit_code
                    safe_log_scalar("worker/exit_code", float(exit_code))
                    err = (
                        exc.to_json_dict(trace_id=trace_ctx.trace_id)
                        if isinstance(exc, SynapseError)
                        else {
                            "status": "error",
                            "exit_code": exit_code,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "trace_id": trace_ctx.trace_id,
                            "agent_fix_hint": (
                                "Provide a valid Traceboard directory inside workspace_root."
                            ),
                        }
                    )
                    _emit_cli_error_payload(err, json_mode)
                    return exit_code

                probe_inst = init_traceboard(
                    str(tb_dir),
                    run_name=f"worker_{manifest.job_id}",
                    engine="synapse-worker",
                    trace_id=trace_ctx.trace_id or None,
                    stage=manifest.stage,
                )
                tb_started_here = probe_inst is not None

        worker = JobBundleWorker(workspace_root=workspace_root)
        result = worker.execute(manifest)
        tb_exit_code = int(result.exit_code)

        result_file = bp / "result_manifest.json"
        result_file.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

        if json_mode:
            payload = {
                "trace_id": trace_ctx.trace_id,
                **result.model_dump(),
            }
            json.dump(payload, sys.stdout, default=str)
            print()
        else:
            status_label = result.status.upper()
            print(
                f"[{status_label}] job={result.job_id} "
                f"stage={result.stage} "
                f"exit_code={result.exit_code}"
            )
            if result.fallback_level:
                print(f"  Fallback: {result.fallback_level}")
            if result.error_type:
                print(f"  Error: {result.error_type}: {result.message}")
            if result.agent_fix_hint:
                print(f"  Hint: {result.agent_fix_hint}")
            for key, entry in result.output_files.items():
                print(
                    f"  {key}: {entry.path} ({entry.size_bytes} bytes, sha256={entry.sha256[:16]}...)"
                )

        return result.exit_code
    finally:
        if tb_started_here:
            shutdown_traceboard(status="finished" if tb_exit_code == 0 else "failed")


__all__ = [
    "CLOSURE_TYPE_MAP",
    "_TRACEPARENT_RE",
    "BundleManifest",
    "ClosureType",
    "EngineSpec",
    "HardwareBudget",
    "JobBundleWorker",
    "OutputFileEntry",
    "OutputSink",
    "PresignedEndpoint",
    "ResultManifest",
    "SceneSpec",
    "_check_ancestor_symlinks",
    "_compute_sha256",
    "_extract_manifest_traceboard_param",
    "_validate_manifest_traceboard_dir",
    "get_registered_stages",
    "register_stage",
    "run_worker_cli",
]
