"""Optional Traceboard integration for structured experiment tracking.

Provides thread-safe, lazy, fail-safe access to the ``aether-traceboard`` Probe SDK.
When ``aether-traceboard`` is not installed, all functions return ``None`` and no-op
cleanly — zero impact on core synapse functionality.

Usage::

    from synapse.observability import init_traceboard, get_probe, safe_log_scalar

    # Initialize once at CLI / worker entry point
    probe = init_traceboard("./tblog", "bench_run_42", spp="64")

    # Use anywhere (safe even without aether-traceboard)
    safe_log_scalar("metrics/psnr", 38.7)
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@runtime_checkable
class StageProtocol(Protocol):
    """Typed contract for a Traceboard execution stage context."""

    def log_metric(self, key: str, value: float) -> None: ...

    def set_pass(self) -> None: ...

    def set_fail(self, reason: str | None = None) -> None: ...

    def set_skip(self, reason: str | None = None) -> None: ...

    def __enter__(self) -> StageProtocol: ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None: ...


@runtime_checkable
class ProbeProtocol(Protocol):
    """Typed contract for an ``aether_traceboard.Probe`` instance."""

    @property
    def active(self) -> bool: ...

    @property
    def step(self) -> int: ...

    def activate(self) -> None: ...

    def deactivate(self) -> None: ...

    def set_step(self, step: int) -> None: ...

    def increment_step(self, delta: int = 1) -> int: ...

    def begin_run(
        self,
        run_name: str,
        engine: str = "unknown",
        *,
        auto_activate: bool = True,
        **config: str,
    ) -> None: ...

    def end_run(self, status: str = "finished") -> None: ...

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None: ...

    def log_image(self, tag: str, image: Any, step: int | None = None) -> None: ...

    def log_histogram(
        self, tag: str, edges: Any, counts: Any, step: int | None = None
    ) -> None: ...

    def log_text(self, tag: str, text: str, step: int | None = None) -> None: ...

    def log_comparison(
        self,
        tag: str,
        image_a: Any,
        image_b: Any,
        label_a: str = "A",
        label_b: str = "B",
        diff_image: Any | None = None,
        metrics: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        passed: bool | None = None,
        step: int | None = None,
    ) -> None: ...

    def log_table(
        self,
        tag: str,
        columns: list[dict[str, str]],
        rows: list[list[Any]],
        step: int | None = None,
    ) -> None: ...

    def log_artifact(
        self,
        tag: str,
        metadata: dict[str, Any],
        blob: bytes,
        step: int | None = None,
    ) -> None: ...

    def log_series_group(
        self, tag: str, series: list[dict[str, Any]], step: int | None = None
    ) -> None: ...

    def log_config(
        self, tag: str, kv: dict[str, str], step: int | None = None
    ) -> None: ...

    def begin_stage(self, name: str, **params: str) -> StageProtocol: ...

    def flush(self) -> None: ...


# Module-level state guarded by _state_lock
_state_lock = threading.RLock()
_probe: ProbeProtocol | None = None
_initialized: bool = False


def _try_import_probe() -> type[Any] | None:
    """Lazily import Probe class, returning None if unavailable."""
    try:
        from aether_traceboard import Probe

        probe_cls: type[Any] = Probe
        return probe_cls
    except ImportError:
        return None


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Return True if *fn* accepts keyword argument *name* (explicitly or via ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def get_probe() -> ProbeProtocol | None:
    """Get the global Traceboard probe instance.

    Returns ``None`` if ``aether-traceboard`` is not installed or
    ``init_traceboard()`` has not been called.
    """
    with _state_lock:
        return _probe


def init_traceboard(
    log_dir: str | Path,
    run_name: str,
    *,
    engine: str = "synapse",
    auto_activate: bool = True,
    trace_id: str | None = None,
    **config: str,
) -> ProbeProtocol | None:
    """Initialize Traceboard atomically with a FileCollector.

    Parameters
    ----------
    log_dir:
        Directory for ``.tbevt`` event files.
    run_name:
        Human-readable name for this run.
    engine:
        Engine identifier (default ``"synapse"``).
    auto_activate:
        Whether to activate the probe immediately.
    trace_id:
        Optional distributed trace ID recorded in ``RUN_START`` config.
    **config:
        Key-value configuration parameters recorded in the run.

    Returns
    -------
    ProbeProtocol or None
        The initialized probe, or ``None`` if ``aether-traceboard``
        is not installed or initialization failed.
    """
    global _probe, _initialized
    with _state_lock:
        if _initialized:
            logger.warning("Traceboard already initialized; ignoring re-init")
            return _probe

        probe_cls = _try_import_probe()
        if probe_cls is None:
            logger.info(
                "aether-traceboard not installed; observability disabled. "
                "Install with: pip install aether-traceboard"
            )
            _probe = None
            _initialized = True
            return None

        local_probe: Any = None
        collector: Any = None
        try:
            from aether_traceboard import FileCollector

            local_probe = probe_cls()
            collector = FileCollector(Path(log_dir))
            local_probe.add_collector(collector)
            if auto_activate:
                local_probe.activate()

            begin_kwargs: dict[str, Any] = {"engine": engine, **config}
            if trace_id is not None:
                begin_kwargs.setdefault("trace_id", trace_id)
            if _accepts_kwarg(local_probe.begin_run, "auto_activate"):
                begin_kwargs["auto_activate"] = auto_activate
            local_probe.begin_run(run_name, **begin_kwargs)

            _probe = local_probe
            _initialized = True
            logger.info(
                "Traceboard initialized: run=%r log_dir=%s", run_name, log_dir
            )
            return _probe
        except Exception:
            logger.warning("Failed to initialize Traceboard", exc_info=True)
            if collector is not None:
                with contextlib.suppress(Exception):
                    collector.on_run_close()
            _probe = None
            _initialized = True
            return None


def shutdown_traceboard(*, status: str = "finished") -> None:
    """End the current run with *status* and reset module state."""
    global _probe, _initialized
    with _state_lock:
        probe = _probe
        _probe = None
        _initialized = False

    if probe is not None:
        try:
            if _accepts_kwarg(probe.end_run, "status"):
                probe.end_run(status=status)
            else:
                probe.end_run()
        except Exception:
            logger.warning("Error during Traceboard shutdown", exc_info=True)


@contextlib.contextmanager
def traceboard_stage(
    name: str, **params: str
) -> Iterator[StageProtocol | None]:
    """Fail-safe context manager for a Traceboard stage.

    Yields ``None`` if Traceboard is inactive or initialization fails.
    """
    p = get_probe()
    if p is None or not p.active:
        yield None
        return

    try:
        stage_cm = p.begin_stage(name, **params)
        entered = stage_cm.__enter__()
    except Exception:
        logger.warning("Failed to begin Traceboard stage %r", name, exc_info=True)
        yield None
        return

    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield entered
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        try:
            stage_cm.__exit__(*exc_info)
        except Exception:
            logger.warning("Failed to exit Traceboard stage %r", name, exc_info=True)


def safe_log_scalar(
    tag: str, value: float, *, step: int | None = None
) -> None:
    """Fail-safe scalar logger that ignores inactive probes and non-finite floats."""
    probe = get_probe()
    if probe is None or not probe.active:
        return
    try:
        fval = float(value)
        if not math.isfinite(fval):
            return
        probe.log_scalar(tag, fval, step=step)
    except Exception:
        logger.debug("Traceboard safe_log_scalar(%r) failed", tag, exc_info=True)


def safe_log_text(tag: str, text: str, *, step: int | None = None) -> None:
    """Fail-safe text logger."""
    probe = get_probe()
    if probe is None or not probe.active:
        return
    try:
        probe.log_text(tag, str(text), step=step)
    except Exception:
        logger.debug("Traceboard safe_log_text(%r) failed", tag, exc_info=True)


def safe_log_table(
    tag: str,
    columns: list[dict[str, str]],
    rows: list[list[Any]],
    *,
    step: int | None = None,
) -> None:
    """Fail-safe table logger coercing cells to strings."""
    probe = get_probe()
    if probe is None or not probe.active:
        return
    try:
        str_rows = [[str(cell) for cell in row] for row in rows]
        probe.log_table(tag, columns, str_rows, step=step)
    except Exception:
        logger.debug("Traceboard safe_log_table(%r) failed", tag, exc_info=True)


def safe_log_artifact(
    tag: str,
    metadata: dict[str, Any],
    blob: bytes,
    *,
    step: int | None = None,
) -> None:
    """Fail-safe artifact logger."""
    probe = get_probe()
    if probe is None or not probe.active:
        return
    try:
        meta = dict(metadata)
        meta.setdefault("size_bytes", len(blob))
        probe.log_artifact(tag, meta, blob, step=step)
    except Exception:
        logger.debug("Traceboard safe_log_artifact(%r) failed", tag, exc_info=True)


def safe_log_comparison(
    tag: str,
    image_a: Any,
    image_b: Any,
    *,
    label_a: str = "A",
    label_b: str = "B",
    diff_image: Any | None = None,
    metrics: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    passed: bool | None = None,
    step: int | None = None,
) -> None:
    """Fail-safe comparison image/metrics logger."""
    probe = get_probe()
    if probe is None or not probe.active:
        return
    try:
        probe.log_comparison(
            tag,
            image_a,
            image_b,
            label_a=label_a,
            label_b=label_b,
            diff_image=diff_image,
            metrics=metrics,
            thresholds=thresholds,
            passed=passed,
            step=step,
        )
    except Exception:
        logger.debug("Traceboard safe_log_comparison(%r) failed", tag, exc_info=True)
