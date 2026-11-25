"""Optional Traceboard integration for structured experiment tracking.

Provides lazy, fail-safe access to the aether-traceboard Probe SDK.
When ``aether-traceboard`` is not installed, all functions return
``None`` and no-op — zero impact on core synapse functionality.

Usage::

    from synapse.observability import init_traceboard, get_probe

    # Initialize once at CLI / worker entry point
    probe = init_traceboard("./tblog", "bench_run_42", spp="64")

    # Use anywhere (safe even without aether-traceboard)
    p = get_probe()
    if p:
        p.log_scalar("metrics/psnr", 38.7)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Module-level state — None until init_traceboard() is called
_probe: Any = None
_initialized: bool = False


def _try_import_probe() -> type[Any] | None:
    """Lazily import Probe class, returning None if unavailable."""
    try:
        from aether_traceboard import Probe

        probe_cls: type[Any] = Probe
        return probe_cls
    except ImportError:
        return None


def get_probe() -> Any:
    """Get the global Traceboard probe singleton.

    Returns ``None`` if ``aether-traceboard`` is not installed or
    ``init_traceboard()`` has not been called.
    """
    return _probe


def init_traceboard(
    log_dir: str | Path,
    run_name: str,
    *,
    engine: str = "synapse",
    auto_activate: bool = True,
    **config: str,
) -> Any:
    """Initialize Traceboard with a FileCollector.

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
    **config:
        Key-value configuration parameters recorded in the run.

    Returns
    -------
    Probe or None
        The initialized probe, or ``None`` if ``aether-traceboard``
        is not installed.
    """
    global _probe, _initialized
    if _initialized:
        logger.warning("Traceboard already initialized; ignoring re-init")
        return _probe

    ProbeCls = _try_import_probe()
    if ProbeCls is None:
        logger.info(
            "aether-traceboard not installed; observability disabled. "
            "Install with: pip install aether-traceboard"
        )
        _initialized = True
        return None

    try:
        from aether_traceboard import FileCollector

        _probe = ProbeCls()
        collector = FileCollector(Path(log_dir))
        _probe.add_collector(collector)
        if auto_activate:
            _probe.activate()
        _probe.begin_run(run_name, engine=engine, **config)
        _initialized = True
        logger.info(
            "Traceboard initialized: run=%r log_dir=%s", run_name, log_dir
        )
        return _probe
    except Exception:
        logger.warning("Failed to initialize Traceboard", exc_info=True)
        _initialized = True
        return None


def shutdown_traceboard() -> None:
    """End the current run and flush all collectors."""
    global _probe, _initialized
    if _probe is not None:
        try:
            _probe.end_run()
        except Exception:
            logger.warning("Error during Traceboard shutdown", exc_info=True)
    _probe = None
    _initialized = False


@contextmanager
def traceboard_stage(
    name: str, **params: str
) -> Iterator[Any]:
    """Context manager for a Traceboard test stage.

    Yields ``None`` if Traceboard is not active.
    """
    p = get_probe()
    if p is None or not p.active:
        yield None
        return
    stage = p.begin_stage(name, **params)
    with stage as ctx:
        yield ctx
