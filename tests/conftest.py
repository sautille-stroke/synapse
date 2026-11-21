"""Shared pytest fixtures and CI configuration for the synapse test suite."""

from __future__ import annotations

import os
from typing import Any, Generator

# Prevent OpenMP / OpenBLAS / PyTorch thread oversubscription under pytest-xdist
# (e.g. 16 workers x 16 threads = 256 threads thrashing 16 cores).
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pytest

# ---------------------------------------------------------------------------
# SYNAPSE_CI_STRICT_DEPS: when set to "1", tests that would normally skip
# due to missing optional dependencies (torch, trimesh, safetensors, etc.)
# will FAIL instead.  This ensures CI catches missing dep installations.
# ---------------------------------------------------------------------------

def _is_strict_deps() -> bool:
    val = os.environ.get("SYNAPSE_CI_STRICT_DEPS", "").strip().lower()
    return val in ("1", "true", "yes", "on")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, Any, None]:
    """Fail-fast on unexpected skipped tests when SYNAPSE_CI_STRICT_DEPS=1.

    Tests explicitly marked ``@pytest.mark.external_data`` or
    ``@pytest.mark.platform_specific`` may still skip when external sibling
    repositories or OS-specific prerequisites are absent.
    """
    outcome = yield
    rep: Any = outcome.get_result()
    if (
        _is_strict_deps()
        and rep.skipped
        and rep.when in ("setup", "call")
        and not getattr(rep, "wasxfail", None)
        and item.get_closest_marker("external_data") is None
        and item.get_closest_marker("platform_specific") is None
    ):
        rep.outcome = "failed"
        rep.longrepr = f"SYNAPSE_CI_STRICT_DEPS=1 forbids skipped tests: {rep.longrepr}"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, Any, None]:
    """Fail collection-time skips when SYNAPSE_CI_STRICT_DEPS=1.

    This catches module-level ``pytest.importorskip(..., allow_module_level=True)``
    calls that silently skip entire test modules during collection.
    """
    outcome = yield
    rep: Any = outcome.get_result()
    if _is_strict_deps() and rep.outcome == "skipped":
        rep.outcome = "failed"
        rep.longrepr = (
            f"SYNAPSE_CI_STRICT_DEPS=1 forbids module-level skips: "
            f"{getattr(rep, 'longrepr', 'unknown')}"
        )


# ---------------------------------------------------------------------------
# Session-scoped software rasterizer cache
# ---------------------------------------------------------------------------

_rasterizer_cache: dict[str, Any] = {}


@pytest.fixture(scope="session")
def shared_rasterizer_cache() -> dict[str, Any]:
    """Session-scoped cache for software rasterizer instances.

    Tests that create a SoftwareRasterizer can store and retrieve instances
    by a key (e.g. "128x128") to avoid repeated construction overhead.
    """
    return _rasterizer_cache


@pytest.fixture(scope="session")
def cached_rasterizer_64() -> Any:
    """Session-cached 64×64 software rasterizer instance."""
    key = "64x64"
    if key not in _rasterizer_cache:
        try:
            from synapse.render_core.rasterizer import SoftwareRasterizer

            _rasterizer_cache[key] = SoftwareRasterizer(width=64, height=64)
        except ImportError as exc:
            if _is_strict_deps():
                raise RuntimeError(
                    f"SYNAPSE_CI_STRICT_DEPS=1: failed to import SoftwareRasterizer: {exc}"
                ) from exc
            return None
    return _rasterizer_cache[key]


@pytest.fixture(scope="session")
def cached_rasterizer_128() -> Any:
    """Session-cached 128×128 software rasterizer instance."""
    key = "128x128"
    if key not in _rasterizer_cache:
        try:
            from synapse.render_core.rasterizer import SoftwareRasterizer

            _rasterizer_cache[key] = SoftwareRasterizer(width=128, height=128)
        except ImportError as exc:
            if _is_strict_deps():
                raise RuntimeError(
                    f"SYNAPSE_CI_STRICT_DEPS=1: failed to import SoftwareRasterizer: {exc}"
                ) from exc
            return None
    return _rasterizer_cache[key]
