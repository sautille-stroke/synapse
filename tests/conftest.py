"""Shared pytest fixtures and CI configuration for the synapse test suite."""

from __future__ import annotations

import os
from typing import Any, Generator

import pytest

# ---------------------------------------------------------------------------
# SYNAPSE_CI_STRICT_DEPS: when set to "1", tests that would normally skip
# due to missing optional dependencies (torch, trimesh, safetensors, etc.)
# will FAIL instead.  This ensures CI catches missing dep installations.
# ---------------------------------------------------------------------------

_STRICT_DEPS = os.environ.get("SYNAPSE_CI_STRICT_DEPS", "0") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, Any, None]:
    """Fail-fast on skipped tests when SYNAPSE_CI_STRICT_DEPS=1."""
    outcome = yield
    rep: Any = outcome.get_result()
    if _STRICT_DEPS and rep.skipped and rep.when in ("setup", "call"):
        rep.outcome = "failed"
        rep.longrepr = (
            f"SYNAPSE_CI_STRICT_DEPS=1 forbids skipped tests: {rep.longrepr}"
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
        except ImportError:
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
        except ImportError:
            return None
    return _rasterizer_cache[key]