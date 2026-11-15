"""Shared pytest fixtures and CI configuration for the synapse test suite."""

from __future__ import annotations

import os
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# SYNAPSE_CI_STRICT_DEPS: when set to "1", tests that would normally skip
# due to missing optional dependencies (torch, trimesh, safetensors, etc.)
# will FAIL instead.  This ensures CI catches missing dep installations.
# ---------------------------------------------------------------------------

_STRICT_DEPS = os.environ.get("SYNAPSE_CI_STRICT_DEPS", "0") == "1"


@pytest.fixture(autouse=True)
def _enforce_strict_deps(request: pytest.FixtureRequest) -> None:  # noqa: PT004
    """When SYNAPSE_CI_STRICT_DEPS=1, convert skips from importorskip into failures.

    This fixture inspects the test outcome after it runs.  If the test was
    skipped via ``pytest.importorskip`` and strict deps is enabled, it raises
    an AssertionError so CI notices the missing dependency.
    """
    yield  # type: ignore[misc]
    if not _STRICT_DEPS:
        return
    # After the test: if it was skipped and the reason mentions an import,
    # re-raise so CI treats it as a failure.
    # NOTE: pytest.importorskip raises Skipped with reason = "could not import ..."
    # We hook into the _call_reprcrash / _skips list indirectly via markers.
    # The actual enforcement happens via the pytest_runtest_makereport hook below.


def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> None:
    """Fail-fast on importorskip skips when SYNAPSE_CI_STRICT_DEPS=1."""
    if not _STRICT_DEPS:
        return
    if call.when == "call" and call.excinfo is not None:
        import _pytest.outcomes

        if call.excinfo.errisinstance(_pytest.outcomes.Skipped):
            reason = str(call.excinfo.value)
            if "could not import" in reason.lower():
                pytest.fail(
                    f"SYNAPSE_CI_STRICT_DEPS=1: test skipped due to missing "
                    f"import, but strict deps requires it to pass. "
                    f"Reason: {reason}",
                    pytrace=False,
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
            from synapse.generators.rasterizer import SoftwareRasterizer

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
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
            from synapse.generators.rasterizer import SoftwareRasterizer

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                _rasterizer_cache[key] = SoftwareRasterizer(width=128, height=128)
        except ImportError:
            return None
    return _rasterizer_cache[key]
