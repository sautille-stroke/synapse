"""Horizon 14+ release parity, wire_guard contract, and version floor tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

import synapse
from synapse.bench.scenario import list_scenarios

from tools.wire_guard import main as wire_guard_main

_MIN_VERSION = (6, 0, 0)


class TestHorizon14ReleaseSeal:
    """Validate release version parity and Horizon 14 CI wire_guard checks."""

    def test_version_parity_and_semver_floor(self) -> None:
        """pyproject.toml and synapse.__version__ must match and be >= 6.0.0."""
        repo_root = Path(__file__).resolve().parents[1]
        pyproject_data = tomllib.loads(
            (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        pyproject_ver = pyproject_data["project"]["version"]
        assert synapse.__version__ == pyproject_ver, (
            f"synapse.__version__={synapse.__version__!r} != pyproject={pyproject_ver!r}"
        )
        major, minor, patch = (int(x) for x in synapse.__version__.split(".")[:3])
        assert (major, minor, patch) >= _MIN_VERSION, (
            f"Version {synapse.__version__} below Horizon 14 floor {_MIN_VERSION}"
        )

    def test_wire_guard_passes_with_horizon14_and_bench50(self) -> None:
        assert len(list_scenarios()) >= 50
        rc = wire_guard_main(["--check"])
        assert rc == 0
