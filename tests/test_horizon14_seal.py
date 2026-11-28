"""Horizon 14 (v6.0.0) final seal, release parity, and wire_guard contract tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

import synapse
from synapse.bench.scenario import list_scenarios

from tools.wire_guard import main as wire_guard_main


class TestHorizon14ReleaseSeal:
    """Validate v6.0.0 version parity and Horizon 14 CI wire_guard checks."""

    def test_version_6_0_0_parity(self) -> None:
        assert synapse.__version__ == "6.0.0"
        repo_root = Path(__file__).resolve().parents[1]
        pyproject_data = tomllib.loads(
            (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert pyproject_data["project"]["version"] == "6.0.0"

    def test_wire_guard_passes_with_horizon14_and_bench50(self) -> None:
        assert len(list_scenarios()) >= 50
        rc = wire_guard_main(["--check"])
        assert rc == 0
