"""Architectural DAG enforcement — zero cycles, zero tier violations.

Wraps ``tools/import_dag_guard.py`` as a pytest test so that CI catches
any future import cycle, tier violation, or lazy import leak.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    """Resolve the repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parent.parent


def _load_guard() -> Any:
    """Load import_dag_guard module dynamically."""
    import importlib.util

    guard_path = _repo_root() / "tools" / "import_dag_guard.py"
    assert guard_path.is_file(), f"import_dag_guard.py not found at {guard_path}"

    spec = importlib.util.spec_from_file_location("import_dag_guard", guard_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_no_import_cycles() -> None:
    """Assert that the synapse import DAG contains no circular dependencies."""
    mod = _load_guard()

    cycles, tier_violations, _edges, errors = mod.validate(_repo_root())

    if errors:
        raise AssertionError(
            f"{len(errors)} fatal error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    if cycles:
        cycle_strs = [" → ".join(scc) + f" → {scc[0]}" for scc in cycles]
        raise AssertionError(
            f"{len(cycles)} import cycle(s) detected:\n"
            + "\n".join(f"  • {s}" for s in cycle_strs)
        )


@pytest.mark.slow
def test_no_tier_violations() -> None:
    """Assert that no import crosses upward in the 5-tier architecture."""
    mod = _load_guard()

    _cycles, tier_violations, _edges, errors = mod.validate(_repo_root())

    if errors:
        raise AssertionError(
            f"{len(errors)} fatal error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    if tier_violations:
        violation_strs = [
            f"{src} (T{s_t}) → {dst} (T{d_t})"
            for src, dst, s_t, d_t in tier_violations
        ]
        raise AssertionError(
            f"{len(tier_violations)} tier violation(s) (upward cross-tier imports):\n"
            + "\n".join(f"  • {s}" for s in violation_strs)
        )


@pytest.mark.slow
def test_import_dag_guard_cli_check_exits_zero() -> None:
    """``python tools/import_dag_guard.py --check`` exits 0."""
    guard_path = _repo_root() / "tools" / "import_dag_guard.py"
    result = subprocess.run(
        [sys.executable, str(guard_path), "--check"],
        capture_output=True,
        text=True,
        cwd=str(_repo_root()),
    )
    assert result.returncode == 0, (
        f"import_dag_guard --check exited with {result.returncode}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_import_dag_guard_detects_dynamic_importlib_bypass(tmp_path: Path) -> None:
    """Anti-silence test: verify import_dag_guard intercepts dynamic importlib & __import__."""
    import ast

    mod = _load_guard()

    # 1. Test _ImportVisitor on importlib.import_module and __import__
    snippet = """
import importlib
def sneaky() -> None:
    mod1 = importlib.import_module("synapse.cli")
    mod2 = __import__("synapse.inspect")
"""
    tree = ast.parse(snippet)
    visitor = mod._ImportVisitor("render_core", "synapse.render_core")
    visitor.visit(tree)

    assert ("render_core", "cli") in visitor.edges, "Failed to intercept importlib.import_module"
    assert ("render_core", "inspect") in visitor.edges, "Failed to intercept __import__"

    # 2. Test mock repo validation catches cycle induced by low-level import
    mock_synapse = tmp_path / "python" / "synapse"
    mock_synapse.mkdir(parents=True)
    (mock_synapse / "errors.py").write_text(
        'import importlib\nmod = importlib.import_module("synapse.cli")\n',
        encoding="utf-8",
    )
    (mock_synapse / "cli").mkdir()
    (mock_synapse / "cli" / "__init__.py").write_text(
        "from synapse import errors\n",
        encoding="utf-8",
    )

    cycles, _tier_v, _edges, _errors = mod.validate(tmp_path)
    assert len(cycles) > 0, "Expected import cycle (cli <-> errors) was not detected!"


def test_relative_import_resolved_by_guard(tmp_path: Path) -> None:
    """Relative imports (``from .. import X``) are resolved to real package paths."""

    mod = _load_guard()

    # Simulate a file in synapse.inspect.schemas doing `from .. import fgpp`
    mock_synapse = tmp_path / "python" / "synapse"
    (mock_synapse / "inspect").mkdir(parents=True)
    (mock_synapse / "inspect" / "__init__.py").write_text("", encoding="utf-8")
    (mock_synapse / "inspect" / "schemas.py").write_text(
        "from ..fgpp import schema\n",
        encoding="utf-8",
    )
    (mock_synapse / "fgpp").mkdir(parents=True)
    (mock_synapse / "fgpp" / "__init__.py").write_text("", encoding="utf-8")
    (mock_synapse / "fgpp" / "schema.py").write_text("", encoding="utf-8")

    edges, _toplevel, _, _ = mod.scan_import_dag(tmp_path)
    # The relative import `from ..fgpp import schema` from synapse.inspect.schemas
    # should resolve to synapse.fgpp and create edge inspect → fgpp
    assert ("inspect", "fgpp") in edges, (
        f"Relative import was not resolved. Edges: {edges}"
    )


def test_non_constant_dynamic_import_caught(tmp_path: Path) -> None:
    """Non-constant dynamic ``importlib.import_module(var)`` treated as failure."""
    mod = _load_guard()

    mock_synapse = tmp_path / "python" / "synapse"
    (mock_synapse / "utils").mkdir(parents=True)
    (mock_synapse / "utils" / "__init__.py").write_text(
        "import importlib\ndef f(name): return importlib.import_module(name)\n",
        encoding="utf-8",
    )

    _, _toplevel, _, errors = mod.scan_import_dag(tmp_path)
    assert any("Non-constant dynamic import" in e for e in errors), (
        f"Expected non-constant dynamic import error. Got: {errors}"
    )


def test_syntax_error_treated_as_failure(tmp_path: Path) -> None:
    """Files with SyntaxError are reported as errors (not silently skipped)."""
    mod = _load_guard()

    mock_synapse = tmp_path / "python" / "synapse"
    (mock_synapse / "broken").mkdir(parents=True)
    (mock_synapse / "broken" / "__init__.py").write_text(
        "def f(\n",  # Intentional SyntaxError
        encoding="utf-8",
    )

    _, _toplevel, _, errors = mod.scan_import_dag(tmp_path)
    assert any("SyntaxError" in e for e in errors), (
        f"Expected SyntaxError. Got: {errors}"
    )


def test_lazy_init_no_eager_cli_pipeline() -> None:
    """``import synapse.fgpp.schema`` must NOT pull in synapse.cli or synapse.pipeline.

    This validates the PEP 562 lazy ``__init__.py`` conversion.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import synapse.fgpp.schema; "
                "has_cli = 'synapse.cli' in sys.modules; "
                "has_pipeline = 'synapse.pipeline' in sys.modules; "
                "print(f'cli={has_cli} pipeline={has_pipeline}'); "
                "assert not has_cli, 'synapse.cli was eagerly loaded!'; "
                "assert not has_pipeline, 'synapse.pipeline was eagerly loaded!'"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(_repo_root()),
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(_repo_root() / "python"),
        },
    )
    assert result.returncode == 0, (
        f"Lazy init test failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )