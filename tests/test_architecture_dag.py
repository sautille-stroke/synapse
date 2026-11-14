"""Architectural DAG enforcement — zero cycles, zero reverse bench deps.

Wraps ``tools/import_dag_guard.py`` as a pytest test so that CI catches
any future import cycle or ``synapse.bench`` reverse dependency.
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """Resolve the repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parent.parent


def test_no_import_cycles() -> None:
    """Assert that the synapse import DAG contains no circular dependencies."""
    import importlib.util

    guard_path = _repo_root() / "tools" / "import_dag_guard.py"
    assert guard_path.is_file(), f"import_dag_guard.py not found at {guard_path}"

    spec = importlib.util.spec_from_file_location("import_dag_guard", guard_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    cycles, bench_reverse_deps, _edges = mod.validate(_repo_root())

    if cycles:
        cycle_strs = [" → ".join(scc) + f" → {scc[0]}" for scc in cycles]
        raise AssertionError(
            f"{len(cycles)} import cycle(s) detected:\n"
            + "\n".join(f"  • {s}" for s in cycle_strs)
        )

    if bench_reverse_deps:
        dep_strs = [f"{u} → {v}" for u, v in bench_reverse_deps]
        raise AssertionError(
            f"{len(bench_reverse_deps)} reverse dependency(ies) into "
            f"synapse.bench:\n" + "\n".join(f"  • {s}" for s in dep_strs)
        )


def test_import_dag_guard_detects_dynamic_importlib_bypass(tmp_path: Path) -> None:
    """Anti-silence test: verify import_dag_guard intercepts dynamic importlib & __import__."""
    import ast
    import importlib.util

    guard_path = _repo_root() / "tools" / "import_dag_guard.py"
    spec = importlib.util.spec_from_file_location("import_dag_guard", guard_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # 1. Test _ImportVisitor on importlib.import_module and __import__
    snippet = """
import importlib
def sneaky():
    mod1 = importlib.import_module("synapse.cli")
    mod2 = __import__("synapse.inspect")
"""
    tree = ast.parse(snippet)
    visitor = mod._ImportVisitor("render_core")
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

    cycles, _reverse, _edges = mod.validate(tmp_path)
    assert len(cycles) > 0, "Expected import cycle (cli <-> errors) was not detected!"

