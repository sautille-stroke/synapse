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
