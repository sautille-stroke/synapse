#!/usr/bin/env python3
"""Import DAG guard — AST-based circular dependency and reverse-dep scanner.

Scans all ``python/synapse/**/*.py`` files, builds a directed graph of
``synapse.<pkg>`` → ``synapse.<pkg>`` imports (both top-level and lazy
local imports inside functions/methods), runs Tarjan's SCC algorithm,
and asserts:

  1. All SCCs are singletons (0 circular dependency cycles).
  2. No edge from a non-``bench`` package into ``synapse.bench``.

Usage::

    python tools/import_dag_guard.py --check   # exit 0 if clean, 1 otherwise
    python tools/import_dag_guard.py            # print the DAG and violations
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# AST import visitor
# ---------------------------------------------------------------------------


def _extract_synapse_pkg(module_name: str) -> str | None:
    """Extract the top-level synapse sub-package from a dotted module name.

    >>> _extract_synapse_pkg("synapse.bench.rasterizer")
    'bench'
    >>> _extract_synapse_pkg("numpy")
    >>> _extract_synapse_pkg("synapse")
    """
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "synapse":
        return parts[1]
    return None


class _ImportVisitor(ast.NodeVisitor):
    """Collect all ``synapse.<pkg>`` imports from an AST."""

    def __init__(self, source_pkg: str) -> None:
        self.source_pkg = source_pkg
        self.edges: set[tuple[str, str]] = set()

    def _record(self, module_name: str | None) -> None:
        if module_name is None:
            return
        target_pkg = _extract_synapse_pkg(module_name)
        if target_pkg and target_pkg != self.source_pkg:
            self.edges.add((self.source_pkg, target_pkg))

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._record(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            self._record(node.module)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Scan all synapse files
# ---------------------------------------------------------------------------


def scan_import_dag(
    root: Path,
) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """Scan ``root/python/synapse/`` and return (edges, pkg_files).

    Returns
    -------
    edges:
        Set of (source_pkg, target_pkg) directed edges.
    pkg_files:
        Mapping from package name to set of .py file paths (for diagnostics).
    """
    synapse_dir = root / "python" / "synapse"
    if not synapse_dir.is_dir():
        print(f"ERROR: {synapse_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    edges: set[tuple[str, str]] = set()
    pkg_files: dict[str, set[str]] = defaultdict(set)

    for py_file in sorted(synapse_dir.rglob("*.py")):
        rel = py_file.relative_to(synapse_dir)
        parts = rel.parts
        if len(parts) < 2:
            # Top-level synapse module (e.g. synapse/errors.py) — no sub-pkg
            continue
        source_pkg = parts[0]
        pkg_files[source_pkg].add(str(rel))

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        visitor = _ImportVisitor(source_pkg)
        visitor.visit(tree)
        edges.update(visitor.edges)

    return edges, dict(pkg_files)


# ---------------------------------------------------------------------------
# Tarjan's SCC algorithm
# ---------------------------------------------------------------------------


def _tarjan_scc(
    vertices: set[str],
    edges: set[tuple[str, str]],
) -> list[list[str]]:
    """Run Tarjan's algorithm and return all SCCs (largest first)."""
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    result: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            result.append(sorted(component))

    for v in sorted(vertices):
        if v not in index:
            strongconnect(v)

    return sorted(result, key=lambda c: (-len(c), c))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    root: Path,
) -> tuple[list[list[str]], list[tuple[str, str]], set[tuple[str, str]]]:
    """Validate the import DAG.

    Returns
    -------
    cycles:
        List of SCCs with more than 1 member (circular dependencies).
    bench_reverse_deps:
        Edges from non-bench packages INTO ``synapse.bench`` that are
        NOT from whitelisted top-level orchestrator packages.
    all_edges:
        Complete edge set for diagnostics.
    """
    edges, pkg_files = scan_import_dag(root)
    all_pkgs = set()
    for u, v in edges:
        all_pkgs.add(u)
        all_pkgs.add(v)

    sccs = _tarjan_scc(all_pkgs, edges)
    cycles = [scc for scc in sccs if len(scc) > 1]

    # Top-level orchestrator packages that are allowed to import from bench
    # (they are consumers, not peers that would create architectural violations)
    # - cli / pipeline / viewer: top-level entry points that compose bench
    # - inspect: the model_evaluator uses bench.model_loader for mesh I/O
    _BENCH_IMPORT_ALLOWED = frozenset({"cli", "pipeline", "viewer", "inspect"})

    bench_reverse_deps = [
        (u, v)
        for (u, v) in edges
        if v == "bench" and u != "bench" and u not in _BENCH_IMPORT_ALLOWED
    ]

    return cycles, sorted(bench_reverse_deps), edges


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python tools/import_dag_guard.py [--check]``."""
    check_mode = "--check" in sys.argv

    # Find repo root: tools/import_dag_guard.py → repo root is parent of tools/
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    cycles, bench_reverse_deps, all_edges = validate(root)

    if not check_mode:
        print("=== Import DAG Edges ===")
        for u, v in sorted(all_edges):
            print(f"  {u} → {v}")
        print()

    if cycles:
        print(f"FAIL: {len(cycles)} circular dependency cycle(s) found:")
        for scc in cycles:
            print(f"  CYCLE: {' → '.join(scc)} → {scc[0]}")
    else:
        if not check_mode:
            print("OK: No circular dependencies found.")

    if bench_reverse_deps:
        print(
            f"FAIL: {len(bench_reverse_deps)} reverse dependency(ies) "
            f"into synapse.bench:"
        )
        for u, v in bench_reverse_deps:
            print(f"  {u} → {v}")
    else:
        if not check_mode:
            print("OK: No reverse dependencies into synapse.bench.")

    if cycles or bench_reverse_deps:
        sys.exit(1)
    else:
        if check_mode:
            print("OK: import DAG is clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
