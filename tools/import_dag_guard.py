#!/usr/bin/env python3
"""Import DAG guard — AST-based 5-Tier dependency and circular-dep scanner.

Scans all ``python/synapse/**/*.py`` files, builds a directed graph of
``synapse.<pkg>`` → ``synapse.<pkg>`` imports (both top-level and lazy
local imports inside functions/methods), runs Tarjan's SCC algorithm,
and asserts:

  1. All SCCs are singletons (0 circular dependency cycles).
  2. No upward tier violation in the 5-tier architecture matrix
     (``tier[src] >= tier[dst]`` must hold for every **module-level**
     import edge).

Lazy imports (inside function/method bodies) still contribute to
cycle detection but are exempt from tier enforcement because they
represent soft runtime coupling, not architectural dependencies.

Tier Map::

    Tier 0: fgpp, errors, telemetry, observability, render_core, bridge
    Tier 1: style, metrics, models
    Tier 2: adapters, composite, recon, refine, inspect
    Tier 3: bench, rsi, training, generators
    Tier 4: pipeline, cli, viewer

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
# 5-Tier architecture map
# ---------------------------------------------------------------------------

TIER_MAP: dict[str, int] = {
    # Tier 0 — Foundation / zero-dep
    "errors": 0,
    "telemetry": 0,
    "observability": 0,
    "fgpp": 0,
    "render_core": 0,
    "bridge": 0,
    # Tier 1 — Shared services
    "style": 1,
    "metrics": 1,
    "models": 1,
    # Tier 2 — Domain algorithms
    "adapters": 2,
    "composite": 2,
    "recon": 2,
    "refine": 2,
    "inspect": 2,
    # Tier 3 — Evaluation, training, generation
    "bench": 3,
    "rsi": 3,
    "training": 3,
    "generators": 3,
    # Tier 4 — Entry points / orchestrators
    "pipeline": 4,
    "cli": 4,
    "viewer": 4,
}

# Packages prefixed with '_' (like _c_geometry) are private C extensions and
# are exempt from tier checks.
_PRIVATE_PREFIX = "_"

# Files allowed to use non-constant ``importlib.import_module(var)`` with an
# explicit ``# dag-guard: external-only`` annotation because they are dynamic
# external C++ extension loaders by design.
_DYNAMIC_IMPORT_ALLOWLIST: frozenset[str] = frozenset({
    "bridge/aether_bridge.py",
})


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


def _resolve_relative_import(
    node_module: str | None,
    node_level: int,
    file_package: str,
) -> str | None:
    """Resolve a relative import to its absolute dotted module name.

    Args:
        node_module: The ``module`` field of ``ImportFrom`` AST node.
        node_level: Number of leading dots (0 = absolute, 1 = current pkg, etc.).
        file_package: The dotted package of the source file
            (e.g. ``synapse.bridge`` for ``synapse/bridge/vg_bridge.py``).

    Returns:
        Fully qualified module name or None if unresolvable.
    """
    if node_level == 0:
        return node_module

    parts = file_package.split(".")
    # Go up `level` packages
    if node_level > len(parts):
        return None
    base = ".".join(parts[: len(parts) - node_level + 1])
    if node_module:
        return f"{base}.{node_module}"
    return base


class _ImportVisitor(ast.NodeVisitor):
    """Collect all ``synapse.<pkg>`` imports from an AST.

    Distinguishes between module-level imports (``is_toplevel=True``)
    and lazy imports inside function/method/class bodies.  Both contribute
    to cycle detection, but only module-level imports are subject to
    tier enforcement.
    """

    def __init__(
        self,
        source_pkg: str,
        file_package: str = "",
        source_lines: list[str] | None = None,
        rel_posix: str = "",
    ) -> None:
        self.source_pkg = source_pkg
        self.file_package = file_package
        self._source_lines = source_lines or []
        self._rel_posix = rel_posix
        # All edges (for cycle detection)
        self.edges: set[tuple[str, str]] = set()
        # Only module-level edges (for tier enforcement)
        self.toplevel_edges: set[tuple[str, str]] = set()
        self.errors: list[str] = []
        self._scope_depth: int = 0  # 0 = module level

    def _has_inline_suppression(self, lineno: int) -> bool:
        """Check whether an allowlisted file line has ``dag-guard: external-only``.

        Non-allowlisted files cannot bypass dynamic import detection via inline comments.
        """
        if self._rel_posix not in _DYNAMIC_IMPORT_ALLOWLIST:
            return False
        if 1 <= lineno <= len(self._source_lines):
            return "dag-guard: external-only" in self._source_lines[lineno - 1]
        return False

    def _record(self, module_name: str | None) -> None:
        if module_name is None:
            return
        target_pkg = _extract_synapse_pkg(module_name)
        if target_pkg and target_pkg != self.source_pkg:
            edge = (self.source_pkg, target_pkg)
            self.edges.add(edge)
            if self._scope_depth == 0:
                self.toplevel_edges.add(edge)

    # Track function/method/class scope depth
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name)
            # Intercept bare `import synapse` in non-__init__.py modules
            # to prevent PEP 562 __getattr__ DAG bypass
            if alias.name == "synapse" and self.source_pkg != "__init__":
                self.errors.append(
                    f"Bare 'import synapse' at line {node.lineno} "
                    f"(bypasses DAG via PEP 562 __getattr__)"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Resolve relative imports (node.level > 0) to absolute paths
        if node.level > 0:
            resolved = _resolve_relative_import(
                node.module, node.level, self.file_package,
            )
            if resolved:
                if resolved == "synapse" and node.names:
                    for alias in node.names:
                        self._record(f"synapse.{alias.name}")
                else:
                    self._record(resolved)
        elif node.module:
            if node.module == "synapse":
                for alias in node.names:
                    self._record(f"synapse.{alias.name}")
            else:
                self._record(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        module_name: str | None = None
        is_dynamic_non_constant = False

        # Detect builtins.__import__("x") — attribute-based __import__ bypass
        is_import_builtin = False
        if isinstance(func, ast.Attribute) and func.attr == "__import__":
            is_import_builtin = True

        if (isinstance(func, ast.Name) and func.id == "__import__") or is_import_builtin:
            # Check positional args
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                module_name = node.args[0].value
            elif node.args:
                is_dynamic_non_constant = True
            # Also check name= keyword argument
            if module_name is None and not is_dynamic_non_constant:
                for kw in node.keywords:
                    if kw.arg == "name":
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            module_name = kw.value.value
                        else:
                            is_dynamic_non_constant = True
                        break
        elif isinstance(func, ast.Name) and func.id == "import_module":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                val = node.args[0].value
                if val.startswith("."):
                    # Relative import_module(".x", pkg) — resolve via 2nd arg
                    pkg_arg: str | None = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                        pkg_arg = node.args[1].value
                    else:
                        for kw in node.keywords:
                            if kw.arg == "package" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                pkg_arg = kw.value.value
                                break
                    if pkg_arg is not None:
                        # Resolve: e.g. import_module(".fgpp", "synapse") -> "synapse.fgpp"
                        level = len(val) - len(val.lstrip("."))
                        rel_mod = val.lstrip(".")
                        parts = pkg_arg.split(".")
                        if level <= len(parts):
                            base = ".".join(parts[: len(parts) - level + 1])
                            module_name = f"{base}.{rel_mod}" if rel_mod else base
                        else:
                            is_dynamic_non_constant = True
                    else:
                        is_dynamic_non_constant = True
                else:
                    module_name = val
            elif node.args and isinstance(node.args[0], ast.JoinedStr):
                first_val = node.args[0].values[0] if node.args[0].values else None
                if isinstance(first_val, ast.Constant) and isinstance(first_val.value, str):
                    if first_val.value.startswith("synapse.") and self.source_pkg == "__init__":
                        # Known aggregator pattern in __init__.py — not a violation
                        pass
                    elif first_val.value.startswith("synapse."):
                        is_dynamic_non_constant = True
                    else:
                        is_dynamic_non_constant = True
                else:
                    is_dynamic_non_constant = True
            elif node.args:
                is_dynamic_non_constant = True
        elif isinstance(func, ast.Attribute) and func.attr == "import_module":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                val = node.args[0].value
                if val.startswith("."):
                    # Relative import_module(".x", pkg) via attribute form
                    pkg_arg_attr: str | None = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                        pkg_arg_attr = node.args[1].value
                    else:
                        for kw in node.keywords:
                            if kw.arg == "package" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                pkg_arg_attr = kw.value.value
                                break
                    if pkg_arg_attr is not None:
                        level = len(val) - len(val.lstrip("."))
                        rel_mod = val.lstrip(".")
                        parts = pkg_arg_attr.split(".")
                        if level <= len(parts):
                            base = ".".join(parts[: len(parts) - level + 1])
                            module_name = f"{base}.{rel_mod}" if rel_mod else base
                        else:
                            is_dynamic_non_constant = True
                    else:
                        is_dynamic_non_constant = True
                else:
                    module_name = val
            elif node.args and isinstance(node.args[0], ast.JoinedStr):
                # f-string: extract constant prefix to see if it starts
                # with "synapse." (common pattern in __init__.py __getattr__)
                first_val = node.args[0].values[0] if node.args[0].values else None
                if isinstance(first_val, ast.Constant) and isinstance(first_val.value, str):
                    if first_val.value.startswith("synapse.") and self.source_pkg == "__init__":
                        # Known aggregator pattern in __init__.py — not a violation
                        pass
                    elif first_val.value.startswith("synapse."):
                        is_dynamic_non_constant = True
                    else:
                        is_dynamic_non_constant = True
                else:
                    is_dynamic_non_constant = True
            elif node.args:
                is_dynamic_non_constant = True

        if is_dynamic_non_constant and not self._has_inline_suppression(node.lineno):
            self.errors.append(
                f"Non-constant dynamic import at line {node.lineno}"
            )

        if module_name is not None:
            self._record(module_name)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Scan all synapse files
# ---------------------------------------------------------------------------


def scan_import_dag(
    root: Path,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, set[str]], list[str]]:
    """Scan ``root/python/synapse/`` and return (all_edges, toplevel_edges, pkg_files, errors).

    Returns
    -------
    all_edges:
        Set of (source_pkg, target_pkg) directed edges (all scopes).
    toplevel_edges:
        Set of (source_pkg, target_pkg) edges from module-level imports only.
    pkg_files:
        Mapping from package name to set of .py file paths (for diagnostics).
    errors:
        List of fatal error strings (SyntaxError, non-constant dynamic imports).
    """
    synapse_dir = root / "python" / "synapse"
    if not synapse_dir.is_dir():
        print(f"ERROR: {synapse_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    all_edges: set[tuple[str, str]] = set()
    toplevel_edges: set[tuple[str, str]] = set()
    pkg_files: dict[str, set[str]] = defaultdict(set)
    all_errors: list[str] = []

    for py_file in sorted(synapse_dir.rglob("*.py")):
        rel = py_file.relative_to(synapse_dir)
        parts = rel.parts

        # Determine the source sub-package
        if len(parts) < 2:
            if py_file.name == "__init__.py":
                # synapse/__init__.py — scan it (no longer skipped).
                source_pkg = "__init__"
            else:
                # Top-level synapse module (e.g. synapse/errors.py)
                source_pkg = py_file.stem
        else:
            source_pkg = parts[0]

        pkg_files[source_pkg].add(str(rel))

        # Compute the dotted package path for relative import resolution
        if rel.parent.parts:
            file_package = "synapse." + ".".join(rel.parent.parts)
        else:
            file_package = "synapse"

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            all_errors.append(f"SyntaxError in {rel}: {exc}")
            continue
        except UnicodeDecodeError:
            continue

        rel_posix = rel.as_posix()
        visitor = _ImportVisitor(
            source_pkg,
            file_package,
            source_lines=source.splitlines(),
            rel_posix=rel_posix,
        )
        visitor.visit(tree)
        all_edges.update(visitor.edges)
        toplevel_edges.update(visitor.toplevel_edges)
        for err in visitor.errors:
            all_errors.append(f"{rel}: {err}")

    return all_edges, toplevel_edges, dict(pkg_files), all_errors


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
# 5-Tier validation
# ---------------------------------------------------------------------------


def check_tier_violations(
    edges: set[tuple[str, str]],
) -> tuple[list[tuple[str, str, int, int]], list[str]]:
    """Check for upward tier violations across all import edges (including local scopes).

    Every import edge in Synapse must satisfy tier[src] >= tier[dst] with zero
    exemptions or upward bypasses.

    Returns:
        (tier_violations, unregistered_package_errors)
    """
    violations: list[tuple[str, str, int, int]] = []
    unregistered: list[str] = []

    for src, dst in sorted(edges):
        # Skip the top-level __init__ aggregator
        if src == "__init__" or dst == "__init__":
            continue
        # Skip private C extensions
        if src.startswith(_PRIVATE_PREFIX) or dst.startswith(_PRIVATE_PREFIX):
            continue

        src_tier = TIER_MAP.get(src)
        dst_tier = TIER_MAP.get(dst)

        if src_tier is None:
            unregistered.append(f"Source package {src!r} is not registered in TIER_MAP")
            continue
        if dst_tier is None:
            unregistered.append(f"Target package {dst!r} is not registered in TIER_MAP")
            continue

        if src_tier < dst_tier:
            violations.append((src, dst, src_tier, dst_tier))

    return violations, sorted(set(unregistered))


# ---------------------------------------------------------------------------
# Validation (public API)
# ---------------------------------------------------------------------------


def validate(
    root: Path,
) -> tuple[list[list[str]], list[tuple[str, str, int, int]], set[tuple[str, str]], list[str]]:
    """Validate the import DAG across all scopes.

    Returns
    -------
    cycles:
        List of SCCs with more than 1 member (circular dependencies).
    tier_violations:
        Edges (from any scope) that violate the 5-tier unidirectional rule.
    all_edges:
        Complete edge set for diagnostics (all scopes).
    errors:
        Fatal errors (SyntaxError, non-constant dynamic imports, unregistered packages).
    """
    all_edges, toplevel_edges, pkg_files, errors = scan_import_dag(root)

    # Verify all discovered sub-packages in pkg_files are registered in TIER_MAP
    for pkg_name in sorted(pkg_files.keys()):
        if pkg_name == "__init__" or pkg_name.startswith(_PRIVATE_PREFIX):
            continue
        if pkg_name not in TIER_MAP:
            errors.append(
                f"Discovered sub-package {pkg_name!r} (files: {sorted(pkg_files[pkg_name])[:3]}) "
                f"is not registered in TIER_MAP"
            )

    all_pkgs = set()
    for u, v in all_edges:
        all_pkgs.add(u)
        all_pkgs.add(v)

    # Check for unregistered packages in TIER_MAP
    for pkg in sorted(all_pkgs):
        if pkg == "__init__" or pkg.startswith(_PRIVATE_PREFIX):
            continue
        if pkg not in TIER_MAP:
            errors.append(f"Package {pkg!r} is not registered in TIER_MAP (add it to TIER_MAP or mark as private)")

    sccs = _tarjan_scc(all_pkgs, all_edges)
    cycles = [scc for scc in sccs if len(scc) > 1]
    tier_violations, unreg = check_tier_violations(all_edges)
    errors = errors + unreg

    return cycles, tier_violations, all_edges, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python tools/import_dag_guard.py [--check]``."""
    check_mode = "--check" in sys.argv

    # Find repo root: tools/import_dag_guard.py → repo root is parent of tools/
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    cycles, tier_violations, all_edges, errors = validate(root)

    if not check_mode:
        print("=== Import DAG Edges ===")
        for u, v in sorted(all_edges):
            print(f"  {u} → {v}")
        print()

    failed = False

    if errors:
        print(f"FAIL: {len(errors)} fatal error(s):")
        for err in errors:
            print(f"  {err}")
        failed = True

    if cycles:
        print(f"FAIL: {len(cycles)} circular dependency cycle(s) found:")
        for scc in cycles:
            print(f"  CYCLE: {' → '.join(scc)} → {scc[0]}")
        failed = True
    else:
        if not check_mode:
            print("OK: No circular dependencies found.")

    if tier_violations:
        print(
            f"FAIL: {len(tier_violations)} tier violation(s) "
            f"(upward cross-tier module-level imports):"
        )
        for src, dst, s_tier, d_tier in tier_violations:
            print(f"  {src} (T{s_tier}) → {dst} (T{d_tier})")
        failed = True
    else:
        if not check_mode:
            print("OK: No tier violations found.")

    if failed:
        sys.exit(1)
    else:
        if check_mode:
            print("OK: import DAG is clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
