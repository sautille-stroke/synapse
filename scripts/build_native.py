#!/usr/bin/env python3
"""Build and verify the upstream C++20 geometry kernel (@aether//python:aether_geo).

Since RFC 0071 (PR-U20), Synapse no longer maintains a local csrc/geometry/
extension; all native geometry operations run through aether's SSOT `aether_geo`
extension. This script locates `AETHER_ROOT` (or sibling `../aether`), builds
`//python:aether_geo` with Bazel, optionally installs the resulting shared
library into `--output-dir`, and runs a Tier-0 smoke test.

Exit codes follow the synapse CLI deterministic status code contract:
  0  — success
  10 — smoke test or validation failure
  30 — backend unavailable (aether checkout or bazel missing, retired direct mode)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _get_repo_root() -> Path:
    """Return the synapse repository root directory."""
    return Path(__file__).resolve().parent.parent


def _find_bazel() -> str | None:
    """Detect bazel or bazelisk executable in PATH."""
    for exe in ("bazel", "bazelisk"):
        path = shutil.which(exe)
        if path:
            return path
    return None


def _find_aether_root(repo_root: Path) -> Path | None:
    """Locate the upstream aether repository checkout.

    Resolution order:
    1. ``AETHER_ROOT`` environment variable (if explicitly set — fail closed if invalid).
    2. Sibling directory ``../aether`` relative to synapse repo root.
    """
    if "AETHER_ROOT" in os.environ:
        env_val = os.environ["AETHER_ROOT"].strip()
        if not env_val:
            return None
        cand = Path(env_val).resolve()
        return cand if (cand / "MODULE.bazel").is_file() or (cand / "python" / "BUILD.bazel").is_file() else None

    for parent in (repo_root.parent, repo_root):
        cand = parent / "aether"
        if (cand / "python" / "BUILD.bazel").is_file():
            return cand
    return None


def _locate_built_aether_geo(aether_root: Path) -> Path | None:
    """Locate built aether_geo shared library inside aether's bazel-bin."""
    bazel_bin_python = aether_root / "bazel-bin" / "python"
    for name in ("aether_geo.so", "aether_geo.pyd", "aether_geo.dylib"):
        cand = bazel_bin_python / name
        if cand.is_file():
            return cand
    if bazel_bin_python.is_dir():
        for cand in sorted(bazel_bin_python.glob("aether_geo*")):
            if cand.is_file() and cand.suffix in {".so", ".pyd", ".dylib"}:
                return cand
    return None


def _build_with_bazel(
    aether_root: Path, bazel_bin: str, quiet: bool = False, timeout: float = 300.0
) -> Path | None:
    """Build @aether//python:aether_geo using Bazel."""
    cmd = [bazel_bin, "build", "//python:aether_geo", "//python:aether_metrics"]
    if not quiet:
        print(f"[build_native] Running Bazel in {aether_root}: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=str(aether_root), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"[build_native] Bazel build timed out after {timeout}s", file=sys.stderr)
        return None
    except Exception as exc:
        if not quiet:
            print(f"[build_native] Bazel invocation error: {exc}", file=sys.stderr)
        return None

    if result.returncode != 0:
        if not quiet:
            print("[build_native] Bazel build failed:", file=sys.stderr)
            if result.stdout:
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        return None

    return _locate_built_aether_geo(aether_root)


def _emit_error(exit_code: int, message: str, agent_fix_hint: str, as_json: bool = False) -> None:
    """Print a structured JSON error for agent consumption."""
    error = {
        "status": "error",
        "error": message,
        "exit_code": exit_code,
        "agent_fix_hint": agent_fix_hint,
    }
    if as_json:
        print(json.dumps(error))
    else:
        print(f"[build_native] Error: {message}", file=sys.stderr)
        print(json.dumps(error), file=sys.stderr)


def _install_extension(built_artifact: Path, target_dir: Path) -> list[Path]:
    """Copy built aether_geo shared library to target_dir with ABI and plain names."""
    target_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if sys.platform == "win32" else ".so"
    )
    installed: list[Path] = []

    abi_target = target_dir / f"aether_geo{ext_suffix}"
    if built_artifact.resolve() != abi_target.resolve():
        shutil.copy2(built_artifact, abi_target)
    installed.append(abi_target)

    plain_ext = ".pyd" if sys.platform == "win32" else ".so"
    plain_target = target_dir / f"aether_geo{plain_ext}"
    if plain_target != abi_target and built_artifact.resolve() != plain_target.resolve():
        shutil.copy2(built_artifact, plain_target)
        installed.append(plain_target)

    return installed


def _load_validate_func(target_dir: Path | None, repo_root: Path, quiet: bool) -> object | None:
    """Load validate_mesh_geometry from target_dir or via aether_bridge."""
    if target_dir is not None:
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
            ".pyd" if sys.platform == "win32" else ".so"
        )
        plain_ext = ".pyd" if sys.platform == "win32" else ".so"
        cands = [target_dir / f"aether_geo{ext_suffix}", target_dir / f"aether_geo{plain_ext}"]
        for cand in cands:
            if cand.is_file():
                try:
                    spec = importlib.util.spec_from_file_location("aether_geo", cand)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        fn = getattr(mod, "validate_mesh_geometry", None)
                        if fn is not None:
                            return fn
                except Exception as exc:
                    if not quiet:
                        print(f"[build_native] Error loading {cand}: {exc}", file=sys.stderr)

    python_root = str(repo_root / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    try:
        from synapse.bridge.aether_bridge import get_aether_geo

        mod = get_aether_geo()
        if mod is not None:
            return getattr(mod, "validate_mesh_geometry", None)
    except ImportError as exc:
        if not quiet:
            print(f"[build_native] Failed to locate aether_geo: {exc}", file=sys.stderr)
    return None


def _smoke_test(target_dir: Path | None, repo_root: Path, quiet: bool = False) -> bool:
    """Verify that aether_geo is loadable and passes a basic tetrahedron check."""
    try:
        import numpy as np
    except ImportError as exc:
        if not quiet:
            print(f"[build_native] Smoke test failed to import numpy: {exc}", file=sys.stderr)
        return False

    validate_func = _load_validate_func(target_dir=target_dir, repo_root=repo_root, quiet=quiet)
    if validate_func is None:
        if not quiet:
            print("[build_native] Smoke test could not load aether_geo.validate_mesh_geometry", file=sys.stderr)
        return False

    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    try:
        res = validate_func(
            vertices=verts,
            faces=faces,
            verts_per_face=3,
            require_watertight=True,
            min_area=1e-8,
            max_aspect_ratio=100.0,
            min_face_ratio=0.01,
        )
        if not res.get("is_manifold") or not res.get("passed"):
            if not quiet:
                print(f"[build_native] Smoke test returned unexpected result: {res}", file=sys.stderr)
            return False
    except Exception as exc:
        if not quiet:
            print(f"[build_native] Smoke test execution error: {exc}", file=sys.stderr)
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify @aether//python:aether_geo C++20 geometry extension."
    )
    parser.add_argument(
        "--force-direct",
        action="store_true",
        help="Retired flag (csrc/geometry migrated to @aether//python:aether_geo).",
    )
    parser.add_argument(
        "--check-only",
        "--check",
        action="store_true",
        dest="check_only",
        help="Only verify existing aether_geo extension integrity.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print build command without executing."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "bazel", "direct", "compiler"],
        default="auto",
        help="Build backend (bazel is canonical; direct/compiler retired after PR-U20).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to copy built extension shared libraries into (default: <repo>/python).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit single JSON envelope output.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Compilation timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    repo_root = _get_repo_root()
    quiet = args.json

    if args.check_only:
        if _smoke_test(target_dir=args.output_dir, repo_root=repo_root, quiet=quiet):
            if args.json:
                print(json.dumps({"status": "ok", "checked": True, "module": "aether_geo"}))
            else:
                print("aether_geo is installed and verified.")
            return 0
        _emit_error(
            30,
            "aether_geo verification failed",
            "Run 'python scripts/build_native.py --backend=bazel' with AETHER_ROOT set",
            as_json=args.json,
        )
        return 30

    aether_root = _find_aether_root(repo_root)
    bazel_bin = _find_bazel() or "bazel"

    if args.force_direct or args.backend in ("direct", "compiler"):
        if args.dry_run:
            if not quiet:
                print(
                    "[build_native] (dry-run) direct compile is retired after PR-U20; "
                    f"use Bazel in {aether_root or '<aether>'}: {bazel_bin} build //python:aether_geo"
                )
            if args.json:
                print(json.dumps({"status": "ok", "backend": "bazel", "dry_run": True}))
            return 0
        _emit_error(
            30,
            "Direct standalone C++ compile is retired after PR-U20 (csrc/geometry moved to @aether//python:aether_geo)",
            "Run 'python scripts/build_native.py --backend=bazel' with AETHER_ROOT or sibling ../aether",
            as_json=args.json,
        )
        return 30

    if args.dry_run:
        target_root = str(aether_root) if aether_root else "<aether>"
        if not quiet:
            print(f"[build_native] (dry-run) cd {target_root} && {bazel_bin} build //python:aether_geo")
        if args.json:
            print(json.dumps({"status": "ok", "backend": "bazel", "target": "//python:aether_geo", "dry_run": True}))
        return 0

    if aether_root is None:
        _emit_error(
            30,
            "Unable to locate upstream aether checkout for //python:aether_geo",
            "Set AETHER_ROOT=/path/to/aether or clone dervgen/aether at ../aether, then run: python scripts/build_native.py --backend=bazel",
            as_json=args.json,
        )
        return 30

    real_bazel = _find_bazel()
    if real_bazel is None:
        _emit_error(
            30,
            "Bazel / bazelisk executable not found in PATH",
            "Install Bazel 9.3.1 (or ensure bazel is in PATH) and run: python scripts/build_native.py --backend=bazel",
            as_json=args.json,
        )
        return 30

    built_artifact = _build_with_bazel(aether_root, real_bazel, quiet=quiet, timeout=args.timeout)
    if built_artifact is None:
        _emit_error(
            30,
            "Bazel build of @aether//python:aether_geo failed",
            "Run 'bazel build //python:aether_geo' inside AETHER_ROOT to inspect compiler errors",
            as_json=args.json,
        )
        return 30

    installed: list[Path] = [built_artifact]
    target_dir = args.output_dir if args.output_dir is not None else repo_root / "python"
    installed = _install_extension(built_artifact, target_dir)

    if _smoke_test(target_dir=target_dir, repo_root=repo_root, quiet=quiet):
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "backend": "bazel",
                        "installed": [str(p) for p in installed],
                        "verified": True,
                    }
                )
            )
        else:
            print("aether_geo native kernel successfully built and verified.")
        return 0

    _emit_error(
        10,
        "aether_geo smoke test failed after build",
        "Verify python ABI matches the Bazel Python toolchain and re-run build_native.py",
        as_json=args.json,
    )
    return 10


if __name__ == "__main__":
    sys.exit(main())
