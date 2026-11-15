#!/usr/bin/env python3
"""Build and bundle the native C++20 geometry kernel (_c_geometry).

This script provides turnkey build automation for the synapse native geometry
accelerator:
1. Detects if Bazel / bazelisk is available. If present, runs
   `bazel build //:_c_geometry` in `csrc/geometry/`.
2. If Bazel is unavailable (e.g., standard pip install or minimal Colab
   environment) or fails, falls back to direct C++20 compiler invocation
   (`clang++` / `g++` / `c++` or MSVC `cl.exe`) using pybind11 and system
   Python headers.
3. Automatically copies and names the built shared library in `python/synapse/`
   matching the active Python ABI extension suffix (e.g.
   `_c_geometry.cpython-312-x86_64-linux-gnu.so` and `_c_geometry.so`).
4. Executes a non-destructive Tier-0 smoke test to verify import integrity.

Exit codes follow the synapse CLI deterministic status code contract:
  0  — success
  10 — argument or contract validation failure
  30 — GPU/backend unavailable (pybind11 missing, no compiler, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _get_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def _get_platform_config() -> str:
    """Return Bazel --config flag based on host platform."""
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "darwin":
        return "macos"
    elif sys.platform == "win32":
        return "windows"
    return "linux"


def _find_bazel() -> str | None:
    """Detect bazel or bazelisk executable in PATH."""
    for exe in ["bazel", "bazelisk"]:
        path = shutil.which(exe)
        if path:
            return path
    return None


def _build_with_bazel(repo_root: Path, bazel_bin: str) -> Path | None:
    """Build _c_geometry using Bazel."""
    csrc_dir = repo_root / "csrc" / "geometry"
    config_flag = f"--config={_get_platform_config()}"
    cmd = [bazel_bin, "build", "//:_c_geometry", config_flag]
    print(f"[build_native] Running Bazel in {csrc_dir}: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(csrc_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print("[build_native] Bazel build failed:")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return None

    # Locate generated output in bazel-bin
    bazel_bin_dir = csrc_dir / "bazel-bin"
    candidates = [
        bazel_bin_dir / "_c_geometry.so",
        bazel_bin_dir / "_c_geometry.pyd",
        bazel_bin_dir / "_c_geometry.dylib",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    # Scan bazel-bin recursively if not in root of bazel-bin
    if bazel_bin_dir.is_dir():
        for f in bazel_bin_dir.rglob("*_c_geometry*"):
            if f.is_file() and f.suffix in {".so", ".pyd", ".dylib"}:
                return f

    return None


def _find_pybind11_include(dry_run: bool = False) -> Path | None:
    """Find pybind11 header directory.

    Does NOT auto-install pybind11 — callers must install it explicitly
    before invoking this script (``pip install pybind11>=2.11.0``).
    Returns None when pybind11 is absent.
    """
    try:
        import pybind11  # type: ignore[import-not-found]
        inc = Path(pybind11.get_include())
        if inc.is_dir() and (inc / "pybind11" / "pybind11.h").is_file():
            return inc
    except ImportError:
        pass

    if dry_run:
        return Path("<pybind11-include>")

    # Check bazel cache for previously downloaded pybind11
    home = Path.home()
    bazel_cache = home / ".cache" / "bazel"
    if bazel_cache.is_dir():
        for h in bazel_cache.rglob("pybind11.h"):
            candidate = h.parent.parent
            if (candidate / "pybind11" / "pybind11.h").is_file():
                return candidate

    return None


def _find_aether_root(repo_root: Path) -> Path | None:
    """Find aether root directory.

    Resolution order:
    1. ``AETHER_ROOT`` environment variable (explicit override).
    2. Sibling directory ``../aether`` (standard dev checkout).
    3. In-tree snapshot ``csrc/aether`` (offline Colab containers).
    """
    if "AETHER_ROOT" in os.environ:
        cand = Path(os.environ["AETHER_ROOT"]).resolve()
        if cand.is_dir():
            return cand
    for cand in [repo_root.parent / "aether", repo_root / "csrc" / "aether"]:
        if cand.is_dir():
            return cand
    return None


def _find_aether_includes(repo_root: Path) -> list[Path]:
    """Collect all required aether include directories for direct compilation.

    Returns include paths for: math, geo, core, and spatial modules.
    Each path is only included if the directory actually exists.
    """
    aether_root = _find_aether_root(repo_root)
    if not aether_root:
        return []

    include_dirs: list[Path] = []
    # Required aether submodules for geometry compilation
    for submod in ["math", "geo", "core", "spatial"]:
        cand = aether_root / submod / "include"
        if cand.is_dir():
            include_dirs.append(cand)
    return include_dirs


def _find_aether_geo_src(repo_root: Path) -> list[Path]:
    """Find aether/geo implementation source files for direct compilation."""
    aether_root = _find_aether_root(repo_root)
    if not aether_root:
        return []

    sources: list[Path] = []
    geo_src_dir = aether_root / "geo" / "src"
    if geo_src_dir.is_dir():
        # mesh_checks.cc is the primary required implementation unit
        mesh_checks = geo_src_dir / "mesh_checks.cc"
        if mesh_checks.is_file():
            sources.append(mesh_checks)
        # mesh_forensics.cc provides optional additional validation
        mesh_forensics = geo_src_dir / "mesh_forensics.cc"
        if mesh_forensics.is_file():
            sources.append(mesh_forensics)
    return sources


def _build_direct_fallback(repo_root: Path, output_file: Path, dry_run: bool = False) -> bool:
    """Directly compile _c_geometry using host C++20 compiler.

    Returns True on success, False on failure. When pybind11 is missing,
    prints a structured JSON error with ``agent_fix_hint`` and returns False.
    """
    csrc_dir = repo_root / "csrc" / "geometry"
    bindings_src = csrc_dir / "src" / "bindings.cc"
    if not bindings_src.is_file():
        print(f"[build_native] Missing bindings source file: {bindings_src}", file=sys.stderr)
        return False

    sources = [bindings_src]
    aether_geo_srcs = _find_aether_geo_src(repo_root)
    if aether_geo_srcs:
        sources.extend(aether_geo_srcs)
    else:
        print("[build_native] Notice: aether/geo/src/ not found; compiling with header declarations only", file=sys.stderr)

    pybind_inc = _find_pybind11_include(dry_run=dry_run)
    if not pybind_inc:
        _emit_error(
            30,
            "pybind11 headers not found in Python environment",
            "pip install 'pybind11>=2.11.0' && python scripts/build_native.py --backend=direct",
        )
        return False

    aether_includes = _find_aether_includes(repo_root)

    includes = [
        f"-I{csrc_dir / 'include'}",
        f"-I{pybind_inc}",
        f"-I{sysconfig.get_path('include')}",
    ]
    platinc = sysconfig.get_path("platinclude")
    if platinc and platinc != sysconfig.get_path("include"):
        includes.append(f"-I{platinc}")
    for inc in aether_includes:
        includes.append(f"-I{inc}")

    is_windows = sys.platform == "win32"
    source_args = [str(s) for s in sources]
    if is_windows:
        compiler = os.environ.get("CXX") or shutil.which("cl")
        if not compiler:
            if dry_run:
                compiler = "cl.exe"
            else:
                _emit_error(
                    30,
                    "MSVC cl.exe not found in PATH",
                    "Install Visual Studio Build Tools and ensure cl.exe is in PATH",
                )
                return False
        cmd = [
            compiler,
            "/O2",
            "/std:c++20",
            "/utf-8",
            "/LD",
            "/EHsc",
            "/DNOMINMAX",
            "/D_CRT_SECURE_NO_WARNINGS",
            *includes,
            *source_args,
            f"/Fe:{output_file}",
        ]
    else:
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        if not compiler:
            if dry_run:
                compiler = "clang++"
            else:
                _emit_error(
                    30,
                    "No C++ compiler (clang++/g++/c++) found in PATH",
                    "apt-get install -y clang && python scripts/build_native.py --backend=direct",
                )
                return False
        cmd = [
            compiler,
            "-O3",
            "-Wall",
            "-shared",
            "-std=c++20",
            "-fPIC",
            "-fvisibility=hidden",
            *includes,
            *source_args,
            "-o",
            str(output_file),
        ]

    print(f"[build_native] Compiling fallback with {compiler}:")
    print(" ", " ".join(cmd))
    if dry_run:
        print("[build_native] (dry-run mode, skipping execution)")
        return True

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[build_native] Fallback compilation failed:", file=sys.stderr)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        return False

    return output_file.is_file()


def _emit_error(exit_code: int, message: str, agent_fix_hint: str) -> None:
    """Print a structured JSON error to stderr for agent consumption."""
    error = {
        "error": message,
        "exit_code": exit_code,
        "agent_fix_hint": agent_fix_hint,
    }
    print(f"[build_native] Error: {message}", file=sys.stderr)
    print(json.dumps(error), file=sys.stderr)


def _install_extension(built_artifact: Path, target_dir: Path) -> list[Path]:
    """Copy built shared library to target directory with ABI and generic names."""
    target_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if sys.platform == "win32" else ".so")
    
    installed = []
    # 1. ABI-specific filename (e.g., _c_geometry.cpython-312-x86_64-linux-gnu.so)
    abi_target = target_dir / f"_c_geometry{ext_suffix}"
    shutil.copy2(built_artifact, abi_target)
    installed.append(abi_target)

    # 2. Plain extension filename for standard lookup
    plain_ext = ".pyd" if sys.platform == "win32" else ".so"
    plain_target = target_dir / f"_c_geometry{plain_ext}"
    if plain_target != abi_target:
        shutil.copy2(built_artifact, plain_target)
        installed.append(plain_target)

    return installed


def _smoke_test() -> bool:
    """Verify that _c_geometry is loadable and passes a basic geometry check."""
    try:
        import numpy as np
        from synapse._c_geometry import validate_mesh_geometry  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"[build_native] Smoke test failed to import: {e}", file=sys.stderr)
        return False

    # Minimal tetrahedron test
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    try:
        res = validate_mesh_geometry(
            vertices=verts,
            faces=faces,
            verts_per_face=3,
            require_watertight=True,
            min_area=1e-8,
            max_aspect_ratio=100.0,
            min_face_ratio=0.01,
        )
        if not res.get("is_manifold") or not res.get("passed"):
            print(f"[build_native] Smoke test returned unexpected result: {res}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[build_native] Smoke test execution error: {e}", file=sys.stderr)
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and install native C++20 _c_geometry extension.")
    parser.add_argument("--force-direct", action="store_true", help="Force direct C++ compilation, bypassing Bazel.")
    parser.add_argument("--check-only", "--check", action="store_true", dest="check_only", help="Only verify existing extension integrity.")
    parser.add_argument("--dry-run", action="store_true", help="Print compiler commands without executing.")
    parser.add_argument("--backend", choices=["auto", "bazel", "direct"], default="auto", help="Build backend to use.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for built .so (default: python/synapse/).")
    args = parser.parse_args()

    repo_root = _get_repo_root()
    target_dir = args.output_dir if args.output_dir else repo_root / "python" / "synapse"

    if args.check_only:
        if _smoke_test():
            print("✅ synapse._c_geometry is installed and verified.")
            return 0
        else:
            print("❌ synapse._c_geometry verification failed.", file=sys.stderr)
            return 30

    built_artifact: Path | None = None

    use_bazel = (args.backend in ("auto", "bazel")) and not args.force_direct and args.backend != "direct"
    bazel_bin = _find_bazel() if use_bazel else None

    # Step 1: Try Bazel
    if bazel_bin:
        print(f"[build_native] Found Bazel at: {bazel_bin}")
        if args.dry_run:
            print(f"[build_native] (dry-run) {bazel_bin} build //:_c_geometry --config={_get_platform_config()}")
            return 0
        built_artifact = _build_with_bazel(repo_root, bazel_bin)
        if built_artifact:
            print(f"[build_native] Bazel build succeeded: {built_artifact}")

    # Step 2: Fallback to direct compilation if Bazel was absent or failed
    if not built_artifact:
        print("[build_native] Falling back to direct C++20 compiler...")
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if sys.platform == "win32" else ".so")
        temp_out = repo_root / "csrc" / "geometry" / f"_c_geometry{ext_suffix}"
        if _build_direct_fallback(repo_root, temp_out, dry_run=args.dry_run):
            if args.dry_run:
                return 0
            built_artifact = temp_out
            print(f"[build_native] Direct compilation succeeded: {built_artifact}")
        else:
            _emit_error(
                30,
                "Both Bazel and fallback compilation failed",
                "pip install 'pybind11>=2.11.0' && python scripts/build_native.py --backend=direct",
            )
            return 30

    # Step 3: Install extension artifacts into python/synapse/
    installed = _install_extension(built_artifact, target_dir)
    for p in installed:
        print(f"[build_native] Installed extension: {p} ({p.stat().st_size} bytes)")

    # Step 4: Run smoke test
    print("[build_native] Running smoke test...")
    # Ensure target_dir is on Python path
    if str(repo_root / "python") not in sys.path:
        sys.path.insert(0, str(repo_root / "python"))

    if _smoke_test():
        print("✅ synapse._c_geometry native kernel successfully built and verified.")
        return 0
    else:
        print("❌ synapse._c_geometry smoke test failed.", file=sys.stderr)
        return 10


if __name__ == "__main__":
    sys.exit(main())
