#!/usr/bin/env python3
"""Build and bundle the native C++20 geometry kernel (_c_geometry).

This script provides turnkey build automation for the synapse native geometry
accelerator:
1. Detects if Bazel / bazelisk is available. If present, runs
   `bazel build //:_c_geometry` in `csrc/geometry/`.
2. If Bazel is unavailable (e.g., standard pip install or minimal Colab
   environment) or fails, falls back to direct C++20 compiler invocation
   (`clang++` / `g++` / `c++` or MSVC `cl.exe`) using pybind11 and system
   Python headers (unless --backend=bazel is explicitly requested).
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
import tempfile
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


def _build_with_bazel(
    repo_root: Path, bazel_bin: str, quiet: bool = False, timeout: float = 300.0
) -> Path | None:
    """Build _c_geometry using Bazel."""
    csrc_dir = repo_root / "csrc" / "geometry"
    cmd = [bazel_bin, "build", "//:_c_geometry"]
    if not quiet:
        print(f"[build_native] Running Bazel in {csrc_dir}: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=str(csrc_dir), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"[build_native] Bazel build timed out after {timeout}s", file=sys.stderr)
        return None
    except Exception as e:
        if not quiet:
            print(f"[build_native] Bazel invocation error: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        if not quiet:
            print("[build_native] Bazel build failed:", file=sys.stderr)
            if result.stdout:
                print(result.stdout, file=sys.stderr)
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
        import pybind11

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
        for fname in ["mesh_checks.cc", "marching_cubes.cc", "subdivision.cc"]:
            cand = geo_src_dir / fname
            if cand.is_file():
                sources.append(cand)
    core_assert = aether_root / "core" / "src" / "assert.cc"
    if core_assert.is_file():
        sources.append(core_assert)
    return sources


def _find_windows_python_lib() -> Path | None:
    """Probe for the Python import library on Windows."""
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    cand_dirs = [
        Path(sysconfig.get_config_var("installed_base") or sys.base_prefix) / "libs",
        Path(sys.prefix) / "libs",
        Path(sys.base_prefix) / "libs",
    ]
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        cand_dirs.append(Path(libdir))
    for d in cand_dirs:
        cand = d / f"python{ver}.lib"
        if cand.is_file():
            return cand
    return None


def _build_direct_fallback(
    repo_root: Path,
    output_file: Path,
    dry_run: bool = False,
    quiet: bool = False,
    as_json: bool = False,
    timeout: float = 300.0,
) -> bool:
    """Directly compile _c_geometry using host C++20 compiler.

    Compiles aether geo sources (mesh_checks.cc, mesh_forensics.cc) with
    ``-fno-exceptions -fno-rtti`` and bindings.cc with ``-fexceptions -frtti``
    into separate object files, then links them into a single shared library.

    Returns True on success, False on failure.
    """
    csrc_dir = repo_root / "csrc" / "geometry"
    bindings_src = csrc_dir / "src" / "bindings.cc"
    if not bindings_src.is_file():
        _emit_error(
            30,
            f"Missing bindings source file: {bindings_src}",
            "Ensure csrc/geometry/src/bindings.cc exists",
            as_json=as_json,
        )
        return False

    aether_geo_srcs = _find_aether_geo_src(repo_root)
    if not aether_geo_srcs and not dry_run:
        _emit_error(
            30,
            "Required @aether//geo source files (mesh_checks.cc) not found for native compilation",
            "Ensure aether submodule or dev checkout is available at ../aether or csrc/aether",
            as_json=as_json,
        )
        return False

    pybind_inc = _find_pybind11_include(dry_run=dry_run)
    if not pybind_inc:
        _emit_error(
            30,
            "pybind11 headers not found in Python environment",
            "pip install 'pybind11>=2.11.0' && python scripts/build_native.py --backend=direct",
            as_json=as_json,
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
                    as_json=as_json,
                )
                return False
    else:
        compiler = (
            os.environ.get("CXX")
            or shutil.which("clang++")
            or shutil.which("g++")
            or shutil.which("c++")
        )
        if not compiler:
            if dry_run:
                compiler = "clang++"
            else:
                _emit_error(
                    30,
                    "No C++ compiler (clang++/g++/c++) found in PATH",
                    "apt-get install -y clang && python scripts/build_native.py --backend=direct",
                    as_json=as_json,
                )
                return False

    # --- Split compilation: aether geo sources vs pybind11 bindings ---
    obj_dir = output_file.parent
    obj_files: list[Path] = []

    def _run_compile(cmd: list[str], label: str) -> bool:
        if not quiet:
            print(f"[build_native] {label}:")
            print(" ", " ".join(cmd))
        if dry_run:
            return True
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if not quiet:
                print(f"[build_native] {label} timed out after {timeout}s", file=sys.stderr)
            return False
        if res.returncode != 0:
            if not quiet:
                print(f"[build_native] {label} failed:", file=sys.stderr)
                if res.stdout:
                    print(res.stdout, file=sys.stderr)
                if res.stderr:
                    print(res.stderr, file=sys.stderr)
            return False
        return True

    if is_windows:
        # --- Step 1a: Compile aether geo sources with /EHs-c- /GR- ---
        for src in aether_geo_srcs or []:
            obj_name = src.stem + ".obj"
            obj_path = obj_dir / obj_name
            cmd = [
                compiler,
                "/c",
                "/O2",
                "/std:c++20",
                "/utf-8",
                "/EHs-c-",
                "/GR-",
                "/D_HAS_EXCEPTIONS=0",
                "/DNOMINMAX",
                "/D_CRT_SECURE_NO_WARNINGS",
                "/DSPDLOG_NO_EXCEPTIONS=1",
                "/DFMT_USE_EXCEPTIONS=0",
                *includes,
                str(src),
                f"/Fo:{obj_path}",
            ]
            if not _run_compile(cmd, f"Compiling {src.name} (no-exceptions)"):
                return False
            obj_files.append(obj_path)

        # --- Step 1b: Compile bindings.cc with /EHsc /GR ---
        bindings_obj = obj_dir / "bindings.obj"
        cmd = [
            compiler,
            "/c",
            "/O2",
            "/std:c++20",
            "/utf-8",
            "/EHsc",
            "/GR",
            "/DNOMINMAX",
            "/D_CRT_SECURE_NO_WARNINGS",
            "/DSYNAPSE_NO_XATLAS",
            *includes,
            str(bindings_src),
            f"/Fo:{bindings_obj}",
        ]
        if not _run_compile(cmd, "Compiling bindings.cc (exceptions+rtti)"):
            return False
        obj_files.append(bindings_obj)

        # --- Step 2: Link ---
        python_lib = _find_windows_python_lib()
        lib_args = ["/link", f"/LIBPATH:{python_lib.parent}", python_lib.name] if python_lib else []
        link_cmd = [
            compiler,
            "/LD",
            *[str(o) for o in obj_files],
            f"/Fe:{output_file}",
            *lib_args,
        ]
        if not _run_compile(link_cmd, "Linking shared library"):
            return False
    else:
        common_flags = [
            "-O3",
            "-Wall",
            "-std=c++20",
            "-fPIC",
            "-fvisibility=hidden",
        ]

        # --- Step 1a: Compile aether geo sources with -fno-exceptions -fno-rtti ---
        for src in aether_geo_srcs or []:
            obj_name = src.stem + ".o"
            obj_path = obj_dir / obj_name
            cmd = [
                compiler,
                "-c",
                *common_flags,
                "-fno-exceptions",
                "-fno-rtti",
                "-DSPDLOG_NO_EXCEPTIONS=1",
                "-DFMT_USE_EXCEPTIONS=0",
                *includes,
                str(src),
                "-o",
                str(obj_path),
            ]
            if not _run_compile(cmd, f"Compiling {src.name} (no-exceptions)"):
                return False
            obj_files.append(obj_path)

        # --- Step 1b: Compile bindings.cc with -fexceptions -frtti ---
        bindings_obj = obj_dir / "bindings.o"
        cmd = [
            compiler,
            "-c",
            *common_flags,
            "-fexceptions",
            "-frtti",
            "-DSYNAPSE_NO_XATLAS",
            *includes,
            str(bindings_src),
            "-o",
            str(bindings_obj),
        ]
        if not _run_compile(cmd, "Compiling bindings.cc (exceptions+rtti)"):
            return False
        obj_files.append(bindings_obj)

        # --- Step 2: Link ---
        link_cmd = [
            compiler,
            "-shared",
            *[str(o) for o in obj_files],
            "-o",
            str(output_file),
        ]
        if sys.platform == "darwin":
            link_cmd.extend(["-Wl,-undefined,dynamic_lookup"])
        if not _run_compile(link_cmd, "Linking shared library"):
            return False

    if dry_run:
        if not quiet:
            print("[build_native] (dry-run mode, skipping execution)")
        return True

    return output_file.is_file()


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
    """Copy built shared library to target directory with ABI and generic names."""
    target_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if sys.platform == "win32" else ".so"
    )

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


def _smoke_test(target_dir: Path | None = None, quiet: bool = False) -> bool:
    """Verify that _c_geometry is loadable and passes a basic geometry check."""
    try:
        import numpy as np
    except ImportError as e:
        if not quiet:
            print(f"[build_native] Smoke test failed to import numpy: {e}", file=sys.stderr)
        return False

    validate_func = None
    if target_dir:
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
            ".pyd" if sys.platform == "win32" else ".so"
        )
        cand_files = [
            target_dir / f"_c_geometry{ext_suffix}",
            target_dir / ("_c_geometry.pyd" if sys.platform == "win32" else "_c_geometry.so"),
        ]
        for cand in cand_files:
            if cand.is_file():
                try:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("_c_geometry", cand)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        validate_func = getattr(mod, "validate_mesh_geometry", None)
                        if validate_func:
                            break
                except Exception as e:
                    if not quiet:
                        print(f"[build_native] Error loading {cand}: {e}", file=sys.stderr)

    if validate_func is None:
        try:
            from synapse._c_geometry import validate_mesh_geometry

            validate_func = validate_mesh_geometry
        except ImportError as e:
            if not quiet:
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
                print(
                    f"[build_native] Smoke test returned unexpected result: {res}", file=sys.stderr
                )
            return False
    except Exception as e:
        if not quiet:
            print(f"[build_native] Smoke test execution error: {e}", file=sys.stderr)
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and install native C++20 _c_geometry extension."
    )
    parser.add_argument(
        "--force-direct", action="store_true", help="Force direct C++ compilation, bypassing Bazel."
    )
    parser.add_argument(
        "--check-only",
        "--check",
        action="store_true",
        dest="check_only",
        help="Only verify existing extension integrity.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print compiler commands without executing."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "bazel", "direct", "compiler"],
        default="auto",
        help="Build backend to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for built .so (default: python/synapse/).",
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
    if args.backend == "compiler":
        args.backend = "direct"
    target_dir = args.output_dir if args.output_dir else repo_root / "python" / "synapse"
    quiet = args.json

    if args.check_only:
        if _smoke_test(target_dir=target_dir, quiet=quiet):
            if args.json:
                print(json.dumps({"status": "ok", "checked": True, "target_dir": str(target_dir)}))
            else:
                print("✅ synapse._c_geometry is installed and verified.")
            return 0
        else:
            _emit_error(
                30,
                "synapse._c_geometry verification failed",
                "Run 'python scripts/build_native.py' to rebuild native kernel",
                as_json=args.json,
            )
            return 30

    built_artifact: Path | None = None
    chosen_backend = "unknown"

    use_bazel = (
        (args.backend in ("auto", "bazel")) and not args.force_direct and args.backend != "direct"
    )
    bazel_bin = _find_bazel() if use_bazel else None

    # Step 1: Try Bazel
    if bazel_bin:
        if not quiet:
            print(f"[build_native] Found Bazel at: {bazel_bin}")
        if args.dry_run:
            if not quiet:
                print(
                    f"[build_native] (dry-run) {bazel_bin} build //:_c_geometry --config={_get_platform_config()}"
                )
            if args.json:
                print(json.dumps({"status": "ok", "backend": "bazel", "dry_run": True}))
            return 0
        built_artifact = _build_with_bazel(repo_root, bazel_bin, quiet=quiet, timeout=args.timeout)
        if built_artifact:
            chosen_backend = "bazel"
            if not quiet:
                print(f"[build_native] Bazel build succeeded: {built_artifact}")

    # If backend was explicitly set to bazel and bazel failed or wasn't found: fail-closed!
    if args.backend == "bazel" and not built_artifact:
        _emit_error(
            30,
            "Bazel build failed and --backend=bazel was explicitly specified (no fallback permitted)",
            "Verify Bazel installation or run 'python scripts/build_native.py --backend=direct'",
            as_json=args.json,
        )
        return 30

    # Step 2: Fallback to direct compilation if Bazel was absent or failed
    if not built_artifact:
        if not quiet:
            print("[build_native] Falling back to direct C++20 compiler...")
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
            ".pyd" if sys.platform == "win32" else ".so"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_out = Path(tmp_dir) / f"_c_geometry{ext_suffix}"
            if _build_direct_fallback(
                repo_root,
                temp_out,
                dry_run=args.dry_run,
                quiet=quiet,
                as_json=False,
                timeout=args.timeout,
            ):
                if args.dry_run:
                    if args.json:
                        print(json.dumps({"status": "ok", "backend": "direct", "dry_run": True}))
                    return 0
                chosen_backend = "direct"
                if not quiet:
                    print(f"[build_native] Direct compilation succeeded: {temp_out}")
                # Install from temp_out while temp directory is alive
                installed = _install_extension(temp_out, target_dir)
            else:
                _emit_error(
                    30,
                    "Both Bazel and fallback compilation failed",
                    "pip install 'pybind11>=2.11.0' && python scripts/build_native.py --backend=direct",
                    as_json=args.json,
                )
                return 30
    else:
        # Install from Bazel output
        installed = _install_extension(built_artifact, target_dir)

    for p in installed:
        if not quiet:
            print(f"[build_native] Installed extension: {p} ({p.stat().st_size} bytes)")

    # Step 4: Run smoke test
    if not quiet:
        print("[build_native] Running smoke test...")
    if str(repo_root / "python") not in sys.path:
        sys.path.insert(0, str(repo_root / "python"))

    if _smoke_test(target_dir=target_dir, quiet=quiet):
        if args.json:
            out_envelope = {
                "status": "ok",
                "backend": chosen_backend,
                "installed": [str(p) for p in installed],
                "verified": True,
            }
            print(json.dumps(out_envelope))
        else:
            print("✅ synapse._c_geometry native kernel successfully built and verified.")
        return 0
    else:
        _emit_error(
            10,
            "synapse._c_geometry smoke test failed",
            "Verify mesh validation logic or rebuild with debug symbols",
            as_json=args.json,
        )
        return 10


if __name__ == "__main__":
    sys.exit(main())
