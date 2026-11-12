#!/usr/bin/env python3
"""wire_guard — cross-language FGPP contract validator.

Validates that the Python ``synapse.fgpp.schema`` constants are
consistent with the C++ header
``aether/fgpp/include/aether/fgpp/fgpp_types.h``.

If the C++ header is not available (no ``aether`` checkout), the tool
falls back to internal consistency checks on the Python schema itself.

Usage::

    python tools/wire_guard.py --check
    python tools/wire_guard.py --aether-header ../aether/fgpp/include/aether/fgpp/fgpp_types.h --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ensure the repo ``python/`` directory is importable when executed
# directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_SRC = _REPO_ROOT / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

from synapse.fgpp.schema import (  # noqa: E402
    DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
    FGPP_BYTES_PER_PIXEL,
    FGPP_CHANNEL_SPECS,
    FGPP_NUM_CHANNELS,
    FGPP_SCHEMA_VERSION,
    MIN_CHARACTER_WHITEBOX_VERTICES,
    PixelType,
    computed_bytes_per_pixel,
    get_channel_names,
)

# ---------------------------------------------------------------------------
# Internal consistency checks (always run)
# ---------------------------------------------------------------------------

_PIXEL_TYPE_BYTES = {PixelType.UINT32: 4, PixelType.HALF: 2, PixelType.FLOAT: 4}


def _check_internal() -> list[str]:
    """Return a list of error descriptions (empty = all OK)."""
    errors: list[str] = []

    # 1. Channel count
    if len(FGPP_CHANNEL_SPECS) != FGPP_NUM_CHANNELS:
        errors.append(
            f"FGPP_NUM_CHANNELS ({FGPP_NUM_CHANNELS}) != "
            f"len(FGPP_CHANNEL_SPECS) ({len(FGPP_CHANNEL_SPECS)})"
        )

    # 2. ASCII sort
    names = get_channel_names()
    if names != sorted(names):
        errors.append("FGPP_CHANNEL_SPECS is NOT sorted in ASCII order")

    # 3. Bytes-per-pixel arithmetic
    computed = computed_bytes_per_pixel()
    if computed != FGPP_BYTES_PER_PIXEL:
        errors.append(
            f"computed bytes-per-pixel ({computed}) != "
            f"FGPP_BYTES_PER_PIXEL ({FGPP_BYTES_PER_PIXEL})"
        )

    # 4. PixelType values
    if PixelType.UINT32 != 0 or PixelType.HALF != 1 or PixelType.FLOAT != 2:
        errors.append("PixelType enum values do not match expected (0, 1, 2)")

    # 5. Schema version
    if FGPP_SCHEMA_VERSION != "fgpp.v1.0":
        errors.append(f"FGPP_SCHEMA_VERSION is '{FGPP_SCHEMA_VERSION}', expected 'fgpp.v1.0'")

    # 6. Whitebox & anchor constants
    if MIN_CHARACTER_WHITEBOX_VERTICES != 1500:
        errors.append(
            f"MIN_CHARACTER_WHITEBOX_VERTICES is {MIN_CHARACTER_WHITEBOX_VERTICES}, expected 1500"
        )
    if DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES != 48:
        errors.append(
            f"DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES is "
            f"{DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES}, expected 48"
        )

    return errors


# ---------------------------------------------------------------------------
# C++ header parsing
# ---------------------------------------------------------------------------

_RE_CHANNEL_NAME = re.compile(r'"([A-Za-z]+\.[A-Z]+)"')
_RE_EXR_SPECS_BODY = re.compile(
    r"getAlphabeticalEXRChannelSpecs\b.*?\{(.*?)\}\s*;\s*\}",
    re.DOTALL,
)
_RE_MIN_VERTS = re.compile(
    r"kMinCharacterWhiteboxVertices\s*=\s*(\d+)"
)
_RE_ANCHOR_INTERVAL = re.compile(
    r"kDefaultPhysicalAnchorIntervalFrames\s*=\s*(\d+)"
)
_RE_PIXEL_TYPE = re.compile(
    r"(\w+)\s*=\s*(\d+)\s*,?\s*//.*PixelType|enum\s+.*PixelType"
)


def _parse_header(path: Path) -> dict:
    """Extract contract values from the C++ header."""
    text = path.read_text(encoding="utf-8")

    # Extract channel names only from the getAlphabeticalEXRChannelSpecs()
    # function body to avoid matching group names (e.g., "BaseColor.RGB")
    # or Doxygen comments.
    channel_names: list[str] = []
    m_body = _RE_EXR_SPECS_BODY.search(text)
    if m_body:
        specs_body = m_body.group(1)
        channel_names = sorted(set(_RE_CHANNEL_NAME.findall(specs_body)))
    else:
        # Fallback: search whole file (less precise)
        channel_names = sorted(set(_RE_CHANNEL_NAME.findall(text)))

    result: dict = {
        "channel_names": channel_names,
        "min_verts": None,
        "anchor_interval": None,
    }

    m = _RE_MIN_VERTS.search(text)
    if m:
        result["min_verts"] = int(m.group(1))

    m = _RE_ANCHOR_INTERVAL.search(text)
    if m:
        result["anchor_interval"] = int(m.group(1))

    return result


def _check_against_header(header_path: Path) -> list[str]:
    """Compare Python schema against parsed C++ header values."""
    errors: list[str] = []
    parsed = _parse_header(header_path)

    # Channel names
    py_names = get_channel_names()
    cpp_names = parsed["channel_names"]
    if cpp_names and cpp_names != py_names:
        errors.append(
            f"Channel name mismatch:\n"
            f"  Python: {py_names}\n"
            f"  C++:    {cpp_names}"
        )

    # Min vertices
    if parsed["min_verts"] is not None:
        if parsed["min_verts"] != MIN_CHARACTER_WHITEBOX_VERTICES:
            errors.append(
                f"kMinCharacterWhiteboxVertices: C++={parsed['min_verts']}, "
                f"Python={MIN_CHARACTER_WHITEBOX_VERTICES}"
            )

    # Anchor interval
    if parsed["anchor_interval"] is not None:
        if parsed["anchor_interval"] != DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES:
            errors.append(
                f"kDefaultPhysicalAnchorIntervalFrames: C++={parsed['anchor_interval']}, "
                f"Python={DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES}"
            )

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_DEFAULT_HEADER_PROBES = [
    Path("../aether/fgpp/include/aether/fgpp/fgpp_types.h"),
    _REPO_ROOT.parent / "aether" / "fgpp" / "include" / "aether" / "fgpp" / "fgpp_types.h",
]


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 1 on contract mismatch."""
    parser = argparse.ArgumentParser(
        description="FGPP cross-language wire format contract validator"
    )
    parser.add_argument(
        "--aether-header",
        type=Path,
        default=None,
        help="Path to aether fgpp_types.h (default: auto-probe)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 on mismatch (CI mode)",
    )
    args = parser.parse_args(argv)

    all_errors: list[str] = []

    # Always run internal consistency
    print("=== FGPP internal consistency check ===")
    internal_errors = _check_internal()
    all_errors.extend(internal_errors)

    if internal_errors:
        for e in internal_errors:
            print(f"  FAIL: {e}")
    else:
        print("  OK: All internal checks passed")
        print(f"  Channels: {FGPP_NUM_CHANNELS}")
        print(f"  Bytes/pixel: {FGPP_BYTES_PER_PIXEL}")
        print(f"  Schema version: {FGPP_SCHEMA_VERSION}")

    # Try to find and compare C++ header
    header_path = args.aether_header
    if header_path is None:
        for probe in _DEFAULT_HEADER_PROBES:
            if probe.exists():
                header_path = probe
                break

    print()
    if header_path is not None and header_path.exists():
        print(f"=== Cross-language check against {header_path} ===")
        cpp_errors = _check_against_header(header_path)
        all_errors.extend(cpp_errors)
        if cpp_errors:
            for e in cpp_errors:
                print(f"  FAIL: {e}")
        else:
            print("  OK: Python schema matches C++ header")
    else:
        print(
            "WARNING: C++ header not found. "
            "Skipping cross-language comparison.\n"
            "  Hint: pass --aether-header <path> or check out aether beside synapse."
        )

    print()
    if all_errors:
        print(f"RESULT: {len(all_errors)} error(s) found")
        return 1
    else:
        print("RESULT: All checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
