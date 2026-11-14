#!/usr/bin/env python3
"""wire_guard — cross-language FGPP contract validator.

Validates that the Python ``synapse.fgpp.schema`` constants are
consistent with the C++ header
``aether/fgpp/include/aether/fgpp/fgpp_types.h``.

Extended checks:
  - Per-channel PixelType validation (FGPP contract §34 B/px).
  - StyleProfile v2.0.0 Pydantic model field alignment.
  - JSON Schema alignment for JobBundle, LoRA, Inspection, RefineSession.

If the C++ header is not available (no ``aether`` checkout), the tool
falls back to internal consistency checks on the Python schema itself.

Usage::

    python tools/wire_guard.py --check
    python tools/wire_guard.py --aether-header ../aether/fgpp/include/aether/fgpp/fgpp_types.h --check
"""

from __future__ import annotations

import argparse
import json
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
# PixelType per-channel validation (TASK 1)
# ---------------------------------------------------------------------------

# Normative PixelType contract — mirrors aether::fgpp::PixelType expectations.
# Cryptomatte.ID: UINT32 (never interpolated/filtered)
# CameraLinearDepth.Z: FLOAT (32-bit)
# WorldNormal.{X,Y,Z}: HALF or FLOAT
# BaseColor.{R,G,B}: HALF
# Roughness/Metallic (RoughnessMetallic.{R,G}): HALF
# DirectRadiance.{R,G,B}: HALF
# MotionVector.{X,Y}: HALF

_EXPECTED_PIXEL_TYPES: dict[str, set[PixelType]] = {
    "Cryptomatte.ID": {PixelType.UINT32},
    "CameraLinearDepth.Z": {PixelType.FLOAT},
    "WorldNormal.X": {PixelType.HALF, PixelType.FLOAT},
    "WorldNormal.Y": {PixelType.HALF, PixelType.FLOAT},
    "WorldNormal.Z": {PixelType.HALF, PixelType.FLOAT},
    "BaseColor.R": {PixelType.HALF},
    "BaseColor.G": {PixelType.HALF},
    "BaseColor.B": {PixelType.HALF},
    "RoughnessMetallic.R": {PixelType.HALF},
    "RoughnessMetallic.G": {PixelType.HALF},
    "DirectRadiance.R": {PixelType.HALF},
    "DirectRadiance.G": {PixelType.HALF},
    "DirectRadiance.B": {PixelType.HALF},
    "MotionVector.X": {PixelType.HALF},
    "MotionVector.Y": {PixelType.HALF},
}


def _check_pixel_types() -> list[str]:
    """Validate per-channel PixelType against the normative contract."""
    errors: list[str] = []
    spec_map: dict[str, PixelType] = {
        name: ptype for name, ptype, _ in FGPP_CHANNEL_SPECS
    }

    for channel_name, allowed_types in _EXPECTED_PIXEL_TYPES.items():
        actual = spec_map.get(channel_name)
        if actual is None:
            errors.append(
                f"PixelType check: channel '{channel_name}' not found in FGPP_CHANNEL_SPECS"
            )
        elif actual not in allowed_types:
            allowed_str = " | ".join(t.name for t in sorted(allowed_types, key=lambda t: t.value))
            errors.append(
                f"PixelType mismatch for '{channel_name}': "
                f"expected {allowed_str}, got {actual.name}"
            )

    # Ensure every channel in FGPP_CHANNEL_SPECS is covered by the contract
    for name, _, _ in FGPP_CHANNEL_SPECS:
        if name not in _EXPECTED_PIXEL_TYPES:
            errors.append(
                f"PixelType check: channel '{name}' has no normative PixelType contract entry"
            )

    return errors


# ---------------------------------------------------------------------------
# StyleProfile v2.0.0 alignment check (TASK 2)
# ---------------------------------------------------------------------------

# Expected v2.0.0 fields for StyleDescriptor (aligned with @aether//style).
_STYLE_PROFILE_V2_EXPECTED_FIELDS: dict[str, str] = {
    "working_colorspace": "str",
    "color_temp_kelvin": "float",
    "tint": "float",
    "exposure_ev": "float",
    "contrast": "float",
    "saturation": "float",
    "key_light_dir_world": "list[float]",
    "key_light_rgb": "list[float]",
    "ambient_ratio": "float",
    "grain_sigma": "float",
    "vignette_strength": "float",
    "tone_curve_lut_1d": "list[float] | None",
}


def _check_style_profile() -> list[str]:
    """Validate StyleDescriptor fields match v2.0.0 contract."""
    errors: list[str] = []

    try:
        from synapse.style.descriptor import StyleDescriptor
    except ImportError as e:
        errors.append(f"Cannot import StyleDescriptor: {e}")
        return errors

    model_fields = set(StyleDescriptor.model_fields.keys())
    expected_fields = set(_STYLE_PROFILE_V2_EXPECTED_FIELDS.keys())

    missing = expected_fields - model_fields
    extra = model_fields - expected_fields

    if missing:
        errors.append(
            f"StyleProfile v2.0.0: missing fields in Python model: {sorted(missing)}"
        )
    if extra:
        # Extra fields are a warning, not an error — we only fail on missing
        pass

    # Validate default colorspace
    try:
        inst = StyleDescriptor()
        if inst.working_colorspace != "ACEScg":
            errors.append(
                f"StyleProfile v2.0.0: default working_colorspace is "
                f"'{inst.working_colorspace}', expected 'ACEScg'"
            )
    except Exception as e:
        errors.append(f"StyleProfile v2.0.0: cannot instantiate default: {e}")

    return errors


# ---------------------------------------------------------------------------
# JSON Schema alignment checks (TASK 3)
# ---------------------------------------------------------------------------

_SCHEMA_DIR = _REPO_ROOT / "schemas"

_SCHEMA_FILES: dict[str, dict[str, str]] = {
    "job_bundle.v1.0.schema.json": {
        "expected_version_field": "schema_version",
        "expected_version_value": "synapse.job_bundle.v1.0",
    },
    "lora.v1.0.schema.json": {
        "expected_version_field": "schema_version",
        "expected_version_value": "synapse.lora.v1.0",
    },
    "inspection.v1.0.schema.json": {
        "expected_version_field": "schema_version",
        "expected_version_value": "2.0.0",
    },
    "refine_session.v1.0.schema.json": {
        "expected_version_field": "schema_version",
        "expected_version_value": "synapse.refine_session.v1.0",
    },
}

# Map schema file to (Python module path, class name, expected required fields)
_SCHEMA_PYTHON_ALIGNMENT: dict[str, tuple[str, str, list[str]]] = {
    "job_bundle.v1.0.schema.json": (
        "synapse.pipeline.bundle_worker",
        "BundleManifest",
        ["schema_version", "job_id", "stage"],
    ),
    "lora.v1.0.schema.json": (
        "synapse.adapters.lora.manifest",
        "LoRAManifest",
        ["schema_version", "name", "category", "rank", "alpha",
         "adapter_path", "adapter_sha256", "target_modules"],
    ),
    "inspection.v1.0.schema.json": (
        "synapse.inspect.schemas",
        "InspectionReport",
        ["asset_id", "asset_name", "inspection_timestamp",
         "overall_quality_score"],
    ),
    "refine_session.v1.0.schema.json": (
        "synapse.refine.pipeline",
        "RefineSessionPayload",
        ["schema_version", "session_id"],
    ),
}


def _check_json_schemas() -> list[str]:
    """Validate JSON schema files exist, are valid JSON, and align with Python models."""
    errors: list[str] = []

    if not _SCHEMA_DIR.is_dir():
        errors.append(f"Schema directory not found: {_SCHEMA_DIR}")
        return errors

    for filename, meta in _SCHEMA_FILES.items():
        schema_path = _SCHEMA_DIR / filename
        if not schema_path.is_file():
            errors.append(f"Schema file missing: {schema_path}")
            continue

        # Parse JSON
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"Schema file '{filename}' is not valid JSON: {e}")
            continue

        # Check it looks like a JSON Schema
        if "$schema" not in schema:
            errors.append(f"Schema '{filename}' missing '$schema' meta-field")

        if schema.get("type") != "object":
            errors.append(f"Schema '{filename}' top-level type is not 'object'")

        # Validate version constant
        props = schema.get("properties", {})
        ver_field = meta["expected_version_field"]
        if ver_field in props:
            ver_prop = props[ver_field]
            const_val = ver_prop.get("const")
            if const_val != meta["expected_version_value"]:
                errors.append(
                    f"Schema '{filename}': {ver_field}.const = "
                    f"'{const_val}', expected '{meta['expected_version_value']}'"
                )

        # Validate required fields align with Python model
        if filename in _SCHEMA_PYTHON_ALIGNMENT:
            _, _, expected_required = _SCHEMA_PYTHON_ALIGNMENT[filename]
            schema_required = set(schema.get("required", []))
            for field in expected_required:
                if field not in schema_required:
                    errors.append(
                        f"Schema '{filename}': required field '{field}' "
                        f"missing from JSON Schema 'required' array"
                    )

        # Validate schema properties cover Python model fields
        if filename in _SCHEMA_PYTHON_ALIGNMENT:
            mod_path, cls_name, _ = _SCHEMA_PYTHON_ALIGNMENT[filename]
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)

                # Get fields from Pydantic model or dataclass
                if hasattr(cls, "model_fields"):
                    # Pydantic v2
                    py_fields = set(cls.model_fields.keys())
                elif hasattr(cls, "__dataclass_fields__"):
                    py_fields = set(cls.__dataclass_fields__.keys())
                else:
                    py_fields = set()

                schema_props = set(props.keys())
                missing_in_schema = py_fields - schema_props
                if missing_in_schema:
                    errors.append(
                        f"Schema '{filename}': Python {cls_name} fields "
                        f"missing from schema properties: {sorted(missing_in_schema)}"
                    )
            except (ImportError, AttributeError) as e:
                errors.append(
                    f"Schema '{filename}': cannot import {mod_path}.{cls_name}: {e}"
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

    # --- Section 1: Internal consistency ---
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

    # --- Section 2: PixelType per-channel validation ---
    print()
    print("=== PixelType per-channel validation ===")
    pt_errors = _check_pixel_types()
    all_errors.extend(pt_errors)
    if pt_errors:
        for e in pt_errors:
            print(f"  FAIL: {e}")
    else:
        print("  OK: All 15 channels have correct PixelType")

    # --- Section 3: StyleProfile v2.0.0 alignment ---
    print()
    print("=== StyleProfile v2.0.0 alignment check ===")
    style_errors = _check_style_profile()
    all_errors.extend(style_errors)
    if style_errors:
        for e in style_errors:
            print(f"  FAIL: {e}")
    else:
        print("  OK: StyleDescriptor matches v2.0.0 contract")

    # --- Section 4: JSON Schema alignment ---
    print()
    print("=== JSON Schema alignment check ===")
    schema_errors = _check_json_schemas()
    all_errors.extend(schema_errors)
    if schema_errors:
        for e in schema_errors:
            print(f"  FAIL: {e}")
    else:
        print(f"  OK: All {len(_SCHEMA_FILES)} schemas validated")

    # --- Section 5: Cross-language C++ header check ---
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
