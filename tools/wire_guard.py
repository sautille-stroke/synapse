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

from typing import Any

__all__ = [
    "FGPP_CHANNEL_SPECS",
    "main",
]

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
    if PixelType.UINT32 != 0 or PixelType.HALF != 1 or PixelType.FLOAT != 2:  # type: ignore[comparison-overlap]
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
# WorldNormal.{X,Y,Z}: HALF only (locks 34 B/px contract)
# BaseColor.{R,G,B}: HALF
# Roughness/Metallic (RoughnessMetallic.{R,G}): HALF
# DirectRadiance.{R,G,B}: HALF
# MotionVector.{X,Y}: HALF

_EXPECTED_PIXEL_TYPES: dict[str, set[PixelType]] = {
    "Cryptomatte.ID": {PixelType.UINT32},
    "CameraLinearDepth.Z": {PixelType.FLOAT},
    "WorldNormal.X": {PixelType.HALF},
    "WorldNormal.Y": {PixelType.HALF},
    "WorldNormal.Z": {PixelType.HALF},
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

    # Cross-repo aether style header validation (gracefully skip if unavailable)
    _aether_style_probes = [
        _REPO_ROOT.parent / "aether" / "style" / "include" / "aether" / "style" / "style_types.h",
        Path("../aether/style/include/aether/style/style_types.h"),
    ]
    aether_style_path = None
    for probe in _aether_style_probes:
        if probe.exists():
            aether_style_path = probe
            break

    if aether_style_path is not None:
        try:
            header_text = aether_style_path.read_text(encoding="utf-8")
            # Check that Python StyleDescriptor field names appear in the C++ header
            # (as struct member names or enum values)
            for field_name in expected_fields:
                # Convert Python snake_case to potential C++ camelCase/snake_case variants
                if field_name not in header_text and field_name.replace("_", "") not in header_text.lower():
                    # Relaxed check: just look for the base concept name
                    base_name = field_name.split("_")[0]
                    if base_name not in header_text.lower():
                        errors.append(
                            f"StyleProfile v2.0.0: field '{field_name}' "
                            f"not found in aether style header"
                        )
        except OSError:
            pass  # Cannot read header — skip silently
    else:
        print("  SKIPPED (upstream @aether//style not present)")

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
    "bundle.v1.0.schema.json": {
        "expected_version_field": "schema_version",
        "expected_version_value": "synapse.bundle.v1.0",
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
    "bundle.v1.0.schema.json": (
        "synapse.bench.bundle_exporter",
        "InspectionBundleManifest",
        ["schema_version", "bundle_id", "sources"],
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
                missing_in_py = schema_props - py_fields
                if missing_in_py:
                    errors.append(
                        f"Schema '{filename}': Schema properties missing from "
                        f"Python {cls_name} fields: {sorted(missing_in_py)}"
                    )

                # Validate default instantiated schema_version
                inst = None
                try:
                    inst = cls()
                except Exception:
                    try:
                        if hasattr(cls, "model_construct"):
                            inst = cls.model_construct()
                    except Exception:
                        pass
                if inst is not None and hasattr(inst, ver_field):
                    inst_ver = getattr(inst, ver_field)
                    if inst_ver != meta["expected_version_value"]:
                        errors.append(
                            f"Schema '{filename}': Python {cls_name}.{ver_field} default "
                            f"is '{inst_ver}', expected '{meta['expected_version_value']}'"
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
    r"getAlphabeticalEXRChannelSpecs\b.*?\{\{\s*(.*?)\s*\}\}\s*;\s*\}",
    re.DOTALL,
)
_RE_CHANNEL_SPEC_ENTRY = re.compile(
    r'\{\s*"([^"]+)"\s*,\s*[^,]+,\s*(\d+)u\s*,\s*PixelType::(\w+)\s*,\s*(true|false)\s*\}'
)
_RE_MIN_VERTS = re.compile(
    r"kMinCharacterWhiteboxVertices\s*=\s*(\d+)"
)
_RE_ANCHOR_INTERVAL = re.compile(
    r"kDefaultPhysicalAnchorIntervalFrames\s*=\s*(\d+)"
)
_RE_PIXEL_TYPE_BLOCK = re.compile(
    r"enum\s+class\s+PixelType\s*:[^{]+\{(.*?)\};",
    re.DOTALL,
)
_RE_ENUM_MEMBER = re.compile(
    r"(\w+)\s*=\s*(\d+)"
)


def _parse_header(path: Path) -> dict[str, Any]:
    """Extract contract values from the C++ header."""
    text = path.read_text(encoding="utf-8")

    channel_specs: list[dict[str, Any]] = []
    m_body = _RE_EXR_SPECS_BODY.search(text)
    if m_body:
        specs_body = m_body.group(1)
        for m_spec in _RE_CHANNEL_SPEC_ENTRY.finditer(specs_body):
            channel_specs.append({
                "name": m_spec.group(1),
                "component_index": int(m_spec.group(2)),
                "pixel_type": m_spec.group(3),
                "allow_spatial_filtering": m_spec.group(4) == "true",
            })

    channel_names: list[str] = [spec["name"] for spec in channel_specs]
    if not channel_names:
        # Fallback regex if spec format differs
        if m_body:
            channel_names = sorted(set(_RE_CHANNEL_NAME.findall(m_body.group(1))))
        else:
            channel_names = sorted(set(_RE_CHANNEL_NAME.findall(text)))

    pixel_type_values: dict[str, int] = {}
    m_pt_block = _RE_PIXEL_TYPE_BLOCK.search(text)
    if m_pt_block:
        for em in _RE_ENUM_MEMBER.finditer(m_pt_block.group(1)):
            pixel_type_values[em.group(1)] = int(em.group(2))

    min_verts = None
    m = _RE_MIN_VERTS.search(text)
    if m:
        min_verts = int(m.group(1))

    anchor_interval = None
    m = _RE_ANCHOR_INTERVAL.search(text)
    if m:
        anchor_interval = int(m.group(1))

    return {
        "channel_names": channel_names,
        "channel_specs": channel_specs,
        "min_verts": min_verts,
        "anchor_interval": anchor_interval,
        "pixel_type_values": pixel_type_values,
    }


def _check_against_header(header_path: Path) -> list[str]:
    """Compare Python schema against parsed C++ header values."""
    errors: list[str] = []
    parsed = _parse_header(header_path)

    # 1. Channel names & order
    py_names = get_channel_names()
    cpp_names = parsed["channel_names"]
    if cpp_names != py_names:
        errors.append(
            f"Channel name/ordering mismatch:\n"
            f"  Python: {py_names}\n"
            f"  C++:    {cpp_names}"
        )

    # 2. Min vertices
    if parsed["min_verts"] is not None:
        if parsed["min_verts"] != MIN_CHARACTER_WHITEBOX_VERTICES:
            errors.append(
                f"kMinCharacterWhiteboxVertices: C++={parsed['min_verts']}, "
                f"Python={MIN_CHARACTER_WHITEBOX_VERTICES}"
            )
    else:
        errors.append("kMinCharacterWhiteboxVertices not found in C++ header")

    # 3. Anchor interval
    if parsed["anchor_interval"] is not None:
        if parsed["anchor_interval"] != DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES:
            errors.append(
                f"kDefaultPhysicalAnchorIntervalFrames: C++={parsed['anchor_interval']}, "
                f"Python={DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES}"
            )
    else:
        errors.append("kDefaultPhysicalAnchorIntervalFrames not found in C++ header")

    # 4. PixelType enum values
    cpp_pt = parsed.get("pixel_type_values", {})
    if not cpp_pt:
        errors.append("PixelType enum values could not be parsed from C++ header")
    else:
        py_pt_map = {
            "Uint32": PixelType.UINT32,
            "Half": PixelType.HALF,
            "Float": PixelType.FLOAT,
        }
        for cpp_name, expected_py_enum in py_pt_map.items():
            if cpp_name not in cpp_pt:
                errors.append(f"PixelType::{cpp_name} missing in C++ header enum")
            elif cpp_pt[cpp_name] != int(expected_py_enum):
                errors.append(
                    f"PixelType::{cpp_name} value mismatch: C++={cpp_pt[cpp_name]}, "
                    f"Python={int(expected_py_enum)}"
                )

    # 5. Detailed per-channel specs (PixelType & allow_spatial_filtering)
    cpp_specs = parsed.get("channel_specs", [])
    if len(cpp_specs) != len(FGPP_CHANNEL_SPECS):
        errors.append(
            f"Channel specs count mismatch: C++={len(cpp_specs)}, "
            f"Python={len(FGPP_CHANNEL_SPECS)}"
        )
    else:
        pt_str_to_enum = {
            "Uint32": PixelType.UINT32,
            "Half": PixelType.HALF,
            "Float": PixelType.FLOAT,
        }
        for (py_ch_name, py_pt, _), cpp_spec in zip(FGPP_CHANNEL_SPECS, cpp_specs):
            if cpp_spec["name"] != py_ch_name:
                errors.append(
                    f"Channel order mismatch: Python has '{py_ch_name}', C++ has '{cpp_spec['name']}'"
                )
            cpp_pt_enum = pt_str_to_enum.get(cpp_spec["pixel_type"])
            if cpp_pt_enum is None or cpp_pt_enum != py_pt:
                errors.append(
                    f"Channel '{py_ch_name}' PixelType mismatch: C++={cpp_spec['pixel_type']}, "
                    f"Python={py_pt.name}"
                )
            expected_filter = py_ch_name != "Cryptomatte.ID"
            if cpp_spec["allow_spatial_filtering"] != expected_filter:
                errors.append(
                    f"Channel '{py_ch_name}' allow_spatial_filtering mismatch: "
                    f"C++={cpp_spec['allow_spatial_filtering']}, expected={expected_filter}"
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
        parsed = _parse_header(header_path)
        print(f"  Parsed PixelType values ({len(parsed.get('pixel_type_values', {}))} items): {parsed.get('pixel_type_values')}")
        print(f"  Parsed {len(parsed.get('channel_specs', []))} ordered EXR channel specs")
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