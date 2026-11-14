"""Tests for wire_guard.py extended contract checks and JSON schemas.

Validates:
  - wire_guard.py --check exits 0
  - PixelType validation catches wrong types
  - Schema validation catches field mismatches
  - Each schema file is valid JSON Schema
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_SRC = _REPO_ROOT / "python"
_TOOLS_DIR = _REPO_ROOT / "tools"
_SCHEMAS_DIR = _REPO_ROOT / "schemas"

if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


# ---------------------------------------------------------------------------
# Import wire_guard internals
# ---------------------------------------------------------------------------

# Import the module directly
sys.path.insert(0, str(_TOOLS_DIR))
import wire_guard  # noqa: E402

# ---------------------------------------------------------------------------
# TASK 5.1: wire_guard.py --check exits 0
# ---------------------------------------------------------------------------


class TestWireGuardCheck:
    """Test that the full wire_guard --check passes."""

    def test_check_exits_zero(self) -> None:
        """wire_guard main(['--check']) returns 0."""
        rc = wire_guard.main(["--check"])
        assert rc == 0, f"wire_guard --check failed with exit code {rc}"

    def test_internal_checks_pass(self) -> None:
        """Internal consistency checks return no errors."""
        errors = wire_guard._check_internal()
        assert errors == [], f"Internal check errors: {errors}"


# ---------------------------------------------------------------------------
# TASK 5.2: PixelType validation catches wrong types
# ---------------------------------------------------------------------------


class TestPixelTypeValidation:
    """Test PixelType per-channel validation."""

    def test_pixel_types_pass_on_correct_spec(self) -> None:
        """Current FGPP_CHANNEL_SPECS has correct PixelTypes."""
        errors = wire_guard._check_pixel_types()
        assert errors == [], f"PixelType errors: {errors}"

    def test_pixel_type_catches_wrong_cryptomatte(self) -> None:
        """Cryptomatte.ID must be UINT32; HALF should fail."""
        from synapse.fgpp.schema import PixelType

        # Patch FGPP_CHANNEL_SPECS to have wrong Cryptomatte type
        bad_specs = [
            (name, PixelType.HALF if name == "Cryptomatte.ID" else pt, group)
            for name, pt, group in wire_guard.FGPP_CHANNEL_SPECS
        ]
        with mock.patch.object(wire_guard, "FGPP_CHANNEL_SPECS", bad_specs):
            errors = wire_guard._check_pixel_types()
        assert any("Cryptomatte.ID" in e for e in errors), (
            f"Expected Cryptomatte.ID PixelType error, got: {errors}"
        )

    def test_pixel_type_catches_wrong_depth(self) -> None:
        """CameraLinearDepth.Z must be FLOAT; HALF should fail."""
        from synapse.fgpp.schema import PixelType

        bad_specs = [
            (name, PixelType.HALF if name == "CameraLinearDepth.Z" else pt, group)
            for name, pt, group in wire_guard.FGPP_CHANNEL_SPECS
        ]
        with mock.patch.object(wire_guard, "FGPP_CHANNEL_SPECS", bad_specs):
            errors = wire_guard._check_pixel_types()
        assert any("CameraLinearDepth.Z" in e for e in errors), (
            f"Expected CameraLinearDepth.Z PixelType error, got: {errors}"
        )

    def test_pixel_type_world_normal_accepts_half_and_float(self) -> None:
        """WorldNormal channels accept both HALF and FLOAT."""
        from synapse.fgpp.schema import PixelType

        # Test with FLOAT — should also be fine
        float_specs = [
            (name, PixelType.FLOAT if name.startswith("WorldNormal.") else pt, group)
            for name, pt, group in wire_guard.FGPP_CHANNEL_SPECS
        ]
        with mock.patch.object(wire_guard, "FGPP_CHANNEL_SPECS", float_specs):
            errors = wire_guard._check_pixel_types()
        normal_errors = [e for e in errors if "WorldNormal" in e]
        assert normal_errors == [], (
            f"WorldNormal.FLOAT should be accepted, got: {normal_errors}"
        )


# ---------------------------------------------------------------------------
# TASK 5.3: StyleProfile alignment
# ---------------------------------------------------------------------------


class TestStyleProfileAlignment:
    """Test StyleProfile v2.0.0 field alignment check."""

    def test_style_profile_passes(self) -> None:
        """StyleDescriptor fields match v2.0.0 contract."""
        errors = wire_guard._check_style_profile()
        assert errors == [], f"StyleProfile errors: {errors}"


# ---------------------------------------------------------------------------
# TASK 5.4: JSON Schema validation
# ---------------------------------------------------------------------------

_SCHEMA_FILENAMES = [
    "job_bundle.v1.0.schema.json",
    "lora.v1.0.schema.json",
    "inspection.v1.0.schema.json",
    "refine_session.v1.0.schema.json",
]


class TestJsonSchemas:
    """Test JSON schema files are valid and aligned."""

    @pytest.mark.parametrize("filename", _SCHEMA_FILENAMES)
    def test_schema_file_exists(self, filename: str) -> None:
        """Each schema file exists in schemas/."""
        path = _SCHEMAS_DIR / filename
        assert path.is_file(), f"Schema file missing: {path}"

    @pytest.mark.parametrize("filename", _SCHEMA_FILENAMES)
    def test_schema_is_valid_json(self, filename: str) -> None:
        """Each schema file is valid JSON."""
        path = _SCHEMAS_DIR / filename
        text = path.read_text(encoding="utf-8")
        schema = json.loads(text)  # Will raise on invalid JSON
        assert isinstance(schema, dict)

    @pytest.mark.parametrize("filename", _SCHEMA_FILENAMES)
    def test_schema_has_meta_fields(self, filename: str) -> None:
        """Each schema has $schema and title fields."""
        path = _SCHEMAS_DIR / filename
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert "$schema" in schema, f"{filename} missing $schema"
        assert "title" in schema, f"{filename} missing title"
        assert schema["type"] == "object", f"{filename} type is not 'object'"

    @pytest.mark.parametrize("filename", _SCHEMA_FILENAMES)
    def test_schema_has_required_array(self, filename: str) -> None:
        """Each schema has a non-empty required array."""
        path = _SCHEMAS_DIR / filename
        schema = json.loads(path.read_text(encoding="utf-8"))
        required = schema.get("required", [])
        assert len(required) > 0, f"{filename} has empty 'required'"

    def test_job_bundle_schema_version_const(self) -> None:
        """JobBundle schema locks version to 'synapse.job_bundle.v1.0'."""
        path = _SCHEMAS_DIR / "job_bundle.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["properties"]["schema_version"]["const"] == "synapse.job_bundle.v1.0"

    def test_lora_schema_version_const(self) -> None:
        """LoRA schema locks version to 'synapse.lora.v1.0'."""
        path = _SCHEMAS_DIR / "lora.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["properties"]["schema_version"]["const"] == "synapse.lora.v1.0"

    def test_lora_schema_adapter_path_pattern(self) -> None:
        """LoRA schema requires .safetensors extension."""
        path = _SCHEMAS_DIR / "lora.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert "pattern" in schema["properties"]["adapter_path"]
        assert "safetensors" in schema["properties"]["adapter_path"]["pattern"]

    def test_lora_schema_sha256_pattern(self) -> None:
        """LoRA schema validates SHA256 hex pattern."""
        path = _SCHEMAS_DIR / "lora.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert "pattern" in schema["properties"]["adapter_sha256"]

    def test_inspection_schema_defect_categories(self) -> None:
        """Inspection schema has correct defect categories."""
        path = _SCHEMAS_DIR / "inspection.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        defect_schema = schema["$defs"]["DefectAnnotation3D"]
        categories = defect_schema["properties"]["category"]["enum"]
        expected = [
            "mesh_crumple",
            "texture_stretch",
            "seam_tearing",
            "floating_debris",
            "anatomical_distortion",
            "plastic_shading",
        ]
        assert categories == expected

    def test_refine_session_regions_enum(self) -> None:
        """Refine session schema lists valid anatomical regions."""
        path = _SCHEMAS_DIR / "refine_session.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        region_enum = schema["properties"]["regions"]["items"]["enum"]
        expected = ["HAND", "FACE", "FOOT", "HAIR", "FULL_BODY"]
        assert region_enum == expected

    def test_refine_session_fusion_methods(self) -> None:
        """Refine session schema lists all 5 fusion methods."""
        path = _SCHEMAS_DIR / "refine_session.v1.0.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        methods = schema["properties"]["fusion_method"]["enum"]
        assert len(methods) == 5
        assert "LAPLACIAN_FREQUENCY" in methods
        assert "POISSON_GRADIENT" in methods

    def test_wire_guard_schema_check_passes(self) -> None:
        """wire_guard._check_json_schemas() returns no errors."""
        errors = wire_guard._check_json_schemas()
        assert errors == [], f"Schema check errors: {errors}"


# ---------------------------------------------------------------------------
# TASK 5.5: Schema catches mismatches (negative tests)
# ---------------------------------------------------------------------------


class TestSchemaCatchesMismatches:
    """Test that wire_guard catches schema misalignments."""

    def test_missing_schema_file_caught(self, tmp_path: Path) -> None:
        """Missing schema file is reported."""
        with mock.patch.object(wire_guard, "_SCHEMA_DIR", tmp_path):
            errors = wire_guard._check_json_schemas()
        assert len(errors) >= 4  # All 4 files missing
        assert any("missing" in e.lower() for e in errors)

    def test_invalid_json_caught(self, tmp_path: Path) -> None:
        """Invalid JSON in schema file is caught."""
        (tmp_path / "job_bundle.v1.0.schema.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        # Create the other 3 as empty to avoid missing-file noise
        for name in _SCHEMA_FILENAMES[1:]:
            (tmp_path / name).write_text("{}", encoding="utf-8")

        with mock.patch.object(wire_guard, "_SCHEMA_DIR", tmp_path):
            errors = wire_guard._check_json_schemas()
        assert any("not valid JSON" in e for e in errors)

    def test_wrong_version_const_caught(self, tmp_path: Path) -> None:
        """Wrong version const in schema is caught."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["schema_version", "job_id", "stage"],
            "properties": {
                "schema_version": {"type": "string", "const": "WRONG_VERSION"},
                "job_id": {"type": "string"},
                "stage": {"type": "string"},
                "inputs": {"type": "object"},
                "outputs": {"type": "object"},
                "params": {"type": "object"},
                "hardware_budget": {"type": "object"},
                "traceparent": {"type": ["string", "null"]},
                "tracestate": {"type": ["string", "null"]},
            },
        }
        (tmp_path / "job_bundle.v1.0.schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )
        # Stub other files to avoid noise
        for name in _SCHEMA_FILENAMES[1:]:
            (tmp_path / name).write_text("{}", encoding="utf-8")

        with mock.patch.object(wire_guard, "_SCHEMA_DIR", tmp_path):
            errors = wire_guard._check_json_schemas()
        assert any("WRONG_VERSION" in e for e in errors)
