"""Tests for InspectionBundle v1.0 schema validator and exporter.

Covers manifest parsing, path-traversal security, bundle validation,
export round-trip, and content listing.  All tests are pure-CPU with
no network or GPU dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synapse.bench.bundle_exporter import (
    INSPECTION_SCHEMA_VERSION,
    InspectionBundleManifest,
    SourceEntry,
    export_inspection_bundle,
    list_bundle_contents,
    validate_inspection_bundle,
)
from synapse.errors import ContractValidationError


# ===================================================================
# Fixtures
# ===================================================================


def _make_minimal_bundle(
    tmp_path: Path,
    *,
    schema_version: str = INSPECTION_SCHEMA_VERSION,
    sources: dict[str, Any] | None = None,
    create_files: bool = True,
) -> Path:
    """Create a minimal bundle directory for testing.

    Returns the bundle root path.
    """
    bundle_dir = tmp_path / "test_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if sources is None:
        sources = {
            "source_model": {
                "path": "source/model.glb",
                "media_type": "model/gltf-binary",
            },
        }

    if create_files:
        for entry in sources.values():
            fp = bundle_dir / entry["path"]
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"\x00" * 64)

    manifest = {
        "schema_version": schema_version,
        "bundle_id": "deadbeef01234567",
        "scenario_name": "test_scenario",
        "created_at": "2026-01-01T00:00:00+00:00",
        "sources": sources,
    }
    manifest_path = bundle_dir / "bundle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return bundle_dir


def _make_full_bundle(tmp_path: Path) -> Path:
    """Create a full round-trip bundle directory."""
    sources: dict[str, Any] = {
        "source_model": {
            "path": "source/model.glb",
            "media_type": "model/gltf-binary",
        },
        "fgpp_0": {
            "path": "fgpp/frame_0000.exr",
            "media_type": "image/x-exr",
        },
        "generated_0": {
            "path": "generated/video_frames/frame_0000.png",
            "media_type": "image/png",
        },
        "reconstructed_model": {
            "path": "reconstructed/model.glb",
            "media_type": "model/gltf-binary",
        },
        "metrics_comparison": {
            "path": "metrics/comparison.json",
            "media_type": "application/json",
        },
        "animation_0": {
            "path": "animations/walk.glb",
            "media_type": "model/gltf-binary",
        },
    }
    return _make_minimal_bundle(
        tmp_path, sources=sources, create_files=True,
    )


# ===================================================================
# TestInspectionBundleManifest
# ===================================================================


class TestInspectionBundleManifest:
    """Tests for the Pydantic manifest model."""

    def test_valid_manifest_parsing(self) -> None:
        """A valid manifest round-trips through the model."""
        data = {
            "schema_version": INSPECTION_SCHEMA_VERSION,
            "bundle_id": "abc123",
            "scenario_name": "demo",
            "created_at": "2026-01-01T00:00:00Z",
            "sources": {
                "model": {
                    "path": "source/model.glb",
                    "media_type": "model/gltf-binary",
                },
            },
        }
        m = InspectionBundleManifest.model_validate(data)
        assert m.schema_version == INSPECTION_SCHEMA_VERSION
        assert m.bundle_id == "abc123"
        assert m.scenario_name == "demo"
        assert "model" in m.sources
        assert m.sources["model"].path == "source/model.glb"

    def test_schema_version_validation(self) -> None:
        """Reject manifests with wrong schema_version."""
        with pytest.raises(Exception):
            InspectionBundleManifest.model_validate({
                "schema_version": "wrong.v9.9",
                "bundle_id": "x",
                "scenario_name": "x",
                "created_at": "2026-01-01T00:00:00Z",
                "sources": {},
            })

    def test_path_traversal_rejection(self) -> None:
        """Paths containing ``..`` must be rejected."""
        with pytest.raises(Exception):
            SourceEntry(
                path="../../../etc/passwd",
                media_type="text/plain",
            )

    def test_absolute_path_rejection(self) -> None:
        """Absolute POSIX paths must be rejected."""
        with pytest.raises(Exception):
            SourceEntry(
                path="/usr/bin/evil",
                media_type="application/octet-stream",
            )

    def test_windows_drive_path_rejection(self) -> None:
        """Windows drive letter paths must be rejected."""
        with pytest.raises(Exception):
            SourceEntry(
                path="C:\\evil",
                media_type="application/octet-stream",
            )

    def test_source_entry_media_type(self) -> None:
        """A valid SourceEntry stores its media_type."""
        entry = SourceEntry(
            path="source/model.glb",
            media_type="model/gltf-binary",
            sha256="aabbcc",
        )
        assert entry.media_type == "model/gltf-binary"
        assert entry.sha256 == "aabbcc"


# ===================================================================
# TestValidateBundle
# ===================================================================


class TestValidateBundle:
    """Tests for :func:`validate_inspection_bundle`."""

    def test_valid_minimal_bundle(self, tmp_path: Path) -> None:
        """A minimal bundle with matching files validates OK."""
        bd = _make_minimal_bundle(tmp_path)
        manifest = validate_inspection_bundle(bd)
        assert manifest.schema_version == INSPECTION_SCHEMA_VERSION
        assert manifest.scenario_name == "test_scenario"
        assert "source_model" in manifest.sources

    def test_valid_full_bundle(self, tmp_path: Path) -> None:
        """A full bundle with all sections validates OK."""
        bd = _make_full_bundle(tmp_path)
        manifest = validate_inspection_bundle(bd)
        assert len(manifest.sources) == 6

    def test_missing_referenced_file(self, tmp_path: Path) -> None:
        """Validation fails when a referenced file is absent."""
        bd = _make_minimal_bundle(
            tmp_path, create_files=False,
        )
        # bundle.json exists but source/model.glb does not
        with pytest.raises(ContractValidationError, match="missing"):
            validate_inspection_bundle(bd)

    def test_missing_bundle_json(self, tmp_path: Path) -> None:
        """Validation fails when bundle.json is absent."""
        bd = tmp_path / "empty_bundle"
        bd.mkdir(parents=True, exist_ok=True)
        with pytest.raises(
            ContractValidationError, match="bundle.json",
        ):
            validate_inspection_bundle(bd)


# ===================================================================
# TestExportBundle
# ===================================================================


class TestExportBundle:
    """Tests for :func:`export_inspection_bundle`."""

    def _create_source_model(self, tmp_path: Path) -> Path:
        """Write a dummy source model file."""
        model_path = tmp_path / "input_model.glb"
        model_path.write_bytes(b"\x00glTF" + b"\x00" * 60)
        return model_path

    def test_export_creates_bundle_json(
        self, tmp_path: Path,
    ) -> None:
        """Exported bundle contains bundle.json."""
        model = self._create_source_model(tmp_path)
        out = tmp_path / "out_bundle"
        export_inspection_bundle(
            scenario_name="s1",
            source_model_path=model,
            output_dir=out,
        )
        assert (out / "bundle.json").is_file()

    def test_export_creates_directory_structure(
        self, tmp_path: Path,
    ) -> None:
        """Exported bundle has the expected sub-directories."""
        model = self._create_source_model(tmp_path)
        out = tmp_path / "out_bundle"
        export_inspection_bundle(
            scenario_name="s2",
            source_model_path=model,
            output_dir=out,
        )
        assert (out / "source").is_dir()
        assert (out / "metrics").is_dir()
        assert (out / "metrics" / "comparison.json").is_file()

    def test_metrics_has_anchor_frames(
        self, tmp_path: Path,
    ) -> None:
        """comparison.json contains anchor_frames field."""
        model = self._create_source_model(tmp_path)
        out = tmp_path / "out_bundle"
        export_inspection_bundle(
            scenario_name="s3",
            source_model_path=model,
            output_dir=out,
            metrics_dict={"psnr": 32.5},
        )
        comp = json.loads(
            (out / "metrics" / "comparison.json").read_text(
                encoding="utf-8",
            ),
        )
        assert "anchor_frames" in comp
        assert comp["anchor_frames"] == [0, 48, 96]
        assert "defect_regions" in comp

    def test_roundtrip_export_then_validate(
        self, tmp_path: Path,
    ) -> None:
        """An exported bundle passes validation."""
        model = self._create_source_model(tmp_path)
        out = tmp_path / "out_bundle"
        manifest = export_inspection_bundle(
            scenario_name="roundtrip",
            source_model_path=model,
            output_dir=out,
        )
        # Validate the just-exported bundle
        validated = validate_inspection_bundle(out)
        assert validated.scenario_name == "roundtrip"
        assert validated.schema_version == INSPECTION_SCHEMA_VERSION
        assert len(validated.sources) == len(manifest.sources)

    def test_export_with_reconstructed_model(
        self, tmp_path: Path,
    ) -> None:
        """Export includes reconstructed model when provided."""
        model = self._create_source_model(tmp_path)
        recon = tmp_path / "recon.glb"
        recon.write_bytes(b"\x00glTF" + b"\x00" * 60)
        out = tmp_path / "out_bundle"
        manifest = export_inspection_bundle(
            scenario_name="with_recon",
            source_model_path=model,
            output_dir=out,
            reconstructed_model_path=recon,
        )
        assert "reconstructed_model" in manifest.sources
        assert (out / "reconstructed" / "recon.glb").is_file()


# ===================================================================
# TestListContents
# ===================================================================


class TestListContents:
    """Tests for :func:`list_bundle_contents`."""

    def test_lists_all_files_with_sizes(
        self, tmp_path: Path,
    ) -> None:
        """Listing returns every file with its byte size."""
        bd = _make_minimal_bundle(tmp_path)
        result = list_bundle_contents(bd)
        assert result["file_count"] >= 2  # bundle.json + model
        for f in result["files"]:
            assert "path" in f
            assert "size_bytes" in f
            assert f["size_bytes"] >= 0

    def test_empty_bundle_returns_minimal_info(
        self, tmp_path: Path,
    ) -> None:
        """An empty directory still returns a valid structure."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = list_bundle_contents(empty)
        assert result["file_count"] == 0
        assert result["total_size_bytes"] == 0
        assert result["files"] == []
