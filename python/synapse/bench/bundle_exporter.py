"""InspectionBundle v1.0 schema validator and exporter.

Provides the ``InspectionBundleManifest`` Pydantic model, path-safe
validation, and utilities to export / validate / list inspection bundles
used by the Synapse-Bench evaluation pipeline.

An inspection bundle is a self-contained directory that packages:

- The source 3D model (``source/model.glb``)
- Optional FGPP EXR ground-truth frames (``fgpp/``)
- Optional generated video frames (``generated/video_frames/``)
- Optional reconstructed model (``reconstructed/model.glb``)
- Optional animations (``animations/``)
- Metrics comparison results (``metrics/comparison.json``)
- A manifest (``bundle.json``) with schema ``inspection.v1.0``
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, field_validator

from synapse.errors import ContractValidationError


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

INSPECTION_SCHEMA_VERSION: str = "inspection.v1.0"
"""Fixed schema version string embedded in every bundle manifest."""

_BUNDLE_MANIFEST_FILENAME: str = "bundle.json"
"""Well-known manifest file name inside every bundle directory."""

# Windows drive-letter pattern, e.g. "C:\" or "D:/"
_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:[/\\]")


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------


class SourceEntry(BaseModel):
    """A single source artefact referenced by the bundle manifest.

    Attributes:
        path: Relative POSIX-style path within the bundle directory.
              Must NOT contain ``..`` segments, start with ``/``, or
              use a Windows drive letter prefix.
        media_type: IANA media (MIME) type, e.g. ``model/gltf-binary``.
        sha256: Optional hex-encoded SHA-256 digest for integrity
                verification.
    """

    path: str
    media_type: str
    sha256: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path_security(cls, v: str) -> str:
        """Reject path-traversal and absolute paths."""
        _assert_safe_relative_path(v)
        return v


class InspectionBundleManifest(BaseModel):
    """Top-level manifest for an InspectionBundle (``bundle.json``).

    Attributes:
        schema_version: Must equal ``inspection.v1.0``.
        bundle_id: Unique identifier (UUID4 hex by default).
        scenario_name: Human-readable scenario label.
        created_at: ISO-8601 UTC timestamp.
        sources: Map of logical names to :class:`SourceEntry` records.
    """

    schema_version: str = INSPECTION_SCHEMA_VERSION
    bundle_id: str = ""
    scenario_name: str = ""
    created_at: str = ""
    sources: dict[str, SourceEntry] = {}

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if v != INSPECTION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{v}'; "
                f"expected '{INSPECTION_SCHEMA_VERSION}'"
            )
        return v


# -------------------------------------------------------------------
# Path security helper
# -------------------------------------------------------------------


def _assert_safe_relative_path(p: str) -> None:
    """Raise :class:`ContractValidationError` if *p* is unsafe.

    A path is rejected when it:
    - Contains ``..`` as a segment (directory traversal).
    - Starts with ``/`` (absolute POSIX path).
    - Starts with a Windows drive letter (e.g. ``C:\\``).
    - Is empty.
    """
    if not p:
        raise ContractValidationError(
            "Empty path in bundle source entry",
            agent_fix_hint="Provide a non-empty relative path.",
        )

    # Absolute POSIX
    if p.startswith("/"):
        raise ContractValidationError(
            f"Absolute POSIX path rejected: '{p}'",
            agent_fix_hint=(
                "Use a relative path within the bundle directory."
            ),
        )

    # Windows drive letter
    if _DRIVE_LETTER_RE.match(p):
        raise ContractValidationError(
            f"Windows absolute path rejected: '{p}'",
            agent_fix_hint=(
                "Use a relative path within the bundle directory."
            ),
        )

    # Traverse segments
    parts = PurePosixPath(p).parts
    if ".." in parts:
        raise ContractValidationError(
            f"Path traversal ('..') rejected: '{p}'",
            agent_fix_hint=(
                "Remove '..' segments from the bundle source path."
            ),
        )


# -------------------------------------------------------------------
# Validate
# -------------------------------------------------------------------


def validate_inspection_bundle(
    bundle_dir: Path,
) -> InspectionBundleManifest:
    """Parse and validate an existing inspection bundle on disk.

    Args:
        bundle_dir: Root directory of the bundle (must contain
            ``bundle.json``).

    Returns:
        Parsed :class:`InspectionBundleManifest`.

    Raises:
        ContractValidationError: On schema mismatch, missing files,
            or path-traversal attempts.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / _BUNDLE_MANIFEST_FILENAME

    if not manifest_path.is_file():
        raise ContractValidationError(
            f"bundle.json not found in '{bundle_dir}'",
            agent_fix_hint=(
                "Ensure the bundle directory contains a valid "
                "bundle.json manifest."
            ),
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractValidationError(
            f"Failed to parse bundle.json: {exc}",
            agent_fix_hint="Fix the JSON syntax in bundle.json.",
        ) from exc

    try:
        manifest = InspectionBundleManifest.model_validate(raw)
    except Exception as exc:
        raise ContractValidationError(
            f"Manifest validation failed: {exc}",
            agent_fix_hint=(
                "Ensure bundle.json conforms to the "
                "inspection.v1.0 schema."
            ),
        ) from exc

    # Verify every referenced path exists on disk
    for logical_name, entry in manifest.sources.items():
        ref_path = bundle_dir / entry.path
        if not ref_path.exists():
            raise ContractValidationError(
                f"Referenced file missing for '{logical_name}': "
                f"'{entry.path}'",
                agent_fix_hint=(
                    f"Add the missing file at "
                    f"'{ref_path}' or remove the entry from "
                    "bundle.json sources."
                ),
            )

    return manifest


# -------------------------------------------------------------------
# Export
# -------------------------------------------------------------------


def export_inspection_bundle(
    scenario_name: str,
    source_model_path: Path,
    output_dir: Path,
    *,
    fgpp_dir: Path | None = None,
    generated_frames: list[Path] | None = None,
    metrics_dict: dict[str, Any] | None = None,
    reconstructed_model_path: Path | None = None,
    animations: list[Path] | None = None,
) -> InspectionBundleManifest:
    """Create a new InspectionBundle on disk.

    The function copies/links source artefacts into a canonical
    directory layout and writes the ``bundle.json`` manifest.

    Args:
        scenario_name: Human-readable scenario label.
        source_model_path: Path to the primary 3D model file
            (typically ``.glb``).
        output_dir: Target directory for the bundle (created if
            absent).
        fgpp_dir: Optional directory of FGPP EXR frames to include.
        generated_frames: Optional list of generated video frame
            image paths.
        metrics_dict: Optional metrics dict to serialise into
            ``metrics/comparison.json``.
        reconstructed_model_path: Optional reconstructed model file.
        animations: Optional list of animation ``.glb`` files.

    Returns:
        The written :class:`InspectionBundleManifest`.
    """
    from synapse.fgpp.schema import (
        DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, SourceEntry] = {}

    # -- source model --
    source_out = output_dir / "source"
    source_out.mkdir(parents=True, exist_ok=True)
    dst_model = source_out / source_model_path.name
    _safe_copy(source_model_path, dst_model)
    sources["source_model"] = SourceEntry(
        path=f"source/{source_model_path.name}",
        media_type=_guess_media_type(source_model_path.name),
    )

    # -- FGPP EXR frames --
    if fgpp_dir is not None:
        fgpp_out = output_dir / "fgpp"
        fgpp_out.mkdir(parents=True, exist_ok=True)
        fgpp_dir = Path(fgpp_dir)
        idx = 0
        for f in sorted(fgpp_dir.iterdir()):
            if f.is_file():
                dst = fgpp_out / f.name
                _safe_copy(f, dst)
                sources[f"fgpp_{idx}"] = SourceEntry(
                    path=f"fgpp/{f.name}",
                    media_type=_guess_media_type(f.name),
                )
                idx += 1

    # -- generated video frames --
    if generated_frames:
        gen_out = output_dir / "generated" / "video_frames"
        gen_out.mkdir(parents=True, exist_ok=True)
        for idx, fp in enumerate(generated_frames):
            fp = Path(fp)
            dst = gen_out / fp.name
            _safe_copy(fp, dst)
            sources[f"generated_{idx}"] = SourceEntry(
                path=f"generated/video_frames/{fp.name}",
                media_type=_guess_media_type(fp.name),
            )

    # -- reconstructed model --
    if reconstructed_model_path is not None:
        recon_out = output_dir / "reconstructed"
        recon_out.mkdir(parents=True, exist_ok=True)
        reconstructed_model_path = Path(reconstructed_model_path)
        dst = recon_out / reconstructed_model_path.name
        _safe_copy(reconstructed_model_path, dst)
        sources["reconstructed_model"] = SourceEntry(
            path=f"reconstructed/{reconstructed_model_path.name}",
            media_type=_guess_media_type(
                reconstructed_model_path.name,
            ),
        )

    # -- animations --
    if animations:
        anim_out = output_dir / "animations"
        anim_out.mkdir(parents=True, exist_ok=True)
        for idx, anim_path in enumerate(animations):
            anim_path = Path(anim_path)
            dst = anim_out / anim_path.name
            _safe_copy(anim_path, dst)
            sources[f"animation_{idx}"] = SourceEntry(
                path=f"animations/{anim_path.name}",
                media_type=_guess_media_type(anim_path.name),
            )

    # -- metrics / comparison.json --
    metrics_out = output_dir / "metrics"
    metrics_out.mkdir(parents=True, exist_ok=True)
    interval = DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES
    comparison_payload: dict[str, Any] = {
        "scenario_name": scenario_name,
        "anchor_frames": [
            i * interval for i in range(3)
        ],  # [0, 48, 96]
        "defect_regions": [],
    }
    if metrics_dict:
        comparison_payload["metrics"] = metrics_dict
    comp_path = metrics_out / "comparison.json"
    comp_path.write_text(
        json.dumps(comparison_payload, indent=2, default=str),
        encoding="utf-8",
    )
    sources["metrics_comparison"] = SourceEntry(
        path="metrics/comparison.json",
        media_type="application/json",
    )

    # -- write manifest --
    manifest = InspectionBundleManifest(
        schema_version=INSPECTION_SCHEMA_VERSION,
        bundle_id=uuid4().hex,
        scenario_name=scenario_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        sources=sources,
    )

    manifest_path = output_dir / _BUNDLE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return manifest


# -------------------------------------------------------------------
# List contents
# -------------------------------------------------------------------


def list_bundle_contents(bundle_dir: Path) -> dict[str, Any]:
    """Return an inventory of all files in *bundle_dir*.

    Args:
        bundle_dir: Root directory of the bundle to inspect.

    Returns:
        A dict with ``"files"`` (list of per-file info dicts) and
        ``"total_size_bytes"`` (aggregate size).
    """
    bundle_dir = Path(bundle_dir)
    files: list[dict[str, Any]] = []
    total_size = 0

    if bundle_dir.is_dir():
        for fp in sorted(bundle_dir.rglob("*")):
            if fp.is_file():
                size = fp.stat().st_size
                rel = fp.relative_to(bundle_dir).as_posix()
                files.append({
                    "path": rel,
                    "size_bytes": size,
                    "media_type": _guess_media_type(fp.name),
                })
                total_size += size

    return {
        "bundle_dir": str(bundle_dir),
        "files": files,
        "total_size_bytes": total_size,
        "file_count": len(files),
    }


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* only if they are distinct files."""
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if src != dst:
        shutil.copy2(src, dst)


_MEDIA_TYPE_MAP: dict[str, str] = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".exr": "image/x-exr",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def _guess_media_type(filename: str) -> str:
    """Return MIME type based on file extension."""
    suffix = Path(filename).suffix.lower()
    return _MEDIA_TYPE_MAP.get(suffix, "application/octet-stream")
