"""Static HTML exporter for Synapse Dataset Viewer (Zero-Server runtime)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.viewer.static_html import VIEWER_HTML_TEMPLATE

__all__ = ["export_static_html"]


def _encode_data_uri(file_path: Path) -> str | None:
    """Encode an image file to a base64 Data URI string."""
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".png":
            mime = "image/png"
        elif suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".svg":
            mime = "image/svg+xml"
        else:
            return None
        raw_bytes = file_path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _check_no_symlinks(path: Path, root: Path) -> None:
    """Recursively check that *path* and all ancestors/children contain no symlinks
    and remain within *root*.

    Raises:
        ContractValidationError: If a symlink or path traversal is detected.
    """
    # Check the path itself and its parents up to root
    current = path
    while True:
        if current.is_symlink():
            raise ContractValidationError(
                f"Symlink detected at: {current}",
                agent_fix_hint="Use real paths, not symlinks, for security.",
            )
        resolved = current.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ContractValidationError(
                f"Path traversal detected: {current} resolves outside {root}",
                agent_fix_hint="Ensure all paths are within the dataset directory.",
            )
        if current == current.parent or current == root:
            break
        current = current.parent

    # Check all children if path is a directory
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ContractValidationError(
                    f"Symlink detected in subdirectory: {child}",
                    agent_fix_hint="Remove symlinks from the dataset directory.",
                )
            if not child.resolve().is_relative_to(root.resolve()):
                raise ContractValidationError(
                    f"Path traversal via child: {child} resolves outside {root}",
                    agent_fix_hint="Ensure all files are within the dataset directory.",
                )


def export_static_html(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained static HTML viewer file with embedded dataset data.

    Embeds manifest.json and base64-encoded images into the HTML template,
    requiring zero running processes or local web servers to view (conforming
    to the Zero-Python server-side rule).

    Args:
        dataset_dir: Directory containing manifest.json and frame/visualization media.
        output_path: Destination path for the .html file. If None, defaults to
            ``dataset_dir / "viewer.html"``.

    Returns:
        Path to the generated self-contained HTML file.
    """
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        raise AssetNotFoundError(
            f"Dataset directory not found: {dataset_path}",
            agent_fix_hint=f"Ensure '{dataset_path}' exists and contains manifest.json.",
        )

    # Reject symlinks in dataset path and all children before resolving
    _check_no_symlinks(dataset_path, dataset_path)

    manifest_file = dataset_path / "manifest.json"
    if not manifest_file.is_file():
        raise AssetNotFoundError(
            f"manifest.json not found in dataset directory: {dataset_path}",
            agent_fix_hint="Generate a dataset first using 'synapse generate'.",
        )

    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractValidationError(
            f"Invalid JSON in manifest.json: {exc}",
            agent_fix_hint=(
                f"Fix the JSON syntax in '{manifest_file}' or "
                "regenerate the dataset with 'synapse generate'."
            ),
        ) from exc

    # Gather images to embed as base64
    embedded_images: dict[str, str] = {}
    image_extensions = {".png", ".jpg", ".jpeg", ".svg"}
    frame_count = 0
    for subdir_name in ["frames", "visualizations"]:
        subdir = dataset_path / subdir_name
        if subdir.is_dir():
            for img_file in sorted(subdir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    rel_key = f"{subdir_name}/{img_file.name}"
                    uri = _encode_data_uri(img_file)
                    if uri:
                        embedded_images[rel_key] = uri
                    if subdir_name == "frames":
                        frame_count += 1

    if frame_count > 9999:
        import warnings

        warnings.warn(
            f"Dataset has {frame_count} frames (>9999). "
            "Frame index display in the viewer may overflow 4-digit fields.",
            stacklevel=2,
        )

    contact_sheet = dataset_path / "contact_sheet.png"
    if contact_sheet.is_file():
        uri = _encode_data_uri(contact_sheet)
        if uri:
            embedded_images["contact_sheet.png"] = uri

    html_content = VIEWER_HTML_TEMPLATE
    # Escape <, >, & in JSON to prevent XSS via </script> tag injection
    # when embedding arbitrary user data inside <script> blocks.
    safe_manifest = (
        json.dumps(manifest_data)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    safe_images = (
        json.dumps(embedded_images)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html_content = html_content.replace(
        "/*__EMBEDDED_MANIFEST__*/ null",
        safe_manifest,
    )
    html_content = html_content.replace(
        "/*__EMBEDDED_IMAGES__*/ {}",
        safe_images,
    )

    if output_path is None:
        target_path = dataset_path / "viewer.html"
    else:
        target_path = Path(output_path).resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(html_content, encoding="utf-8")
    return target_path
