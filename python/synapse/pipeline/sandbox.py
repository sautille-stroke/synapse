"""Workspace sandbox path validation and SHA-256 helpers for JobBundleWorker."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SandboxViolation:
    """Structured error returned when a manifest violates workspace sandboxing."""

    exit_code: int
    error_type: str
    message: str
    agent_fix_hint: str


def _compute_sha256(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _check_ancestor_symlinks(p: Path) -> str | None:
    """Check all ancestor components of *p* for symlink traversal.

    Returns an error message string if any ancestor is a symlink,
    or ``None`` if the path is clean.
    """
    parts = p.parts
    for i in range(1, len(parts) + 1):
        ancestor = Path(*parts[:i])
        if ancestor.is_symlink():
            return (
                f"Symlink traversal blocked: ancestor '{ancestor}' "
                f"in path '{p}' is a symlink."
            )
    return None


def _extract_manifest_traceboard_param(params: dict[str, Any]) -> Any:
    """Extract manifest-configured Traceboard log directory parameter if set."""
    for key in ("_traceboard_dir", "traceboard_dir"):
        val = params.get(key)
        if val is not None:
            return val
    tb_cfg = params.get("traceboard")
    if isinstance(tb_cfg, dict) and tb_cfg.get("log_dir") is not None:
        return tb_cfg.get("log_dir")
    return None


def _validate_manifest_traceboard_dir(
    tb_param: Any,
    workspace_root: str | None,
) -> Path:
    """Validate and sandbox manifest-configured Traceboard output directory."""
    from synapse.errors import ContractValidationError

    if not isinstance(tb_param, str) or not tb_param.strip():
        raise ContractValidationError(
            "Traceboard log directory must be a non-empty string path.",
            agent_fix_hint=(
                "Set a valid directory path for traceboard_dir (or _traceboard_dir)."
            ),
        )

    raw = Path(tb_param)
    if workspace_root is not None:
        ws_root = Path(workspace_root).resolve()
        if not raw.is_absolute():
            raw = ws_root / raw
        if raw.is_symlink():
            raise ContractValidationError(
                f"Symlinks are forbidden in Traceboard path: '{tb_param}' is a symlink.",
                agent_fix_hint=(
                    f"Use a real non-symlink directory inside workspace root '{ws_root}'."
                ),
            )
        ancestor_err = _check_ancestor_symlinks(raw)
        if ancestor_err:
            raise ContractValidationError(
                ancestor_err,
                agent_fix_hint=(
                    f"Remove symlinks from Traceboard path ancestors inside '{ws_root}'."
                ),
            )
        resolved = raw.resolve()
        if not resolved.is_relative_to(ws_root):
            raise ContractValidationError(
                f"Traceboard log directory escapes workspace sandbox: "
                f"'{tb_param}' resolves to '{resolved}' which is outside "
                f"workspace root '{ws_root}'.",
                agent_fix_hint=(
                    f"Ensure Traceboard log directory resolves inside workspace root '{ws_root}'."
                ),
            )
        return resolved

    if raw.is_symlink():
        raise ContractValidationError(
            f"Symlinks are forbidden in Traceboard path: '{tb_param}' is a symlink.",
            agent_fix_hint="Use a real non-symlink directory path.",
        )
    return raw


def validate_io_mappings(
    inputs: dict[str, str],
    outputs: dict[str, str],
    workspace_root: Path | None,
) -> SandboxViolation | None:
    """Validate, resolve symlinks, and normalize ``inputs`` / ``outputs`` in-place."""
    for label, mapping in [("inputs", inputs), ("outputs", outputs)]:
        for key, val in mapping.items():
            if not val or not val.strip():
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=(
                        f"Empty path in {label}['{key}']. "
                        "Paths must be non-empty strings."
                    ),
                    agent_fix_hint=f"Provide a non-empty path for {label}['{key}'].",
                )

            raw = Path(val)
            if not raw.is_absolute() and workspace_root is not None:
                raw = workspace_root / raw

            if raw.is_symlink():
                msg = (
                    f"Symlinks are forbidden in {label}: "
                    f"'{key}' -> '{val}' is a symlink. "
                    "Use real paths only."
                )
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=msg,
                    agent_fix_hint=msg,
                )

            ancestor_err = _check_ancestor_symlinks(raw)
            if ancestor_err:
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=ancestor_err,
                    agent_fix_hint=(
                        "Remove symlinks from path ancestors. Use real paths only."
                    ),
                )

            resolved = raw.resolve()
            if workspace_root is not None and not resolved.is_relative_to(workspace_root):
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=(
                        f"Path escapes workspace sandbox: "
                        f"{label}['{key}'] = '{val}' resolves to "
                        f"'{resolved}' which is outside workspace "
                        f"root '{workspace_root}'."
                    ),
                    agent_fix_hint=(
                        f"Ensure all {label} paths are within "
                        f"the workspace root '{workspace_root}'."
                    ),
                )

            mapping[key] = str(resolved)
    return None


def validate_params_paths(
    params: dict[str, Any],
    workspace_root: Path | None,
) -> SandboxViolation | None:
    """Validate path-like values in ``params`` against symlinks and workspace bounds."""
    for pkey, pval in params.items():
        if not isinstance(pval, str) or pkey.startswith("_"):
            continue
        is_path_key = pkey.endswith(("_path", "_dir", "_file"))
        if os.sep in pval or "/" in pval or is_path_key:
            raw_p = Path(pval)
            if not raw_p.is_absolute() and workspace_root is not None:
                raw_p = workspace_root / raw_p
            if raw_p.is_symlink():
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=(
                        f"Symlinks are forbidden in params: "
                        f"'{pkey}' -> '{pval}' is a symlink."
                    ),
                    agent_fix_hint="Use real paths only in params.",
                )
            ancestor_err = _check_ancestor_symlinks(raw_p)
            if ancestor_err:
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=ancestor_err,
                    agent_fix_hint="Remove symlinks from path ancestors.",
                )
            if workspace_root is not None:
                resolved_p = raw_p.resolve()
                if not resolved_p.is_relative_to(workspace_root):
                    return SandboxViolation(
                        exit_code=10,
                        error_type="ContractValidationError",
                        message=(
                            f"Path escapes workspace sandbox: "
                            f"params['{pkey}'] = '{pval}' resolves to "
                            f"'{resolved_p}' which is outside workspace "
                            f"root '{workspace_root}'."
                        ),
                        agent_fix_hint=(
                            f"Ensure all params paths are within "
                            f"the workspace root '{workspace_root}'."
                        ),
                    )
    return None


def validate_deferred_asset_manifest(
    deferred_manifest: Any,
    workspace_root: Path | None,
) -> SandboxViolation | None:
    """Verify sandbox bounds and SHA-256 checksums for deferred ``asset_manifest`` entries."""
    if not deferred_manifest:
        return None

    for entry in deferred_manifest:
        if not isinstance(entry, dict):
            continue
        expected_sha = entry.get("sha256", "")
        asset_path = entry.get("path", "") or entry.get("logical_path", "")
        if expected_sha and asset_path:
            raw_asset = Path(asset_path)
            if not raw_asset.is_absolute() and workspace_root is not None:
                raw_asset = workspace_root / raw_asset
            if raw_asset.is_symlink() or _check_ancestor_symlinks(raw_asset):
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=f"Symlinks are forbidden in asset_manifest: '{asset_path}'.",
                    agent_fix_hint="Use real non-symlink paths in asset_manifest.",
                )
            p = raw_asset.resolve()
            if workspace_root is not None and not p.is_relative_to(workspace_root):
                return SandboxViolation(
                    exit_code=10,
                    error_type="ContractValidationError",
                    message=(
                        f"Asset path escapes workspace sandbox: "
                        f"'{asset_path}' resolves to '{p}' which is "
                        f"outside workspace root '{workspace_root}'."
                    ),
                    agent_fix_hint=(
                        f"Ensure all asset paths are within "
                        f"the workspace root '{workspace_root}'."
                    ),
                )
            if p.is_file():
                actual = _compute_sha256(p)
                if actual != expected_sha:
                    return SandboxViolation(
                        exit_code=20,
                        error_type="AssetNotFoundError",
                        message=(
                            f"SHA-256 mismatch for asset '{asset_path}': "
                            f"expected {expected_sha[:16]}..., "
                            f"got {actual[:16]}..."
                        ),
                        agent_fix_hint=(
                            f"Re-download or re-export '{asset_path}' "
                            "with the correct contents."
                        ),
                    )
            elif not p.exists():
                return SandboxViolation(
                    exit_code=20,
                    error_type="AssetNotFoundError",
                    message=f"Asset file not found: {asset_path!r}",
                    agent_fix_hint=(
                        f"Ensure '{asset_path}' exists on disk "
                        "before submitting the JobBundle."
                    ),
                )
    return None
