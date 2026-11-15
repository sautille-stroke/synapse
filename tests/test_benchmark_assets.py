"""Unit tests for RFC 0063 / RFC 0062 Benchmark Asset Preparation Pipeline.

Tests:
1. Zero ValueError on external --target-dir mounts (Colab /content/... regression test).
2. Canonical 16-dimension VBench prompt suite completeness.
3. Fail-closed behavior when local/remote data is absent without --allow-synthetic-smoke.
4. Repository manifest protection (never overwrite benchmark_assets_manifest.json by default).
5. RFC 0062 L-Asset manifest structure and package registry validation.
6. stage_from_remote_huggingface returns non-empty download list with mock HF client.
7. stage_from_remote_huggingface rejects SHA-256 mismatch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_benchmark_assets.py"
L_ASSET_MANIFEST = REPO_ROOT / "assets" / "assets_manifest.json"
REPO_MANIFEST = REPO_ROOT / "data" / "manifests" / "benchmark_assets_manifest.json"

sys.path.insert(0, str(REPO_ROOT))


def test_rfc0062_l_asset_manifest_structure() -> None:
    """Verify that assets/assets_manifest.json exists and satisfies RFC 0062."""
    assert L_ASSET_MANIFEST.is_file(), f"Missing {L_ASSET_MANIFEST}"
    with L_ASSET_MANIFEST.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == "1.0.0"
    assert data.get("rfc") == "RFC-0062"
    assert "mirrors" in data and len(data["mirrors"]) >= 2
    # At least one mirror must be a valid https:// URL
    https_mirrors = [m for m in data["mirrors"] if m.startswith("https://")]
    assert len(https_mirrors) >= 1, "At least one https:// mirror URL required"
    assert "packages" in data

    packages = data["packages"]
    for pkg_name in ["gso_starter", "objaverse_eval", "mixamo_dynamics", "vbench_prompts"]:
        assert pkg_name in packages, f"Missing package: {pkg_name}"
        pkg = packages[pkg_name]
        assert "version" in pkg
        assert "sha256" in pkg
        assert "install_dir" in pkg
        assert "size_bytes" in pkg
        # RFC 0062 vital_file field for per-file SHA-256 verification
        assert "vital_file" in pkg, f"Package {pkg_name} missing vital_file field"
        vital = pkg["vital_file"]
        assert "name" in vital, f"vital_file for {pkg_name} missing 'name'"
        assert "sha256" in vital, f"vital_file for {pkg_name} missing 'sha256'"
        assert len(vital["sha256"]) == 64, f"vital_file sha256 must be 64 hex chars"
        # hf_repo and hf_filename for remote distribution
        assert "hf_repo" in pkg, f"Package {pkg_name} missing hf_repo"
        assert "hf_filename" in pkg, f"Package {pkg_name} missing hf_filename"


def test_prepare_benchmark_assets_external_target_dir(tmp_path: Path) -> None:
    """Verify prepare_benchmark_assets.py runs without ValueError on external target-dir."""
    external_target = tmp_path / "colab_simulated_mount" / "datasets" / "benchmarks"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--tier", "starter_dev",
        "--target-dir", str(external_target),
        "--allow-synthetic-smoke",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with {result.returncode}: {result.stderr}"

    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert output["tier"] == "starter_dev"

    manifest_file = external_target / "staged_manifest.json"
    assert manifest_file.is_file(), "staged_manifest.json must be written in target_dir"

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["tier"] == "starter_dev"
    assert manifest["total_assets"] >= 2
    for asset in manifest["assets"]:
        rel_p = asset["relative_path"]
        assert not rel_p.startswith("/"), f"relative_path should not be absolute: {rel_p}"
        assert ".." not in rel_p, f"relative_path should not contain '..': {rel_p}"
        # File must exist inside external_target
        staged_file = external_target / rel_p
        assert staged_file.is_file(), f"Referenced asset does not exist: {staged_file}"


def test_vbench_16_dimensions_completeness(tmp_path: Path) -> None:
    """Verify that all 16 canonical VBench dimensions are generated."""
    target_dir = tmp_path / "benchmarks"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--target-dir", str(target_dir),
        "--allow-synthetic-smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    prompts_file = target_dir / "vbench_prompts" / "prompts_per_dimension.json"
    assert prompts_file.is_file(), "prompts_per_dimension.json must exist"

    with prompts_file.open("r", encoding="utf-8") as f:
        vbench_data = json.load(f)

    assert vbench_data.get("total_dimensions") == 16
    expected_dims = [
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "object_class",
        "multiple_objects",
        "color",
        "spatial_relationship",
        "scene",
        "temporal_style",
        "overall_consistency",
        "human_action",
        "appearance_style",
    ]
    dims = vbench_data["dimensions"]
    for dim in expected_dims:
        assert dim in dims, f"Missing VBench dimension: {dim}"
        items = dims[dim]
        assert len(items) >= 2, f"Dimension {dim} must have at least 2 prompts"
        for item in items:
            assert "prompt" in item
            assert "quality_target" in item
            assert 0.0 < item["quality_target"] <= 1.0


def test_tier_aliases_normalization(tmp_path: Path) -> None:
    """Verify tier alias normalization ('starter' -> 'starter_dev')."""
    target_dir = tmp_path / "tier_test"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--tier", "starter",
        "--target-dir", str(target_dir),
        "--allow-synthetic-smoke",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert output["tier"] == "starter_dev"


def test_fail_closed_without_synthetic_smoke(tmp_path: Path) -> None:
    """Verify fail-closed behavior (exit 20) when data is missing and smoke flag is absent."""
    empty_source = tmp_path / "empty_source"
    empty_source.mkdir()
    target_dir = tmp_path / "target"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--source-root", str(empty_source),
        "--target-dir", str(target_dir),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 20, f"Expected exit code 20, got {result.returncode}"

    output = json.loads(result.stdout)
    assert output["exit_code"] == 20
    assert "agent_fix_hint" in output


def test_repo_manifest_never_modified_by_default(tmp_path: Path) -> None:
    """Verify that default runs do NOT touch data/manifests/benchmark_assets_manifest.json."""
    if not REPO_MANIFEST.is_file():
        pytest.skip("Repo manifest does not exist in working directory")

    mtime_before = REPO_MANIFEST.stat().st_mtime
    target_dir = tmp_path / "target_isolation"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--target-dir", str(target_dir),
        "--allow-synthetic-smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    mtime_after = REPO_MANIFEST.stat().st_mtime
    assert mtime_before == mtime_after, "Default run must NOT touch data/manifests/benchmark_assets_manifest.json"


def _create_mock_hf_file(target_root: Path, pkg_name: str, filename: str, content: bytes) -> Path:
    """Helper to create a mock file simulating HF download."""
    install_dir = target_root / pkg_name
    install_dir.mkdir(parents=True, exist_ok=True)
    file_path = install_dir / filename
    file_path.write_bytes(content)
    return file_path


def test_stage_from_remote_huggingface_returns_nonempty(tmp_path: Path) -> None:
    """Verify stage_from_remote_huggingface returns non-empty download list with mock HF client."""
    import hashlib

    from scripts.prepare_benchmark_assets import stage_from_remote_huggingface

    target_root = tmp_path / "hf_staging"
    target_root.mkdir()

    # Create a fake vital file that matches the manifest SHA-256
    test_content = b"mock stanford bunny OBJ content for testing\n"
    test_sha = hashlib.sha256(test_content).hexdigest()

    # Patch the manifest to use our test SHA
    mock_manifest = {
        "packages": {
            "gso_starter": {
                "hf_repo": "sautille-stroke/synapse-benchmark-assets",
                "hf_filename": "gso_starter/stanford_bunny.obj",
                "install_dir": "data/benchmarks/gso_starter",
                "category": "3d_reconstruction",
                "vital_file": {
                    "name": "stanford_bunny.obj",
                    "sha256": test_sha,
                },
            }
        }
    }

    def mock_hf_hub_download(
        repo_id: str, filename: str, repo_type: str, token: Any = None, local_dir: str = "",
    ) -> str:
        """Simulate downloading a file from HuggingFace Hub."""
        dl_path = Path(local_dir) / Path(filename).name
        dl_path.parent.mkdir(parents=True, exist_ok=True)
        dl_path.write_bytes(test_content)
        return str(dl_path)

    with (
        mock.patch(
            "scripts.prepare_benchmark_assets._load_assets_manifest",
            return_value=mock_manifest,
        ),
        mock.patch(
            "scripts.prepare_benchmark_assets.hf_hub_download",
            side_effect=mock_hf_hub_download,
            create=True,
        ),
        mock.patch.dict(
            "sys.modules",
            {"huggingface_hub": mock.MagicMock(hf_hub_download=mock_hf_hub_download)},
        ),
    ):
        records = stage_from_remote_huggingface(target_root, tier="starter_dev", verbose=False)

    assert len(records) > 0, "stage_from_remote_huggingface must return non-empty list"
    for rec in records:
        assert "sha256" in rec
        assert "relative_path" in rec
        assert rec["size_bytes"] > 0


def test_stage_from_remote_huggingface_rejects_sha256_mismatch(tmp_path: Path) -> None:
    """Verify stage_from_remote_huggingface raises ValueError on SHA-256 mismatch."""
    from scripts.prepare_benchmark_assets import stage_from_remote_huggingface

    target_root = tmp_path / "hf_mismatch"
    target_root.mkdir()

    test_content = b"corrupted content\n"
    # Manifest declares a different SHA
    wrong_sha = "0" * 64

    mock_manifest = {
        "packages": {
            "gso_starter": {
                "hf_repo": "sautille-stroke/synapse-benchmark-assets",
                "hf_filename": "gso_starter/stanford_bunny.obj",
                "install_dir": "data/benchmarks/gso_starter",
                "category": "3d_reconstruction",
                "vital_file": {
                    "name": "stanford_bunny.obj",
                    "sha256": wrong_sha,
                },
            }
        }
    }

    def mock_hf_hub_download(
        repo_id: str, filename: str, repo_type: str, token: Any = None, local_dir: str = "",
    ) -> str:
        dl_path = Path(local_dir) / Path(filename).name
        dl_path.parent.mkdir(parents=True, exist_ok=True)
        dl_path.write_bytes(test_content)
        return str(dl_path)

    with (
        mock.patch(
            "scripts.prepare_benchmark_assets._load_assets_manifest",
            return_value=mock_manifest,
        ),
        mock.patch.dict(
            "sys.modules",
            {"huggingface_hub": mock.MagicMock(hf_hub_download=mock_hf_hub_download)},
        ),
    ):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            stage_from_remote_huggingface(target_root, tier="starter_dev", verbose=False)
