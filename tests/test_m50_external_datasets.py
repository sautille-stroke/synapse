"""Tests for M50: External 3D datasets contract, normalization, Stage 0 gate, and 40 scenarios."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synapse.adapters.mock import MockVideoAdapter
from synapse.bench.registry import (
    AssetRegistry,
    ExternalDatasetSpec,
    ScenarioRegistry,
    filter_stage0_quality,
    normalize_external_mesh,
)
from synapse.bench.scenario import (
    ALLOWED_EXTERNAL_LICENSES,
    Asset,
    AssetSource,
    load_scenario,
    validate_external_license,
)
from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.pipeline.bench_runner import BenchPipelineRunner


def test_scenario_catalogue_expanded_to_40_with_external_sources() -> None:
    """PR-208/209: built-in scenario registry expands from 30 to 40 with 10 external assets."""
    reg = ScenarioRegistry()
    names = reg.list_scenarios()
    assert len(names) >= 40

    sources_seen: set[AssetSource] = set()
    ext_count = 0
    for sid in names:
        sc = load_scenario(sid)
        if sc.asset.source in {
            AssetSource.GSO,
            AssetSource.ABO,
            AssetSource.OMNIOBJECT3D,
            AssetSource.THINGI10K,
        }:
            ext_count += 1
            sources_seen.add(sc.asset.source)
            assert sc.asset.license in ALLOWED_EXTERNAL_LICENSES
            assert sc.asset.fallback_procedural
    assert ext_count == 10
    assert sources_seen == {
        AssetSource.GSO,
        AssetSource.ABO,
        AssetSource.OMNIOBJECT3D,
        AssetSource.THINGI10K,
    }


def test_external_license_whitelist_enforced() -> None:
    """PR-207: permissive licenses pass while non-whitelisted licenses raise exit 10."""
    validate_external_license(AssetSource.GSO, "CC-BY-4.0")
    validate_external_license(AssetSource.THINGI10K, "CC0-1.0")
    ExternalDatasetSpec(source=AssetSource.ABO, asset_id="abo_1", license="Apache-2.0")

    with pytest.raises(ContractValidationError) as exc_info:
        Asset(source=AssetSource.GSO, id="bad_license_obj", license="GPL-3.0")
    assert exc_info.value.exit_code == 10

    with pytest.raises(ContractValidationError):
        Asset(source=AssetSource.ABO, id="prop_obj", license="Proprietary")


def test_normalize_external_mesh_z_up_to_y_up_and_unit_sphere() -> None:
    """PR-207: Z-up external coordinates rotate cleanly to Y-up centered unit sphere."""
    # Box shifted in Z-up space: [2..4, 0..2, 0..2]
    verts = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [4.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    norm_v, norm_f = normalize_external_mesh(verts, faces, up_axis="Z", target_radius=1.0)
    assert norm_v.shape == verts.shape
    assert norm_f.shape == faces.shape
    center = 0.5 * (norm_v.min(axis=0) + norm_v.max(axis=0))
    np.testing.assert_allclose(center, np.zeros(3), atol=1e-5)
    max_r = float(np.max(np.linalg.norm(norm_v, axis=1)))
    assert max_r == pytest.approx(1.0, abs=1e-5)

    bad_verts = verts.copy()
    bad_verts[0, 0] = np.nan
    with pytest.raises(ContractValidationError):
        normalize_external_mesh(bad_verts, faces)


def test_filter_stage0_quality_gates_corrupt_and_degenerate_meshes() -> None:
    """PR-207: Stage 0 gate passes clean tetrahedron and rejects degenerate/NaN meshes."""
    tet_v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    tet_f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    ok, reasons = filter_stage0_quality(tet_v, tet_f, require_manifold=True, require_watertight=True)
    assert ok is True
    assert reasons == []

    # Degenerate zero-area triangle
    bad_v = tet_v.copy()
    bad_v[1] = bad_v[0]
    ok_bad, reasons_bad = filter_stage0_quality(bad_v, tet_f)
    assert ok_bad is False
    assert any("degenerate_zero_area_faces" in r for r in reasons_bad)


def test_asset_registry_external_offline_fallback_and_local_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-207: resolve_with_metadata falls back offline cleanly, fails when strict, verifies SHA."""
    reg = AssetRegistry()
    ext_asset = Asset(
        source=AssetSource.GSO,
        id="gso_nonexistent_local_asset",
        license="CC-BY-4.0",
        fallback_procedural="sphere",
    )
    resolved, meta = reg.resolve_with_metadata(ext_asset, strict_external=False)
    assert resolved.is_file()
    assert meta["fallback_used"] is True

    with pytest.raises(AssetNotFoundError) as exc_info:
        reg.resolve_with_metadata(ext_asset, strict_external=True)
    assert exc_info.value.exit_code == 20

    # Local external dir with SHA-256 verification
    ext_dir = tmp_path / "ext" / "gso"
    ext_dir.mkdir(parents=True)
    obj_path = ext_dir / "gso_custom_cup.obj"
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(obj_path.read_bytes()).hexdigest()
    monkeypatch.setenv("SYNAPSE_EXTERNAL_DATASETS_DIR", str(tmp_path / "ext"))

    local_asset = Asset(
        source=AssetSource.GSO,
        id="gso_custom_cup",
        license="CC-BY-4.0",
        sha256=digest,
        fallback_procedural="cube",
    )
    p_local, m_local = reg.resolve_with_metadata(local_asset, strict_external=True)
    assert p_local == obj_path
    assert m_local["resolved_via"] == "local_external"
    assert m_local["fallback_used"] is False

    bad_sha_asset = Asset(
        source=AssetSource.GSO,
        id="gso_custom_cup",
        license="CC-BY-4.0",
        sha256="0" * 64,
        fallback_procedural="cube",
    )
    with pytest.raises(ContractValidationError):
        reg.resolve_with_metadata(bad_sha_asset, strict_external=False)


def test_bench_runner_runs_external_scenarios_offline() -> None:
    """PR-209: BenchPipelineRunner executes GSO and Thingi10K scenarios end-to-end offline."""
    runner = BenchPipelineRunner(MockVideoAdapter())
    for sid in ("t1_ext_gso_ceramic_mug", "t4_ext_thingi10k_lattice_stress"):
        report = runner.run_scenario(sid, seed=42, resolution=(32, 32), num_frames=4)
        assert report.error is None, f"{sid} failed: {report.error}"
        assert report.num_frames == 4
        assert report.external_dataset_meta["stage0_passed"] is True
        d = report.to_dict()
        assert "external_dataset_meta" in d


def test_prepare_benchmark_assets_external_manifest_cli(tmp_path: Path) -> None:
    """PR-209: prepare_benchmark_assets.py --manifest-only & --offline-fallback work offline."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "prepare_benchmark_assets.py"
    out_manifest = tmp_path / "ext_manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "all",
            "--manifest-only",
            "--target-dir",
            str(tmp_path),
            "--manifest-out",
            str(out_manifest),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "ok"
    assert data["count"] == 10
    assert out_manifest.is_file()
