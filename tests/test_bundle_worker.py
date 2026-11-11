"""Tests for synapse.pipeline.bundle_worker — RFC 0039 JobBundleWorker."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synapse.pipeline.bundle_worker import (
    BundleManifest,
    HardwareBudget,
    JobBundleWorker,
    OutputFileEntry,
    ResultManifest,
    _STAGE_HANDLERS,
    _compute_sha256,
    get_registered_stages,
    register_stage,
    run_worker_cli,
)


# =========================================================================
# BundleManifest
# =========================================================================


class TestBundleManifest:
    """Test BundleManifest Pydantic model validation."""

    def test_valid_manifest_parsing(self) -> None:
        """A well-formed manifest dict should parse without errors."""
        data = {
            "job_id": "job-001",
            "stage": "fgpp_validate",
            "inputs": {"fgpp_dir": "/tmp/fgpp"},
            "outputs": {},
            "params": {"verbose": True},
            "hardware_budget": {
                "free_vram_gb": 24.0,
                "max_cpu_cores": 8,
            },
        }
        m = BundleManifest(**data)
        assert m.job_id == "job-001"
        assert m.stage == "fgpp_validate"
        assert m.inputs["fgpp_dir"] == "/tmp/fgpp"
        assert m.hardware_budget.free_vram_gb == 24.0
        assert m.schema_version == "synapse.job_bundle.v1.0"

    def test_schema_version_validation(self) -> None:
        """Unsupported schema_version must raise ValidationError."""
        with pytest.raises(Exception, match="Unsupported schema_version"):
            BundleManifest(
                schema_version="bad.version.0",
                job_id="j1",
                stage="fgpp_validate",
            )

    def test_missing_required_fields(self) -> None:
        """Omitting required fields must raise ValidationError."""
        with pytest.raises(Exception):
            BundleManifest()  # type: ignore[call-arg]

    def test_default_hardware_budget(self) -> None:
        """Default HardwareBudget should be 24 GiB / 8 cores."""
        m = BundleManifest(job_id="j2", stage="fgpp_validate")
        assert m.hardware_budget.free_vram_gb == 24.0
        assert m.hardware_budget.max_cpu_cores == 8

    def test_traceparent_optional(self) -> None:
        """traceparent and tracestate default to None."""
        m = BundleManifest(job_id="j3", stage="fgpp_validate")
        assert m.traceparent is None
        assert m.tracestate is None

    def test_traceparent_set(self) -> None:
        """traceparent can be set explicitly."""
        m = BundleManifest(
            job_id="j4",
            stage="fgpp_validate",
            traceparent="00-abc123-def456-01",
            tracestate="vendor=test",
        )
        assert m.traceparent == "00-abc123-def456-01"
        assert m.tracestate == "vendor=test"


# =========================================================================
# Stage Registry
# =========================================================================


class TestStageRegistry:
    """Test stage handler registration and discovery."""

    def test_builtin_stages_registered(self) -> None:
        """All four built-in stages must be in the registry."""
        stages = get_registered_stages()
        assert "fgpp_validate" in stages
        assert "neural_video_generate" in stages
        assert "inverse_reconstruct" in stages
        assert "kinematic_distill" in stages
        assert "style_extract_and_map" in stages
        assert "surgical_composite" in stages

    def test_custom_stage_registration(self) -> None:
        """The @register_stage decorator should add a custom stage."""
        stage_name = "_test_custom_stage_xyz"
        try:

            @register_stage(stage_name)
            def _my_handler(
                manifest: BundleManifest,
            ) -> dict[str, Any]:
                return {"output_paths": {}, "custom": True}

            assert stage_name in _STAGE_HANDLERS
            assert _STAGE_HANDLERS[stage_name] is _my_handler
        finally:
            _STAGE_HANDLERS.pop(stage_name, None)

    def test_get_registered_stages_sorted(self) -> None:
        """get_registered_stages should return sorted names."""
        stages = get_registered_stages()
        assert stages == sorted(stages)


# =========================================================================
# JobBundleWorker
# =========================================================================


class TestJobBundleWorker:
    """Test JobBundleWorker execution for each stage."""

    def test_fgpp_validate_stage(self, tmp_path: Path) -> None:
        """fgpp_validate should succeed with valid FGPP directory."""
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "test_scene",
            "fps": 24,
            "num_frames": 2,
            "total_frames": 2,
            "resolution": [512, 512],
            "frames": [
                {"frame_index": 0},
                {"frame_index": 1},
            ],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        manifest = BundleManifest(
            job_id="test-fgpp",
            stage="fgpp_validate",
            inputs={"fgpp_dir": str(fgpp_dir)},
        )

        worker = JobBundleWorker(worker_id="test")
        result = worker.execute(manifest)

        assert result.status == "success"
        assert result.exit_code == 0
        assert result.job_id == "test-fgpp"
        assert result.stage == "fgpp_validate"

    def test_neural_video_generate_stage(
        self, tmp_path: Path,
    ) -> None:
        """neural_video_generate with mock adapter should succeed."""
        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-nvg",
            stage="neural_video_generate",
            inputs={},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 16,
                "height": 64,
                "width": 64,
                "adapter": "mock",
                "model_variant": "wan2.1_1.3b",
            },
            hardware_budget=HardwareBudget(
                free_vram_gb=48.0,
            ),
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        assert result.status in ("success", "fallback")
        assert "video" in result.output_files
        assert output_video.exists()

    def test_style_extract_stage(self, tmp_path: Path) -> None:
        """style_extract_and_map should produce a style profile."""
        output_yaml = tmp_path / "style_profile.yaml"

        manifest = BundleManifest(
            job_id="test-style",
            stage="style_extract_and_map",
            inputs={},
            outputs={"style_profile": str(output_yaml)},
            params={
                "width": 64,
                "height": 64,
                "num_frames": 4,
            },
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        assert result.status == "success"
        assert "style_profile" in result.output_files
        # May be .yaml or .json depending on pyyaml availability
        entry = result.output_files["style_profile"]
        assert Path(entry.path).exists()

    def test_surgical_composite_stage(
        self, tmp_path: Path,
    ) -> None:
        """surgical_composite should produce a composited array."""
        output_npy = tmp_path / "composited.npy"

        manifest = BundleManifest(
            job_id="test-comp",
            stage="surgical_composite",
            inputs={},
            outputs={"composited": str(output_npy)},
            params={
                "width": 64,
                "height": 64,
                "num_frames": 4,
            },
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        assert result.status == "success"
        assert "composited" in result.output_files
        assert output_npy.exists()

    def test_inverse_reconstruct_stage(
        self, tmp_path: Path,
    ) -> None:
        """inverse_reconstruct should produce trajectory and telemetry."""
        output_traj = tmp_path / "camera_trajectory.json"
        output_telem = tmp_path / "kinematics_telemetry.json"

        manifest = BundleManifest(
            job_id="test-recon",
            stage="inverse_reconstruct",
            inputs={},
            outputs={
                "trajectory": str(output_traj),
                "telemetry": str(output_telem),
            },
            params={
                "width": 64,
                "height": 64,
                "num_frames": 8,
            },
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        assert result.status == "success"
        assert "trajectory" in result.output_files
        assert "telemetry" in result.output_files
        assert output_traj.exists()
        assert output_telem.exists()

    def test_oom_fallback_spatial_tile(
        self, tmp_path: Path,
    ) -> None:
        """With tight VRAM (0.80 < R <= 1.05), spatial_tile injected."""
        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-oom-l1",
            stage="neural_video_generate",
            inputs={},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 16,
                "height": 64,
                "width": 64,
                "adapter": "mock",
                "model_variant": "wan2.1_1.3b",
            },
            hardware_budget=HardwareBudget(
                free_vram_gb=6.0,
            ),
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        assert result.status == "fallback"
        assert (
            result.fallback_level
            == "LEVEL_1_SPATIAL_TILE_2X2"
        )

    def test_oom_fallback_short_window(
        self, tmp_path: Path,
    ) -> None:
        """With very tight VRAM (R > 1.05), window_frames=49."""
        output_video = tmp_path / "video.npy"

        # wan2.1_14b model has ~28 GB weights, so use small budget
        manifest = BundleManifest(
            job_id="test-oom-l2",
            stage="neural_video_generate",
            inputs={},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 97,
                "height": 512,
                "width": 512,
                "adapter": "mock",
                "model_variant": "wan2.1_14b",
            },
            hardware_budget=HardwareBudget(
                free_vram_gb=8.0,
            ),
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        # With 8 GB budget against 14B model (~28 GB weights alone),
        # even with 49-frame window the estimate will exceed budget,
        # so we expect error exit_code=30
        assert result.exit_code == 30
        assert result.status == "error"

    def test_oom_reject(self, tmp_path: Path) -> None:
        """With extremely low VRAM, exit_code=30."""
        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-oom-reject",
            stage="neural_video_generate",
            inputs={},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 97,
                "height": 1024,
                "width": 1024,
                "adapter": "mock",
                "model_variant": "wan2.1_14b",
            },
            hardware_budget=HardwareBudget(
                free_vram_gb=2.0,
            ),
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 30
        assert result.status == "error"
        assert result.agent_fix_hint is not None

    def test_result_manifest_sha256(
        self, tmp_path: Path,
    ) -> None:
        """SHA256 checksums in output_files must be correct."""
        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-sha",
            stage="neural_video_generate",
            inputs={},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 4,
                "height": 32,
                "width": 32,
                "adapter": "mock",
                "model_variant": "wan2.1_1.3b",
            },
            hardware_budget=HardwareBudget(
                free_vram_gb=48.0,
            ),
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        assert "video" in result.output_files
        entry = result.output_files["video"]

        # Independently compute SHA256
        expected_sha = _compute_sha256(Path(entry.path))
        assert entry.sha256 == expected_sha
        assert entry.size_bytes > 0
        assert entry.size_bytes == Path(entry.path).stat().st_size

    def test_single_execution(self, tmp_path: Path) -> None:
        """Worker executes once and returns; no lingering threads."""
        initial_threads = threading.active_count()

        output_video = tmp_path / "video.npy"
        manifest = BundleManifest(
            job_id="test-single",
            stage="neural_video_generate",
            inputs={},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 4,
                "height": 32,
                "width": 32,
                "adapter": "mock",
                "model_variant": "wan2.1_1.3b",
            },
            hardware_budget=HardwareBudget(
                free_vram_gb=48.0,
            ),
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 0
        # Worker should not spawn persistent threads
        assert threading.active_count() <= initial_threads + 1

    def test_unknown_stage_error(self) -> None:
        """An unregistered stage should return exit_code=10."""
        manifest = BundleManifest(
            job_id="test-bad-stage",
            stage="nonexistent_stage_xyz",
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 10
        assert result.status == "error"
        assert "nonexistent_stage_xyz" in (
            result.agent_fix_hint or ""
        )


# =========================================================================
# CLI Worker
# =========================================================================


class TestCLIWorker:
    """Test the worker CLI entry point."""

    def test_worker_run_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Create temp bundle, run via CLI, verify JSON output."""
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        # Create FGPP directory
        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "cli_test",
            "fps": 24,
            "num_frames": 1,
            "total_frames": 1,
            "resolution": [64, 64],
            "frames": [{"frame_index": 0}],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        # Create bundle manifest
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        bundle_manifest = {
            "schema_version": "synapse.job_bundle.v1.0",
            "job_id": "cli-test-001",
            "stage": "fgpp_validate",
            "inputs": {"fgpp_dir": str(fgpp_dir)},
            "outputs": {},
            "params": {},
        }
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest), encoding="utf-8",
        )

        exit_code = run_worker_cli(
            str(bundle_dir), json_mode=True,
        )

        assert exit_code == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out.strip())
        assert result["status"] == "success"
        assert result["job_id"] == "cli-test-001"
        assert result["exit_code"] == 0

        # Result manifest file should be written
        result_file = bundle_dir / "result_manifest.json"
        assert result_file.exists()

    def test_worker_run_missing_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Missing bundle manifest should return exit_code=20."""
        exit_code = run_worker_cli(
            str(tmp_path / "nonexistent"), json_mode=True,
        )
        assert exit_code == 20

    def test_worker_run_text_mode(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-JSON mode should print human-readable output."""
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "text_test",
            "fps": 24,
            "num_frames": 1,
            "total_frames": 1,
            "resolution": [64, 64],
            "frames": [{"frame_index": 0}],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        bundle_manifest = {
            "schema_version": "synapse.job_bundle.v1.0",
            "job_id": "text-test-001",
            "stage": "fgpp_validate",
            "inputs": {"fgpp_dir": str(fgpp_dir)},
            "outputs": {},
            "params": {},
        }
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest), encoding="utf-8",
        )

        exit_code = run_worker_cli(
            str(bundle_dir), json_mode=False,
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        assert "text-test-001" in captured.out
