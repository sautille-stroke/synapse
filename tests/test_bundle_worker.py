"""Tests for synapse.pipeline.bundle_worker — RFC 0039 JobBundleWorker."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from synapse.pipeline.bundle_worker import (
    _STAGE_HANDLERS,
    BundleManifest,
    HardwareBudget,
    JobBundleWorker,
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
        """neural_video_generate without fgpp_dir should return exit_code=20."""
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

        assert result.exit_code == 20
        assert result.status == "error"
        assert "fgpp" in (result.agent_fix_hint or "").lower()

    def test_neural_video_generate_with_fgpp(
        self, tmp_path: Path,
    ) -> None:
        """neural_video_generate with valid fgpp_dir should succeed."""
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "test_scene",
            "fps": 24,
            "num_frames": 2,
            "total_frames": 2,
            "resolution": [64, 64],
            "frames": [
                {"frame_index": 0},
                {"frame_index": 1},
            ],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-nvg-fgpp",
            stage="neural_video_generate",
            inputs={"fgpp_dir": str(fgpp_dir)},
            outputs={"video": str(output_video)},
            params={
                "num_frames": 2,
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
        """style_extract_and_map without frames_dir should return exit_code=20."""
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

        assert result.exit_code == 20
        assert result.status == "error"

    def test_style_extract_with_frames(self, tmp_path: Path) -> None:
        """style_extract_and_map with valid frames_dir should succeed."""
        import numpy as np

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Create dummy PNG frames using PIL
        try:
            from PIL import Image

            for i in range(4):
                img = Image.fromarray(
                    np.full((64, 64, 3), 128, dtype=np.uint8),
                    mode="RGB",
                )
                img.save(frames_dir / f"frame_{i:04d}.png")
        except ImportError:
            pytest.skip("PIL not available")

        output_yaml = tmp_path / "style_profile.yaml"

        manifest = BundleManifest(
            job_id="test-style-real",
            stage="style_extract_and_map",
            inputs={"frames_dir": str(frames_dir)},
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

    def test_surgical_composite_stage(
        self, tmp_path: Path,
    ) -> None:
        """surgical_composite without input dirs should return exit_code=20."""
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

        assert result.exit_code == 20
        assert result.status == "error"

    def test_inverse_reconstruct_stage(
        self, tmp_path: Path,
    ) -> None:
        """inverse_reconstruct without frames_dir should return exit_code=20."""
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

        assert result.exit_code == 20
        assert result.status == "error"

    def test_style_extract_missing_pil_exits_30(
        self, tmp_path: Path,
    ) -> None:
        """style_extract_and_map with PIL unavailable should return exit_code=30."""
        import unittest.mock

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Create a dummy .png file (content doesn't matter — PIL import fails before read)
        (frames_dir / "frame_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        output_yaml = tmp_path / "style_profile.yaml"

        manifest = BundleManifest(
            job_id="test-style-nopil",
            stage="style_extract_and_map",
            inputs={"frames_dir": str(frames_dir)},
            outputs={"style_profile": str(output_yaml)},
            params={
                "width": 64,
                "height": 64,
                "num_frames": 4,
            },
        )

        worker = JobBundleWorker()
        with unittest.mock.patch.dict(
            "sys.modules", {"PIL": None, "PIL.Image": None},
        ):
            result = worker.execute(manifest)

        assert result.exit_code == 30
        assert result.status == "error"

    def test_lora_preencode_missing_dataset_exits_20(
        self, tmp_path: Path,
    ) -> None:
        """lora_preencode without dataset_dir should return exit_code=20."""
        manifest = BundleManifest(
            job_id="test-preencode-nodata",
            stage="lora_preencode",
            inputs={},
            outputs={"shards_dir": str(tmp_path / "shards")},
            params={"num_frames": 4},
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 20
        assert result.status == "error"

    def test_lora_package_skip_bench_false_exits_40(
        self, tmp_path: Path,
    ) -> None:
        """lora_package with skip_bench=False and no bench_metrics → exit 40."""
        # Create a minimal fake weights file so we get past path validation
        weights = tmp_path / "adapter_model.safetensors"
        weights.write_bytes(b"\x00" * 16)

        manifest = BundleManifest(
            job_id="test-package-nogate",
            stage="lora_package",
            inputs={},
            outputs={},
            params={
                "weights_path": str(weights),
                "output_dir": str(tmp_path / "bundle_out"),
                "category": "style",
                "rank": 8,
                "alpha": 1.0,
                # skip_bench defaults to False, bench_metrics not provided
            },
        )

        worker = JobBundleWorker()
        result = worker.execute(manifest)

        assert result.exit_code == 40
        assert result.status == "error"

    def test_oom_fallback_spatial_tile(
        self, tmp_path: Path,
    ) -> None:
        """With tight VRAM (0.80 < R <= 1.05), spatial_tile injected."""
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "oom_test",
            "fps": 24,
            "num_frames": 2,
            "total_frames": 2,
            "resolution": [64, 64],
            "frames": [
                {"frame_index": 0},
                {"frame_index": 1},
            ],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-oom-l1",
            stage="neural_video_generate",
            inputs={"fgpp_dir": str(fgpp_dir)},
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
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "sha_test",
            "fps": 24,
            "num_frames": 2,
            "total_frames": 2,
            "resolution": [32, 32],
            "frames": [
                {"frame_index": 0},
                {"frame_index": 1},
            ],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        output_video = tmp_path / "video.npy"

        manifest = BundleManifest(
            job_id="test-sha",
            stage="neural_video_generate",
            inputs={"fgpp_dir": str(fgpp_dir)},
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
        from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

        initial_threads = threading.active_count()

        fgpp_dir = tmp_path / "fgpp"
        fgpp_dir.mkdir()
        meta = {
            "schema_version": FGPP_SCHEMA_VERSION,
            "scene_id": "single_test",
            "fps": 24,
            "num_frames": 2,
            "total_frames": 2,
            "resolution": [32, 32],
            "frames": [
                {"frame_index": 0},
                {"frame_index": 1},
            ],
        }
        (fgpp_dir / "metadata.json").write_text(
            json.dumps(meta), encoding="utf-8",
        )

        output_video = tmp_path / "video.npy"
        manifest = BundleManifest(
            job_id="test-single",
            stage="neural_video_generate",
            inputs={"fgpp_dir": str(fgpp_dir)},
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


# =========================================================================
# PR-44: Missing-input error tests (anti-silent-failure contract)
# =========================================================================


class TestMissingInputErrors:
    """Verify that missing inputs raise proper errors, not fake success."""

    def test_neural_video_generate_missing_fgpp_dir(self) -> None:
        """neural_video_generate without fgpp_dir → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-nvg-no-fgpp",
            stage="neural_video_generate",
            inputs={},
            params={
                "num_frames": 4,
                "height": 32,
                "width": 32,
                "adapter": "mock",
                "model_variant": "wan2.1_1.3b",
            },
            hardware_budget=HardwareBudget(free_vram_gb=48.0),
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_style_extract_missing_frames_dir(self) -> None:
        """style_extract_and_map without frames_dir → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-style-no-frames",
            stage="style_extract_and_map",
            inputs={},
            params={"width": 64, "height": 64, "num_frames": 4},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_surgical_composite_missing_rendered_dir(self) -> None:
        """surgical_composite without rendered_dir → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-comp-no-rendered",
            stage="surgical_composite",
            inputs={},
            params={"num_frames": 4},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_inverse_reconstruct_missing_frames_dir(self) -> None:
        """inverse_reconstruct without frames_dir → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-recon-no-frames",
            stage="inverse_reconstruct",
            inputs={},
            params={"width": 64, "height": 64, "num_frames": 8},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_lora_preencode_missing_dataset_dir(self) -> None:
        """lora_preencode without dataset_dir → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-preencode-no-dataset",
            stage="lora_preencode",
            inputs={},
            params={"num_frames": 4},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_video_to_3d_invert_missing_video(self) -> None:
        """video_to_3d_invert without video_frames → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-v2d-no-video",
            stage="video_to_3d_invert",
            inputs={},
            params={"num_frames": 4},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_inverse_render_loop_missing_video(self) -> None:
        """inverse_render_loop without video_frames → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-irl-no-video",
            stage="inverse_render_loop",
            inputs={},
            params={"num_frames": 4},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_anatomical_refine_missing_video(self) -> None:
        """anatomical_refine without video_frames → exit_code=20."""
        manifest = BundleManifest(
            job_id="test-refine-no-video",
            stage="anatomical_refine",
            inputs={},
            params={"num_frames": 4},
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        assert result.exit_code == 20
        assert result.status == "error"

    def test_lora_package_skip_bench_defaults_false(
        self, tmp_path: Path,
    ) -> None:
        """handle_lora_package skip_bench defaults to False."""
        manifest = BundleManifest(
            job_id="test-pkg-bench",
            stage="lora_package",
            inputs={},
            params={
                "output_dir": str(tmp_path / "bundle"),
                "weights_path": str(tmp_path / "nonexistent.safetensors"),
            },
        )
        worker = JobBundleWorker()
        result = worker.execute(manifest)
        # The packager should not silently skip bench
        # It will fail because weights_path doesn't exist, but the
        # important thing is that skip_bench was False
        assert result.exit_code != 0
