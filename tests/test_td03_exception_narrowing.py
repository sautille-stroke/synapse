"""Regression tests for PR-TD03: narrowed exception handling and observable warnings."""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from synapse.errors import ContractValidationError
from synapse.pipeline.bundle_worker import BundleManifest
from synapse.refine.pipeline import LocalRefinementPipeline
from synapse.refine.types import AnatomicalRegion, RefinementROI
from synapse.training.feature_store import iter_feature_shards


def test_bundle_worker_rejects_malformed_scene_spec_and_output_sink() -> None:
    """Invalid SceneSpec or OutputSink typed fields must raise ContractValidationError."""
    bad_scene_bundle = {
        "job_id": "job-td03-bad-scene",
        "closure_type": "neural_video",
        "scene_spec": {"resolution": "not-a-list"},
    }
    with pytest.raises(ContractValidationError, match="Invalid scene_spec"):
        BundleManifest.from_rfc0039_job_bundle(bad_scene_bundle)

    bad_sink_bundle = {
        "job_id": "job-td03-bad-sink",
        "closure_type": "neural_video",
        "output_sink": {"channels": 12345},
    }
    with pytest.raises(ContractValidationError, match="Invalid output_sink"):
        BundleManifest.from_rfc0039_job_bundle(bad_sink_bundle)


def test_bundle_worker_logs_malformed_optional_engine_and_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed optional EngineSpec/HardwareBudget emits warnings rather than silent pass."""
    bundle = {
        "job_id": "job-td03-opt-warn",
        "closure_type": "neural_video",
        "engine": {"name": 123},
        "hardware_budget": {"free_vram_gb": "not-a-float"},
    }
    with caplog.at_level(logging.WARNING, logger="synapse.pipeline"):
        manifest = BundleManifest.from_rfc0039_job_bundle(bundle)
    assert manifest.job_id == "job-td03-opt-warn"
    assert "EngineSpec" in caplog.text
    assert "HardwareBudget" in caplog.text


def test_feature_store_rejects_corrupt_meta_json(tmp_path: Path) -> None:
    """Corrupt meta.json in a feature shard must raise ContractValidationError fail-closed."""
    shard_path = tmp_path / "shard-0000.tar"
    lat_buf = io.BytesIO()
    np.save(lat_buf, np.zeros((4, 8, 8), dtype=np.float32))
    prm_buf = io.BytesIO()
    np.save(prm_buf, np.zeros((16,), dtype=np.float32))

    with tarfile.open(shard_path, "w") as tf:
        for name, data in (
            ("sample_0000.latent.npy", lat_buf.getvalue()),
            ("sample_0000.prompt.npy", prm_buf.getvalue()),
            ("sample_0000.meta.json", b"{corrupt-json"),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    with pytest.raises(ContractValidationError, match="Corrupt meta.json"):
        list(iter_feature_shards([shard_path]))


def test_refine_foundary_failure_raises_when_strict_and_warns_otherwise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Foundary inference errors must not be silently swallowed in strict mode and must warn otherwise."""
    frames = np.full((1, 64, 64, 3), 0.5, dtype=np.float32)
    roi = RefinementROI(
        bbox_xyxy=np.array([16.0, 16.0, 48.0, 48.0], dtype=np.float32),
        crop_acescg=np.full((32, 32, 3), 0.5, dtype=np.float32),
        region=AnatomicalRegion.HAND,
        confidence=0.9,
        anatomical_quality_score=0.4,
        frame_index=0,
    )

    class _FailingFoundary:
        @staticmethod
        def infer(service: str, inputs: dict[str, Any] | None = None) -> Any:
            _ = (service, inputs)
            raise RuntimeError("foundary transport failure")

    # Non-strict logs warning
    pipe_non_strict = LocalRefinementPipeline(strict=False)
    with caplog.at_level(logging.WARNING, logger="synapse.refine.pipeline"):
        pipe_non_strict.run(frames, [[roi]], foundary_client=_FailingFoundary())
    assert "Foundary 'flux' refinement candidate failed" in caplog.text

    # Strict re-raises immediately when priors/refiner succeed up to Foundary dispatch
    from synapse.refine.hand import HandRefiner

    pipe_strict = LocalRefinementPipeline(strict=True)
    pipe_strict.hand_refiner = HandRefiner(strict=False)
    with pytest.raises(RuntimeError, match="foundary transport failure"):
        pipe_strict.run(
            frames,
            [[roi]],
            foundary_client=_FailingFoundary(),
        )


def test_bench_runner_logs_corrupt_npy_frame(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt .npy frame in FGPP dir logs a warning instead of silent continue."""
    from synapse.pipeline.bench_runner import _try_load_exr_sequence

    fgpp_dir = tmp_path / "fgpp"
    fgpp_dir.mkdir()
    (fgpp_dir / "beauty_0000.npy").write_bytes(b"not-a-valid-npy")
    with caplog.at_level(logging.WARNING, logger="synapse.pipeline.bench_runner"):
        res = _try_load_exr_sequence(fgpp_dir, "beauty")
    assert res is None
    assert "Skipping unreadable FGPP frame" in caplog.text


def test_cli_bench_logs_fallback_when_scenario_unresolved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing scenario asset resolution during bundle export logs a warning before cube fallback."""
    import argparse
    from unittest import mock

    from synapse.cli._bench import _cmd_bench_run
    from synapse.pipeline.bench_runner import ScenarioRunReport

    fake_report = ScenarioRunReport(
        scenario_name="nonexistent_scenario_td03",
        adapter_id="mock",
        metrics={"psnr_db": 35.0},
        num_frames=4,
        resolution=(64, 64),
        seed=0,
    )
    args = argparse.Namespace(
        adapter="mock",
        profile=None,
        tb_media=False,
        scenario=None,
        resolution=64,
        seed=0,
        export_bundle=str(tmp_path / "export"),
        json=True,
        gate=False,
    )
    class _DummyAdapter:
        adapter_id = "mock"

    with (
        mock.patch("synapse.adapters.MockVideoAdapter", return_value=_DummyAdapter()),
        mock.patch("synapse.pipeline.bench_runner.BenchPipelineRunner") as runner_cls,
        caplog.at_level(logging.WARNING, logger="synapse.cli._bench"),
    ):
        runner_cls.return_value.run_all.return_value = [fake_report]
        _cmd_bench_run(args)

    assert "falling back to 'cube'" in caplog.text
