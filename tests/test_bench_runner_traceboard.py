"""End-to-end Traceboard tests for BenchPipelineRunner, CLI matrix, CPU verifier, and Colab runner."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from synapse import observability as obs
from synapse.adapters.mock import MockVideoAdapter
from synapse.pipeline.bench_runner import BenchPipelineRunner


def _reset_obs() -> Any:
    obs.shutdown_traceboard()
    obs._probe = None
    obs._initialized = False
    return obs


def _has_aether_traceboard() -> bool:
    try:
        import aether_traceboard  # noqa: F401

        return True
    except ImportError:
        return False


def _load_traceboard_reader() -> tuple[Any, Any, Any, Any, Any]:
    from aether_traceboard._types import EventType, StageStatus
    from aether_traceboard.event import unpack_run_end, unpack_stage_end
    from aether_traceboard.reader import read_file

    return EventType, StageStatus, unpack_run_end, unpack_stage_end, read_file


@contextlib.contextmanager
def _typed_fallback_probe() -> Any:
    """Attach typed ProbeProtocol / StageProtocol mocks when aether-traceboard is absent."""
    stages: list[tuple[str, MagicMock]] = []
    probe = MagicMock(spec=obs.ProbeProtocol)
    probe.active = True

    def _begin_stage(name: str, **_params: str) -> MagicMock:
        st = MagicMock(spec=obs.StageProtocol)
        st.__enter__ = MagicMock(return_value=st)
        st.__exit__ = MagicMock(return_value=False)
        stages.append((name, st))
        return st

    probe.begin_stage.side_effect = _begin_stage
    with obs._state_lock:
        obs._probe = probe
        obs._initialized = True
    try:
        yield probe, stages
    finally:
        obs._probe = None
        obs._initialized = False


class _FailingGateMockAdapter(MockVideoAdapter):
    """Deterministic adapter outputting dark zeroed frames that fail quality gates."""

    def generate(self, conditions: Any, config: Any = None) -> Any:
        result = super().generate(conditions, config)
        result.frames_srgb_u8 = np.zeros_like(result.frames_srgb_u8)
        return result


class _RaisingAdapter(MockVideoAdapter):
    """Adapter that raises during generation to test STAGE_END(FAIL)."""

    def generate(self, conditions: Any, config: Any = None) -> Any:
        raise RuntimeError("simulated adapter generation failure")


class TestBenchRunnerTraceboardE2E:
    """Gate TB-7: Verify PASS, gate failure, and runtime exception states end-to-end."""

    def test_run_all_pass_emits_valid_tbevt(self, tmp_path: Path) -> None:
        _reset_obs()
        tb_dir = tmp_path / "bench_pass"
        runner = BenchPipelineRunner(adapter=MockVideoAdapter(), gate_profile=None)

        if _has_aether_traceboard():
            EventType, StageStatus, unpack_run_end, unpack_stage_end, read_file = (
                _load_traceboard_reader()
            )
            probe = obs.init_traceboard(str(tb_dir), "bench_pass_run", engine="synapse-bench")
            assert probe is not None
            try:
                reports = runner.run_all(
                    scenarios=["t1_rigid_cube_orbit"],
                    resolution=(64, 64),
                    num_frames=5,
                )
            finally:
                obs.shutdown_traceboard(status="finished")

            assert len(reports) == 1
            assert reports[0].error is None
            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            assert not stats.truncated_tail
            assert recs[0].event_type == EventType.RUN_START
            assert recs[-1].event_type == EventType.RUN_END
            assert unpack_run_end(recs[-1].payload) == "finished"

            stage_ends = [r for r in recs if r.event_type == EventType.STAGE_END]
            assert len(stage_ends) == 2
            for se in stage_ends:
                _, meta = unpack_stage_end(se.payload)
                assert meta is not None
                assert meta["status"] == int(StageStatus.PASS)

            scalar_tags = {r.tag for r in recs if r.event_type == EventType.SCALAR}
            assert "bench/scenario/t1_rigid_cube_orbit/passed" in scalar_tags
            assert "bench/scenario/t1_rigid_cube_orbit/num_frames" in scalar_tags
            assert any(
                t.startswith("bench/scenario/t1_rigid_cube_orbit/metrics/")
                for t in scalar_tags
            )
            assert "bench/summary/total_scenarios" in scalar_tags
            assert "bench/pillar/roundtrip" in scalar_tags
        else:
            with _typed_fallback_probe() as (mock_probe, stages):
                reports = runner.run_all(
                    scenarios=["t1_rigid_cube_orbit"],
                    resolution=(64, 64),
                    num_frames=5,
                )
            assert len(reports) == 1
            assert reports[0].error is None
            assert [name for name, _ in stages] == ["bench/run_all", "bench/t1_rigid_cube_orbit"]
            stages[1][1].set_pass.assert_called_once()
            logged_tags = {c.args[0] for c in mock_probe.log_scalar.call_args_list}
            assert "bench/scenario/t1_rigid_cube_orbit/passed" in logged_tags
            assert "bench/pillar/roundtrip" in logged_tags

    def test_gate_failure_emits_stage_fail_with_gate_failed_reason(self, tmp_path: Path) -> None:
        _reset_obs()
        tb_dir = tmp_path / "bench_gate_fail"
        runner = BenchPipelineRunner(
            adapter=_FailingGateMockAdapter(),
            gate_profile="track_b_video",
        )

        if _has_aether_traceboard():
            EventType, StageStatus, unpack_run_end, unpack_stage_end, read_file = (
                _load_traceboard_reader()
            )
            probe = obs.init_traceboard(str(tb_dir), "bench_gate_fail", engine="synapse-bench")
            assert probe is not None
            try:
                report = runner.run_scenario(
                    scenario_name="t1_rigid_cube_orbit",
                    resolution=(64, 64),
                    num_frames=5,
                )
            finally:
                obs.shutdown_traceboard(status="failed")

            assert report.error is None
            assert report.gate_verdict is not None
            assert report.gate_verdict.passed is False

            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            stage_ends = [r for r in recs if r.event_type == EventType.STAGE_END]
            assert len(stage_ends) == 1
            _, meta = unpack_stage_end(stage_ends[0].payload)
            assert meta is not None
            assert meta["status"] == int(StageStatus.FAIL)
            assert meta["reason"] == "gate_failed"
            assert unpack_run_end(recs[-1].payload) == "failed"
        else:
            with _typed_fallback_probe() as (_mock_probe, stages):
                report = runner.run_scenario(
                    scenario_name="t1_rigid_cube_orbit",
                    resolution=(64, 64),
                    num_frames=5,
                )
            assert report.error is None
            assert report.gate_verdict is not None
            assert report.gate_verdict.passed is False
            stages[0][1].set_fail.assert_called_once_with("gate_failed")

    def test_runtime_exception_in_adapter_emits_stage_fail_with_reason(
        self, tmp_path: Path
    ) -> None:
        _reset_obs()
        tb_dir = tmp_path / "bench_exc"
        runner = BenchPipelineRunner(
            adapter=_RaisingAdapter(),
            gate_profile="track_b_video",
        )

        if _has_aether_traceboard():
            EventType, StageStatus, unpack_run_end, unpack_stage_end, read_file = (
                _load_traceboard_reader()
            )
            obs.init_traceboard(str(tb_dir), "bench_exc", engine="synapse-bench")
            try:
                report = runner.run_scenario(
                    scenario_name="t1_rigid_cube_orbit",
                    resolution=(64, 64),
                    num_frames=5,
                )
            finally:
                obs.shutdown_traceboard(status="failed")

            assert report.error is not None
            assert "simulated adapter generation failure" in report.error

            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            stage_ends = [r for r in recs if r.event_type == EventType.STAGE_END]
            assert len(stage_ends) == 1
            _, meta = unpack_stage_end(stage_ends[0].payload)
            assert meta is not None
            assert meta["status"] == int(StageStatus.FAIL)
            assert "simulated adapter generation failure" in meta["reason"]
            assert unpack_run_end(recs[-1].payload) == "failed"
        else:
            with _typed_fallback_probe() as (_mock_probe, stages):
                report = runner.run_scenario(
                    scenario_name="t1_rigid_cube_orbit",
                    resolution=(64, 64),
                    num_frames=5,
                )
            assert report.error is not None
            assert "simulated adapter generation failure" in report.error
            stages[0][1].set_fail.assert_called_once()
            assert "simulated adapter generation failure" in stages[0][1].set_fail.call_args.args[0]


class TestCliTraceboardMatrixE2E:
    """Gate TB-6: Validate CLI --traceboard across flag placements and subcommands."""

    def test_cli_traceboard_full_matrix(self, tmp_path: Path) -> None:
        from synapse.cli import main_bench, main_synapse

        _reset_obs()
        invocations: list[tuple[Path, int, str, int]] = [
            (
                tmp_path / "m1",
                main_synapse(["--traceboard", str(tmp_path / "m1"), "bench", "list", "--json"]),
                "synapse_bench_list.",
                0,
            ),
            (
                tmp_path / "m2",
                main_synapse(["bench", "--traceboard", str(tmp_path / "m2"), "list", "--json"]),
                "synapse_bench_list.",
                0,
            ),
            (
                tmp_path / "m3",
                main_bench(["--traceboard", str(tmp_path / "m3"), "list", "--json"]),
                "synapse_bench_list.",
                0,
            ),
            (
                tmp_path / "m4",
                main_synapse(
                    [
                        "bench",
                        "run",
                        "--scenario",
                        "t1_rigid_cube_orbit",
                        "--adapter",
                        "mock",
                        "--resolution",
                        "64",
                        "--json",
                        "--traceboard",
                        str(tmp_path / "m4"),
                    ]
                ),
                "synapse_bench_run.",
                1,
            ),
        ]

        if _has_aether_traceboard():
            EventType, StageStatus, unpack_run_end, unpack_stage_end, read_file = (
                _load_traceboard_reader()
            )
            for out_dir, rc, prefix, min_stages in invocations:
                assert rc == 0
                files = sorted(out_dir.glob("*.tbevt"))
                assert len(files) == 1
                assert files[0].name.startswith(prefix)
                _hdr, recs, stats = read_file(files[0])
                assert stats.crc_errors == 0
                assert not stats.truncated_tail
                assert recs[0].event_type == EventType.RUN_START
                assert recs[-1].event_type == EventType.RUN_END
                assert unpack_run_end(recs[-1].payload) == "finished"
                stage_ends = [r for r in recs if r.event_type == EventType.STAGE_END]
                assert len(stage_ends) >= min_stages
                for se in stage_ends:
                    _, meta = unpack_stage_end(se.payload)
                    assert meta is not None
                    assert meta["status"] == int(StageStatus.PASS)
        else:
            for out_dir, rc, _prefix, _min_stages in invocations:
                assert rc == 0
                assert not list(out_dir.glob("*.tbevt"))
            assert obs.get_probe() is None


class TestRunnerScriptsTraceboard:
    """Gate TB-9: Validate run_cpu_verification finally flush & colab NVMe staging sync."""

    def test_cpu_verification_early_exit_20_flushes_run_end(self, tmp_path: Path) -> None:
        from scripts import run_cpu_verification

        _reset_obs()
        empty_ds = tmp_path / "empty_datasets"
        empty_ds.mkdir()
        tb_dir = tmp_path / "cpu_tb"
        report_path = tmp_path / "report.json"

        argv = [
            "run_cpu_verification.py",
            "--dataset-root",
            str(empty_ds),
            "--report-out",
            str(report_path),
            "--traceboard",
            str(tb_dir),
        ]
        if _has_aether_traceboard():
            EventType, _StageStatus, unpack_run_end, _unpack_stage_end, read_file = (
                _load_traceboard_reader()
            )
            with patch.object(sys, "argv", argv):
                rc = run_cpu_verification.main()

            assert rc == 20
            assert obs.get_probe() is None
            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            assert recs[0].event_type == EventType.RUN_START
            assert recs[-1].event_type == EventType.RUN_END
            assert unpack_run_end(recs[-1].payload) == "failed"
        else:
            with (
                patch.object(obs, "init_traceboard", return_value=MagicMock(spec=obs.ProbeProtocol)),
                patch.object(obs, "shutdown_traceboard") as m_shut,
                patch.object(sys, "argv", argv),
            ):
                rc = run_cpu_verification.main()
            assert rc == 20
            m_shut.assert_called_once_with(status="failed")

    def test_colab_runner_stages_nvme_and_atomically_syncs_drive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts import colab_runner

        _reset_obs()
        nvme_tmp = tmp_path / "nvme_tmp"
        nvme_tmp.mkdir()
        monkeypatch.setenv("TMPDIR", str(nvme_tmp))

        drive_job = (
            tmp_path
            / "drive"
            / "MyDrive"
            / "dervgen"
            / "work"
            / "tester"
            / "synapse"
            / "2026-11-26"
            / "job_tb_01"
        )
        observed_staging: list[Path | None] = []
        observed_drive: list[Path | None] = []

        def _fake_suite(
            dataset_dir: Path,
            checkpoint_dir: Path,
            experiments: list[str],
            mode: str = "auto",
            video_path: Path | None = None,
            tb_staging_dir: Path | None = None,
            tb_drive_dir: Path | None = None,
        ) -> dict[str, Any]:
            observed_staging.append(tb_staging_dir)
            observed_drive.append(tb_drive_dir)
            with obs.traceboard_stage("colab/e1", mode=mode) as st:
                obs.safe_log_scalar("colab/e1/psnr_db", 34.5, step=0)
                if st is not None:
                    st.set_pass()
            colab_runner._sync_traceboard_to_drive(tb_staging_dir, tb_drive_dir)
            # Active write target must stay off FUSE Drive path
            assert tb_staging_dir is not None
            assert "/drive/" not in str(tb_staging_dir).replace("\\", "/")
            assert "MyDrive" not in str(tb_staging_dir)
            assert tb_drive_dir == drive_job / "traceboard"
            return {"mode": mode, "completed": {"e1": {"status": "PASS", "psnr_db": 34.5}}}

        argv = [
            "colab_runner.py",
            "--bundle",
            str(tmp_path / "nonexistent_bundle.tar.gz"),
            "--dataset-dir",
            str(tmp_path / "ds"),
            "--job-dir",
            str(drive_job),
            "--user",
            "tester",
            "--job-id",
            "job_tb_01",
            "--exp",
            "e1",
            "--mode",
            "mock",
            "--traceboard",
        ]
        with (
            patch.object(colab_runner, "run_colab_suite", side_effect=_fake_suite),
            patch.object(sys, "argv", argv),
        ):
            rc = colab_runner.main()

        assert rc == 0
        assert len(observed_staging) == 1
        job_meta = json.loads((drive_job / "job.json").read_text(encoding="utf-8"))
        assert job_meta["status"] == "COMPLETED"

        if _has_aether_traceboard():
            EventType, StageStatus, unpack_run_end, unpack_stage_end, read_file = (
                _load_traceboard_reader()
            )
            drive_tb = drive_job / "traceboard"
            assert drive_tb.is_dir()
            assert list(drive_tb.glob("*.tmp")) == []
            tbevt_files = list(drive_tb.glob("*.tbevt"))
            assert len(tbevt_files) == 1

            _hdr, recs, stats = read_file(tbevt_files[0])
            assert stats.crc_errors == 0
            assert not stats.truncated_tail
            assert recs[0].event_type == EventType.RUN_START
            assert recs[-1].event_type == EventType.RUN_END
            assert unpack_run_end(recs[-1].payload) == "finished"
            stage_ends = [r for r in recs if r.event_type == EventType.STAGE_END]
            assert len(stage_ends) == 1
            _, meta = unpack_stage_end(stage_ends[0].payload)
            assert meta is not None
            assert meta["status"] == int(StageStatus.PASS)
