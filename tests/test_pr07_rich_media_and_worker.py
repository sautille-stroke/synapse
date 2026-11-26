"""Tests for PR-TR07: bundle_worker, ConditionScheduler, refine comparison, and bench rich media telemetry."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from synapse import observability as obs
from synapse.adapters.mock import MockVideoAdapter
from synapse.adapters.scheduler import (
    CausalWindowPlan,
    ConditionScheduler,
    DiscontinuityDetector,
)
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.pipeline.bench_runner import BenchPipelineRunner
from synapse.pipeline.bundle_worker import (
    _STAGE_HANDLERS,
    BundleManifest,
    run_worker_cli,
)
from synapse.refine.pipeline import LocalRefinementPipeline
from synapse.refine.types import AnatomicalRegion, RefinementROI


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


def _load_reader() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    from aether_traceboard._types import EventType, StageStatus
    from aether_traceboard.event import (
        unpack_comparison,
        unpack_histogram,
        unpack_image,
        unpack_run_end,
        unpack_scalar,
        unpack_stage_end,
    )
    from aether_traceboard.reader import read_file

    return (
        EventType,
        StageStatus,
        unpack_comparison,
        unpack_histogram,
        unpack_image,
        unpack_run_end,
        (unpack_scalar, unpack_stage_end, read_file),
    )


@contextlib.contextmanager
def _typed_fallback_probe() -> Any:
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


class _DepthMockAdapter(MockVideoAdapter):
    """Mock adapter that also returns predicted depth for residual histogram tests."""

    def generate(self, conditions: Any, config: Any = None) -> Any:
        result = super().generate(conditions, config)
        n_f, h, w = result.frames_srgb_u8.shape[:3]
        result.metadata = dict(result.metadata or {})
        depth_arr = np.linspace(1.5, 3.5, n_f * h * w, dtype=np.float32).reshape(n_f, h, w)
        result.metadata["depth"] = depth_arr
        return result


class TestBundleWorkerTraceboard:
    """Validate SYNAPSE_TRACEBOARD_DIR, JobBundle traceboard config, and worker status codes."""

    def test_worker_cli_records_stage_pass_and_trace_id_from_job_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_obs()
        stage_name = "_test_tr07_ok_stage"
        _STAGE_HANDLERS[stage_name] = lambda m: {"output_paths": {}, "fallback_level": None}
        tb_dir = tmp_path / "worker_tb"
        bundle_dir = tmp_path / "bundle_ok"
        bundle_dir.mkdir()
        manifest_path = bundle_dir / "bundle_manifest.json"
        trace_hex = "0af7651916cd43dd8448eb211c80319c"
        manifest = BundleManifest(
            job_id="job_tr07_ok",
            stage=stage_name,
            traceparent=f"00-{trace_hex}-b7ad6b7169203331-01",
            params={"_traceboard_dir": str(tb_dir)},
        )
        manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

        try:
            if _has_aether_traceboard():
                (
                    EventType,
                    StageStatus,
                    _uc,
                    _uh,
                    _ui,
                    unpack_run_end,
                    (unpack_scalar, unpack_stage_end, read_file),
                ) = _load_reader()
                rc = run_worker_cli(str(bundle_dir), json_mode=False)
                assert rc == 0
                assert obs.get_probe() is None
                files = list(tb_dir.glob("*.tbevt"))
                assert len(files) == 1
                _hdr, recs, stats = read_file(files[0])
                assert stats.crc_errors == 0
                assert unpack_run_end(recs[-1].payload) == "finished"
                se = [r for r in recs if r.event_type == EventType.STAGE_END][0]
                _, meta = unpack_stage_end(se.payload)
                assert meta is not None and meta["status"] == int(StageStatus.PASS)
                exit_vals = [
                    unpack_scalar(r.payload)
                    for r in recs
                    if r.event_type == EventType.SCALAR and r.tag == "worker/exit_code"
                ]
                assert exit_vals == [0.0]
            else:
                with _typed_fallback_probe() as (mock_probe, stages):
                    rc = run_worker_cli(str(bundle_dir), json_mode=False)
                assert rc == 0
                assert stages[0][0] == f"worker/{stage_name}"
                stages[0][1].set_pass.assert_called_once()
                mock_probe.log_scalar.assert_any_call("worker/exit_code", 0.0, step=None)
        finally:
            _STAGE_HANDLERS.pop(stage_name, None)

    def test_worker_cli_env_var_records_error_codes_10_and_20(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reset_obs()
        tb_dir_20 = tmp_path / "worker_env_20"
        monkeypatch.setenv("SYNAPSE_TRACEBOARD_DIR", str(tb_dir_20))

        if _has_aether_traceboard():
            EventType, _ss, _uc, _uh, _ui, unpack_run_end, (unpack_scalar, unpack_stage_end, read_file) = (
                _load_reader()
            )
            rc20 = run_worker_cli(str(tmp_path / "missing_bundle"), json_mode=False)
            assert rc20 == 20
            files20 = list(tb_dir_20.glob("*.tbevt"))
            assert len(files20) == 1
            _hdr, recs20, stats20 = read_file(files20[0])
            assert stats20.crc_errors == 0
            assert unpack_run_end(recs20[-1].payload) == "failed"
            assert any(
                r.tag == "worker/exit_code" and unpack_scalar(r.payload) == 20.0
                for r in recs20
                if r.event_type == EventType.SCALAR
            )

            tb_dir_10 = tmp_path / "worker_env_10"
            monkeypatch.setenv("SYNAPSE_TRACEBOARD_DIR", str(tb_dir_10))
            unknown_bundle = tmp_path / "unknown_stage_bundle"
            unknown_bundle.mkdir()
            (unknown_bundle / "bundle_manifest.json").write_text(
                BundleManifest(job_id="job_bad", stage="no_such_stage").model_dump_json(),
                encoding="utf-8",
            )
            rc10 = run_worker_cli(str(unknown_bundle), json_mode=False)
            assert rc10 == 10
            files10 = list(tb_dir_10.glob("*.tbevt"))
            assert len(files10) == 1
            _hdr10, recs10, stats10 = read_file(files10[0])
            assert stats10.crc_errors == 0
            assert unpack_run_end(recs10[-1].payload) == "failed"
            se10 = [r for r in recs10 if r.event_type == EventType.STAGE_END][0]
            _, meta10 = unpack_stage_end(se10.payload)
            assert meta10 is not None and "exit_10" in meta10["reason"]
        else:
            with patch.object(obs, "init_traceboard", return_value=None):
                rc20 = run_worker_cli(str(tmp_path / "missing_bundle"), json_mode=False)
            assert rc20 == 20

    def test_worker_cli_rejects_manifest_traceboard_dir_escaping_workspace(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _reset_obs()
        ws = (tmp_path / "workspace").resolve()
        ws.mkdir()
        outside_tb = tmp_path.resolve() / "outside_tb"
        bundle_dir = ws / "bundle_escape"
        bundle_dir.mkdir()
        manifest = BundleManifest(
            job_id="job_escape",
            stage="no_such_stage",
            params={"_traceboard_dir": "../outside_tb"},
        )
        (bundle_dir / "bundle_manifest.json").write_text(
            manifest.model_dump_json(), encoding="utf-8"
        )

        rc = run_worker_cli(str(bundle_dir), json_mode=True, workspace_root=str(ws))
        assert rc == 10
        assert not outside_tb.exists()
        assert obs.get_probe() is None
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["exit_code"] == 10
        assert payload["error_type"] == "ContractValidationError"
        assert payload["agent_fix_hint"]

        symlink_tb = ws / "tb_symlink"
        symlink_tb.symlink_to(outside_tb)
        (bundle_dir / "bundle_manifest.json").write_text(
            BundleManifest(
                job_id="job_escape_sym",
                stage="no_such_stage",
                params={"traceboard": {"log_dir": str(symlink_tb)}},
            ).model_dump_json(),
            encoding="utf-8",
        )
        rc_sym = run_worker_cli(str(bundle_dir), json_mode=False, workspace_root=str(ws))
        assert rc_sym == 10
        assert not outside_tb.exists()

    def test_worker_cli_allows_manifest_traceboard_dir_inside_workspace(
        self, tmp_path: Path
    ) -> None:
        _reset_obs()
        stage_name = "_test_tr_sandbox_ok"
        _STAGE_HANDLERS[stage_name] = lambda _m: {"output_paths": {}, "fallback_level": None}
        ws = (tmp_path / "workspace_ok").resolve()
        ws.mkdir()
        bundle_dir = ws / "bundle_ok"
        bundle_dir.mkdir()
        (bundle_dir / "bundle_manifest.json").write_text(
            BundleManifest(
                job_id="job_sandbox_ok",
                stage=stage_name,
                params={"traceboard_dir": "tb_inside"},
            ).model_dump_json(),
            encoding="utf-8",
        )

        try:
            if _has_aether_traceboard():
                rc = run_worker_cli(str(bundle_dir), json_mode=False, workspace_root=str(ws))
                assert rc == 0
                assert len(list((ws / "tb_inside").glob("*.tbevt"))) == 1
            else:
                with _typed_fallback_probe() as (mock_probe, _stages):
                    rc = run_worker_cli(str(bundle_dir), json_mode=False, workspace_root=str(ws))
                assert rc == 0
                mock_probe.log_scalar.assert_any_call("worker/exit_code", 0.0, step=None)
        finally:
            _STAGE_HANDLERS.pop(stage_name, None)
            _reset_obs()


class TestConditionSchedulerTraceboard:
    """Validate per-window ConditionScheduler monotonic step telemetry."""

    def test_scheduler_logs_monotonic_windows_and_shot_cut(self, tmp_path: Path) -> None:
        _reset_obs()
        cam = CameraParams(fx=16.0, fy=16.0, cx=8.0, cy=8.0, width=16, height=16)
        frames: list[FGPPFrame] = []
        n_frames = 140
        for t in range(n_frames):
            depth = np.full((16, 16), 2.5, dtype=np.float32)
            mv = np.zeros((16, 16, 2), dtype=np.float32)
            wn = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (16, 16, 1))
            cid = np.ones((16, 16), dtype=np.uint32)
            if t >= 70:
                mv[:] = 50.0
                wn = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (16, 16, 1))
                cid[:] = 2
            frames.append(
                FGPPFrame(
                    camera=cam,
                    depth=depth,
                    frame_index=t,
                    motion_vector=mv,
                    world_normal=wn,
                    cryptomatte_id=cid,
                )
            )
        seq = FGPPSequence(frames=frames, fps=24.0, scene_id="sched_test")
        scheduler = ConditionScheduler(
            plan=CausalWindowPlan(window_frames=65, stride_frames=48, overlap_frames=17, total_frames=n_frames),
            detector=DiscontinuityDetector(threshold=0.5),
            anchor_interval=48,
        )

        if _has_aether_traceboard():
            EventType, _ss, _uc, _uh, _ui, _ure, (unpack_scalar, _use, read_file) = (
                _load_reader()
            )
            tb_dir = tmp_path / "sched_tb"
            obs.init_traceboard(str(tb_dir), "sched_run")
            try:
                windows = scheduler.schedule(seq)
            finally:
                obs.shutdown_traceboard()
            assert len(windows) >= 2
            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            idx_recs = [
                (r.step, unpack_scalar(r.payload))
                for r in recs
                if r.event_type == EventType.SCALAR and r.tag == "adapter/scheduler/window_idx"
            ]
            assert [s for s, _ in idx_recs] == list(range(len(windows)))
            assert [v for _, v in idx_recs] == [float(i) for i in range(len(windows))]
            reanchor_vals = [
                unpack_scalar(r.payload)
                for r in recs
                if r.event_type == EventType.SCALAR
                and r.tag == "adapter/scheduler/shot_cut_reanchor"
            ]
            assert 1.0 in reanchor_vals
        else:
            with _typed_fallback_probe() as (mock_probe, _stages):
                windows = scheduler.schedule(seq)
            assert len(windows) >= 2
            mock_probe.log_scalar.assert_any_call("adapter/scheduler/window_idx", 0.0, step=0)


class TestRichMediaEventsEndToEnd:
    """Verify IMAGE, COMPARISON, HISTOGRAM, and ARTIFACT events are genuinely produced and bounded."""

    def test_refine_pipeline_emits_bounded_comparison_event(self, tmp_path: Path) -> None:
        _reset_obs()
        frames = np.full((2, 600, 600, 3), 0.35, dtype=np.float32)
        frames[:, 50:550, 50:550, :] = 0.72
        roi = RefinementROI(
            bbox_xyxy=np.array([10.0, 10.0, 590.0, 590.0], dtype=np.float32),
            crop_acescg=np.full((580, 580, 3), 0.55, dtype=np.float32),
            region=AnatomicalRegion.HAND,
            confidence=0.95,
            anatomical_quality_score=0.5,
            frame_index=0,
        )
        pipe = LocalRefinementPipeline(strict=False)

        if _has_aether_traceboard():
            EventType, _ss, unpack_comparison, _uh, unpack_image, _ure, (_us, _use, read_file) = (
                _load_reader()
            )
            tb_dir = tmp_path / "refine_tb"
            obs.init_traceboard(str(tb_dir), "refine_comp")
            try:
                res = pipe.run(frames, [[roi], []])
            finally:
                obs.shutdown_traceboard()
            assert res.num_patches_fused == 1
            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            comp_recs = [r for r in recs if r.event_type == EventType.COMPARISON]
            assert len(comp_recs) == 1
            assert comp_recs[0].tag == "refine/anatomical_patch"
            meta = unpack_comparison(comp_recs[0].payload)
            assert meta["width"] <= 512 and meta["height"] <= 512
            img_recs = [r for r in recs if r.event_type == EventType.IMAGE]
            assert {r.tag for r in img_recs} >= {
                "refine/anatomical_patch/a",
                "refine/anatomical_patch/b",
            }
            w, h, channels, _fmt, pixel_bytes = unpack_image(img_recs[0].payload)
            assert w <= 512 and h <= 512 and channels == 3
            assert int(np.sum(np.frombuffer(pixel_bytes, dtype=np.uint8))) > 0
        else:
            with _typed_fallback_probe() as (mock_probe, _stages):
                res = pipe.run(frames, [[roi], []])
            assert res.num_patches_fused == 1
            mock_probe.log_comparison.assert_called_once()
            img_a = mock_probe.log_comparison.call_args.args[1]
            assert img_a.dtype == np.uint8
            assert max(img_a.shape[:2]) <= 512

    def test_refine_pipeline_strict_gate_failure_logs_telemetry_before_raising(
        self, tmp_path: Path
    ) -> None:
        from synapse.errors import QualityGateError
        from synapse.refine.gates import GateThresholds
        from synapse.refine.hand import HandRefiner

        _reset_obs()
        frames = np.full((2, 128, 128, 3), 0.35, dtype=np.float32)
        frames[:, 16:112, 16:112, :] = 0.72
        roi = RefinementROI(
            bbox_xyxy=np.array([8.0, 8.0, 120.0, 120.0], dtype=np.float32),
            crop_acescg=np.full((112, 112, 3), 0.55, dtype=np.float32),
            region=AnatomicalRegion.HAND,
            confidence=0.95,
            anatomical_quality_score=0.5,
            frame_index=0,
        )
        pipe = LocalRefinementPipeline(
            gate_thresholds=GateThresholds(min_psnr_db=100.0),
            strict=True,
        )
        pipe.hand_refiner = HandRefiner(strict=False)

        if _has_aether_traceboard():
            EventType, _ss, unpack_comparison, _uh, _ui, _ure, (unpack_scalar, _use, read_file) = (
                _load_reader()
            )
            tb_dir = tmp_path / "refine_strict_tb"
            obs.init_traceboard(str(tb_dir), "refine_strict_fail")
            try:
                with pytest.raises(QualityGateError):
                    pipe.run(frames, [[roi], []])
            finally:
                obs.shutdown_traceboard(status="failed")

            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            passed_vals = [
                unpack_scalar(r.payload)
                for r in recs
                if r.event_type == EventType.SCALAR and r.tag == "refine/passed"
            ]
            assert passed_vals == [0.0]
            comp_recs = [
                r
                for r in recs
                if r.event_type == EventType.COMPARISON and r.tag == "refine/anatomical_patch"
            ]
            assert len(comp_recs) == 1
            meta = unpack_comparison(comp_recs[0].payload)
            assert meta["pass"] is False
        else:
            with _typed_fallback_probe() as (mock_probe, _stages):
                with pytest.raises(QualityGateError):
                    pipe.run(frames, [[roi], []])
            mock_probe.log_scalar.assert_any_call("refine/passed", 0.0, step=None)
            mock_probe.log_comparison.assert_called_once()
            assert mock_probe.log_comparison.call_args.kwargs.get("passed") is False

    def test_bench_runner_emits_bounded_images_and_depth_histogram(self, tmp_path: Path) -> None:
        _reset_obs()
        runner = BenchPipelineRunner(
            adapter=_DepthMockAdapter(),
            gate_profile=None,
            log_images=True,
            log_histograms=True,
            media_max_dim=32,
            max_keyframe_images=2,
        )

        if _has_aether_traceboard():
            EventType, _ss, _uc, unpack_histogram, unpack_image, _ure, (_us, _use, read_file) = (
                _load_reader()
            )
            tb_dir = tmp_path / "bench_media_tb"
            obs.init_traceboard(str(tb_dir), "bench_media")
            try:
                reports = runner.run_all(
                    scenarios=["t1_rigid_cube_orbit"],
                    resolution=(64, 64),
                    num_frames=5,
                )
            finally:
                obs.shutdown_traceboard()
            assert len(reports) == 1 and reports[0].error is None
            assert "depth_rmse_m" in reports[0].metrics
            files = list(tb_dir.glob("*.tbevt"))
            assert len(files) == 1
            assert files[0].stat().st_size < 256 * 1024  # compact bounded size
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            imgs = [r for r in recs if r.event_type == EventType.IMAGE]
            assert len(imgs) == 2
            for ir in imgs:
                w, h, _channels, _fmt, pixel_bytes = unpack_image(ir.payload)
                assert w <= 32 and h <= 32
                assert int(np.sum(np.frombuffer(pixel_bytes, dtype=np.uint8))) > 0
            hists = [r for r in recs if r.event_type == EventType.HISTOGRAM]
            assert len(hists) == 1
            assert hists[0].tag == "bench/scenario/t1_rigid_cube_orbit/depth_residual_hist"
            num_bins, edges_bytes, counts_bytes = unpack_histogram(hists[0].payload)
            edges = np.frombuffer(edges_bytes, dtype="<f8")
            counts = np.frombuffer(counts_bytes, dtype="<f8")
            assert len(edges) == num_bins + 1 and len(counts) == num_bins
            assert float(np.sum(counts)) > 0.0
        else:
            with _typed_fallback_probe() as (mock_probe, _stages):
                reports = runner.run_all(
                    scenarios=["t1_rigid_cube_orbit"],
                    resolution=(64, 64),
                    num_frames=5,
                )
            assert len(reports) == 1 and reports[0].error is None
            assert mock_probe.log_image.call_count == 2
            assert mock_probe.log_histogram.call_count == 1

    def test_render_comparison_html_logs_artifact(self, tmp_path: Path) -> None:
        from synapse.bench.report import render_comparison_html

        _reset_obs()
        comparison_data = {
            "overall_winner": "model_a",
            "track_winners": {"track_a": "model_a", "track_b": "model_a", "track_c": "model_a", "track_d": "model_a"},
            "rows": [
                {
                    "model_name": "model_a",
                    "overall_score": 0.91,
                    "track_scores": {"track_a": {"score": 0.9}, "track_b": {"score": 0.92}},
                }
            ],
        }
        out_html = tmp_path / "leaderboard.html"

        if _has_aether_traceboard():
            EventType, _ss, _uc, _uh, _ui, _ure, (_us, _use, read_file) = _load_reader()
            from aether_traceboard.event import unpack_artifact

            tb_dir = tmp_path / "cmp_art_tb"
            obs.init_traceboard(str(tb_dir), "cmp_art")
            try:
                html_str = render_comparison_html(comparison_data, output_path=out_html)
            finally:
                obs.shutdown_traceboard()
            assert out_html.is_file()
            files = list(tb_dir.glob("*.tbevt"))
            _hdr, recs, stats = read_file(files[0])
            assert stats.crc_errors == 0
            arts = [r for r in recs if r.event_type == EventType.ARTIFACT]
            assert any(r.tag == "bench/compare_report" for r in arts)
            art_meta, blob = unpack_artifact(
                [r for r in arts if r.tag == "bench/compare_report"][0].payload
            )
            assert art_meta["mime_type"] == "text/html"
            assert art_meta["size_bytes"] == len(html_str.encode("utf-8"))
            assert blob.decode("utf-8") == html_str
        else:
            with _typed_fallback_probe() as (mock_probe, _stages):
                render_comparison_html(comparison_data, output_path=out_html)
            mock_probe.log_artifact.assert_called_once()
