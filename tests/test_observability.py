"""Comprehensive tests for synapse.observability — Traceboard integration layer.

Coverage matrix:
- get_probe() before/after init, with/without aether-traceboard
- init_traceboard() success, double-init, missing-dependency, exception paths
- shutdown_traceboard() with probe, without probe, with exception
- traceboard_stage() with active probe, inactive probe, no probe
- End-to-end: write .tbevt → read back → verify records
- CLI --traceboard flag integration
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Helper: reset module state ────────────────────────────────────────


def _reset_obs() -> Any:
    """Import and reset synapse.observability module state."""
    import synapse.observability as obs

    obs._probe = None
    obs._initialized = False
    return obs


# ══════════════════════════════════════════════════════════════════════
# get_probe()
# ══════════════════════════════════════════════════════════════════════


class TestGetProbe:
    """Test get_probe() behavior."""

    def test_returns_none_before_init(self) -> None:
        obs = _reset_obs()
        assert obs.get_probe() is None

    def test_returns_none_without_aether_traceboard(self) -> None:
        obs = _reset_obs()
        with patch.object(obs, "_try_import_probe", return_value=None):
            obs.init_traceboard("/tmp/tblog", "test_run")
            assert obs.get_probe() is None

    def test_returns_probe_after_successful_init(self, tmp_path: Path) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)},
            ):
                obs.init_traceboard(str(tmp_path), "test_run")
                assert obs.get_probe() is mock_probe_instance

    def test_returns_none_after_shutdown(self) -> None:
        obs = _reset_obs()
        obs._probe = MagicMock(spec=obs.ProbeProtocol)
        obs._initialized = True
        obs.shutdown_traceboard()
        assert obs.get_probe() is None


# ══════════════════════════════════════════════════════════════════════
# init_traceboard()
# ══════════════════════════════════════════════════════════════════════


class TestInitTraceboard:
    """Test init_traceboard() with mocked aether_traceboard."""

    def test_init_success(self, tmp_path: Path) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)},
            ):
                result = obs.init_traceboard(str(tmp_path), "test_run", spp="64")

        assert result is mock_probe_instance
        assert obs._initialized is True
        mock_probe_instance.begin_run.assert_called_once()
        mock_probe_instance.activate.assert_called_once()
        mock_probe_instance.add_collector.assert_called_once()

    def test_init_no_auto_activate(self, tmp_path: Path) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)},
            ):
                result = obs.init_traceboard(
                    str(tmp_path), "test_run", auto_activate=False
                )

        assert result is mock_probe_instance
        mock_probe_instance.activate.assert_not_called()

    def test_init_custom_engine(self, tmp_path: Path) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)},
            ):
                obs.init_traceboard(
                    str(tmp_path), "run_1", engine="fulgur", backend="vulkan"
                )

        # Verify engine and config kwargs are forwarded
        call_args = mock_probe_instance.begin_run.call_args
        assert call_args[0][0] == "run_1"
        assert call_args[1]["engine"] == "fulgur"
        assert call_args[1]["backend"] == "vulkan"

    def test_double_init_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        obs = _reset_obs()
        obs._initialized = True  # Already initialized
        with caplog.at_level(logging.WARNING):
            obs.init_traceboard("/tmp/tblog", "test_run")
        assert "already initialized" in caplog.text

    def test_double_init_returns_existing_probe(self) -> None:
        obs = _reset_obs()
        existing_probe = MagicMock(spec=obs.ProbeProtocol)
        obs._probe = existing_probe
        obs._initialized = True
        result = obs.init_traceboard("/tmp/tblog", "test_run")
        assert result is existing_probe

    def test_init_missing_dependency_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        obs = _reset_obs()
        with patch.object(obs, "_try_import_probe", return_value=None):
            with caplog.at_level(logging.INFO):
                result = obs.init_traceboard("/tmp/tblog", "test_run")
        assert result is None
        assert obs._initialized is True
        assert "not installed" in caplog.text

    def test_init_exception_during_setup(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_instance.add_collector.side_effect = RuntimeError("disk full")
        mock_probe_cls.return_value = mock_probe_instance

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=MagicMock())},
            ):
                with caplog.at_level(logging.WARNING):
                    result = obs.init_traceboard(str(tmp_path), "test_run")

        assert result is None
        assert obs._initialized is True
        assert "Failed to initialize" in caplog.text

    def test_init_with_path_object(self, tmp_path: Path) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)},
            ):
                result = obs.init_traceboard(tmp_path, "test_run")  # Path, not str

        assert result is mock_probe_instance


# ══════════════════════════════════════════════════════════════════════
# shutdown_traceboard()
# ══════════════════════════════════════════════════════════════════════


class TestShutdown:
    """Test shutdown_traceboard() behavior."""

    def test_shutdown_resets_state(self) -> None:
        obs = _reset_obs()
        obs._probe = MagicMock(spec=obs.ProbeProtocol)
        obs._initialized = True
        obs.shutdown_traceboard()
        assert obs._probe is None
        assert obs._initialized is False

    def test_shutdown_calls_end_run(self) -> None:
        obs = _reset_obs()
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        obs._probe = mock_probe
        obs._initialized = True
        obs.shutdown_traceboard()
        mock_probe.end_run.assert_called_once()

    def test_shutdown_noop_when_no_probe(self) -> None:
        obs = _reset_obs()
        obs.shutdown_traceboard()  # Should not raise
        assert obs._probe is None
        assert obs._initialized is False

    def test_shutdown_handles_end_run_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        obs = _reset_obs()
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        mock_probe.end_run.side_effect = RuntimeError("flush failed")
        obs._probe = mock_probe
        obs._initialized = True

        with caplog.at_level(logging.WARNING):
            obs.shutdown_traceboard()

        assert obs._probe is None
        assert obs._initialized is False
        assert "Error during Traceboard shutdown" in caplog.text

    def test_shutdown_idempotent(self) -> None:
        obs = _reset_obs()
        obs._probe = MagicMock(spec=obs.ProbeProtocol)
        obs._initialized = True
        obs.shutdown_traceboard()
        obs.shutdown_traceboard()  # Second call should be safe
        assert obs._probe is None
        assert obs._initialized is False


# ══════════════════════════════════════════════════════════════════════
# traceboard_stage()
# ══════════════════════════════════════════════════════════════════════


class TestTraceboardStage:
    """Test traceboard_stage() context manager."""

    def test_yields_none_when_no_probe(self) -> None:
        obs = _reset_obs()
        with obs.traceboard_stage("test_stage") as ctx:
            assert ctx is None

    def test_yields_none_when_probe_inactive(self) -> None:
        obs = _reset_obs()
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        mock_probe.active = False
        obs._probe = mock_probe
        with obs.traceboard_stage("test_stage") as ctx:
            assert ctx is None

    def test_yields_stage_context_when_active(self) -> None:
        obs = _reset_obs()
        mock_stage = MagicMock(spec=obs.StageProtocol)
        mock_stage.__enter__ = MagicMock(return_value=mock_stage)
        mock_stage.__exit__ = MagicMock(return_value=False)
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        mock_probe.active = True
        mock_probe.begin_stage.return_value = mock_stage
        obs._probe = mock_probe
        with obs.traceboard_stage("test_stage", key="val") as ctx:
            assert ctx is mock_stage
        mock_probe.begin_stage.assert_called_once_with("test_stage", key="val")

    def test_stage_with_multiple_params(self) -> None:
        obs = _reset_obs()
        mock_stage = MagicMock(spec=obs.StageProtocol)
        mock_stage.__enter__ = MagicMock(return_value=mock_stage)
        mock_stage.__exit__ = MagicMock(return_value=False)
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        mock_probe.active = True
        mock_probe.begin_stage.return_value = mock_stage
        obs._probe = mock_probe
        with obs.traceboard_stage("train", epoch="1", lr="0.001", batch="32") as ctx:
            assert ctx is mock_stage
        mock_probe.begin_stage.assert_called_once_with(
            "train", epoch="1", lr="0.001", batch="32"
        )


# ══════════════════════════════════════════════════════════════════════
# _try_import_probe()
# ══════════════════════════════════════════════════════════════════════


class TestTryImportProbe:
    """Test the lazy import helper."""

    def test_returns_none_when_not_installed(self) -> None:
        import synapse.observability as obs

        with patch.dict("sys.modules", {"aether_traceboard": None}):
            # Force re-evaluation by clearing cached import
            result = obs._try_import_probe()
            # Will return None because importing None raises ImportError
            # Actually sys.modules[key] = None blocks the import
            assert result is None

    def test_returns_probe_class_when_installed(self) -> None:
        import synapse.observability as obs

        mock_module = MagicMock()
        mock_probe_cls = type("Probe", (), {})
        mock_module.Probe = mock_probe_cls

        with patch.dict("sys.modules", {"aether_traceboard": mock_module}):
            result = obs._try_import_probe()
            assert result is mock_probe_cls


# ══════════════════════════════════════════════════════════════════════
# End-to-End Integration (with real aether-traceboard if available)
# ══════════════════════════════════════════════════════════════════════


def _has_aether_traceboard() -> bool:
    """Check if aether-traceboard is importable."""
    try:
        import aether_traceboard  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.external_data
@pytest.mark.skipif(
    not _has_aether_traceboard(),
    reason="aether-traceboard not installed (external sibling package)",
)
class TestEndToEndIntegration:
    """Integration tests using real aether-traceboard SDK."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Init → log events → shutdown → read back .tbevt file."""
        obs = _reset_obs()

        # Initialize
        probe = obs.init_traceboard(
            str(tmp_path), "e2e_test_run", engine="synapse-test", spp="64"
        )
        assert probe is not None
        assert obs.get_probe() is probe
        assert probe.active is True

        # Log some events
        probe.log_scalar("metrics/psnr", 38.7)
        probe.log_scalar("metrics/ssim", 0.998)
        probe.log_text("log/info", "Integration test complete")
        probe.set_step(1)
        probe.log_scalar("metrics/psnr", 39.1)

        # Shutdown
        obs.shutdown_traceboard()
        assert obs.get_probe() is None

        # Verify .tbevt file was created
        tbevt_files = list(tmp_path.glob("*.tbevt"))
        assert len(tbevt_files) == 1, f"Expected 1 .tbevt file, found {len(tbevt_files)}"

        # Read back and verify
        from aether_traceboard.reader import read_events

        records = read_events(tbevt_files[0])
        assert len(records) >= 5  # RUN_START + 3 scalars + 1 text + RUN_END (at least)

        # Verify RUN_START is first
        from aether_traceboard._types import EventType

        assert records[0].event_type == EventType.RUN_START
        assert records[0].tag == "e2e_test_run"

        # Verify RUN_END is last
        assert records[-1].event_type == EventType.RUN_END

        # Verify scalar events exist
        scalar_events = [r for r in records if r.event_type == EventType.SCALAR]
        assert len(scalar_events) >= 3

    def test_stage_context_e2e(self, tmp_path: Path) -> None:
        """Test traceboard_stage() writes real STAGE_BEGIN/END events."""
        obs = _reset_obs()
        probe = obs.init_traceboard(str(tmp_path), "stage_e2e")
        assert probe is not None

        with obs.traceboard_stage("validation", split="val") as ctx:
            assert ctx is not None
            ctx.log_metric("accuracy", 0.95)
            ctx.set_pass()

        obs.shutdown_traceboard()

        # Read back
        from aether_traceboard._types import EventType
        from aether_traceboard.reader import read_events

        records = read_events(list(tmp_path.glob("*.tbevt"))[0])
        stage_begins = [r for r in records if r.event_type == EventType.STAGE_BEGIN]
        stage_ends = [r for r in records if r.event_type == EventType.STAGE_END]
        assert len(stage_begins) == 1
        assert len(stage_ends) == 1
        assert stage_begins[0].tag == "validation"

    def test_reinit_after_shutdown(self, tmp_path: Path) -> None:
        """After shutdown + reset, can init again."""
        obs = _reset_obs()

        # First lifecycle
        probe1 = obs.init_traceboard(str(tmp_path / "run1"), "first_run")
        assert probe1 is not None
        obs.shutdown_traceboard()

        # Second lifecycle
        probe2 = obs.init_traceboard(str(tmp_path / "run2"), "second_run")
        assert probe2 is not None
        obs.shutdown_traceboard()

        # Both runs should have created .tbevt files
        run1_files = list((tmp_path / "run1").glob("*.tbevt"))
        run2_files = list((tmp_path / "run2").glob("*.tbevt"))
        assert len(run1_files) == 1
        assert len(run2_files) == 1

    def test_crc_integrity_of_generated_file(self, tmp_path: Path) -> None:
        """Every record in the generated .tbevt should have valid CRC-32."""
        obs = _reset_obs()
        probe = obs.init_traceboard(str(tmp_path), "crc_test")
        assert probe is not None

        # Write diverse events
        probe.log_scalar("a", 1.0)
        probe.log_text("b", "hello")
        probe.log_scalar("c", 42.0)
        obs.shutdown_traceboard()

        # Raw CRC verification
        from aether_traceboard._crc32 import compute_crc32
        from aether_traceboard.event import (
            _FILE_HEADER_FMT,
            _RECORD_HEADER_FMT,
            unpack_record_header,
        )

        data = list(tmp_path.glob("*.tbevt"))[0].read_bytes()
        pos = _FILE_HEADER_FMT.size
        idx = 0
        while pos + _RECORD_HEADER_FMT.size <= len(data):
            payload_len, stored_crc, _, _, _tag_len, _ = unpack_record_header(
                data[pos : pos + _RECORD_HEADER_FMT.size]
            )
            body_start = pos + _RECORD_HEADER_FMT.size
            body_end = body_start + payload_len
            if body_end > len(data):
                break
            body = data[body_start:body_end]
            actual_crc = compute_crc32(body)
            assert actual_crc == stored_crc, (
                f"Record[{idx}] CRC mismatch: stored=0x{stored_crc:08X}, "
                f"computed=0x{actual_crc:08X}"
            )
            pos = body_end
            idx += 1

        assert idx >= 4  # RUN_START + scalar + text + scalar + RUN_END


# ══════════════════════════════════════════════════════════════════════
# Lazy Import Isolation (synapse.__init__)
# ══════════════════════════════════════════════════════════════════════


class TestLazyImportIsolation:
    """Verify that `import synapse` does NOT trigger traceboard init."""

    def test_import_synapse_does_not_init_traceboard(self) -> None:
        """PEP 562 compliance: import synapse must be side-effect-free."""
        obs = _reset_obs()
        # After importing synapse, _initialized should still be False
        import synapse  # noqa: F401

        assert obs._initialized is False
        assert obs._probe is None

    def test_accessing_observability_submodule_does_not_init(self) -> None:
        """Accessing synapse.observability should not auto-init the probe."""
        obs = _reset_obs()
        from synapse import observability  # noqa: F401

        assert obs._initialized is False
        assert obs._probe is None


# ══════════════════════════════════════════════════════════════════════
# CLI --traceboard Integration
# ══════════════════════════════════════════════════════════════════════


class TestCliTraceboardFlag:
    """Test CLI argument parser integration."""

    def test_add_traceboard_args(self) -> None:
        """add_traceboard_args() adds --traceboard to parser."""
        import argparse

        from synapse.cli.common import add_traceboard_args

        parser = argparse.ArgumentParser()
        add_traceboard_args(parser)
        args = parser.parse_args(["--traceboard", "/tmp/tblog"])
        assert args.traceboard == "/tmp/tblog"

    def test_traceboard_default_is_none(self) -> None:
        """--traceboard defaults to None when not specified."""
        import argparse

        from synapse.cli.common import add_traceboard_args

        parser = argparse.ArgumentParser()
        add_traceboard_args(parser)
        args = parser.parse_args([])
        assert args.traceboard is None


class TestHardenedObservabilityAndCliRouting:
    """Tests added by PR-TR04 for locking, atomic rollback, CLI routing, and bench failure."""

    def test_init_exception_leaves_no_partial_probe(self, tmp_path: Path) -> None:
        obs = _reset_obs()
        mock_probe_cls = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_instance = MagicMock(spec=obs.ProbeProtocol)
        mock_probe_instance.begin_run.side_effect = RuntimeError("begin_run exploded")
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector = MagicMock()

        with (
            patch.object(obs, "_try_import_probe", return_value=mock_probe_cls),
            patch.dict(
                "sys.modules",
                {"aether_traceboard": MagicMock(FileCollector=MagicMock(return_value=mock_collector))},
            ),
        ):
            result = obs.init_traceboard(str(tmp_path), "broken_run", trace_id="tr_123")

        assert result is None
        assert obs.get_probe() is None
        assert obs._initialized is True
        mock_collector.on_run_close.assert_called_once()

    def test_traceboard_stage_isolates_stage_errors_and_preserves_caller_exc(self) -> None:
        obs = _reset_obs()
        mock_stage = MagicMock(spec=obs.StageProtocol)
        mock_stage.__enter__ = MagicMock(return_value=mock_stage)
        mock_stage.__exit__ = MagicMock(side_effect=RuntimeError("stage exit error"))
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        mock_probe.active = True
        mock_probe.begin_stage.return_value = mock_stage
        obs._probe = mock_probe

        with pytest.raises(ValueError, match="domain failure"):
            with obs.traceboard_stage("failing_stage") as st:
                assert st is mock_stage
                raise ValueError("domain failure")

    def test_safe_log_scalar_ignores_nan_and_inf(self) -> None:
        obs = _reset_obs()
        mock_probe = MagicMock(spec=obs.ProbeProtocol)
        mock_probe.active = True
        obs._probe = mock_probe

        obs.safe_log_scalar("x/nan", float("nan"))
        obs.safe_log_scalar("x/inf", float("inf"))
        obs.safe_log_scalar("x/ok", 1.25, step=3)

        mock_probe.log_scalar.assert_called_once_with("x/ok", 1.25, step=3)

    def test_cli_traceboard_subcommand_placement_and_no_command_leak(self, tmp_path: Path) -> None:
        from synapse.cli import main_bench, main_synapse

        obs = _reset_obs()
        tb_empty = tmp_path / "empty_cmd"
        assert main_synapse(["--traceboard", str(tb_empty)]) == 0
        assert not tb_empty.exists()
        assert obs.get_probe() is None

        tb_after = tmp_path / "after_subcmd"
        if _has_aether_traceboard():
            assert main_synapse(["bench", "list", "--json", "--traceboard", str(tb_after)]) == 0
            assert tb_after.exists()
            assert len(list(tb_after.glob("synapse_bench_list.*.tbevt"))) == 1
        else:
            with (
                patch.object(obs, "init_traceboard") as m_init,
                patch.object(obs, "shutdown_traceboard") as m_shut,
            ):
                assert main_synapse(["bench", "list", "--json", "--traceboard", str(tb_after)]) == 0
                m_init.assert_called_once()
                m_shut.assert_called_once_with(status="finished")
        assert obs.get_probe() is None

        tb_bench_bin = tmp_path / "synapse_bench_bin"
        if _has_aether_traceboard():
            assert main_bench(["list", "--json", "--traceboard", str(tb_bench_bin)]) == 0
            assert len(list(tb_bench_bin.glob("synapse_bench_list.*.tbevt"))) == 1
        else:
            with (
                patch.object(obs, "init_traceboard") as m_init,
                patch.object(obs, "shutdown_traceboard") as m_shut,
            ):
                assert main_bench(["list", "--json", "--traceboard", str(tb_bench_bin)]) == 0
                m_init.assert_called_once()
                m_shut.assert_called_once_with(status="finished")
        assert obs.get_probe() is None

    def test_bench_runner_error_calls_set_fail_reason_cleanly(self, tmp_path: Path) -> None:
        from synapse.adapters.mock import MockVideoAdapter
        from synapse.pipeline.bench_runner import BenchPipelineRunner

        obs = _reset_obs()
        tb_dir = tmp_path / "bench_err"
        obs.init_traceboard(str(tb_dir), "bench_err_run")
        runner = BenchPipelineRunner(adapter=MockVideoAdapter(), gate_profile="T1")
        try:
            reports = runner.run_all(scenarios=["nonexistent_scenario_xyz"])
        finally:
            obs.shutdown_traceboard(status="failed")

        assert len(reports) == 1
        assert reports[0].error is not None
        files = list(tb_dir.glob("*.tbevt"))
        if _has_aether_traceboard():
            assert len(files) == 1
            from aether_traceboard._types import EventType, StageStatus
            from aether_traceboard.event import unpack_run_end, unpack_stage_end
            from aether_traceboard.reader import read_events

            recs = read_events(files[0])
            stage_ends = [r for r in recs if r.event_type == EventType.STAGE_END]
            assert len(stage_ends) == 2
            for se in stage_ends:
                _, meta = unpack_stage_end(se.payload)
                assert meta is not None
                assert meta["status"] == int(StageStatus.FAIL)
                assert "reason" in meta
            run_ends = [r for r in recs if r.event_type == EventType.RUN_END]
            assert unpack_run_end(run_ends[-1].payload) == "failed"
        else:
            assert files == []
            assert obs.get_probe() is None
