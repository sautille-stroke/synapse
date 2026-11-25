"""Tests for synapse.observability — Traceboard integration layer."""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestGetProbe:
    """Test get_probe() behavior."""

    def test_returns_none_before_init(self) -> None:
        # Reset state
        import synapse.observability as obs
        from synapse.observability import get_probe
        obs._probe = None
        obs._initialized = False
        assert get_probe() is None

    def test_returns_none_without_aether_traceboard(self) -> None:
        import synapse.observability as obs
        obs._probe = None
        obs._initialized = False
        with patch.object(obs, "_try_import_probe", return_value=None):
            result = obs.init_traceboard("/tmp/tblog", "test_run")
            assert result is None
            assert obs.get_probe() is None

    def test_double_init_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        import synapse.observability as obs
        obs._probe = None
        obs._initialized = True  # Already initialized
        with caplog.at_level(logging.WARNING):
            obs.init_traceboard("/tmp/tblog", "test_run")
        assert "already initialized" in caplog.text


class TestShutdown:
    """Test shutdown_traceboard() behavior."""

    def test_shutdown_resets_state(self) -> None:
        import synapse.observability as obs
        obs._probe = MagicMock()
        obs._initialized = True
        obs.shutdown_traceboard()
        assert obs._probe is None
        assert obs._initialized is False

    def test_shutdown_calls_end_run(self) -> None:
        import synapse.observability as obs
        mock_probe = MagicMock()
        obs._probe = mock_probe
        obs._initialized = True
        obs.shutdown_traceboard()
        mock_probe.end_run.assert_called_once()

    def test_shutdown_noop_when_no_probe(self) -> None:
        import synapse.observability as obs
        obs._probe = None
        obs._initialized = False
        obs.shutdown_traceboard()  # Should not raise


class TestTraceboardStage:
    """Test traceboard_stage() context manager."""

    def test_yields_none_when_no_probe(self) -> None:
        import synapse.observability as obs
        obs._probe = None
        with obs.traceboard_stage("test_stage") as ctx:
            assert ctx is None

    def test_yields_none_when_probe_inactive(self) -> None:
        import synapse.observability as obs
        mock_probe = MagicMock()
        mock_probe.active = False
        obs._probe = mock_probe
        with obs.traceboard_stage("test_stage") as ctx:
            assert ctx is None

    def test_yields_stage_context_when_active(self) -> None:
        import synapse.observability as obs
        mock_stage = MagicMock()
        mock_stage.__enter__ = MagicMock(return_value=mock_stage)
        mock_stage.__exit__ = MagicMock(return_value=False)
        mock_probe = MagicMock()
        mock_probe.active = True
        mock_probe.begin_stage.return_value = mock_stage
        obs._probe = mock_probe
        with obs.traceboard_stage("test_stage", key="val") as ctx:
            assert ctx is mock_stage
        mock_probe.begin_stage.assert_called_once_with("test_stage", key="val")


class TestInitTraceboard:
    """Test init_traceboard() with mocked aether_traceboard."""

    def test_init_success(self, tmp_path: Any) -> None:
        import synapse.observability as obs
        obs._probe = None
        obs._initialized = False

        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock()
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        def side_effect():
            return mock_probe_cls

        with patch.object(obs, "_try_import_probe", side_effect=lambda: mock_probe_cls):
            with patch.dict("sys.modules", {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)}):
                result = obs.init_traceboard(str(tmp_path), "test_run", spp="64")

        assert result is mock_probe_instance
        assert obs._initialized is True
        mock_probe_instance.begin_run.assert_called_once()
        mock_probe_instance.activate.assert_called_once()
        mock_probe_instance.add_collector.assert_called_once()

    def test_init_no_auto_activate(self, tmp_path: Any) -> None:
        import synapse.observability as obs
        obs._probe = None
        obs._initialized = False

        mock_probe_cls = MagicMock()
        mock_probe_instance = MagicMock()
        mock_probe_cls.return_value = mock_probe_instance
        mock_collector_cls = MagicMock()

        with patch.object(obs, "_try_import_probe", return_value=mock_probe_cls):
            with patch.dict("sys.modules", {"aether_traceboard": MagicMock(FileCollector=mock_collector_cls)}):
                result = obs.init_traceboard(
                    str(tmp_path), "test_run", auto_activate=False
                )

        assert result is mock_probe_instance
        mock_probe_instance.activate.assert_not_called()

