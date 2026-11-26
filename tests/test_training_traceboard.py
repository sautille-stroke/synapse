"""Tests for Traceboard instrumentation in synapse.training modules.

Verifies that training modules correctly integrate with the Traceboard
observability layer without requiring aether-traceboard to be installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _can_import_torch() -> bool:
    """Check if torch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


_needs_torch = pytest.mark.skipif(
    not _can_import_torch(),
    reason="torch not available",
)


def _can_import_safetensors() -> bool:
    """Check if safetensors is available."""
    try:
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


_needs_safetensors = pytest.mark.skipif(
    not _can_import_safetensors(),
    reason="safetensors not available",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_observability() -> Iterator[None]:
    """Reset observability module state before each test."""
    import synapse.observability as obs

    obs._probe = None
    obs._initialized = False
    yield
    obs._probe = None
    obs._initialized = False


@pytest.fixture
def mock_probe() -> MagicMock:
    """Create a typed mock Traceboard probe with active=True."""
    from synapse.observability import ProbeProtocol

    probe = MagicMock(spec=ProbeProtocol)
    probe.active = True
    return probe


@pytest.fixture
def _set_probe(mock_probe: MagicMock) -> Iterator[None]:
    """Install mock probe into observability module."""
    import synapse.observability as obs

    obs._probe = mock_probe
    obs._initialized = True
    yield
    obs._probe = None
    obs._initialized = False


# ---------------------------------------------------------------------------
# trainer.py: train_step logging
# ---------------------------------------------------------------------------


class TestTrainerTrainStepLogging:
    """Verify train_step() logs scalars to Traceboard."""

    @_needs_torch
    @pytest.mark.usefixtures("_set_probe")
    def test_train_step_logs_loss_and_step(self, mock_probe: MagicMock) -> None:
        """train_step should log loss, step, and lr_multiplier."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        config = LoRATrainConfig(category="style", rank=4, max_steps=10)
        trainer = StatelessLoRATrainer(config)

        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        trainer.inject_lora(model, ["0"])
        trainer.freeze_backbone(model)

        batch = {
            "latent": torch.randn(1, 8),
            "noise": torch.randn(1, 8),
        }
        mock_probe.log_scalar.reset_mock()
        result = trainer.train_step(model, batch, step=5)

        assert "loss" in result
        assert result["step"] == 5

        # Verify Traceboard step + scalars logged with explicit step=5
        mock_probe.set_step.assert_called_with(5)
        scalar_tags = [
            call.args[0] for call in mock_probe.log_scalar.call_args_list
        ]
        assert "train/loss" in scalar_tags
        assert "train/step" in scalar_tags
        assert "train/lr_multiplier" in scalar_tags
        for call in mock_probe.log_scalar.call_args_list:
            assert call.kwargs.get("step") == 5


# ---------------------------------------------------------------------------
# trainer.py: inject_lora logging
# ---------------------------------------------------------------------------


class TestTrainerInjectLoraLogging:
    """Verify inject_lora() logs injection stats."""

    @_needs_torch
    @pytest.mark.usefixtures("_set_probe")
    def test_inject_lora_logs_params(self, mock_probe: MagicMock) -> None:
        """inject_lora should log injected layers and param counts."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        config = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(config)

        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        injected = trainer.inject_lora(model, ["0"])

        assert injected == 1
        scalar_tags = [
            call.args[0] for call in mock_probe.log_scalar.call_args_list
        ]
        assert "train/lora_injected_layers" in scalar_tags
        assert "train/total_params" in scalar_tags
        assert "train/trainable_params" in scalar_tags


# ---------------------------------------------------------------------------
# trainer.py: save_state_dict logging
# ---------------------------------------------------------------------------


class TestTrainerCheckpointLogging:
    """Verify save_state_dict() logs checkpoint step."""

    @_needs_torch
    @pytest.mark.usefixtures("_set_probe")
    def test_save_state_dict_logs_step(
        self, mock_probe: MagicMock, tmp_path: Path
    ) -> None:
        """save_state_dict should log checkpoint_step scalar."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        config = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(config)
        trainer._step_count = 42

        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        checkpoint_path = tmp_path / "ckpt.pt"
        trainer.save_state_dict(checkpoint_path, model)

        assert checkpoint_path.exists()
        scalar_tags = [
            call.args[0] for call in mock_probe.log_scalar.call_args_list
        ]
        assert "train/checkpoint_step" in scalar_tags
        # Verify correct step value and step kwarg
        for call in mock_probe.log_scalar.call_args_list:
            if call.args[0] == "train/checkpoint_step":
                assert call.args[1] == 42.0
                assert call.kwargs.get("step") == 42


# ---------------------------------------------------------------------------
# No-probe safety: all training ops work without Traceboard
# ---------------------------------------------------------------------------


class TestTrainerWithoutTraceboard:
    """Verify training works fine without any Traceboard probe."""

    @_needs_torch
    def test_train_step_without_probe(self) -> None:
        """train_step must not error when no probe is initialized."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        config = LoRATrainConfig(category="style", rank=4, max_steps=10)
        trainer = StatelessLoRATrainer(config)

        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        trainer.inject_lora(model, ["0"])
        trainer.freeze_backbone(model)

        batch = {
            "latent": torch.randn(1, 8),
            "noise": torch.randn(1, 8),
        }
        result = trainer.train_step(model, batch, step=0)
        assert "loss" in result
        assert result["step"] == 0


# ---------------------------------------------------------------------------
# packager.py: package result logging
# ---------------------------------------------------------------------------


class TestPackagerLogging:
    """Verify package_and_gate_lora_bundle logs packaging results."""

    @_needs_torch
    @_needs_safetensors
    @pytest.mark.usefixtures("_set_probe")
    def test_package_logs_gate_result(
        self, mock_probe: MagicMock, tmp_path: Path
    ) -> None:
        """Successful packaging should log package_passed scalar."""
        import torch
        from safetensors.torch import save_file
        from synapse.training.packager import package_and_gate_lora_bundle

        # Create a minimal valid safetensors file
        weights_path = tmp_path / "adapter.safetensors"
        tensors = {
            "cross_attn.to_k.lora_A.weight": torch.randn(4, 8),
            "cross_attn.to_k.lora_B.weight": torch.randn(8, 4),
        }
        save_file(tensors, str(weights_path))

        output_dir = tmp_path / "bundle"
        result = package_and_gate_lora_bundle(
            weights_path=weights_path,
            output_dir=output_dir,
            category="style",
            rank=4,
            bench_metrics={
                "chamfer_mm": 2.5,
                "normal_angular_error_deg": 6.0,
                "border_ring_ciede2000": 1.2,
            },
        )
        assert result["passed"] is True

        scalar_tags = [
            call.args[0] for call in mock_probe.log_scalar.call_args_list
        ]
        assert "train/package_passed" in scalar_tags


# ---------------------------------------------------------------------------
# losses.py: DPO loss decomposition logging
# ---------------------------------------------------------------------------


class TestDPOLossLogging:
    """Verify DiffusionDPOLoss logs preferred/rejected errors."""

    @_needs_torch
    @pytest.mark.usefixtures("_set_probe")
    def test_dpo_loss_logs_components(self, mock_probe: MagicMock) -> None:
        """DPO loss should log preferred and rejected error scalars."""
        import torch
        from synapse.training.losses import DiffusionDPOLoss

        dpo = DiffusionDPOLoss(beta=500.0)
        result = dpo(
            v_theta_w=torch.randn(2, 4),
            v_theta_l=torch.randn(2, 4),
            v_ref_w=torch.randn(2, 4),
            v_ref_l=torch.randn(2, 4),
            target_w=torch.randn(2, 4),
            target_l=torch.randn(2, 4),
        )
        assert result.ndim == 0  # scalar

        scalar_tags = [
            call.args[0] for call in mock_probe.log_scalar.call_args_list
        ]
        assert "train/dpo_preferred_err" in scalar_tags
        assert "train/dpo_rejected_err" in scalar_tags

    @_needs_torch
    @pytest.mark.usefixtures("_set_probe")
    def test_trainer_dpo_step_forwards_step_to_loss_logging(
        self, mock_probe: MagicMock
    ) -> None:
        """StatelessLoRATrainer DPO train_step must forward step to DPO loss scalars."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        config = LoRATrainConfig(
            category="style",
            rank=4,
            max_steps=10,
            loss_mode="dpo",
            dpo_beta=500.0,
        )
        trainer = StatelessLoRATrainer(config)

        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        trainer.inject_lora(model, ["0"])
        trainer.freeze_backbone(model)

        batch = {
            "chosen": torch.randn(1, 8),
            "rejected": torch.randn(1, 8),
            "noise": torch.randn(1, 8),
        }
        mock_probe.log_scalar.reset_mock()
        result = trainer.train_step(model, batch, step=7)

        assert "loss" in result
        assert result["step"] == 7

        dpo_pref_calls = [
            c
            for c in mock_probe.log_scalar.call_args_list
            if c.args[0] == "train/dpo_preferred_err"
        ]
        dpo_rej_calls = [
            c
            for c in mock_probe.log_scalar.call_args_list
            if c.args[0] == "train/dpo_rejected_err"
        ]
        assert len(dpo_pref_calls) == 1
        assert len(dpo_rej_calls) == 1
        assert dpo_pref_calls[0].kwargs.get("step") == 7
        assert dpo_rej_calls[0].kwargs.get("step") == 7
