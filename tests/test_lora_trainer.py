"""Tests for PR-18: StatelessLoRATrainer + losses.

Covers:
- flow_matching_loss, latent_edge_alignment_loss, diffusion_dpo_loss
- LoRATrainConfig validation
- StatelessLoRATrainer: freeze, inject, train_step, export
- lora_train stage registration
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from synapse.training.losses import (
    diffusion_dpo_loss,
    flow_matching_loss,
    latent_edge_alignment_loss,
    normal_reconstruction_loss,
)

# -------------------------------------------------------------------
# Loss function tests (numpy-only, no torch needed)
# -------------------------------------------------------------------


class TestFlowMatchingLoss:
    """Tests for flow_matching_loss."""

    def test_perfect_prediction(self) -> None:
        """If v_pred == x_1 - x_0, loss should be 0."""
        x_0 = np.random.randn(4, 16)
        x_1 = np.random.randn(4, 16)
        v_pred = x_1 - x_0
        loss = flow_matching_loss(v_pred, x_1, x_0)
        assert loss == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_loss(self) -> None:
        """Random prediction should give positive loss."""
        v_pred = np.ones((4, 8))
        x_1 = np.zeros((4, 8))
        x_0 = np.zeros((4, 8))
        # target = x_1 - x_0 = 0, diff = 1, MSE = 1.0
        loss = flow_matching_loss(v_pred, x_1, x_0)
        assert loss == pytest.approx(1.0, abs=1e-10)

    def test_manual_mse(self) -> None:
        """Verify MSE computation against manual calculation."""
        v_pred = np.array([1.0, 2.0, 3.0])
        x_1 = np.array([2.0, 3.0, 4.0])
        x_0 = np.array([0.0, 0.0, 0.0])
        # target = [2,3,4], diff = [-1,-1,-1], MSE = 1.0
        loss = flow_matching_loss(v_pred, x_1, x_0)
        assert loss == pytest.approx(1.0, abs=1e-10)


class TestLatentEdgeAlignmentLoss:
    """Tests for latent_edge_alignment_loss."""

    def test_flat_input_near_zero(self) -> None:
        """Flat (constant) inputs should have ~0 edge loss."""
        z = np.ones((16, 16)) * 0.5
        d = np.ones((16, 16)) * 0.5
        loss = latent_edge_alignment_loss(z, d)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_edged_input_nonzero(self) -> None:
        """Strong edges should produce nonzero loss."""
        z = np.zeros((16, 16))
        z[4:12, 4:12] = 1.0  # sharp square edge
        d = np.ones((16, 16)) * 0.5  # flat
        loss = latent_edge_alignment_loss(z, d)
        assert loss > 0.01

    def test_identical_edges_zero(self) -> None:
        """Identical edge patterns should give ~0 loss."""
        pattern = np.zeros((16, 16))
        pattern[8:, :] = 1.0  # horizontal step
        loss = latent_edge_alignment_loss(pattern, pattern)
        assert loss == pytest.approx(0.0, abs=1e-10)




class TestNormalReconstructionLoss:
    """Unit tests for normal_reconstruction_loss (HP-04)."""

    def test_identical_normals_zero(self) -> None:
        """Identical predicted and GT normals should give ~0 loss."""
        n = np.array([[[0, 0, 1], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]]],
                     dtype=np.float64)
        loss = normal_reconstruction_loss(n, n)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_opposite_normals_two(self) -> None:
        """Opposite normals should give loss close to 2."""
        n_pred = np.array([[[0, 0, 1]]], dtype=np.float64)
        n_gt = np.array([[[0, 0, -1]]], dtype=np.float64)
        loss = normal_reconstruction_loss(n_pred, n_gt)
        assert loss == pytest.approx(2.0, abs=1e-6)

    def test_perpendicular_normals_one(self) -> None:
        """Perpendicular normals should give loss of 1."""
        n_pred = np.array([[[1, 0, 0]]], dtype=np.float64)
        n_gt = np.array([[[0, 1, 0]]], dtype=np.float64)
        loss = normal_reconstruction_loss(n_pred, n_gt)
        assert loss == pytest.approx(1.0, abs=1e-10)

    def test_channel_first_format(self) -> None:
        """[3, H, W] format should produce same result as [H, W, 3]."""
        n_hwc = np.random.randn(4, 4, 3)
        n_chw = np.transpose(n_hwc, (2, 0, 1))
        gt = np.random.randn(4, 4, 3)
        gt_chw = np.transpose(gt, (2, 0, 1))
        loss_hwc = normal_reconstruction_loss(n_hwc, gt)
        loss_chw = normal_reconstruction_loss(n_chw, gt_chw)
        assert loss_hwc == pytest.approx(loss_chw, abs=1e-10)

    def test_unnormalised_inputs(self) -> None:
        """Non-unit-length inputs should still work (internal L2 norm)."""
        n_pred = np.array([[[10, 0, 0]]], dtype=np.float64)
        n_gt = np.array([[[5, 0, 0]]], dtype=np.float64)
        loss = normal_reconstruction_loss(n_pred, n_gt)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_mask_restricts_to_foreground(self) -> None:
        """With a mask, loss is computed only over foreground pixels."""
        # 2x2 grid: top-left = identical, top-right = opposite
        n_pred = np.array(
            [[[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]],
            dtype=np.float64,
        )
        n_gt = np.array(
            [[[0, 0, 1], [0, 0, -1]], [[0, 0, 1], [0, 0, -1]]],
            dtype=np.float64,
        )
        # Without mask: mean of (0, 2, 0, 2) = 1.0
        loss_full = normal_reconstruction_loss(n_pred, n_gt)
        assert loss_full == pytest.approx(1.0, abs=1e-6)

        # Mask only the identical pixels → loss = 0
        mask = np.array([[1, 0], [1, 0]], dtype=np.float64)
        loss_fg = normal_reconstruction_loss(n_pred, n_gt, mask=mask)
        assert loss_fg == pytest.approx(0.0, abs=1e-6)

        # Mask only the opposite pixels → loss = 2
        mask_opp = np.array([[0, 1], [0, 1]], dtype=np.float64)
        loss_opp = normal_reconstruction_loss(n_pred, n_gt, mask=mask_opp)
        assert loss_opp == pytest.approx(2.0, abs=1e-6)

    def test_mask_none_backward_compat(self) -> None:
        """Passing mask=None should produce same result as no mask."""
        n = np.random.default_rng(7).standard_normal((4, 4, 3))
        gt = np.random.default_rng(8).standard_normal((4, 4, 3))
        loss_no_mask = normal_reconstruction_loss(n, gt)
        loss_none = normal_reconstruction_loss(n, gt, mask=None)
        assert loss_no_mask == pytest.approx(loss_none, abs=1e-12)

    def test_mask_all_zero_returns_zero(self) -> None:
        """All-zero mask → no valid pixels → loss = 0."""
        n_pred = np.array([[[0, 0, 1]]], dtype=np.float64)
        n_gt = np.array([[[0, 0, -1]]], dtype=np.float64)
        mask = np.array([[0]], dtype=np.float64)
        loss = normal_reconstruction_loss(n_pred, n_gt, mask=mask)
        assert loss == pytest.approx(0.0, abs=1e-12)

class TestDiffusionDPOLoss:
    """Tests for diffusion_dpo_loss."""

    def test_equal_errors_gives_log2(self) -> None:
        """Equal errors → sigmoid(0) = 0.5 → -log(0.5) = ln2."""
        v = np.array([1.0, 0.0])
        loss = diffusion_dpo_loss(v, v, v, v)
        assert loss == pytest.approx(
            np.log(2.0), abs=1e-6,
        )

    def test_loss_is_nonnegative(self) -> None:
        """DPO loss is always non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            v1 = rng.randn(8)
            v2 = rng.randn(8)
            v3 = rng.randn(8)
            v4 = rng.randn(8)
            loss = diffusion_dpo_loss(v1, v2, v3, v4)
            assert loss >= 0.0, f"Negative DPO loss: {loss}"

    def test_asymmetric_behaviour(self) -> None:
        """Swapping chosen/rejected changes the loss."""
        v_theta = np.array([1.0, 0.5])
        v_ref = np.array([1.0, 0.0])
        v_zero = np.array([0.0, 0.0])

        loss_a = diffusion_dpo_loss(
            v_theta, v_zero, v_ref, v_ref,
        )
        loss_b = diffusion_dpo_loss(
            v_zero, v_theta, v_ref, v_ref,
        )
        # The two losses should differ
        assert loss_a != pytest.approx(loss_b, abs=1e-3)

    def test_directional_preference_monotonicity(self) -> None:
        """Improving policy accuracy on chosen vs rejected strictly lowers DPO loss."""
        target_w = np.array([1.0, 1.0])
        target_l = np.array([-1.0, -1.0])
        v_ref_w = np.array([0.5, 0.5])
        v_ref_l = np.array([-0.5, -0.5])

        # Policy == reference -> loss is exactly ln(2)
        loss_init = diffusion_dpo_loss(
            v_ref_w, v_ref_l, v_ref_w, v_ref_l,
            beta=10.0, target_w=target_w, target_l=target_l,
        )
        assert loss_init == pytest.approx(np.log(2.0), abs=1e-6)

        # Policy improves on chosen (closer to target_w than ref) -> loss < ln(2)
        v_better_w = np.array([0.9, 0.9])
        loss_better = diffusion_dpo_loss(
            v_better_w, v_ref_l, v_ref_w, v_ref_l,
            beta=10.0, target_w=target_w, target_l=target_l,
        )
        assert loss_better < np.log(2.0)

        # Policy degrades on chosen (further from target_w than ref) -> loss > ln(2)
        v_worse_w = np.array([0.0, 0.0])
        loss_worse = diffusion_dpo_loss(
            v_worse_w, v_ref_l, v_ref_w, v_ref_l,
            beta=10.0, target_w=target_w, target_l=target_l,
        )
        assert loss_worse > np.log(2.0)


# -------------------------------------------------------------------
# LoRATrainConfig tests
# -------------------------------------------------------------------


class TestLoRATrainConfig:
    """Tests for LoRATrainConfig validation."""

    def test_valid_config(self) -> None:
        """Valid configuration should parse without error."""
        from synapse.training.trainer import LoRATrainConfig

        cfg = LoRATrainConfig(
            category="style", rank=8, alpha=1.0,
        )
        assert cfg.rank == 8
        assert cfg.loss_mode == "sft"

    def test_invalid_rank_not_power_of_two(self) -> None:
        """Non-power-of-2 rank should raise ValidationError."""
        from synapse.training.trainer import LoRATrainConfig

        with pytest.raises(Exception, match="power of 2"):
            LoRATrainConfig(category="style", rank=6)

    def test_rank_boundaries(self) -> None:
        """Edge rank values 1 and 64 should be valid."""
        from synapse.training.trainer import LoRATrainConfig

        cfg1 = LoRATrainConfig(category="motion", rank=1)
        assert cfg1.rank == 1
        cfg64 = LoRATrainConfig(category="control", rank=64)
        assert cfg64.rank == 64


# -------------------------------------------------------------------
# Torch-dependent trainer tests
# -------------------------------------------------------------------


@pytest.fixture()
def _require_torch():
    """Skip if torch is not installed."""
    pytest.importorskip("torch")


@pytest.mark.usefixtures("_require_torch")
class TestStatelessLoRATrainer:
    """Tests for StatelessLoRATrainer (requires torch)."""

    @staticmethod
    def _make_toy_model():
        """Build a tiny 2-layer MLP for testing."""
        import torch

        class ToyDiT(torch.nn.Module):
            def __init__(self, dim: int = 64) -> None:
                super().__init__()
                self.ffn = torch.nn.Sequential(
                    torch.nn.Linear(dim, dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(dim * 4, dim),
                )

            def forward(self, x):
                return self.ffn(x)

        return ToyDiT(dim=64)

    def test_freeze_backbone(self) -> None:
        """Freeze sets all requires_grad to False."""
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        model = self._make_toy_model()
        cfg = LoRATrainConfig(category="style")
        trainer = StatelessLoRATrainer(cfg)
        info = trainer.freeze_backbone(model)

        assert info["trainable_params"] == 0
        assert info["trainable_ratio"] == 0.0

    def test_inject_lora_identity_at_step0(self) -> None:
        """B=0 ensures output unchanged at step 0."""
        import torch
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        model = self._make_toy_model()
        x = torch.randn(2, 64)

        # Get baseline output
        model.eval()
        with torch.no_grad():
            baseline = model(x).clone()

        cfg = LoRATrainConfig(category="style", rank=8)
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        count = trainer.inject_lora(
            model, target_patterns=["ffn.0", "ffn.2"],
        )

        assert count == 2

        # After injection with B=0, output should match baseline
        model.eval()
        with torch.no_grad():
            after = model(x)

        diff = (after - baseline).abs().max().item()
        assert diff < 1e-6, f"Step-0 output changed by {diff}"

    def test_inject_lora_count(self) -> None:
        """Correct number of layers are injected."""
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        model = self._make_toy_model()
        cfg = LoRATrainConfig(category="control", rank=4)
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)

        # Pattern matches both linear layers in ffn
        count = trainer.inject_lora(
            model, target_patterns=["ffn.0", "ffn.2"],
        )
        assert count == 2

        # Pattern matching only first layer
        model2 = self._make_toy_model()
        trainer2 = StatelessLoRATrainer(cfg)
        trainer2.freeze_backbone(model2)
        count2 = trainer2.inject_lora(
            model2, target_patterns=["ffn.0"],
        )
        assert count2 == 1

    def test_train_step_loss_decreases(self) -> None:
        """SFT training should decrease loss significantly."""
        import torch
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        torch.manual_seed(42)
        model = self._make_toy_model()

        cfg = LoRATrainConfig(
            category="style",
            rank=16,
            alpha=16.0,
            lr_spatial=5e-3,
            max_steps=80,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model, target_patterns=["ffn.0", "ffn.2"],
        )

        # Fixed target data for consistent signal
        x_1 = torch.randn(8, 64)
        x_0 = torch.randn(8, 64)

        losses: list[float] = []
        for step in range(80):
            batch = {"latent": x_1, "noise": x_0}
            result = trainer.train_step(model, batch, step)
            losses.append(result["loss"])

        initial = losses[0]
        final = losses[-1]

        assert final < initial * 0.8, (
            f"Loss did not decrease >20%: "
            f"{initial:.4f} -> {final:.4f}"
        )

    def test_frozen_weights_unchanged(self) -> None:
        """Base weights should be bitwise identical after training."""
        import torch
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        torch.manual_seed(123)
        model = self._make_toy_model()

        # Snapshot original weights before LoRA injection
        orig_weights: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            orig_weights[name] = param.data.clone()

        cfg = LoRATrainConfig(
            category="identity", rank=4, lr_spatial=1e-3,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model, target_patterns=["ffn.0", "ffn.2"],
        )

        x_1 = torch.randn(4, 64)
        x_0 = torch.randn(4, 64)

        for step in range(10):
            trainer.train_step(
                model, {"latent": x_1, "noise": x_0}, step,
            )

        # After training, check that the original Linear weights
        # inside LoRA wrappers are unchanged.
        for name, child in model.named_modules():
            if hasattr(child, "orig") and hasattr(child, "lora_A"):
                # This is a LoRA wrapper; check orig's params
                orig_w = child.orig.weight.data
                # Find matching key by looking at shape
                for key, snap in orig_weights.items():
                    if (
                        key.endswith(".weight")
                        and snap.shape == orig_w.shape
                    ):
                        diff = (
                            orig_w - snap
                        ).abs().max().item()
                        assert diff == 0.0, (
                            f"Base weight {key} changed "
                            f"by {diff}"
                        )
                        break

    def test_export_checkpoint(self, tmp_path) -> None:
        """Export should create a safetensors file."""
        pytest.importorskip("safetensors")

        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        model = self._make_toy_model()
        cfg = LoRATrainConfig(
            category="style", rank=4,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model, target_patterns=["ffn.0", "ffn.2"],
        )

        out_path = tmp_path / "lora.safetensors"
        result = trainer.export_checkpoint(model, out_path)

        assert result.exists()
        assert result.stat().st_size > 0

        # Load back and verify keys contain lora_A/lora_B and metadata
        from safetensors import safe_open
        from safetensors.torch import load_file

        loaded = load_file(str(result))
        keys = list(loaded.keys())
        assert any(k.endswith(".lora_A.weight") for k in keys)
        assert any(k.endswith(".lora_B.weight") for k in keys)

        with safe_open(str(result), framework="pt") as f:
            meta = f.metadata()
            assert meta is not None
            assert meta.get("category") == "style"
            assert meta.get("rank") == "4"

    def test_inject_lora_idempotent_and_order_independent(self) -> None:
        """Calling inject_lora twice or before freeze_backbone works cleanly."""
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        model = self._make_toy_model()
        cfg = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(cfg)

        # Inject first, then freeze_backbone -> LoRA params remain trainable
        c1 = trainer.inject_lora(model, target_patterns=["ffn.0", "ffn.2"])
        assert c1 == 2
        # Second call with same patterns must not double-wrap .orig
        c2 = trainer.inject_lora(model, target_patterns=["ffn.0", "ffn.2"])
        assert c2 == 0

        info = trainer.freeze_backbone(model)
        assert info["trainable_params"] > 0

    def test_dpo_step_reference_parity_and_preference_learning(self) -> None:
        """At step 0 (B=0), DPO loss equals ln(2); training decreases DPO loss."""
        import torch
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        torch.manual_seed(7)
        model = self._make_toy_model()
        cfg = LoRATrainConfig(
            category="motion",
            rank=8,
            alpha=8.0,
            lr_spatial=5e-3,
            loss_mode="dpo",
            dpo_beta=10.0,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn.0", "ffn.2"])

        chosen = torch.randn(4, 64)
        rejected = -chosen.clone()
        noise = torch.zeros(4, 64)
        fixed_t = torch.full((4, 1), 0.5)

        batch = {
            "chosen": chosen,
            "rejected": rejected,
            "noise": noise,
            "t": fixed_t,
        }
        step0 = trainer.train_step(model, batch, 0)
        assert step0["loss"] == pytest.approx(float(np.log(2.0)), abs=1e-5)

        for s in range(1, 25):
            last = trainer.train_step(model, batch, s)
        assert last["loss"] < step0["loss"] * 0.9, (
            f"DPO loss did not decrease: {step0['loss']:.4f} -> {last['loss']:.4f}"
        )

    def test_spatial_and_temporal_lr_groups_and_grad_accum(self) -> None:
        """Verify separate lr_spatial / lr_temporal optimizer groups and grad_accum_steps."""
        import torch
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        class SpatialTemporalModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spatial_attn = torch.nn.Linear(16, 16)
                self.temporal_attn = torch.nn.Linear(16, 16)

            def forward(self, x: Any) -> Any:
                return self.spatial_attn(x) + self.temporal_attn(x)

        model = SpatialTemporalModel()
        cfg = LoRATrainConfig(
            category="motion",
            rank=4,
            lr_spatial=1e-3,
            lr_temporal=2e-4,
            grad_accum_steps=2,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model, target_patterns=["spatial_attn", "temporal_attn"]
        )

        x_1 = torch.randn(4, 16)
        x_0 = torch.randn(4, 16)

        # Step 0 (micro_step 1 of 2): gradients accumulated, B still 0
        trainer.train_step(model, {"latent": x_1, "noise": x_0}, 0)
        b_after_half = model.spatial_attn.lora_B.data.abs().max().item()
        assert b_after_half == 0.0

        # Step 1 (micro_step 2 of 2): optimizer steps, B becomes non-zero
        trainer.train_step(model, {"latent": x_1, "noise": x_0}, 1)
        b_after_full = model.spatial_attn.lora_B.data.abs().max().item()
        assert b_after_full > 0.0

        assert len(trainer._optimizer.param_groups) == 2
        lrs = {pg["lr"] for pg in trainer._optimizer.param_groups}
        assert lrs == {1e-3, 2e-4}


# -------------------------------------------------------------------
# Stage registration test
# -------------------------------------------------------------------


class TestStageRegistration:
    """Test lora_train stage is registered."""

    def test_lora_train_stage_registered(self) -> None:
        """'lora_train' should appear in registered stages."""
        # Force import to trigger registration
        import synapse.training.trainer  # noqa: F401
        from synapse.pipeline.bundle_worker import (
            get_registered_stages,
        )

        stages = get_registered_stages()
        assert "lora_train" in stages
