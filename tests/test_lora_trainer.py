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

pytestmark = pytest.mark.slow

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
        n = np.array([[[0, 0, 1], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]]], dtype=np.float64)
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
            np.log(2.0),
            abs=1e-6,
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
            v_theta,
            v_zero,
            v_ref,
            v_ref,
        )
        loss_b = diffusion_dpo_loss(
            v_zero,
            v_theta,
            v_ref,
            v_ref,
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
            v_ref_w,
            v_ref_l,
            v_ref_w,
            v_ref_l,
            beta=10.0,
            target_w=target_w,
            target_l=target_l,
        )
        assert loss_init == pytest.approx(np.log(2.0), abs=1e-6)

        # Policy improves on chosen (closer to target_w than ref) -> loss < ln(2)
        v_better_w = np.array([0.9, 0.9])
        loss_better = diffusion_dpo_loss(
            v_better_w,
            v_ref_l,
            v_ref_w,
            v_ref_l,
            beta=10.0,
            target_w=target_w,
            target_l=target_l,
        )
        assert loss_better < np.log(2.0)

        # Policy degrades on chosen (further from target_w than ref) -> loss > ln(2)
        v_worse_w = np.array([0.0, 0.0])
        loss_worse = diffusion_dpo_loss(
            v_worse_w,
            v_ref_l,
            v_ref_w,
            v_ref_l,
            beta=10.0,
            target_w=target_w,
            target_l=target_l,
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
            category="style",
            rank=8,
            alpha=1.0,
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


@pytest.fixture
def _require_torch() -> None:
    """Skip if torch is not installed."""
    pytest.importorskip("torch")


@pytest.mark.slow
@pytest.mark.usefixtures("_require_torch")
class TestStatelessLoRATrainer:
    """Tests for StatelessLoRATrainer (requires torch)."""

    @staticmethod
    def _make_toy_model() -> Any:
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

            def forward(self, x: Any) -> Any:
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
            model,
            target_patterns=["ffn.0", "ffn.2"],
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
            model,
            target_patterns=["ffn.0", "ffn.2"],
        )
        assert count == 2

        # Pattern matching only first layer
        model2 = self._make_toy_model()
        trainer2 = StatelessLoRATrainer(cfg)
        trainer2.freeze_backbone(model2)
        count2 = trainer2.inject_lora(
            model2,
            target_patterns=["ffn.0"],
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
            model,
            target_patterns=["ffn.0", "ffn.2"],
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

        assert final < initial * 0.8, f"Loss did not decrease >20%: {initial:.4f} -> {final:.4f}"

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
            category="identity",
            rank=4,
            lr_spatial=1e-3,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model,
            target_patterns=["ffn.0", "ffn.2"],
        )

        x_1 = torch.randn(4, 64)
        x_0 = torch.randn(4, 64)

        for step in range(10):
            trainer.train_step(
                model,
                {"latent": x_1, "noise": x_0},
                step,
            )

        # After training, check that the original Linear weights
        # inside LoRA wrappers are unchanged.
        for name, child in model.named_modules():
            if hasattr(child, "orig") and hasattr(child, "lora_A"):
                # This is a LoRA wrapper; check orig's params
                orig_w = child.orig.weight.data
                # Find matching key by looking at shape
                for key, snap in orig_weights.items():
                    if key.endswith(".weight") and snap.shape == orig_w.shape:
                        diff = (orig_w - snap).abs().max().item()
                        assert diff == 0.0, f"Base weight {key} changed by {diff}"
                        break

    def test_export_checkpoint(self, tmp_path: Any) -> None:
        """Export should create a safetensors file."""
        pytest.importorskip("safetensors")

        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        model = self._make_toy_model()
        cfg = LoRATrainConfig(
            category="style",
            rank=4,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model,
            target_patterns=["ffn.0", "ffn.2"],
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
        trainer.inject_lora(model, target_patterns=["spatial_attn", "temporal_attn"])

        x_1 = torch.randn(4, 16)
        x_0 = torch.randn(4, 16)

        # Step 0 (micro_step 1 of 2): gradients accumulated, B still 0
        trainer.train_step(model, {"latent": x_1, "noise": x_0}, 0)
        b_after_half = model.spatial_attn.lora_B.data.abs().max().item()  # type: ignore[operator]
        assert b_after_half == 0.0

        # Step 1 (micro_step 2 of 2): optimizer steps, B becomes non-zero
        trainer.train_step(model, {"latent": x_1, "noise": x_0}, 1)
        b_after_full = model.spatial_attn.lora_B.data.abs().max().item()  # type: ignore[operator]
        assert b_after_full > 0.0

        assert len(trainer._optimizer.param_groups) == 2
        lrs = sorted(pg["lr"] for pg in trainer._optimizer.param_groups)
        # Cosine LR scheduling may adjust the exact values slightly;
        # verify relative ordering and approximate magnitude.
        assert lrs[0] == pytest.approx(2e-4, rel=0.01)
        assert lrs[1] == pytest.approx(1e-3, rel=0.01)

    def test_train_step_produces_gradients(self) -> None:
        """PR-77: After train_step(), LoRA parameter .grad is not None and norm > 0."""
        import torch
        from synapse.training.trainer import (
            LoRATrainConfig,
            StatelessLoRATrainer,
        )

        torch.manual_seed(77)
        model = self._make_toy_model()

        cfg = LoRATrainConfig(
            category="style",
            rank=8,
            alpha=8.0,
            lr_spatial=1e-3,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(
            model,
            target_patterns=["ffn.0", "ffn.2"],
        )

        x_1 = torch.randn(4, 64)
        x_0 = torch.randn(4, 64)

        # Single train step
        trainer.train_step(model, {"latent": x_1, "noise": x_0}, 0)

        # Check LoRA parameters have gradients
        lora_params = [
            (name, p)
            for name, p in model.named_parameters()
            if p.requires_grad and ("lora_A" in name or "lora_B" in name)
        ]
        assert len(lora_params) > 0, "No LoRA parameters found"

        for name, p in lora_params:
            # After optimizer.step() + zero_grad(), grads are zeroed.
            # But the parameter values should have changed from initial.
            # For lora_A (initialised with N(0, 0.01)) the values
            # should be different after the step.
            pass  # grad was zeroed after step; check param mutation below

        # Better: verify that after a step, lora_B (initially 0) has
        # non-zero values (proving backprop + optimizer actually ran).
        has_nonzero_lora_b = False
        for name, child in model.named_modules():
            if hasattr(child, "lora_B"):
                if child.lora_B.data.abs().max().item() > 0:
                    has_nonzero_lora_b = True
                    break
        assert has_nonzero_lora_b, "lora_B remained zero after train_step — backprop did not run"

    def test_differentiable_flow_matching_loss_class(self) -> None:
        """PR-77: FlowMatchingLoss class preserves autograd gradient chain."""
        import torch
        from synapse.training.losses import FlowMatchingLoss

        v_pred = torch.randn(4, 16, requires_grad=True)
        x_1 = torch.randn(4, 16)
        x_0 = torch.randn(4, 16)

        loss_fn = FlowMatchingLoss()
        loss = loss_fn(v_pred, x_1, x_0)
        loss.backward()

        assert v_pred.grad is not None
        assert v_pred.grad.norm().item() > 0

    def test_differentiable_dpo_loss_class(self) -> None:
        """PR-77: DiffusionDPOLoss class preserves autograd gradient chain."""
        import torch
        from synapse.training.losses import DiffusionDPOLoss

        v_tw = torch.randn(4, 16, requires_grad=True)
        v_tl = torch.randn(4, 16, requires_grad=True)
        v_rw = torch.randn(4, 16)
        v_rl = torch.randn(4, 16)
        target_w = torch.randn(4, 16)
        target_l = torch.randn(4, 16)

        loss_fn = DiffusionDPOLoss(beta=10.0)
        loss = loss_fn(v_tw, v_tl, v_rw, v_rl, target_w, target_l)
        loss.backward()

        assert v_tw.grad is not None
        assert v_tw.grad.norm().item() > 0
        assert v_tl.grad is not None

    def test_differentiable_edge_preserving_loss_4d_and_5d(self) -> None:
        """PR-99: EdgePreservingLoss supports 4D [B, C, H, W] and 5D [B, C, T, H, W]."""
        import torch
        from synapse.training.losses import EdgePreservingLoss

        loss_fn = EdgePreservingLoss()

        # 4D: [B=2, C=3, H=16, W=16]
        z_4d = torch.randn(2, 3, 16, 16, requires_grad=True)
        d_4d = torch.randn(2, 1, 16, 16)
        loss_4d = loss_fn(z_4d, d_4d)
        loss_4d.backward()
        assert z_4d.grad is not None
        assert torch.isfinite(z_4d.grad).all()

        # 5D: [B=2, C=3, T=4, H=16, W=16]
        z_5d = torch.randn(2, 3, 4, 16, 16, requires_grad=True)
        d_5d = torch.randn(2, 1, 4, 16, 16)
        loss_5d = loss_fn(z_5d, d_5d)
        loss_5d.backward()
        assert z_5d.grad is not None
        assert torch.isfinite(z_5d.grad).all()

    def test_edge_preserving_loss_gradient_finite_on_identical(self) -> None:
        """PR-99: EdgePreservingLoss sqrt(+1e-8) ensures finite gradients on identical inputs."""
        import torch
        from synapse.training.losses import EdgePreservingLoss

        loss_fn = EdgePreservingLoss()
        z = (torch.ones(2, 1, 8, 8) * 0.5).requires_grad_(True)
        d = torch.ones(2, 1, 8, 8) * 0.5
        loss = loss_fn(z, d)
        loss.backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_normal_reconstruction_loss_zero_mask_gradient(self) -> None:
        """PR-99: NormalReconstructionLoss with empty foreground mask returns differentiable 0."""
        import torch
        from synapse.training.losses import NormalReconstructionLoss

        loss_fn = NormalReconstructionLoss()
        n_pred = torch.randn(8, 8, 3, requires_grad=True)
        n_gt = torch.randn(8, 8, 3)
        empty_mask = torch.zeros(8, 8)

        loss = loss_fn(n_pred, n_gt, mask=empty_mask)
        assert loss.item() == 0.0
        loss.backward()
        assert n_pred.grad is not None
        assert torch.isfinite(n_pred.grad).all()

    def test_diffusion_dpo_loss_batch_per_sample_reduction(self) -> None:
        """PR-99: DiffusionDPOLoss reduces per-sample across batch."""
        import torch
        from synapse.training.losses import DiffusionDPOLoss

        loss_fn = DiffusionDPOLoss(beta=10.0)
        v_tw = torch.randn(4, 16, requires_grad=True)
        v_tl = torch.randn(4, 16, requires_grad=True)
        v_rw = torch.randn(4, 16)
        v_rl = torch.randn(4, 16)
        target_w = torch.randn(4, 16)
        target_l = torch.randn(4, 16)

        loss = loss_fn(v_tw, v_tl, v_rw, v_rl, target_w, target_l)
        assert torch.isfinite(loss)
        loss.backward()
        assert v_tw.grad is not None
        assert torch.isfinite(v_tw.grad).all()

    def test_trainer_flush_gradients(self) -> None:
        """PR-99: StatelessLoRATrainer.flush_gradients steps on leftover accumulation."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        model = self._make_toy_model()
        cfg = LoRATrainConfig(
            category="style",
            grad_accum_steps=4,
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn.0"])

        batch = {"latent": torch.randn(2, 64), "noise": torch.randn(2, 64)}
        trainer.train_step(model, batch, 0)  # micro_step 1 of 4
        assert trainer._micro_step == 1
        trainer.flush_gradients(model)
        assert trainer._micro_step == 0


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


# -------------------------------------------------------------------
# PR-117: fnmatch matching, GradScaler, fgpp_cond extraction tests
# -------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("_require_torch")
class TestPR117TrainerEnhancements:
    """PR-117: Unified fnmatch matching, GradScaler, fgpp_cond extraction."""

    @staticmethod
    def _make_toy_model() -> Any:
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

            def forward(self, x: Any) -> Any:
                return self.ffn(x)

        return ToyDiT(dim=64)

    def test_fnmatch_glob_patterns(self) -> None:
        """PR-117: fnmatch glob patterns (e.g., 'ffn.*') match layers correctly."""
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        model = self._make_toy_model()
        cfg = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)

        # 'ffn.*' glob should match both ffn.0 and ffn.2
        count = trainer.inject_lora(model, target_patterns=["ffn.*"])
        assert count == 2, f"Expected 2 layers matched by 'ffn.*', got {count}"

    def test_fnmatch_no_match_raises_contract_error(self) -> None:
        """PR-117: inject_lora raises ContractValidationError when 0 layers match."""
        from synapse.errors import ContractValidationError
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        model = self._make_toy_model()
        cfg = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)

        with pytest.raises(ContractValidationError, match="matched 0 layers"):
            trainer.inject_lora(model, target_patterns=["nonexistent_layer_xyz.*"])

    def test_gradscaler_created_for_fp16(self) -> None:
        """PR-117: GradScaler is created when mixed_precision='fp16' and CUDA available."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        torch.manual_seed(117)
        model = self._make_toy_model()
        cfg = LoRATrainConfig(
            category="style",
            rank=4,
            mixed_precision="fp16",
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn.0", "ffn.2"])

        batch = {
            "latent": torch.randn(2, 64),
            "noise": torch.randn(2, 64),
        }
        result = trainer.train_step(model, batch, 0)
        assert "loss" in result
        # GradScaler is only created on CUDA; on CPU fp16 autocast is disabled
        if torch.cuda.is_available():
            assert hasattr(trainer, "_grad_scaler")
        else:
            # On CPU, autocast is disabled so scaler is not created
            assert not hasattr(trainer, "_grad_scaler")

    def test_batch_device_migration(self) -> None:
        """PR-117: batch tensors are migrated to same device as module params."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        torch.manual_seed(117)
        model = self._make_toy_model()
        cfg = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn.0", "ffn.2"])

        # Batch tensors on CPU (default) — should be auto-migrated
        batch = {
            "latent": torch.randn(2, 64),
            "noise": torch.randn(2, 64),
        }
        result = trainer.train_step(model, batch, 0)
        assert "loss" in result
        assert np.isfinite(result["loss"])

    def test_sft_step_stores_t_in_batch(self) -> None:
        """PR-186: train_step does not mutate caller batch['t'] and samples fresh t each step."""
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        torch.manual_seed(117)
        seen_t: list[torch.Tensor] = []

        class _TimeAwareModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ffn = torch.nn.Sequential(
                    torch.nn.Linear(64, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 64),
                )

            def forward(self, x: Any, t: Any = None) -> Any:
                if t is not None:
                    seen_t.append(t.detach().clone())
                return self.ffn(x)

        model = _TimeAwareModel()
        cfg = LoRATrainConfig(category="style", rank=4)
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn.0", "ffn.2"])

        batch: dict[str, Any] = {
            "latent": torch.randn(4, 64),
            "noise": torch.randn(4, 64),
        }
        assert "t" not in batch
        trainer.train_step(model, batch, 0)
        trainer.train_step(model, batch, 1)
        # Caller batch must NEVER be mutated (prevents frozen timestep bug)
        assert "t" not in batch
        assert len(seen_t) == 2
        assert not torch.allclose(seen_t[0], seen_t[1]), "Each train_step must sample fresh t"

    def test_extract_auxiliary_from_raw_fgpp(self) -> None:
        """PR-117 / PR-140: _extract_auxiliary_from_fgpp extracts depth/normal from [T,H,W,15].

        Post-PR-140 the method stays on-device and applies T→τ temporal compression.
        """
        import torch
        from synapse.training.trainer import StatelessLoRATrainer

        # Raw FGPP: [T, H, W, 15]
        t, h, w = 4, 32, 32
        fgpp = np.zeros((t, h, w, 15), dtype=np.float32)
        fgpp[..., 0] = np.linspace(1.0, 10.0, h * w).reshape(1, h, w)  # depth
        fgpp[..., 4] = 0.0  # normal x
        fgpp[..., 5] = 0.0  # normal y
        fgpp[..., 6] = 1.0  # normal z

        batch: dict[str, Any] = {
            "fgpp_cond": torch.as_tensor(fgpp),
        }

        StatelessLoRATrainer._extract_auxiliary_from_fgpp(batch)

        assert "depth" in batch
        assert "normal" in batch
        # Temporal compression T=4 → τ=(4-1)//4+1=1; spatial 8× downsample
        tau = (t - 1) // 4 + 1
        assert batch["depth"].shape == (1, tau, h // 8, w // 8)
        assert batch["normal"].shape == (3, tau, h // 8, w // 8)

    def test_extract_auxiliary_from_wan_7ch(self) -> None:
        """PR-117 / PR-140: _extract_auxiliary_from_fgpp extracts depth/normal from [7,T,H,W].

        Post-PR-140 the method stays on-device and applies T→τ temporal compression.
        """
        import torch
        from synapse.training.trainer import StatelessLoRATrainer

        # Wan control: [7, T, H, W]
        t, h, w = 4, 32, 32
        fgpp = np.zeros((7, t, h, w), dtype=np.float32)
        fgpp[0] = np.linspace(1.0, 10.0, h * w).reshape(1, h, w)  # depth
        fgpp[1:4] = np.array([0.0, 0.0, 1.0]).reshape(3, 1, 1, 1)  # normal

        batch: dict[str, Any] = {
            "fgpp_cond": torch.as_tensor(fgpp),
        }

        StatelessLoRATrainer._extract_auxiliary_from_fgpp(batch)

        assert "depth" in batch
        assert "normal" in batch
        tau = (t - 1) // 4 + 1
        assert batch["depth"].shape == (1, tau, h // 8, w // 8)
        assert batch["normal"].shape == (3, tau, h // 8, w // 8)

    def test_extract_auxiliary_no_overwrite(self) -> None:
        """PR-117: _extract_auxiliary_from_fgpp does not overwrite existing keys."""
        import torch
        from synapse.training.trainer import StatelessLoRATrainer

        sentinel = torch.ones(1)
        batch: dict[str, Any] = {
            "fgpp_cond": np.zeros((4, 32, 32, 15), dtype=np.float32),
            "depth": sentinel,
            "normal": sentinel,
        }

        StatelessLoRATrainer._extract_auxiliary_from_fgpp(batch)

        # Should NOT overwrite existing keys
        assert batch["depth"] is sentinel
        assert batch["normal"] is sentinel


class TestNormalReconstructionLoss4D5D:
    """PR-117: NormalReconstructionLoss supports 4D/5D tensors and channel truncation."""

    def test_batch_4d_format(self) -> None:
        """[B, 3, H, W] batch format produces valid loss."""
        n_pred = np.random.randn(2, 3, 8, 8)
        n_gt = np.random.randn(2, 3, 8, 8)
        loss = normal_reconstruction_loss(n_pred, n_gt)
        assert 0.0 <= loss <= 2.0

    def test_video_5d_format(self) -> None:
        """[B, 3, T, H, W] video format produces valid loss."""
        torch = pytest.importorskip("torch")
        from synapse.training.losses import NormalReconstructionLoss

        loss_fn = NormalReconstructionLoss()
        n_pred = torch.randn(1, 3, 4, 8, 8)
        n_gt = torch.randn(1, 3, 4, 8, 8)
        loss = loss_fn(n_pred, n_gt)
        assert 0.0 <= float(loss) <= 2.0

    def test_channel_truncation(self) -> None:
        """Inputs with >3 channels are truncated to 3."""
        n_pred = np.random.randn(8, 8, 5)  # 5 channels
        n_gt = np.random.randn(8, 8, 5)
        loss = normal_reconstruction_loss(n_pred, n_gt)
        assert 0.0 <= loss <= 2.0

    def test_mask_device_consistency(self) -> None:
        """PR-117: mask tensor is moved to same device as loss_map."""
        torch = pytest.importorskip("torch")
        from synapse.training.losses import NormalReconstructionLoss

        loss_fn = NormalReconstructionLoss()
        n_pred = torch.randn(8, 8, 3, requires_grad=True)
        n_gt = torch.randn(8, 8, 3)
        mask = np.ones((8, 8), dtype=np.float64)  # numpy mask
        loss = loss_fn(n_pred, n_gt, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert n_pred.grad is not None


@pytest.mark.slow
class TestPR186CheckpointAndLossHardening:
    """PR-186: Full LoRA/probe checkpoint parity, bf16 precision, 5D normal mask, replicate Sobel."""

    def test_checkpoint_save_reset_load_exact_parity(self, tmp_path: Any) -> None:
        """save_state_dict -> zero LoRA weights -> load_state_dict restores output < 1e-7."""
        pytest.importorskip("torch")
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        torch.manual_seed(186)

        class _SmallModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ffn = torch.nn.Linear(16, 16)

            def forward(self, x: Any) -> Any:
                return self.ffn(x)

        model = _SmallModel()
        trainer = StatelessLoRATrainer(LoRATrainConfig(category="style", rank=4, alpha=4.0))
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn"])

        batch = {"latent": torch.randn(2, 16), "noise": torch.randn(2, 16)}
        for step in range(3):
            trainer.train_step(model, batch, step)

        probe_x = torch.randn(2, 16)
        with torch.no_grad():
            expected_out = model(probe_x).clone()

        ckpt = tmp_path / "trainer_state.pt"
        trainer.save_state_dict(ckpt)

        # Corrupt / reset LoRA weights to zeros
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.zero_()
            corrupted_out = model(probe_x).clone()
        assert float((expected_out - corrupted_out).abs().max().item()) > 1e-5

        # Restore into fresh trainer and verify exact parity < 1e-7
        restored_trainer = StatelessLoRATrainer(LoRATrainConfig(category="style", rank=4, alpha=4.0))
        step_restored = restored_trainer.load_state_dict(ckpt, model)
        assert step_restored == 2
        with torch.no_grad():
            restored_out = model(probe_x).clone()
        assert float((expected_out - restored_out).abs().max().item()) < 1e-7

    def test_bf16_flow_and_dpo_losses_promoted_to_float32(self) -> None:
        """FlowMatchingLoss and DiffusionDPOLoss compute in float32 to prevent bf16 cancellation."""
        pytest.importorskip("torch")
        import torch
        from synapse.training.losses import DiffusionDPOLoss, FlowMatchingLoss

        fm = FlowMatchingLoss()
        x1 = torch.ones(2, 8, dtype=torch.bfloat16)
        x0 = torch.zeros(2, 8, dtype=torch.bfloat16)
        v = torch.full((2, 8), 0.5, dtype=torch.bfloat16, requires_grad=True)
        loss_fm = fm(v, x1, x0)
        assert loss_fm.dtype == torch.float32
        assert abs(float(loss_fm.item()) - 0.25) < 1e-5

        dpo = DiffusionDPOLoss(beta=500.0)
        t_w = torch.ones(2, 8, dtype=torch.bfloat16)
        t_l = -torch.ones(2, 8, dtype=torch.bfloat16)
        v_ref_w = torch.full((2, 8), 0.5, dtype=torch.bfloat16)
        v_ref_l = torch.full((2, 8), -0.5, dtype=torch.bfloat16)
        loss_dpo = dpo(v_ref_w, v_ref_l, v_ref_w, v_ref_l, t_w, t_l)
        assert loss_dpo.dtype == torch.float32
        assert abs(float(loss_dpo.item()) - float(np.log(2.0))) < 1e-5

    def test_normal_loss_5d_gt3_channels_and_5d_mask(self) -> None:
        """NormalReconstructionLoss handles 5D C>3 tensors and 5D [B,1,T,H,W] masks."""
        pytest.importorskip("torch")
        import torch
        from synapse.training.losses import NormalReconstructionLoss

        loss_fn = NormalReconstructionLoss()
        n_pred = torch.randn(2, 6, 3, 4, 4, requires_grad=True)
        n_gt = torch.randn(2, 6, 3, 4, 4)
        mask_5d = torch.ones(2, 1, 3, 4, 4)
        loss = loss_fn(n_pred, n_gt, mask=mask_5d)
        assert torch.isfinite(loss)
        loss.backward()
        assert n_pred.grad is not None

    def test_edge_preserving_replicate_padding_zero_border_artifact(self) -> None:
        """Constant non-zero 2D image must have near-zero Sobel gradient at borders."""
        pytest.importorskip("torch")
        import torch
        import torch.nn.functional as torch_f
        from synapse.training.losses import EdgePreservingLoss

        flat = torch.full((1, 1, 8, 8), 5.0, dtype=torch.float32)
        mag = EdgePreservingLoss._sobel_magnitude_batch(flat, torch_f)
        assert float(mag.max().item()) < 1e-3

    def test_auto_instantiated_geometry_probe_and_checkpoint_roundtrip(
        self, tmp_path: Any
    ) -> None:
        """normal_loss_weight > 0 auto-creates LatentGeometryProbe, trains & persists it."""
        pytest.importorskip("torch")
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        class _VideoModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ffn = torch.nn.Linear(4, 4)

            def forward(self, x: Any) -> Any:
                return self.ffn(x)

        model = _VideoModel()
        cfg = LoRATrainConfig(
            category="control", rank=4, alpha=4.0, normal_loss_weight=0.5, lr_spatial=1e-2
        )
        trainer = StatelessLoRATrainer(cfg)
        trainer.freeze_backbone(model)
        trainer.inject_lora(model, target_patterns=["ffn"])

        batch = {
            "latent": torch.randn(1, 4, 2, 4, 4),
            "noise": torch.randn(1, 4, 2, 4, 4),
            "normal_gt": torch.randn(1, 3, 2, 4, 4),
        }
        res = trainer.train_step(model, batch, 0)
        assert np.isfinite(res["loss"])
        assert trainer._geometry_probe is not None

        ckpt = tmp_path / "probe_ckpt.pt"
        trainer.save_state_dict(ckpt)
        orig_probe_w = trainer._geometry_probe._conv.weight.detach().clone()

        trainer2 = StatelessLoRATrainer(cfg)
        trainer2.load_state_dict(ckpt, model)
        assert trainer2._geometry_probe is not None
        torch.testing.assert_close(trainer2._geometry_probe._conv.weight, orig_probe_w)


@pytest.mark.slow
class TestPRTD01BackboneLoading:
    """PR-TD01: Real --base-model loading fails closed and never falls back to _WhitelistDiT."""

    def test_resolve_training_backbone_rejects_ambiguous_or_invalid(
        self, tmp_path: Any
    ) -> None:
        from pathlib import Path

        from synapse.errors import (
            AssetNotFoundError,
            BackendUnavailableError,
            ContractValidationError,
            SynapseError,
        )
        from synapse.training.backbone import resolve_training_backbone

        base = Path(str(tmp_path))

        # Neither base_model nor mock_backbone -> exit 50
        with pytest.raises(SynapseError) as exc_50:
            resolve_training_backbone(base_model_path=None, mock_backbone=False)
        assert exc_50.value.exit_code == 50

        # Both specified -> exit 10
        with pytest.raises(ContractValidationError):
            resolve_training_backbone(base_model_path=base, mock_backbone=True)

        # Nonexistent base_model -> exit 20
        with pytest.raises(AssetNotFoundError):
            resolve_training_backbone(base_model_path=base / "missing_ckpt", mock_backbone=False)

        # Directory without config.json -> exit 10 (never falls back to _WhitelistDiT)
        empty_dir = base / "empty_ckpt"
        empty_dir.mkdir()
        with pytest.raises(ContractValidationError):
            resolve_training_backbone(base_model_path=empty_dir, mock_backbone=False)

        # Malformed config.json -> exit 10
        bad_cfg_dir = base / "bad_cfg"
        bad_cfg_dir.mkdir()
        (bad_cfg_dir / "config.json").write_text("{not-json", encoding="utf-8")
        with pytest.raises(ContractValidationError):
            resolve_training_backbone(base_model_path=bad_cfg_dir, mock_backbone=False)

        # Valid config.json without diffusers -> exit 30 (unless diffusers installed)
        valid_dir = base / "valid_cfg"
        (valid_dir / "transformer").mkdir(parents=True)
        (valid_dir / "transformer" / "config.json").write_text(
            '{"_class_name": "WanTransformer3DModel"}', encoding="utf-8"
        )
        import sys

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "diffusers", None)
            with pytest.raises(BackendUnavailableError) as exc_diff:
                resolve_training_backbone(base_model_path=valid_dir, mock_backbone=False)
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            assert "diffusers" in str(exc_diff.value)

    def test_stage_and_cli_train_real_loaded_backbone(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both handle_lora_train and CLI lora train use the loaded real model, not _WhitelistDiT."""
        pytest.importorskip("torch")
        import io
        import json
        import sys
        import types
        from pathlib import Path

        import torch
        from synapse.cli import main_synapse
        from synapse.pipeline.bundle_worker import BundleManifest
        from synapse.pipeline.stages.training import handle_lora_train
        from synapse.training.feature_store import (
            FeatureStoreBuilder,
            LinearProjectionEncoder,
            LinearProjectionTextEncoder,
        )

        root = Path(str(tmp_path))
        shards_dir = root / "shards"
        builder = FeatureStoreBuilder(LinearProjectionEncoder(), LinearProjectionTextEncoder())
        frames = np.full((4, 32, 32, 3), 120, dtype=np.uint8)
        builder.add_sample("sample_0000", frames, "sks character studio")
        builder.build_shards(shards_dir)

        ckpt_dir = root / "wan_ckpt"
        (ckpt_dir / "transformer").mkdir(parents=True)
        (ckpt_dir / "transformer" / "config.json").write_text(
            '{"_class_name": "WanTransformer3DModel"}', encoding="utf-8"
        )

        loaded_calls: list[str] = []

        class _DiffusersWanStub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {
                                "attn1": torch.nn.ModuleDict(
                                    {
                                        "to_q": torch.nn.Linear(64, 64),
                                        "to_k": torch.nn.Linear(64, 64),
                                        "to_v": torch.nn.Linear(64, 64),
                                    }
                                ),
                                "attn2": torch.nn.ModuleDict(
                                    {
                                        "to_k": torch.nn.Linear(64, 64),
                                        "to_v": torch.nn.Linear(64, 64),
                                    }
                                ),
                                "ffn": torch.nn.ModuleDict(
                                    {"proj": torch.nn.Linear(64, 64)}
                                ),
                            }
                        )
                    ]
                )

            def forward(self, x: Any, t: Any = None) -> Any:
                _ = t
                orig_shape = x.shape
                h = x.reshape(orig_shape[0], -1, 64)
                blk: Any = self.blocks[0]
                h = h + blk["attn2"]["to_k"](h) + blk["attn2"]["to_v"](h)
                h = h + blk["ffn"]["proj"](h)
                return h.reshape(orig_shape)

        class _WanLoader:
            @staticmethod
            def from_pretrained(model_dir: str, **kwargs: Any) -> torch.nn.Module:
                _ = kwargs
                loaded_calls.append(model_dir)
                return _DiffusersWanStub()

        fake_diffusers = types.SimpleNamespace(WanTransformer3DModel=_WanLoader)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

        # 1. Stage handler uses real loaded DiffusersWanStub (attn2, NOT cross_attn)
        stage_out = root / "stage_weights"
        manifest = BundleManifest(
            job_id="td01-stage",
            stage="lora_train",
            inputs={"shards_dir": str(shards_dir)},
            outputs={"weights_dir": str(stage_out)},
            params={
                "base_model": str(ckpt_dir),
                "train_config": {"category": "style", "rank": 4, "max_steps": 2},
            },
        )
        res = handle_lora_train(manifest)
        assert res["backbone_kind"] == "real"
        assert res["mock_backbone"] is False
        assert loaded_calls == [str(ckpt_dir / "transformer")]

        from safetensors.numpy import load_file

        stage_weights = load_file(str(stage_out / "adapter_model.safetensors"))
        assert any("attn2.to_k" in k for k in stage_weights), (
            f"Expected Diffusers attn2 keys from real backbone, got {list(stage_weights)}"
        )
        assert not any("cross_attn" in k for k in stage_weights)

        # 2. CLI handler uses real loaded DiffusersWanStub as well
        cli_out = root / "cli_weights"
        old = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            rc = main_synapse(
                [
                    "--json",
                    "lora",
                    "train",
                    "--base-model",
                    str(ckpt_dir),
                    "--shards-dir",
                    str(shards_dir),
                    "--output",
                    str(cli_out),
                    "--config",
                    '{"category": "style", "rank": 4, "max_steps": 2}',
                ]
            )
        finally:
            sys.stdout = old
        assert rc == 0
        cli_payload = json.loads(buf.getvalue())
        assert cli_payload["backbone_kind"] == "real"
        assert cli_payload["mock_backbone"] is False
        assert len(loaded_calls) == 2
        cli_weights = load_file(str(cli_out / "adapter_model.safetensors"))
        assert any("attn2.to_k" in k for k in cli_weights)

    def test_real_diffusers_wan_signature_forwards_encoder_hidden_states_and_unwraps_sample(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR-TD01 review fix: handles real WanTransformer3DModel(hidden_states, timestep, encoder_hidden_states) & ModelOutput."""
        pytest.importorskip("torch")
        import sys
        import types
        from pathlib import Path

        import torch
        from synapse.pipeline.bundle_worker import BundleManifest
        from synapse.pipeline.stages.training import handle_lora_train
        from synapse.training.feature_store import (
            FeatureStoreBuilder,
            LinearProjectionEncoder,
            LinearProjectionTextEncoder,
        )

        root = Path(str(tmp_path))
        shards_dir = root / "shards"
        builder = FeatureStoreBuilder(LinearProjectionEncoder(), LinearProjectionTextEncoder())
        frames = np.full((4, 32, 32, 3), 120, dtype=np.uint8)
        builder.add_sample("sample_0000", frames, "sks hero studio")
        builder.build_shards(shards_dir)

        ckpt_dir = root / "wan_real_sig_ckpt"
        (ckpt_dir / "transformer").mkdir(parents=True)
        (ckpt_dir / "transformer" / "config.json").write_text(
            '{"_class_name": "WanTransformer3DModel"}', encoding="utf-8"
        )

        observed: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]] = []

        class _ExactDiffusersWan(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {
                                "attn2": torch.nn.ModuleDict(
                                    {
                                        "to_k": torch.nn.Linear(64, 64),
                                        "to_v": torch.nn.Linear(64, 64),
                                    }
                                ),
                                "ffn": torch.nn.ModuleDict(
                                    {"proj": torch.nn.Linear(64, 64)}
                                ),
                            }
                        )
                    ]
                )
                self.txt_proj = torch.nn.Linear(512, 64)

            def forward(
                self,
                hidden_states: Any,
                timestep: Any,
                encoder_hidden_states: Any,
                return_dict: bool = True,
            ) -> Any:
                observed.append(
                    (
                        tuple(hidden_states.shape),
                        tuple(timestep.shape),
                        tuple(encoder_hidden_states.shape),
                        return_dict,
                    )
                )
                assert encoder_hidden_states.ndim == 3
                orig_shape = hidden_states.shape
                h = hidden_states.reshape(orig_shape[0], -1, 64)
                txt = self.txt_proj(encoder_hidden_states)
                blk: Any = self.blocks[0]
                h = h + blk["attn2"]["to_k"](h) + blk["attn2"]["to_v"](txt.expand(-1, h.shape[1], -1))
                out = (h + blk["ffn"]["proj"](h)).reshape(orig_shape)
                if return_dict:
                    return types.SimpleNamespace(sample=out)
                return (out,)

        class _WanLoader:
            @staticmethod
            def from_pretrained(_model_dir: str, **kwargs: Any) -> torch.nn.Module:
                _ = kwargs
                return _ExactDiffusersWan()

        monkeypatch.setitem(
            sys.modules, "diffusers", types.SimpleNamespace(WanTransformer3DModel=_WanLoader)
        )

        out_dir = root / "real_sig_weights"
        res = handle_lora_train(
            BundleManifest(
                job_id="td01-exact-diffusers-sig",
                stage="lora_train",
                inputs={"shards_dir": str(shards_dir)},
                outputs={"weights_dir": str(out_dir)},
                params={
                    "base_model": str(ckpt_dir),
                    "train_config": {"category": "style", "rank": 4, "max_steps": 2},
                },
            )
        )
        assert res["backbone_kind"] == "real"
        assert len(observed) == 2
        assert all(enc_shape == (1, 1, 512) for _, _, enc_shape, _ in observed)
        assert all(len(ts_shape) == 1 for _, ts_shape, _, _ in observed)
        assert all(rd is False for _, _, _, rd in observed)

