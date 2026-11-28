"""Unit tests for M58 Mask-Aware LoRA loss and NF4 QLoRA adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synapse.observability import TraceSession
from synapse.training import (
    CATEGORY_DEFAULT_TARGETS,
    NF4_CODEBOOK,
    FlowMatchingLoss,
    LoRATrainConfig,
    StatelessLoRATrainer,
    build_mock_dit_backbone,
    create_qlora_linear,
    dequantize_nf4_blockwise,
    flow_matching_loss,
    quantize_nf4_blockwise,
)


class TestMaskAwareConfigAndNumpyFlowLoss:
    """Tests that execute everywhere without requiring PyTorch."""

    def test_lora_train_config_defaults_and_validation(self) -> None:
        cfg = LoRATrainConfig(category="identity")
        assert cfg.mask_aware is False
        assert cfg.mask_background_weight == 0.0
        assert cfg.use_qlora is False
        assert cfg.quantization_bits == 16
        assert cfg.qlora_block_size == 64

        qlora_cfg = LoRATrainConfig(
            category="style",
            rank=16,
            mask_aware=True,
            mask_background_weight=0.1,
            use_qlora=True,
            quantization_bits=4,
            qlora_block_size=32,
        )
        assert qlora_cfg.use_qlora is True
        assert qlora_cfg.mask_aware is True

        with pytest.raises(ValueError, match="qlora_block_size must be even"):
            LoRATrainConfig(category="style", qlora_block_size=33)

        with pytest.raises(ValueError):
            LoRATrainConfig(category="style", mask_background_weight=1.5)

    def test_numpy_flow_matching_loss_mask_weighting(self) -> None:
        x_1 = np.zeros((1, 4, 2, 4, 4), dtype=np.float32)
        x_0 = np.zeros_like(x_1)
        # Error is only in background (right half w >= 2)
        v_pred = np.zeros_like(x_1)
        v_pred[:, :, :, :, 2:] = 3.0

        fg_mask = np.zeros((1, 1, 2, 4, 4), dtype=np.float32)
        fg_mask[:, :, :, :, :2] = 1.0

        # Background weight = 0 -> zero loss despite background error
        assert flow_matching_loss(v_pred, x_1, x_0, mask=fg_mask, background_weight=0.0) == 0.0
        # Background weight = 1 -> standard unweighted MSE = 0.5 * 9 = 4.5
        assert flow_matching_loss(
            v_pred, x_1, x_0, mask=fg_mask, background_weight=1.0
        ) == pytest.approx(4.5, abs=1e-6)
        # All-zero mask with background_weight = 0 returns safe 0.0
        zero_mask = np.zeros_like(fg_mask)
        assert flow_matching_loss(v_pred, x_1, x_0, mask=zero_mask, background_weight=0.0) == 0.0


class TestFlowMatchingLossMaskingTorch:
    """Differentiable masked FlowMatchingLoss tests (require torch)."""

    def test_torch_masked_loss_matches_numpy_and_handles_zero_mask(self) -> None:
        torch = pytest.importorskip("torch")
        loss_fn = FlowMatchingLoss()

        x_1 = torch.ones((1, 4, 2, 4, 4), dtype=torch.float32)
        x_0 = torch.zeros_like(x_1)
        v_pred = torch.ones_like(x_1)
        v_pred[:, :, :, :, 2:] = 5.0  # background error = 4^2 = 16
        v_pred.requires_grad_(True)

        fg_mask = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        fg_mask[:, :, :, :2] = 1.0

        loss_bg0 = loss_fn(v_pred, x_1, x_0, mask=fg_mask, background_weight=0.0)
        assert float(loss_bg0.item()) == pytest.approx(0.0, abs=1e-6)
        loss_bg0.backward()
        assert v_pred.grad is not None
        assert float(v_pred.grad[:, :, :, :, :2].abs().sum().item()) == pytest.approx(0.0, abs=1e-6)
        assert float(v_pred.grad[:, :, :, :, 2:].abs().sum().item()) == pytest.approx(0.0, abs=1e-6)

        v_pred2 = v_pred.detach().clone().requires_grad_(True)
        loss_bg_half = loss_fn(v_pred2, x_1, x_0, mask=fg_mask, background_weight=0.2)
        expected = flow_matching_loss(
            v_pred2.detach().numpy(),
            x_1.numpy(),
            x_0.numpy(),
            mask=fg_mask.numpy(),
            background_weight=0.2,
        )
        assert float(loss_bg_half.item()) == pytest.approx(expected, rel=1e-5, abs=1e-5)

        # All-zero mask must remain finite with zero gradient
        v_zero = v_pred.detach().clone().requires_grad_(True)
        z_mask = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        loss_zero = loss_fn(v_zero, x_1, x_0, mask=z_mask, background_weight=0.0)
        assert torch.isfinite(loss_zero)
        assert float(loss_zero.item()) == 0.0
        loss_zero.backward()
        assert v_zero.grad is not None
        assert torch.all(torch.isfinite(v_zero.grad))

    def test_pixel_resolution_mask_auto_pools_to_latent_shape(self) -> None:
        torch = pytest.importorskip("torch")
        loss_fn = FlowMatchingLoss()
        x_1 = torch.zeros((1, 16, 2, 4, 4), dtype=torch.float32)
        x_0 = torch.zeros_like(x_1)
        v_pred = torch.ones_like(x_1)
        px_mask = torch.zeros((1, 1, 5, 32, 32), dtype=torch.float32)
        px_mask[:, :, :, :16, :16] = 1.0
        loss = loss_fn(v_pred, x_1, x_0, mask=px_mask, background_weight=0.0)
        assert torch.isfinite(loss)
        assert float(loss.item()) == pytest.approx(1.0, rel=1e-4)


class TestNF4QLoRAQuantization:
    """Tests for NF4 quantization, compression, and QLoRALinear."""

    def test_nf4_codebook_and_quantize_roundtrip(self) -> None:
        torch = pytest.importorskip("torch")
        assert len(NF4_CODEBOOK) == 16
        assert NF4_CODEBOOK[0] == -1.0
        assert NF4_CODEBOOK[-1] == 1.0
        assert 0.0 in NF4_CODEBOOK
        assert list(NF4_CODEBOOK) == sorted(NF4_CODEBOOK)

        gen = torch.Generator().manual_seed(42)
        w = torch.randn((128, 64), generator=gen, dtype=torch.float32) * 0.5
        qt = quantize_nf4_blockwise(w, block_size=64)
        recon = dequantize_nf4_blockwise(qt)
        assert tuple(recon.shape) == (128, 64)
        mse = float(torch.mean((w - recon) ** 2).item())
        assert mse < 5e-3
        assert qt.compression_ratio_vs_fp16() >= 3.0

        # Non-block-aligned shape + all-zero matrix edge case
        z = torch.zeros((17, 23), dtype=torch.float32)
        q_z = quantize_nf4_blockwise(z, block_size=64)
        r_z = dequantize_nf4_blockwise(q_z)
        assert tuple(r_z.shape) == (17, 23)
        assert torch.all(r_z == 0.0)

    def test_qlora_linear_identity_and_frozen_base(self) -> None:
        torch = pytest.importorskip("torch")
        base = torch.nn.Linear(64, 32, bias=True)
        qmod = create_qlora_linear(base, rank=4, alpha=4.0, block_size=64)
        x = torch.randn((2, 64), dtype=torch.float32)

        # B starts at zero -> lora enabled == lora disabled at step 0
        y_enabled = qmod(x)
        qmod.lora_enabled = False
        y_disabled = qmod(x)
        assert torch.allclose(y_enabled, y_disabled, atol=1e-6)

        qmod.lora_enabled = True
        loss = qmod(x).pow(2).mean()
        loss.backward()
        assert qmod.lora_A.grad is not None
        assert qmod.lora_B.grad is not None
        assert not hasattr(qmod.qweight, "grad") or qmod.qweight.grad is None
        assert qmod.weight_memory_bytes() < base.weight.numel() * 2


class TestTrainerMaskAwareAndQLoRAIntegration:
    """End-to-end tests in StatelessLoRATrainer with mask_aware + use_qlora."""

    def test_qlora_and_mask_aware_training_cycle_and_checkpoint(
        self,
        tmp_path: Path,
    ) -> None:
        torch = pytest.importorskip("torch")
        model = build_mock_dit_backbone(dim=64)
        cfg = LoRATrainConfig(
            category="identity",
            rank=4,
            alpha=4.0,
            max_steps=4,
            mixed_precision="fp32",
            mask_aware=True,
            mask_background_weight=0.05,
            use_qlora=True,
            quantization_bits=4,
            qlora_block_size=64,
            max_trainable_ratio=0.5,
        )
        trainer = StatelessLoRATrainer(cfg)
        targets = CATEGORY_DEFAULT_TARGETS["identity"]
        injected = trainer.inject_lora(model, targets)
        assert injected >= 1
        # Idempotent 2nd call must not double-wrap QLoRALinear layers
        assert trainer.inject_lora(model, targets) == 0

        freeze_stats = trainer.freeze_backbone(model)
        assert freeze_stats["trainable_ratio"] < 0.5
        assert freeze_stats["total_params"] > freeze_stats["trainable_params"]

        # Raw FGPP [1, T=5, H=16, W=16, 15] with Cryptomatte channel 14 active on left half
        fgpp = torch.zeros((1, 5, 16, 16, 15), dtype=torch.float32)
        fgpp[..., :8, 14] = 42.0
        batch = {
            "latent": torch.ones((1, 16, 2, 2, 2), dtype=torch.float32) * 0.25,
            "noise": torch.zeros((1, 16, 2, 2, 2), dtype=torch.float32),
            "fgpp_cond": fgpp,
        }

        sess = TraceSession(run_name="test-m58-qlora", metrics_dir=tmp_path / "trace")
        with sess:
            res = trainer.train_step(model, batch, step=1)
        assert np.isfinite(res["loss"])

        metrics_jsonl = list((tmp_path / "trace").glob("*.jsonl"))
        assert len(metrics_jsonl) == 1
        content = metrics_jsonl[0].read_text(encoding="utf-8")
        assert "train/mask_coverage" in content

        ckpt_path = tmp_path / "resume" / "trainer.pt"
        saved_step_path = trainer.save_state_dict(ckpt_path, module=model)
        assert saved_step_path.is_file()

        trainer2 = StatelessLoRATrainer(cfg)
        model2 = build_mock_dit_backbone(dim=64)
        trainer2.inject_lora(model2, targets)
        restored_step = trainer2.load_state_dict(saved_step_path, model2)
        assert restored_step == 1

        sf_path = trainer2.export_checkpoint(model2, tmp_path / "export" / "adapter.safetensors")
        assert sf_path.is_file()
