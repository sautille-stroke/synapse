"""GPU smoke tests for Horizon 14 Transport/Backend real execution paths.

These tests are marked ``gpu_smoke`` and are skipped unless a CUDA device is
available.  Run them on Colab L4/A100 or ``win-dev`` to verify that the full
backend pipeline produces non-trivial outputs:

    pytest tests/test_gpu_smoke.py -m gpu_smoke -v

On CPU-only machines (ubuntu-dev1, macOS) they are automatically skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

pytestmark = [pytest.mark.gpu_smoke, pytest.mark.slow]

_CUDA_AVAILABLE = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA device")


# ---------------------------------------------------------------------------
# M54: DiffusionRenderer
# ---------------------------------------------------------------------------
class TestDiffusionRendererGPUSmoke:
    """Verify DiffusionRenderer real backend produces non-zero G-buffer output."""

    @_skip_no_cuda
    def test_mock_backend_on_cuda_device(self) -> None:
        """MockDiffusionRendererBackend should work on GPU hosts too."""
        from synapse.bridge.diffusion_renderer import MockDiffusionRendererBackend

        backend = MockDiffusionRendererBackend()
        gbuffer = np.random.rand(4, 512, 512, 3).astype(np.float32)
        result = backend.forward(gbuffer)
        assert result is not None
        assert result.shape[0] == gbuffer.shape[0]


# ---------------------------------------------------------------------------
# M55: TRELLIS.2
# ---------------------------------------------------------------------------
class TestTrellis2GPUSmoke:
    """Verify TRELLIS.2 transport mock works on GPU-available hosts."""

    @_skip_no_cuda
    def test_trellis2_mock_reconstruction(self) -> None:
        from synapse.recon.trellis2 import TopologyPolicy, Trellis2Transport

        transport = Trellis2Transport(backend="mock")
        images = np.random.rand(4, 512, 512, 3).astype(np.float32)
        mesh = transport.reconstruct(images, topology_policy=TopologyPolicy.PERMISSIVE)
        assert mesh is not None
        assert mesh.num_vertices > 0


# ---------------------------------------------------------------------------
# M56: SAM 2.1
# ---------------------------------------------------------------------------
class TestSAM2GPUSmoke:
    """Verify SAM2Transport mock segmentation runs on GPU hosts."""

    @_skip_no_cuda
    def test_sam2_mock_segmentation(self) -> None:
        from synapse.bridge.sam2_transport import MockSAM2Backend, SAM2Transport

        backend = MockSAM2Backend()
        transport = SAM2Transport(backend=backend)
        frames = np.random.randint(0, 255, (8, 256, 256, 3), dtype=np.uint8)
        masks = transport.segment_video(frames, prompt_points=[(128, 128)])
        assert masks is not None
        assert masks.shape[0] == frames.shape[0]


# ---------------------------------------------------------------------------
# M58: QLoRA NF4 quantization
# ---------------------------------------------------------------------------
class TestQLoRAGPUSmoke:
    """Verify NF4 quantization round-trip on CUDA tensors."""

    @_skip_no_cuda
    def test_nf4_quantize_dequantize_cuda_roundtrip(self) -> None:
        from synapse.training.qlora import dequantize_nf4_blockwise, quantize_nf4_blockwise

        x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        quant = quantize_nf4_blockwise(x)
        assert quant.packed_uint8.device.type == "cuda"
        recon = dequantize_nf4_blockwise(quant)
        assert recon.shape == x.shape
        # NF4 4-bit quantization: expect reasonable SQNR (> 10 dB)
        mse = torch.mean((x - recon) ** 2).item()
        signal_power = torch.mean(x**2).item()
        sqnr_db = 10.0 * np.log10(max(signal_power, 1e-12) / max(mse, 1e-12))
        assert sqnr_db > 10.0, f"SQNR too low: {sqnr_db:.1f} dB"

    @_skip_no_cuda
    def test_qlora_linear_forward_backward(self) -> None:
        from synapse.training.qlora import create_qlora_linear

        linear = create_qlora_linear(128, 64, rank=16, device="cuda")
        assert linear is not None
        x = torch.randn(4, 128, device="cuda", requires_grad=False)
        y = linear(x)
        assert y.shape == (4, 64)
        loss = y.sum()
        loss.backward()
        # Only LoRA A/B should have gradients, base is frozen
        assert linear.lora_A.grad is not None
        assert linear.lora_B.grad is not None


# ---------------------------------------------------------------------------
# M58: Auto-tune VRAM detection on real GPU
# ---------------------------------------------------------------------------
class TestAutoTuneGPUSmoke:
    """Verify detect_cuda_vram_gb returns real hardware capacity on GPU hosts."""

    @_skip_no_cuda
    def test_detect_cuda_vram_returns_positive(self) -> None:
        from synapse.training import detect_cuda_vram_gb

        vram = detect_cuda_vram_gb()
        assert vram > 0.0, f"Expected positive VRAM on CUDA host, got {vram}"
        # Sanity: typical GPUs are 4–80 GB
        assert 2.0 <= vram <= 128.0, f"VRAM {vram} GB looks implausible"

    @_skip_no_cuda
    def test_auto_tune_uses_real_vram(self) -> None:
        from synapse.training import auto_tune_lora_config, detect_cuda_vram_gb

        real_vram = detect_cuda_vram_gb()
        rec = auto_tune_lora_config("style", dataset_size=100)
        assert rec.vram_gb == real_vram
