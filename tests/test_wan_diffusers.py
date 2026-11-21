"""Tests for DiffusersWanBackend ControlNeXtPatchifyStem, CFG broadcast, and VAE stats."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from synapse.adapters.base import PreparedConditionPayload, VideoGenerationConfig
from synapse.adapters.wan_diffusers import (
    DiffusersWanBackend,
    _build_controlnext_patchify_stem,
)
from synapse.errors import AssetNotFoundError, BackendUnavailableError


def _make_stub_transformer(hidden_dim: int = 32) -> Any:
    import torch

    class _StubTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=4,
                attention_head_dim=hidden_dim // 4,
            )
            self.patch_embed = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            with torch.no_grad():
                self.patch_embed.weight.copy_(torch.eye(hidden_dim))

        def forward(self, x: Any) -> Any:
            return self.patch_embed(x)

    return _StubTransformer()


def test_control_hook_fail_closed_without_weights() -> None:
    """Missing control_weights_path must raise BackendUnavailableError (exit 30)."""
    pytest.importorskip("torch")
    transformer = _make_stub_transformer(hidden_dim=32)

    class _StubPipe:
        device = "cpu"

        def __call__(self, **kwargs: Any) -> Any:
            return SimpleNamespace(frames=[np.ones((5, 16, 16, 3), dtype=np.float32) * 0.5])

    backend = DiffusersWanBackend(pipeline=_StubPipe(), transformer=transformer)
    payload = PreparedConditionPayload(
        control_tensor=np.ones((7, 5, 16, 16), dtype=np.float32),
        text_prompt="test",
    )
    cfg = VideoGenerationConfig(num_frames=5, width=16, height=16)

    with pytest.raises(BackendUnavailableError, match="ControlNeXt weights are required"):
        backend.submit(payload, cfg)


def test_control_hook_missing_weight_file_raises_asset_not_found(tmp_path: Path) -> None:
    """Non-existent control_weights_path raises AssetNotFoundError (exit 20)."""
    pytest.importorskip("torch")
    transformer = _make_stub_transformer(hidden_dim=32)

    class _StubPipe:
        device = "cpu"

        def __call__(self, **kwargs: Any) -> Any:
            return SimpleNamespace(frames=[np.ones((5, 16, 16, 3), dtype=np.float32) * 0.5])

    backend = DiffusersWanBackend(
        pipeline=_StubPipe(),
        transformer=transformer,
        control_weights_path=tmp_path / "missing.safetensors",
    )
    payload = PreparedConditionPayload(
        control_tensor=np.ones((7, 5, 16, 16), dtype=np.float32),
    )
    cfg = VideoGenerationConfig(num_frames=5, width=16, height=16)

    with pytest.raises(AssetNotFoundError):
        backend.submit(payload, cfg)


def test_controlnext_patchify_stem_weights_cfg_broadcast_and_memory(tmp_path: Path) -> None:
    """Loads pretrained ControlNeXt .safetensors, downsamples by (4,16,16), broadcasts CFG B=2."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    hidden_dim = 32
    stem = _build_controlnext_patchify_stem(
        in_channels=7,
        hidden_dim=hidden_dim,
        spatial_stride_h=16,
        spatial_stride_w=16,
        temporal_stride=4,
    )
    with torch.no_grad():
        stem.proj.weight.fill_(0.05)

    weights_path = tmp_path / "controlnext.safetensors"
    save_file(stem.state_dict(), str(weights_path))

    transformer = _make_stub_transformer(hidden_dim=hidden_dim)
    captured_out: list[Any] = []

    class _CfgStubPipe:
        device = "cpu"

        def __call__(self, **kwargs: Any) -> Any:
            # Simulate CFG batch B=2, L = tau(2) * H'(2) * W'(2) = 8 tokens
            tokens = torch.zeros(2, 8, hidden_dim, dtype=torch.float32)
            out = transformer(tokens)
            captured_out.append(out.detach().clone())
            return SimpleNamespace(frames=[np.ones((5, 32, 32, 3), dtype=np.float32) * 0.5])

    backend = DiffusersWanBackend(
        pipeline=_CfgStubPipe(),
        transformer=transformer,
        control_weights_path=weights_path,
        control_scale=1.0,
    )
    payload = PreparedConditionPayload(
        control_tensor=np.ones((7, 5, 32, 32), dtype=np.float32),
        text_prompt="studio lighting",
    )
    cfg = VideoGenerationConfig(num_frames=5, width=32, height=32)
    frames = backend.submit(payload, cfg)

    assert frames.shape == (5, 32, 32, 3)
    assert len(captured_out) == 1
    # Residual must be non-zero and identically broadcast across both CFG batch items
    assert captured_out[0].shape == (2, 8, hidden_dim)
    assert float(captured_out[0].abs().max().item()) > 1.0
    torch.testing.assert_close(captured_out[0][0], captured_out[0][1])
    # Hook must be cleanly detached after submit()
    assert len(transformer.patch_embed._forward_hooks) == 0


def test_vae_warm_start_latents_mean_std_normalization() -> None:
    """AutoencoderKLWan latents_mean/latents_std are normalized on encode and denormalized on decode."""
    pytest.importorskip("torch")
    import torch

    class _StubVae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(
                latents_mean=[1.0, -2.0, 0.5, 0.25],
                latents_std=[2.0, 4.0, 0.5, 0.25],
            )
            self.last_decode_input: Any = None

        def encode(self, x: Any) -> Any:
            # Return known constant raw latents [1, 4, 2, 2, 2]
            raw = torch.tensor([3.0, 2.0, 1.0, 0.5], dtype=x.dtype, device=x.device).view(
                1, 4, 1, 1, 1
            )
            return raw.expand(1, 4, 2, 2, 2).clone()

        def decode(self, z: Any) -> Any:
            self.last_decode_input = z.detach().clone()
            return SimpleNamespace(sample=torch.zeros(1, 3, 4, 16, 16, dtype=z.dtype))

    vae = _StubVae()

    class _LatentPipe:
        device = "cpu"

        def __call__(self, **kwargs: Any) -> Any:
            # Normalized latent [1, 4, 2, 2, 2] all ones -> denorm should equal mean + std
            return SimpleNamespace(frames=torch.ones(1, 4, 2, 2, 2, dtype=torch.float32))

    backend = DiffusersWanBackend(pipeline=_LatentPipe(), vae=vae)
    ws_frame = np.full((16, 16, 3), 128, dtype=np.uint8)
    enc = backend._encode_warm_start_latents([ws_frame, ws_frame])
    # (raw - mean) / std = ([3-1]/2, [2-(-2)]/4, [1-0.5]/0.5, [0.5-0.25]/0.25) = (1, 1, 1, 1)
    torch.testing.assert_close(enc, torch.ones_like(enc), atol=1e-5, rtol=1e-5)

    # Full intermediate window decode must denormalize latents * std + mean
    payload = PreparedConditionPayload(
        extra={"intermediate_window": True, "warm_start_frames": [ws_frame, ws_frame]},
    )
    cfg = VideoGenerationConfig(num_frames=4, width=16, height=16)
    backend.submit(payload, cfg)
    assert vae.last_decode_input is not None
    expected_denorm = torch.tensor([3.0, 2.0, 1.0, 0.5]).view(1, 4, 1, 1, 1).expand(1, 4, 2, 2, 2)
    torch.testing.assert_close(vae.last_decode_input, expected_denorm, atol=1e-5, rtol=1e-5)
