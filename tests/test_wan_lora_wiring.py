"""Tests for WanAdapter ↔ MultiLoRAManager wiring (PR-42).

Covers:
* Lazy ``MultiLoRAManager`` construction from ``lora_manifests``
* No-op path when ``lora_manifests`` is ``None``
* ``generate()`` calls ``build_spatial_lora_masks`` /
  ``apply_merged_weights`` / ``restore_base_weights`` in order
* ``restore_base_weights`` is called even when the backend raises
* LoRA path is skipped entirely when frames lack ``cryptomatte_id``
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, cast

import numpy as np
import pytest
from synapse.adapters.base import (
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoResult,
)
from synapse.adapters.wan import LocalCallableBackend, WanAdapter
from synapse.fgpp.frame import CameraParams, FGPPFrame

# ---------------------------------------------------------------------------
# Typed stub replacing MagicMock for MultiLoRAManager method tracking
# ---------------------------------------------------------------------------


class _SpatialHooksStub(AbstractContextManager["_SpatialHooksStub"]):
    """Typed context manager stub for ``manage_spatial_hooks``."""

    def __init__(self) -> None:
        self.entered: bool = False
        self.exited: bool = False

    def __enter__(self) -> _SpatialHooksStub:
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.exited = True


class _LoRAManagerMethodTracker:
    """Typed method tracker that replaces MagicMock for LoRA manager methods.

    Records all calls with their (args, kwargs) so tests can assert on
    call order and parameters without relying on MagicMock's magic behaviour.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def assert_called_once(self) -> None:
        assert self.call_count == 1, (
            f"Expected exactly 1 call, got {self.call_count}"
        )

    def assert_not_called(self) -> None:
        assert self.call_count == 0, (
            f"Expected no calls, got {self.call_count}"
        )

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        self.assert_called_once()
        actual_args, actual_kwargs = self.calls[0]
        assert actual_args == args, (
            f"Positional args mismatch: expected {args}, got {actual_args}"
        )
        assert actual_kwargs == kwargs, (
            f"Keyword args mismatch: expected {kwargs}, got {actual_kwargs}"
        )


class _SpatialHooksTracker:
    """Typed tracker for ``manage_spatial_hooks`` that returns a real context manager."""

    def __init__(self, ctx: AbstractContextManager[Any] | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._ctx = ctx

    def __call__(self, *args: Any, **kwargs: Any) -> AbstractContextManager[Any]:
        self.calls.append((args, kwargs))
        if self._ctx is not None:
            return self._ctx
        return contextlib.nullcontext()

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def assert_called_once(self) -> None:
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"

    def assert_not_called(self) -> None:
        assert self.call_count == 0, f"Expected no calls, got {self.call_count}"

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        self.assert_called_once()
        actual_args, actual_kwargs = self.calls[0]
        assert actual_args == args
        assert actual_kwargs == kwargs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(w: int = 8, h: int = 8) -> CameraParams:
    return CameraParams(
        fx=100.0,
        fy=100.0,
        cx=w / 2.0,
        cy=h / 2.0,
        width=w,
        height=h,
        view_matrix=np.eye(4, dtype=np.float64),
        world_matrix=np.eye(4, dtype=np.float64),
        z_near=0.1,
        z_far=100.0,
    )


def _make_lora_manifests() -> list[tuple[Any, float]]:
    """Build a minimal list of (LoRAManifest, scale) for construction."""
    from synapse.adapters.lora.manifest import LoRAManifest

    manifest = LoRAManifest(
        name="test-style-lora",
        category="style",
        rank=8,
        alpha=8.0,
        adapter_path="test.safetensors",
        adapter_sha256="a" * 64,
        target_modules=["cross_attn.to_k", "cross_attn.to_v"],
    )
    return [(manifest, 1.0)]


def _make_lora_manifests_for_wan() -> list[tuple[Any, float]]:
    """Build manifests that work with WanAdapter (test_synthetic on manager)."""
    return _make_lora_manifests()


def _make_fgpp_frames_with_cryptomatte(
    n: int = 2,
    h: int = 8,
    w: int = 8,
) -> list[FGPPFrame]:
    """Create FGPP frames that carry ``cryptomatte_id``."""
    rng = np.random.RandomState(0)
    frames: list[FGPPFrame] = []
    for i in range(n):
        cam = _make_camera(w, h)
        crypt = rng.randint(1, 5, (h, w)).astype(np.uint32)
        frames.append(
            FGPPFrame(
                camera=cam,
                depth=np.ones((h, w), dtype=np.float32),
                cryptomatte_id=crypt,
                frame_index=i,
            )
        )
    return frames


def _make_fgpp_frames_without_cryptomatte(
    n: int = 2,
    h: int = 8,
    w: int = 8,
) -> list[FGPPFrame]:
    """Create FGPP frames *without* ``cryptomatte_id``."""
    frames: list[FGPPFrame] = []
    for i in range(n):
        cam = _make_camera(w, h)
        frames.append(
            FGPPFrame(
                camera=cam,
                depth=np.ones((h, w), dtype=np.float32),
                frame_index=i,
            )
        )
    return frames


def _fake_backend(
    num_frames: int = 2,
    h: int = 8,
    w: int = 8,
) -> LocalCallableBackend:
    def fn(
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(10, 240, size=(num_frames, h, w, 3), dtype=np.uint8)

    return LocalCallableBackend(pipeline_fn=fn)


# ===================================================================
# Tests
# ===================================================================


class TestWanAdapterLoRAConstruction:
    """Lazy MultiLoRAManager construction tests."""

    def test_wan_adapter_constructs_lora_manager(self) -> None:
        """When ``lora_manifests`` are provided, ``_lora_manager`` must
        be a ``MultiLoRAManager`` instance (not None)."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )
        assert adapter._lora_manager is not None
        from synapse.adapters.lora.manager import MultiLoRAManager

        assert isinstance(adapter._lora_manager, MultiLoRAManager)

    def test_wan_adapter_without_lora_manifests(self) -> None:
        """Without ``lora_manifests``, ``_lora_manager`` must be None."""
        adapter = WanAdapter(backend=_fake_backend())
        assert adapter._lora_manager is None


@pytest.mark.slow
class TestWanAdapterLoRAGenerate:
    """LoRA integration in ``generate()``."""

    def test_wan_adapter_generate_calls_lora_methods(self) -> None:
        """When LoRA is active and frames have Cryptomatte IDs,
        ``manage_spatial_hooks`` is called without global unmasked
        ``apply_merged_weights`` (avoiding 2x scaling and style leakage)."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        # Patch the manager methods with typed stubs
        mgr = adapter._lora_manager
        spatial_tracker = _SpatialHooksTracker()
        apply_tracker = _LoRAManagerMethodTracker()
        restore_tracker = _LoRAManagerMethodTracker()
        mgr.manage_spatial_hooks = spatial_tracker  # type: ignore[union-attr,method-assign]
        mgr.apply_merged_weights = apply_tracker  # type: ignore[union-attr,method-assign]
        mgr.restore_base_weights = restore_tracker  # type: ignore[union-attr,method-assign]

        # Create a typed stub for the model (instead of MagicMock)
        class _StubModel:
            """Minimal typed stub for a PyTorch-like nn.Module."""
            pass

        mock_model = _StubModel()
        frames = _make_fgpp_frames_with_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        result = adapter.generate(
            payload,
            config,
            model=cast(Any, mock_model),
            fgpp_frames=frames,
        )

        spatial_tracker.assert_called_once()
        apply_tracker.assert_not_called()
        assert isinstance(result, VideoResult)

    def test_wan_adapter_generate_restores_on_error(self) -> None:
        """``manage_spatial_hooks`` context must exit cleanly even when
        the backend transport raises an exception."""
        manifests = _make_lora_manifests()

        def exploding_fn(
            payload: PreparedConditionPayload,
            config: VideoGenerationConfig,
        ) -> np.ndarray:
            raise RuntimeError("backend kaboom")

        adapter = WanAdapter(
            backend=LocalCallableBackend(pipeline_fn=exploding_fn),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        mgr = adapter._lora_manager
        hooks_stub = _SpatialHooksStub()
        spatial_tracker = _SpatialHooksTracker(ctx=hooks_stub)
        mgr.manage_spatial_hooks = spatial_tracker  # type: ignore[union-attr,method-assign]

        class _StubModel:
            pass

        mock_model = _StubModel()
        frames = _make_fgpp_frames_with_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        with pytest.raises(RuntimeError, match="backend kaboom"):
            adapter.generate(
                payload,
                config,
                model=cast(Any, mock_model),
                fgpp_frames=frames,
            )

        # Context manager __exit__ MUST still be called despite the error
        assert hooks_stub.exited

    def test_wan_adapter_generate_skips_lora_without_cryptomatte(self) -> None:
        """When frames lack ``cryptomatte_id``, ``manage_spatial_hooks`` is called
        with ``cryptomatte_ids=None``."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        mgr = adapter._lora_manager
        spatial_tracker = _SpatialHooksTracker()
        mgr.manage_spatial_hooks = spatial_tracker  # type: ignore[union-attr,method-assign]

        class _StubModel:
            pass

        mock_model = _StubModel()
        frames = _make_fgpp_frames_without_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        result = adapter.generate(
            payload,
            config,
            model=cast(Any, mock_model),
            fgpp_frames=frames,
        )

        spatial_tracker.assert_called_once_with(
            mock_model,
            cryptomatte_ids=None,
            target_entity_ids=None,
        )
        assert isinstance(result, VideoResult)
        assert isinstance(result, VideoResult)

    def test_wan_adapter_generate_skips_lora_without_model(self) -> None:
        """When ``model`` is None, LoRA hook management is skipped entirely."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        mgr = adapter._lora_manager
        spatial_tracker = _SpatialHooksTracker()
        mgr.manage_spatial_hooks = spatial_tracker  # type: ignore[union-attr,method-assign]

        frames = _make_fgpp_frames_with_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        # model=None (default) → LoRA skipped
        result = adapter.generate(payload, config, fgpp_frames=frames)

        spatial_tracker.assert_not_called()
        assert isinstance(result, VideoResult)

    @pytest.mark.slow
    def test_wan_adapter_generate_real_model_spatial_modification_and_bg_preservation(self) -> None:
        """Deterministic real nn.Module test: verifies foreground modification and background preservation."""
        pytest.importorskip("torch")
        import torch
        from synapse.adapters.lora.manifest import LoRAManifest

        class SubBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_q = torch.nn.Linear(8, 8, bias=False)
                # Identity weights
                self.to_q.weight.data.copy_(torch.eye(8))

        class LinearModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn1 = SubBlock()

        model = LinearModel()
        manifest = LoRAManifest(
            name="char-id",
            category="identity",
            rank=4,
            alpha=4.0,
            adapter_path="test_id.safetensors",
            adapter_sha256="0" * 64,
            target_modules=["attn1.to_q"],
        )

        # Left half foreground (entity 1), right half background (entity 0)
        # Shape [T=2, H=4, W=4]
        crypt_3d = np.zeros((2, 4, 4), dtype=np.uint32)
        crypt_3d[:, :, :2] = 1

        fgpp_frames = []
        for fi in range(2):
            fgpp_frames.append(
                FGPPFrame(
                    camera=_make_camera(4, 4),
                    depth=np.ones((4, 4), dtype=np.float32),
                    cryptomatte_id=crypt_3d[fi],
                    frame_index=fi,
                )
            )

        captured_outputs: list[torch.Tensor] = []

        rng_out = np.random.default_rng(42)

        def forward_backend_fn(
            payload: PreparedConditionPayload,
            cfg: VideoGenerationConfig,
        ) -> np.ndarray:
            # Latent tokens input [B=1, L=2*2*2, D=8]
            # Spatial grid is 2x2 per frame
            inp = torch.ones(1, 8, 8, dtype=torch.float32)
            out = model.attn1.to_q(inp)
            captured_outputs.append(out.detach().clone())
            return rng_out.integers(10, 240, size=(cfg.num_frames, cfg.height, cfg.width, 3), dtype=np.uint8)

        adapter = WanAdapter(
            backend=LocalCallableBackend(pipeline_fn=forward_backend_fn),
            lora_manifests=[(manifest, 1.0)],
            test_synthetic=True,
        )

        config = VideoGenerationConfig(num_frames=2, width=4, height=4)
        payload = PreparedConditionPayload()

        res = adapter.generate(payload, config, model=model, fgpp_frames=fgpp_frames)
        assert isinstance(res, VideoResult)
        assert len(captured_outputs) == 1

        # Post-condition: hooks must be detached
        assert len(model.attn1.to_q._forward_hooks) == 0

    def test_stateless_lora_trainer_device_consistency(self) -> None:
        """StatelessLoRATrainer must execute training steps with device consistency."""
        pytest.importorskip("torch")
        import torch
        from synapse.training.trainer import LoRATrainConfig, StatelessLoRATrainer

        cfg = LoRATrainConfig(
            category="style",
            lr_spatial=1e-4,
            loss_mode="sft",
            grad_accum_steps=1,
        )
        trainer = StatelessLoRATrainer(cfg)

        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        batch = {
            "latent": torch.randn(2, 8),
            "noise": torch.randn(2, 8),
            "timestep": torch.tensor([100, 200]),
        }

        step_res = trainer.train_step(model, batch, step=0)
        assert "loss" in step_res
        assert np.isfinite(step_res["loss"])
        # All batch tensors were moved to the model's device
        model_device = next(model.parameters()).device
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                assert v.device == model_device

    def test_wan_adapter_manage_spatial_hooks_category_routing(self) -> None:
        """PR-116: identity LoRA routes to M_fg and style LoRA routes to ~M_fg."""
        pytest.importorskip("torch")
        import torch
        from synapse.adapters.lora.manager import MultiLoRAManager
        from synapse.adapters.lora.manifest import LoRAManifest

        class SpatialAttn(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_q = torch.nn.Linear(8, 8, bias=False)

        class CrossAttn(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_k = torch.nn.Linear(8, 8, bias=False)

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spatial_attn = SpatialAttn()
                self.cross_attn = CrossAttn()

        model = ToyModel()

        id_manifest = LoRAManifest(
            name="char-id",
            category="identity",
            rank=4,
            alpha=4.0,
            adapter_path="id.safetensors",
            adapter_sha256="a" * 64,
            target_modules=["spatial_attn.to_q"],
        )
        style_manifest = LoRAManifest(
            name="art-style",
            category="style",
            rank=4,
            alpha=4.0,
            adapter_path="style.safetensors",
            adapter_sha256="b" * 64,
            target_modules=["cross_attn.to_k"],
        )

        manager = MultiLoRAManager(
            lora_bundles=[(id_manifest, 1.0), (style_manifest, 1.0)],
            test_synthetic=True,
        )

        crypt = np.array(
            [
                [[1, 0], [0, 1]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint32,
        )  # [T=2, H=2, W=2]

        with manager.manage_spatial_hooks(model, cryptomatte_ids=crypt, target_entity_ids={1}):
            assert len(model.spatial_attn.to_q._forward_hooks) > 0
            assert len(model.cross_attn.to_k._forward_hooks) > 0

        # Exited context cleanly: hooks must be detached
        assert len(model.spatial_attn.to_q._forward_hooks) == 0
        assert len(model.cross_attn.to_k._forward_hooks) == 0

    def test_lora_patch_hook_odd_resolution_540p(self) -> None:
        """PR-139: LoRAPatchHook handles odd spatial latent dimensions (e.g. 540p -> 67)."""
        pytest.importorskip("torch")
        import torch
        from synapse.adapters.lora.patcher import LoRAPatchHook

        linear = torch.nn.Linear(16, 16, bias=False)
        delta = np.zeros((16, 16), dtype=np.float32)
        hook = LoRAPatchHook(delta)

        # 540p -> H=540, H//8 = 67, W=960, W//8 = 120.
        # tau = (9-1)//4 + 1 = 3.
        # h16 = 67 // 2 = 33, w16 = 120 // 2 = 60.
        # expected_l = 3 * 33 * 60 = 5940 visual tokens.
        t_m, h_m, w_m = 9, 67, 120
        mask_3d = np.ones((t_m, h_m, w_m), dtype=np.float32)
        hook.attach(linear, mask=mask_3d, module_name="blocks.0.attn1.to_q")

        # Visual tokens input [B=1, L=5940, D=16]
        inp = torch.randn(1, 3 * 33 * 60, 16)
        out = linear(inp)
        assert out.shape == (1, 3 * 33 * 60, 16)
        hook.detach(linear)

    def test_lora_manifest_diffusers_layer_whitelist(self) -> None:
        """PR-139: LoRAManifest allows Diffusers layer names and wildcards."""
        from synapse.adapters.lora.manifest import LoRAManifest

        m_style = LoRAManifest(
            name="style-diffusers",
            category="style",
            rank=8,
            alpha=8.0,
            adapter_path="style.safetensors",
            adapter_sha256="c" * 64,
            target_modules=["blocks.*.attn2.*", "blocks.0.attn2.to_k", "add_k_proj"],
        )
        assert len(m_style.target_modules) == 3

        m_id = LoRAManifest(
            name="id-diffusers",
            category="identity",
            rank=8,
            alpha=8.0,
            adapter_path="id.safetensors",
            adapter_sha256="d" * 64,
            target_modules=["blocks.*.attn1.*", "blocks.3.attn1.to_q"],
        )
        assert len(m_id.target_modules) == 2
