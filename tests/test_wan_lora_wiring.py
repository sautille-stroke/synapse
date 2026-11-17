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

from typing import Any
from unittest.mock import MagicMock

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
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(w: int = 8, h: int = 8) -> CameraParams:
    return CameraParams(
        fx=100.0, fy=100.0,
        cx=w / 2.0, cy=h / 2.0,
        width=w, height=h,
        view_matrix=np.eye(4, dtype=np.float64),
        world_matrix=np.eye(4, dtype=np.float64),
        z_near=0.1, z_far=100.0,
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
    n: int = 2, h: int = 8, w: int = 8,
) -> list[FGPPFrame]:
    """Create FGPP frames that carry ``cryptomatte_id``."""
    rng = np.random.RandomState(0)
    frames: list[FGPPFrame] = []
    for i in range(n):
        cam = _make_camera(w, h)
        crypt = rng.randint(1, 5, (h, w)).astype(np.uint32)
        frames.append(FGPPFrame(
            camera=cam,
            depth=np.ones((h, w), dtype=np.float32),
            cryptomatte_id=crypt,
            frame_index=i,
        ))
    return frames


def _make_fgpp_frames_without_cryptomatte(
    n: int = 2, h: int = 8, w: int = 8,
) -> list[FGPPFrame]:
    """Create FGPP frames *without* ``cryptomatte_id``."""
    frames: list[FGPPFrame] = []
    for i in range(n):
        cam = _make_camera(w, h)
        frames.append(FGPPFrame(
            camera=cam,
            depth=np.ones((h, w), dtype=np.float32),
            frame_index=i,
        ))
    return frames


def _fake_backend(
    num_frames: int = 2, h: int = 8, w: int = 8,
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


class TestWanAdapterLoRAGenerate:
    """LoRA integration in ``generate()``."""

    def test_wan_adapter_generate_calls_lora_methods(self) -> None:
        """When LoRA is active and frames have Cryptomatte IDs,
        ``build_spatial_lora_masks``, ``apply_merged_weights``, and
        ``restore_base_weights`` must all be called."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        # Patch the manager methods
        mgr = adapter._lora_manager
        mgr.build_spatial_lora_masks = MagicMock()  # type: ignore[union-attr]
        mgr.apply_merged_weights = MagicMock()  # type: ignore[union-attr]
        mgr.restore_base_weights = MagicMock()  # type: ignore[union-attr]

        mock_model = MagicMock()
        frames = _make_fgpp_frames_with_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        result = adapter.generate(
            payload, config, model=mock_model, fgpp_frames=frames,
        )

        mgr.build_spatial_lora_masks.assert_called_once()  # type: ignore[union-attr]
        mgr.apply_merged_weights.assert_called_once_with(mock_model)  # type: ignore[union-attr]
        mgr.restore_base_weights.assert_called_once_with(mock_model)  # type: ignore[union-attr]
        assert isinstance(result, VideoResult)

    def test_wan_adapter_generate_restores_on_error(self) -> None:
        """``restore_base_weights`` must be called even when the backend
        transport raises an exception."""
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
        mgr.build_spatial_lora_masks = MagicMock()  # type: ignore[union-attr]
        mgr.apply_merged_weights = MagicMock()  # type: ignore[union-attr]
        mgr.restore_base_weights = MagicMock()  # type: ignore[union-attr]

        mock_model = MagicMock()
        frames = _make_fgpp_frames_with_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        with pytest.raises(RuntimeError, match="backend kaboom"):
            adapter.generate(
                payload, config, model=mock_model, fgpp_frames=frames,
            )

        # restore MUST still be called despite the error
        mgr.restore_base_weights.assert_called_once_with(mock_model)  # type: ignore[union-attr]

    def test_wan_adapter_generate_skips_lora_without_cryptomatte(self) -> None:
        """When frames lack ``cryptomatte_id``, LoRA methods must NOT
        be called even if a manager is present."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        mgr = adapter._lora_manager
        mgr.build_spatial_lora_masks = MagicMock()  # type: ignore[union-attr]
        mgr.apply_merged_weights = MagicMock()  # type: ignore[union-attr]
        mgr.restore_base_weights = MagicMock()  # type: ignore[union-attr]

        mock_model = MagicMock()
        frames = _make_fgpp_frames_without_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        result = adapter.generate(
            payload, config, model=mock_model, fgpp_frames=frames,
        )

        mgr.build_spatial_lora_masks.assert_not_called()  # type: ignore[union-attr]
        mgr.apply_merged_weights.assert_not_called()  # type: ignore[union-attr]
        mgr.restore_base_weights.assert_not_called()  # type: ignore[union-attr]
        assert isinstance(result, VideoResult)



    def test_wan_adapter_generate_skips_lora_without_model(self) -> None:
        """When ``model`` is None, LoRA methods must NOT be called."""
        manifests = _make_lora_manifests()
        adapter = WanAdapter(
            backend=_fake_backend(),
            lora_manifests=manifests,
            test_synthetic=True,
        )

        mgr = adapter._lora_manager
        mgr.build_spatial_lora_masks = MagicMock()  # type: ignore[union-attr]
        mgr.apply_merged_weights = MagicMock()  # type: ignore[union-attr]
        mgr.restore_base_weights = MagicMock()  # type: ignore[union-attr]

        frames = _make_fgpp_frames_with_cryptomatte()
        config = VideoGenerationConfig(num_frames=2, width=8, height=8)
        payload = PreparedConditionPayload()

        # model=None (default) → LoRA skipped
        result = adapter.generate(payload, config, fgpp_frames=frames)

        mgr.build_spatial_lora_masks.assert_not_called()  # type: ignore[union-attr]
        mgr.apply_merged_weights.assert_not_called()  # type: ignore[union-attr]
        mgr.restore_base_weights.assert_not_called()  # type: ignore[union-attr]
        assert isinstance(result, VideoResult)

    def test_wan_adapter_manage_spatial_hooks_category_routing(self) -> None:
        """PR-116: identity LoRA routes to M_fg and style LoRA routes to ~M_fg."""
        torch = pytest.importorskip("torch")
        from synapse.adapters.lora.manager import MultiLoRAManager
        from synapse.adapters.lora.manifest import LoRAManifest

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spatial = torch.nn.Linear(8, 8, bias=False)
                self.cross = torch.nn.Linear(8, 8, bias=False)

        model = ToyModel()

        id_manifest = LoRAManifest(
            name="char-id",
            category="identity",
            rank=4,
            alpha=4.0,
            adapter_path="id.safetensors",
            adapter_sha256="a" * 64,
            target_modules=["spatial"],
        )
        style_manifest = LoRAManifest(
            name="art-style",
            category="style",
            rank=4,
            alpha=4.0,
            adapter_path="style.safetensors",
            adapter_sha256="b" * 64,
            target_modules=["cross"],
        )

        manager = MultiLoRAManager(
            lora_bundles=[(id_manifest, 1.0), (style_manifest, 1.0)],
            test_synthetic=True,
        )

        crypt = np.array([
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
        ], dtype=np.uint32)  # [T=2, H=2, W=2]

        with manager.manage_spatial_hooks(model, cryptomatte_ids=crypt, target_entity_ids={1}):
            assert len(model.spatial._forward_hooks) > 0
            assert len(model.cross._forward_hooks) > 0

        # Exited context cleanly: hooks must be detached
        assert len(model.spatial._forward_hooks) == 0
        assert len(model.cross._forward_hooks) == 0


