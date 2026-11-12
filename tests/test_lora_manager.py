"""Tests for PR-16 LoRA manifest, manager, orthogonalisation and patcher.

Covers:
* LoRAManifest Pydantic validation (safe format, category whitelists, SHA256)
* MultiLoRAManager weight apply / restore on toy nn.Linear
* Cryptomatte spatial mask construction and dominant-entity voting
* Gram-Schmidt orthogonalisation of conflicting LoRA deltas
* LoRAPatchHook attach / detach / mask-gated output
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from synapse.adapters.lora.manifest import LoRAManifest
from synapse.adapters.lora.manager import MultiLoRAManager


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_SHA = "a" * 64


def _make_manifest(
    name: str = "test-lora",
    category: str = "style",
    target_modules: list[str] | None = None,
    adapter_path: str = "model.safetensors",
    sha: str = _VALID_SHA,
    rank: int = 16,
    alpha: float = 16.0,
) -> LoRAManifest:
    if target_modules is None:
        target_modules = ["cross_attn.to_k"]
    return LoRAManifest(
        name=name,
        category=category,
        rank=rank,
        alpha=alpha,
        adapter_path=adapter_path,
        adapter_sha256=sha,
        target_modules=target_modules,
    )


# ===================================================================
# TestLoRAManifest
# ===================================================================


class TestLoRAManifest:
    """Validation tests for ``LoRAManifest``."""

    def test_valid_manifest_parses(self) -> None:
        """A well-formed manifest should parse without error."""
        m = _make_manifest()
        assert m.name == "test-lora"
        assert m.schema_version == "synapse.lora.v1.0"
        assert m.rank == 16

    def test_reject_bin_adapter(self) -> None:
        """Adapter path ending in .bin must be rejected."""
        with pytest.raises(ValueError, match="safetensors"):
            _make_manifest(adapter_path="model.bin")

    def test_reject_pt_adapter(self) -> None:
        """Adapter path ending in .pt must be rejected."""
        with pytest.raises(ValueError, match="safetensors"):
            _make_manifest(adapter_path="weights.pt")

    def test_style_accepts_valid_targets(self) -> None:
        """Style category should accept cross_attn and ffn targets."""
        m = _make_manifest(
            category="style",
            target_modules=[
                "cross_attn.to_k",
                "cross_attn.to_v",
                "ffn.linear1",
            ],
        )
        assert len(m.target_modules) == 3

    def test_style_rejects_spatial_attn(self) -> None:
        """Style category must reject spatial_attn targets."""
        with pytest.raises(ValueError, match="not allowed"):
            _make_manifest(
                category="style",
                target_modules=["spatial_attn.proj"],
            )

    def test_identity_accepts_spatial_and_cross(self) -> None:
        """Identity category accepts spatial_attn and cross_attn."""
        m = _make_manifest(
            category="identity",
            target_modules=[
                "spatial_attn.proj",
                "cross_attn.to_k",
            ],
        )
        assert len(m.target_modules) == 2

    def test_motion_rejects_ffn(self) -> None:
        """Motion category must reject ffn targets."""
        with pytest.raises(ValueError, match="not allowed"):
            _make_manifest(
                category="motion",
                target_modules=["ffn.linear1"],
            )

    def test_sha256_format_validation(self) -> None:
        """SHA256 must be 64-char lowercase hex."""
        with pytest.raises(ValueError, match="64-char"):
            _make_manifest(sha="ZZZZ")


# ===================================================================
# TestMultiLoRAManager
# ===================================================================


class TestMultiLoRAManager:
    """Tests for ``MultiLoRAManager`` weight merging and masking."""

    @staticmethod
    def _wrap_linear(
        torch: Any, linear: Any, name: str = "ffn.proj",
    ) -> Any:
        """Wrap *linear* in a parent Module at the given dotted path."""
        parts = name.split(".")
        root = torch.nn.Module()
        parent = root
        for part in parts[:-1]:
            child = torch.nn.Module()
            parent.add_module(part, child)
            parent = child
        parent.add_module(parts[-1], linear)
        return root

    def test_apply_and_restore_single_lora(self) -> None:
        """Apply one LoRA then restore; error must be < 1e-7."""
        torch = pytest.importorskip("torch")

        linear = torch.nn.Linear(8, 4, bias=False)
        original = linear.weight.data.clone()
        model = self._wrap_linear(torch, linear)

        manifest = _make_manifest(
            target_modules=["ffn.proj"],
            category="control",
        )
        mgr = MultiLoRAManager(lora_bundles=[(manifest, 1.0)])
        mgr.apply_merged_weights(model)
        mgr.restore_base_weights(model)

        max_err = (linear.weight.data - original).abs().max().item()
        assert max_err < 1e-7

    def test_apply_and_restore_two_loras(self) -> None:
        """Apply two LoRAs then restore; error must be < 1e-7."""
        torch = pytest.importorskip("torch")

        linear = torch.nn.Linear(8, 4, bias=False)
        original = linear.weight.data.clone()
        model = self._wrap_linear(torch, linear)

        m1 = _make_manifest(
            name="lora-a",
            target_modules=["ffn.proj"],
            category="control",
        )
        m2 = _make_manifest(
            name="lora-b",
            target_modules=["ffn.proj"],
            category="control",
        )
        mgr = MultiLoRAManager(
            lora_bundles=[(m1, 0.5), (m2, 0.8)],
        )
        mgr.apply_merged_weights(model)
        mgr.restore_base_weights(model)

        max_err = (linear.weight.data - original).abs().max().item()
        assert max_err < 1e-7

    def test_weights_change_after_apply(self) -> None:
        """Weights must differ from original after apply."""
        torch = pytest.importorskip("torch")

        linear = torch.nn.Linear(16, 8, bias=False)
        original = linear.weight.data.clone()
        model = self._wrap_linear(torch, linear)

        manifest = _make_manifest(
            target_modules=["ffn.proj"],
            category="whitebox",
            rank=8,
            alpha=8.0,
        )
        mgr = MultiLoRAManager(lora_bundles=[(manifest, 1.0)])
        mgr.apply_merged_weights(model)

        diff = (linear.weight.data - original).abs().max().item()
        assert diff > 1e-10, "Weights should change after LoRA apply"
        mgr.restore_base_weights(model)

    def test_spatial_mask_from_cryptomatte(self) -> None:
        """Mask shape should be [T, H/8, W/8]."""
        ids = np.zeros((2, 16, 16), dtype=np.uint32)
        mgr = MultiLoRAManager(lora_bundles=[])
        mask = mgr.build_spatial_lora_masks(
            ids, target_entity_ids={1}, spatial_block=8,
        )
        assert mask.shape == (2, 2, 2)
        assert mask.dtype == bool

    def test_foreground_mask_isolates_entity(self) -> None:
        """Foreground mask True only where target entity dominates."""
        ids = np.zeros((1, 8, 16), dtype=np.uint32)
        ids[0, :, 8:] = 42  # right half is entity 42
        mgr = MultiLoRAManager(lora_bundles=[])
        mask = mgr.build_spatial_lora_masks(
            ids, target_entity_ids={42}, spatial_block=8,
        )
        # mask shape: [1, 1, 2]
        assert mask[0, 0, 0] is np.bool_(False)
        assert mask[0, 0, 1] is np.bool_(True)

    def test_background_is_complement(self) -> None:
        """Background mask should be ~foreground."""
        ids = np.full((1, 8, 8), fill_value=5, dtype=np.uint32)
        mgr = MultiLoRAManager(lora_bundles=[])
        fg = mgr.build_spatial_lora_masks(
            ids, target_entity_ids={5}, spatial_block=8,
        )
        bg = ~fg
        assert fg[0, 0, 0] is np.bool_(True)
        assert bg[0, 0, 0] is np.bool_(False)

    def test_mask_downsampling_correctness(self) -> None:
        """Dominant-entity voting in 8x8 blocks."""
        ids = np.zeros((1, 8, 8), dtype=np.uint32)
        # Put entity 7 in 33 of 64 pixels (majority)
        ids[0, :5, :7] = 7  # 35 pixels of entity 7
        mgr = MultiLoRAManager(lora_bundles=[])
        mask = mgr.build_spatial_lora_masks(
            ids, target_entity_ids={7}, spatial_block=8,
        )
        assert mask[0, 0, 0] is np.bool_(True)


# ===================================================================
# TestOrthogonalization
# ===================================================================


class TestOrthogonalization:
    """Tests for ``orthogonalize_conflicting_loras``."""

    def test_parallel_loras_get_projected(self) -> None:
        """Perfectly parallel deltas should yield near-zero result."""
        a = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([[2.0, 0.0], [0.0, 0.0]])

        result = MultiLoRAManager.orthogonalize_conflicting_loras(
            a, b, threshold=0.25,
        )
        assert np.linalg.norm(result) < 1e-10

    def test_orthogonal_loras_unchanged(self) -> None:
        """Already orthogonal deltas should be unchanged."""
        a = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([[0.0, 1.0], [0.0, 0.0]])

        result = MultiLoRAManager.orthogonalize_conflicting_loras(
            a, b, threshold=0.25,
        )
        np.testing.assert_allclose(result, b, atol=1e-10)

    def test_below_threshold_unchanged(self) -> None:
        """Deltas with cosine sim below threshold stay unchanged."""
        rng = np.random.RandomState(0)
        a = rng.randn(4, 4)
        b = rng.randn(4, 4)

        # Use a very high threshold to ensure no projection
        result = MultiLoRAManager.orthogonalize_conflicting_loras(
            a, b, threshold=0.99,
        )
        np.testing.assert_allclose(result, b, atol=1e-10)

    def test_projection_reduces_cosine_similarity(self) -> None:
        """Projection should reduce cosine similarity."""
        a = np.array([[3.0, 1.0], [0.5, 0.0]])
        b = np.array([[2.0, 0.5], [0.3, 0.0]])

        a_flat = a.ravel()
        b_flat = b.ravel()
        cos_before = abs(
            np.dot(a_flat, b_flat)
            / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        )

        result = MultiLoRAManager.orthogonalize_conflicting_loras(
            a, b, threshold=0.25,
        )
        r_flat = result.ravel()
        cos_after = abs(
            np.dot(a_flat, r_flat)
            / (np.linalg.norm(a_flat) * np.linalg.norm(r_flat) + 1e-12)
        )
        assert cos_after < cos_before


# ===================================================================
# TestPatcher
# ===================================================================


class TestPatcher:
    """Tests for ``LoRAPatchHook``."""

    def test_hook_modifies_output(self) -> None:
        """Attached hook should alter the linear layer's output."""
        torch = pytest.importorskip("torch")
        from synapse.adapters.lora.patcher import LoRAPatchHook

        linear = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(linear.weight)
        x = torch.ones(1, 4)

        out_before = linear(x).detach().clone()

        delta = torch.randn(4, 4) * 0.1
        hook = LoRAPatchHook(lora_delta=delta)
        hook.attach(linear)

        out_after = linear(x).detach()
        diff = (out_after - out_before).abs().max().item()
        assert diff > 1e-5, "Hook should modify output"

        hook.detach(linear)

    def test_hook_detaches_cleanly(self) -> None:
        """After detach, output should match original."""
        torch = pytest.importorskip("torch")
        from synapse.adapters.lora.patcher import LoRAPatchHook

        linear = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(linear.weight)
        x = torch.ones(1, 4)

        out_original = linear(x).detach().clone()

        delta = torch.randn(4, 4) * 0.1
        hook = LoRAPatchHook(lora_delta=delta)
        hook.attach(linear)
        hook.detach(linear)

        out_after_detach = linear(x).detach()
        np.testing.assert_allclose(
            out_after_detach.numpy(),
            out_original.numpy(),
            atol=1e-6,
        )

    def test_mask_gated_output(self) -> None:
        """Mask-gated hook should only affect masked positions."""
        torch = pytest.importorskip("torch")
        from synapse.adapters.lora.patcher import LoRAPatchHook

        linear = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(linear.weight)

        # Batch of 2 tokens
        x = torch.ones(2, 4)
        out_before = linear(x).detach().clone()

        delta = torch.ones(4, 4) * 0.5
        # Mask: first token active (1.0), second zeroed (0.0)
        mask = torch.tensor([1.0, 0.0])
        hook = LoRAPatchHook(lora_delta=delta)
        hook.attach(linear, mask=mask)

        out_after = linear(x).detach()

        # First token should differ
        diff_0 = (out_after[0] - out_before[0]).abs().max().item()
        assert diff_0 > 0.1, "Masked-in token should be modified"

        # Second token should be unchanged
        diff_1 = (out_after[1] - out_before[1]).abs().max().item()
        assert diff_1 < 1e-6, "Masked-out token should be unmodified"

        hook.detach(linear)
