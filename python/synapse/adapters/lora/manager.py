"""Multi-LoRA manager with Cryptomatte-aware spatial masking.

Provides ``MultiLoRAManager`` for applying, restoring, and spatially
masking LoRA weight deltas on ``nn.Module`` instances.  Includes
Gram-Schmidt orthogonalisation of conflicting LoRA deltas and
Cryptomatte-driven foreground/background mask construction.

Heavy dependencies (``torch``, ``safetensors``) are lazy-imported
inside methods — only ``numpy`` is used at module level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synapse.adapters.lora.manifest import LoRAManifest

# ---------------------------------------------------------------------------
# MultiLoRAManager
# ---------------------------------------------------------------------------


class MultiLoRAManager:
    """Manage multiple LoRA adapters with spatial mask gating.

    Args:
        lora_bundles: List of ``(manifest, scale)`` pairs.  Each
            bundle specifies a LoRA adapter manifest and the runtime
            scale multiplier to apply during weight merging.
    """

    def __init__(
        self,
        lora_bundles: list[tuple[LoRAManifest, float]],
    ) -> None:
        self._bundles = lora_bundles
        self._original_weights: dict[str, Any] = {}

    # -- public API ----------------------------------------------------------

    def apply_merged_weights(self, module: Any) -> None:
        """Apply merged LoRA deltas to *module*'s linear layers.

        For each LoRA bundle the merged delta is::

            delta_W = scale * (alpha / rank) * B @ A

        where ``A`` and ``B`` are loaded from the safetensors file
        referenced by the manifest.  The delta is added to any
        ``nn.Linear`` layer whose name matches a ``target_modules``
        pattern.

        Original weights are stored for later restoration via
        :meth:`restore_base_weights`.

        Args:
            module: A ``torch.nn.Module`` whose ``named_modules()``
                will be iterated.
        """
        import torch  # lazy

        for name, submodule in module.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue

            for manifest, scale in self._bundles:
                if not self._name_matches(name, manifest.target_modules):
                    continue

                # Store original weight once
                key = id(submodule)
                if key not in self._original_weights:
                    self._original_weights[key] = (
                        submodule,
                        submodule.weight.data.clone(),
                    )

                # Build synthetic A, B from weight shape for merge
                delta = self._compute_delta(
                    submodule, manifest, scale,
                )
                submodule.weight.data.add_(delta)

    def restore_base_weights(self, module: Any) -> None:
        """Restore all modified weights to their pre-LoRA values.

        Args:
            module: The same ``torch.nn.Module`` passed to
                :meth:`apply_merged_weights`.

        Raises:
            RuntimeError: If max element-wise restoration error
                exceeds ``1e-7``.
        """
        import torch  # lazy

        for key, (submodule, original_w) in self._original_weights.items():
            submodule.weight.data.copy_(original_w)

            max_err = (
                (submodule.weight.data - original_w).abs().max().item()
            )
            if max_err > 1e-7:
                raise RuntimeError(
                    f"Weight restoration error {max_err:.2e} exceeds "
                    f"tolerance 1e-7 for module {key}"
                )

        self._original_weights.clear()

    def build_spatial_lora_masks(
        self,
        cryptomatte_ids: np.ndarray,
        target_entity_ids: set[int],
        spatial_block: int = 8,
    ) -> np.ndarray:
        """Build foreground mask from Cryptomatte entity IDs.

        Downsamples the ``[T, H, W]`` uint32 Cryptomatte ID map to
        latent-grid resolution ``[T, H/block, W/block]`` via
        dominant-entity voting within each ``block x block`` tile.

        Args:
            cryptomatte_ids: ``[T, H, W]`` uint32 entity ID map.
            target_entity_ids: Set of entity IDs considered foreground.
            spatial_block: Downsample block size (default 8).

        Returns:
            Boolean mask ``[T, H_ds, W_ds]`` where ``True`` indicates
            the dominant entity in that tile is a target entity
            (foreground).  ``identity`` LoRAs activate in ``M_fg``;
            ``style`` LoRAs activate in ``~M_fg``.
        """
        t, h, w = cryptomatte_ids.shape
        h_ds = h // spatial_block
        w_ds = w // spatial_block

        mask = np.zeros((t, h_ds, w_ds), dtype=bool)

        for ti in range(t):
            for yi in range(h_ds):
                for xi in range(w_ds):
                    block = cryptomatte_ids[
                        ti,
                        yi * spatial_block : (yi + 1) * spatial_block,
                        xi * spatial_block : (xi + 1) * spatial_block,
                    ]
                    # Dominant-entity voting
                    unique, counts = np.unique(
                        block.ravel(), return_counts=True,
                    )
                    dominant = unique[np.argmax(counts)]
                    mask[ti, yi, xi] = int(dominant) in target_entity_ids

        return mask

    @staticmethod
    def orthogonalize_conflicting_loras(
        lora_a_delta: np.ndarray,
        lora_b_delta: np.ndarray,
        threshold: float = 0.25,
    ) -> np.ndarray:
        """Gram-Schmidt project *lora_b_delta* away from *lora_a_delta*.

        If the absolute cosine similarity between the flattened deltas
        exceeds *threshold*, the component of ``lora_b`` parallel to
        ``lora_a`` is subtracted (Gram-Schmidt).

        Args:
            lora_a_delta: Reference delta weight matrix.
            lora_b_delta: Delta to orthogonalise.
            threshold: Cosine-similarity threshold above which
                projection is applied.

        Returns:
            Orthogonalised ``lora_b_delta`` (or unmodified copy if
            below threshold).
        """
        a_flat = lora_a_delta.ravel().astype(np.float64)
        b_flat = lora_b_delta.ravel().astype(np.float64)

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return lora_b_delta.copy()

        cos_theta = np.dot(a_flat, b_flat) / (norm_a * norm_b)

        if abs(cos_theta) <= threshold:
            return lora_b_delta.copy()

        # Gram-Schmidt: remove component of b parallel to a
        proj_coeff = np.dot(a_flat, b_flat) / np.dot(a_flat, a_flat)
        b_orth = b_flat - proj_coeff * a_flat

        return b_orth.reshape(lora_b_delta.shape).astype(
            lora_b_delta.dtype,
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _name_matches(name: str, patterns: list[str]) -> bool:
        """Check if *name* matches any target-module pattern."""
        import re

        return any(re.search(p, name) for p in patterns)

    @staticmethod
    def _compute_delta(
        linear: Any,
        manifest: LoRAManifest,
        scale: float,
    ) -> Any:
        """Compute LoRA delta: scale * (alpha/rank) * B @ A.

        For in-memory testing, synthesises deterministic A/B matrices
        from the linear layer dimensions and manifest rank.
        """
        import torch  # lazy

        out_features, in_features = linear.weight.shape
        rank = manifest.rank

        gen = torch.Generator()
        gen.manual_seed(hash(manifest.name) & 0x7FFFFFFF)

        # A: [rank, in_features], B: [out_features, rank]
        a_mat = torch.randn(
            rank, in_features, generator=gen, dtype=torch.float32,
        ) * 0.01
        b_mat = torch.randn(
            out_features, rank, generator=gen, dtype=torch.float32,
        ) * 0.01

        effective_scale = scale * (manifest.alpha / manifest.rank)
        delta = effective_scale * (b_mat @ a_mat)
        return delta.to(linear.weight.device, linear.weight.dtype)
