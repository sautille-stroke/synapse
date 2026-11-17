"""Multi-LoRA manager with Cryptomatte-aware spatial masking.

Provides ``MultiLoRAManager`` for applying, restoring, and spatially
masking LoRA weight deltas on ``nn.Module`` instances.  Includes
Gram-Schmidt orthogonalisation of conflicting LoRA deltas and
Cryptomatte-driven foreground/background mask construction.

Heavy dependencies (``torch``, ``safetensors``) are lazy-imported
inside methods — only ``numpy`` is used at module level.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Iterator
from typing import Any

import numpy as np

from synapse.adapters.lora.manifest import LoRAManifest
from synapse.errors import AssetNotFoundError, ContractValidationError

# ---------------------------------------------------------------------------
# MultiLoRAManager
# ---------------------------------------------------------------------------


class MultiLoRAManager:
    """Manage multiple LoRA adapters with spatial mask gating.

    Args:
        lora_bundles: List of ``(manifest, scale)`` pairs.  Each
            bundle specifies a LoRA adapter manifest and the runtime
            scale multiplier to apply during weight merging.
        test_synthetic: If ``True``, synthesise deterministic A/B
            matrices from the layer shape instead of reading real
            ``.safetensors`` files.  **Only for unit tests.**
    """

    def __init__(
        self,
        lora_bundles: list[tuple[LoRAManifest, float]],
        *,
        test_synthetic: bool = False,
    ) -> None:
        self._bundles = lora_bundles
        self._original_weights: dict[int, tuple[Any, Any]] = {}
        self._test_synthetic = test_synthetic
        # Cache of loaded safetensors state dicts keyed by adapter_path
        self._loaded_state_dicts: dict[str, dict[str, Any]] = {}

    # -- public API ----------------------------------------------------------

    def load_lora(self, manifest: LoRAManifest) -> dict[str, Any]:
        """Load and verify a ``.safetensors`` LoRA adapter file.

        Reads the file at ``manifest.adapter_path``, verifies its
        SHA-256 against ``manifest.adapter_sha256``, and returns the
        state dict of tensors.

        Args:
            manifest: LoRA manifest specifying the adapter file and
                expected SHA-256.

        Returns:
            A dict mapping layer keys to tensors.

        Raises:
            AssetNotFoundError: If the file does not exist (exit 20).
            ContractValidationError: If SHA-256 mismatch (exit 10).
        """
        path = manifest.adapter_path
        if path in self._loaded_state_dicts:
            return self._loaded_state_dicts[path]

        if not os.path.isfile(path):
            raise AssetNotFoundError(
                f"LoRA adapter file not found: {path}",
                agent_fix_hint=(
                    f"Ensure the .safetensors file exists at "
                    f"'{path}'. Re-download or re-export the adapter."
                ),
                details={"adapter_path": path},
            )

        # SHA-256 verification
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        actual_sha = sha.hexdigest()

        if actual_sha != manifest.adapter_sha256:
            raise ContractValidationError(
                f"SHA-256 mismatch for '{path}': "
                f"expected {manifest.adapter_sha256}, "
                f"got {actual_sha}",
                agent_fix_hint=(
                    "Re-download the adapter file or update the "
                    "manifest SHA-256 hash."
                ),
                details={
                    "adapter_path": path,
                    "expected_sha256": manifest.adapter_sha256,
                    "actual_sha256": actual_sha,
                },
            )

        # Lazy import safetensors
        from safetensors.torch import load_file
        state_dict = load_file(path)
        self._loaded_state_dicts[path] = state_dict
        return state_dict

    def apply_lora_stack(self, module: Any) -> None:
        """Apply merged and orthogonalised LoRA deltas to *module*.

        For each consecutive pair of LoRA bundles, checks cosine
        similarity and orthogonalises if conflicting.  Then applies
        the combined deltas to matching ``nn.Linear`` layers.

        Original weights are stored for later restoration via
        :meth:`restore_base_weights`.

        Args:
            module: A ``torch.nn.Module`` whose ``named_modules()``
                will be iterated.
        """
        import torch  # lazy

        # Collect per-layer deltas from all bundles
        layer_deltas: dict[str, list[np.ndarray]] = {}

        for name, submodule in module.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue

            for manifest, scale in self._bundles:
                if not self._name_matches(name, manifest.target_modules):
                    continue

                delta = self._compute_delta(
                    submodule, manifest, scale, module_name=name,
                )
                delta_np = delta.detach().cpu().float().numpy()
                if name not in layer_deltas:
                    layer_deltas[name] = []
                layer_deltas[name].append(delta_np)

        # Orthogonalise pairwise via Modified Gram-Schmidt then apply combined delta
        for name, submodule in module.named_modules():
            if name not in layer_deltas:
                continue

            deltas = layer_deltas[name]
            n_deltas = len(deltas)
            if n_deltas >= 2:
                # Full iterative Modified Gram-Schmidt across all delta pairs
                for i in range(n_deltas):
                    for j in range(i + 1, n_deltas):
                        deltas[j] = self.orthogonalize_conflicting_loras(
                            deltas[i], deltas[j],
                        )

            combined = sum(deltas[1:], deltas[0])

            key = id(submodule)
            if key not in self._original_weights:
                self._original_weights[key] = (
                    submodule,
                    submodule.weight.data.clone(),
                )

            combined_t = torch.tensor(
                combined,
                dtype=submodule.weight.dtype,
                device=submodule.weight.device,
            )
            submodule.weight.data.add_(combined_t)

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

                delta = self._compute_delta(
                    submodule, manifest, scale, module_name=name,
                )
                submodule.weight.data.add_(delta)

    def restore_base_weights(self, module: Any) -> None:
        """Restore all modified weights to their pre-LoRA values.

        Args:
            module: The same ``torch.nn.Module`` passed to
                :meth:`apply_merged_weights`.

        Raises:
            ContractValidationError: If NaN/Inf weights are detected.
            RuntimeError: If max element-wise restoration error
                exceeds ``1e-7``.
        """
        import torch  # lazy

        for key, (submodule, original_w) in self._original_weights.items():
            if not torch.isfinite(submodule.weight.data).all():
                raise ContractValidationError(
                    f"Weight corruption detected (NaN/Inf) in module {key}",
                    agent_fix_hint="Check LoRA training hyperparameters or scaling factors.",
                )

            submodule.weight.data.copy_(original_w)

            if not torch.isfinite(submodule.weight.data).all():
                raise ContractValidationError(
                    f"Restored weight contains NaN/Inf in module {key}",
                    agent_fix_hint="Original weight was corrupted.",
                )

            max_err = (
                (submodule.weight.data - original_w).abs().max().item()
            )
            if max_err > 1e-7:
                raise RuntimeError(
                    f"Weight restoration error {max_err:.2e} exceeds "
                    f"tolerance 1e-7 for module {key}"
                )

        self._original_weights.clear()

    @contextlib.contextmanager
    def manage_spatial_hooks(
        self,
        module: Any,
        cryptomatte_ids: np.ndarray | None = None,
        target_entity_ids: set[int] | None = None,
    ) -> Iterator[None]:
        """Context manager to attach spatial LoRAPatchHooks with iterative
        Modified Gram-Schmidt orthogonalisation, guaranteeing cleanup on exit.

        Routes bundles by category:
        - "identity": foreground mask M_fg (target entities)
        - "style": background mask ~M_fg
        - other: no spatial mask (None)
        """
        import torch  # lazy

        from synapse.adapters.lora.patcher import LoRAPatchHook

        attached_hooks: list[tuple[LoRAPatchHook, Any]] = []

        try:
            # Build spatial masks if cryptomatte IDs are present
            m_fg: np.ndarray | None = None
            if cryptomatte_ids is not None:
                if target_entity_ids is None or len(target_entity_ids) == 0:
                    target_entity_ids = {
                        int(v) for v in np.unique(cryptomatte_ids) if int(v) != 0
                    }
                m_fg = self.build_spatial_lora_masks(
                    cryptomatte_ids, target_entity_ids
                )

            # Collect deltas per submodule name across all bundles
            layer_deltas: dict[str, list[tuple[LoRAManifest, np.ndarray, Any]]] = {}

            for name, submodule in module.named_modules():
                if not isinstance(submodule, torch.nn.Linear):
                    continue

                for manifest, scale in self._bundles:
                    if not self._name_matches(name, manifest.target_modules):
                        continue

                    delta = self._compute_delta(
                        submodule, manifest, scale, module_name=name,
                    )
                    delta_np = delta.detach().cpu().float().numpy()
                    if name not in layer_deltas:
                        layer_deltas[name] = []
                    layer_deltas[name].append((manifest, delta_np, submodule))

            # Orthogonalize deltas across conflicting pairs and attach hooks
            for name, bundle_list in layer_deltas.items():
                if len(bundle_list) >= 2:
                    deltas = [b[1] for b in bundle_list]
                    for i in range(len(deltas)):
                        for j in range(i + 1, len(deltas)):
                            deltas[j] = self.orthogonalize_conflicting_loras(
                                deltas[i], deltas[j]
                            )
                    bundle_list = [
                        (b[0], deltas[idx], b[2])
                        for idx, b in enumerate(bundle_list)
                    ]

                for manifest, delta_np, submodule in bundle_list:
                    # Select category-specific mask
                    mask_to_use: Any | None = None
                    if m_fg is not None:
                        if manifest.category == "identity":
                            mask_to_use = m_fg.astype(np.float32)
                        elif manifest.category == "style":
                            mask_to_use = (~m_fg).astype(np.float32)

                    hook = LoRAPatchHook(delta_np)
                    hook.attach(submodule, mask=mask_to_use, module_name=name)
                    attached_hooks.append((hook, submodule))

            yield
        finally:
            for hook, submodule in attached_hooks:
                hook.detach(submodule)
            attached_hooks.clear()

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

        if not target_entity_ids:
            return np.zeros((t, h_ds, w_ds), dtype=bool)

        try:
            from scipy import stats  # lazy

            cropped = cryptomatte_ids[:, :h_ds * spatial_block, :w_ds * spatial_block]
            blocks = (
                cropped.reshape(t, h_ds, spatial_block, w_ds, spatial_block)
                .transpose(0, 1, 3, 2, 4)
                .reshape(t, h_ds, w_ds, spatial_block * spatial_block)
            )
            dominant = np.asarray(
                stats.mode(blocks, axis=-1, keepdims=False).mode
            ).reshape(t, h_ds, w_ds)
            return np.isin(dominant, list(target_entity_ids))
        except ImportError:
            mask = np.zeros((t, h_ds, w_ds), dtype=bool)
            for ti in range(t):
                for yi in range(h_ds):
                    for xi in range(w_ds):
                        block = cryptomatte_ids[
                            ti,
                            yi * spatial_block : (yi + 1) * spatial_block,
                            xi * spatial_block : (xi + 1) * spatial_block,
                        ]
                        unique, counts = np.unique(
                            block.ravel(), return_counts=True,
                        )
                        dominant_val = unique[np.argmax(counts)]
                        mask[ti, yi, xi] = int(dominant_val) in target_entity_ids
            return mask

    @staticmethod
    def orthogonalize_conflicting_loras(
        lora_a_delta: np.ndarray,
        lora_b_delta: np.ndarray,
        threshold: float = 0.25,
    ) -> np.ndarray:
        """Modified Gram-Schmidt project *lora_b_delta* away from *lora_a_delta*.

        If the absolute cosine similarity between the flattened deltas
        exceeds *threshold*, the component of ``lora_b`` parallel to
        ``lora_a`` is subtracted (Modified Gram-Schmidt with re-projection).

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
            return np.asarray(lora_b_delta.copy())

        cos_theta = np.dot(a_flat, b_flat) / (norm_a * norm_b)

        if abs(cos_theta) <= threshold:
            return np.asarray(lora_b_delta.copy())

        # Modified Gram-Schmidt projection + iterative refinement
        denom = np.dot(a_flat, a_flat)
        proj_coeff = np.dot(a_flat, b_flat) / denom
        b_orth = b_flat - proj_coeff * a_flat

        residual_coeff = np.dot(a_flat, b_orth) / denom
        b_orth = b_orth - residual_coeff * a_flat

        return np.asarray(
            b_orth.reshape(lora_b_delta.shape).astype(lora_b_delta.dtype)
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _name_matches(name: str, patterns: list[str]) -> bool:
        """Check if *name* matches any target-module pattern."""
        import fnmatch
        import re

        for p in patterns:
            if fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name, f"*{p}*"):
                return True
            safe_pattern = re.escape(p).replace(r"\*", ".*")
            if re.search(safe_pattern, name):
                return True
        return False

    def _compute_delta(
        self,
        linear: Any,
        manifest: LoRAManifest,
        scale: float,
        module_name: str | None = None,
    ) -> Any:
        """Compute LoRA delta: scale * (alpha/rank) * B @ A.

        When ``test_synthetic`` is ``True``, synthesises deterministic
        A/B matrices from the linear layer dimensions and manifest rank.
        Otherwise reads real ``lora_A``/``lora_B`` tensors from the
        ``.safetensors`` file.
        """
        import torch  # lazy

        out_features, in_features = linear.weight.shape
        rank = manifest.rank

        if self._test_synthetic:
            gen = torch.Generator()
            gen.manual_seed(hash(manifest.name) & 0x7FFFFFFF)

            # A: [rank, in_features], B: [out_features, rank]
            a_mat = torch.randn(
                rank, in_features, generator=gen, dtype=torch.float32,
            ) * 0.01
            b_mat = torch.randn(
                out_features, rank, generator=gen, dtype=torch.float32,
            ) * 0.01
        else:
            state_dict = self.load_lora(manifest)

            # Find matching A and B tensors in the state dict
            a_mat, b_mat = _extract_lora_ab(
                state_dict, in_features, out_features, rank, module_name=module_name,
            )

        effective_scale = scale * (manifest.alpha / manifest.rank)
        delta = effective_scale * (b_mat @ a_mat)
        return delta.to(linear.weight.device, linear.weight.dtype)


def _extract_lora_ab(
    state_dict: dict[str, Any],
    in_features: int,
    out_features: int,
    rank: int,
    module_name: str | None = None,
) -> tuple[Any, Any]:
    """Extract lora_A and lora_B tensors from a safetensors state dict.

    Searches for keys containing ``lora_A`` and ``lora_B`` (or
    ``lora_down`` / ``lora_up``) matching *module_name*.

    Returns:
        Tuple of ``(a_mat, b_mat)`` tensors.

    Raises:
        ContractValidationError: If ambiguous keys or dimension mismatch.
    """
    import torch  # lazy

    filtered_dict = state_dict
    if module_name:
        matching = {k: v for k, v in state_dict.items() if module_name in k}
        if not matching:
            # Try leaf component (e.g. "spatial_attn.to_q")
            parts = module_name.split(".")
            if len(parts) >= 2:
                short_name = ".".join(parts[-2:])
                matching = {k: v for k, v in state_dict.items() if short_name in k}
        if matching:
            filtered_dict = matching

    a_candidates: list[str] = []
    b_candidates: list[str] = []

    for key, tensor in filtered_dict.items():
        lk = key.lower()
        if "lora_a" in lk or "lora_down" in lk:
            if tensor.shape == torch.Size([rank, in_features]):
                a_candidates.append(key)
            elif module_name and module_name in key:
                raise ContractValidationError(
                    f"Dimension mismatch for lora_A key '{key}': "
                    f"expected [{rank}, {in_features}], got {list(tensor.shape)}",
                    agent_fix_hint="Verify adapter target_modules and rank configuration.",
                )
        elif "lora_b" in lk or "lora_up" in lk:
            if tensor.shape == torch.Size([out_features, rank]):
                b_candidates.append(key)
            elif module_name and module_name in key:
                raise ContractValidationError(
                    f"Dimension mismatch for lora_B key '{key}': "
                    f"expected [{out_features}, {rank}], got {list(tensor.shape)}",
                    agent_fix_hint="Verify adapter target_modules and rank configuration.",
                )

    if len(a_candidates) > 1 and module_name:
        exact_a = [k for k in a_candidates if module_name in k]
        if len(exact_a) == 1:
            a_candidates = exact_a
        elif len(exact_a) > 1:
            raise ContractValidationError(
                f"Ambiguous lora_A candidate keys for module '{module_name}': {exact_a}",
                agent_fix_hint="Ensure adapter keys uniquely identify target modules.",
            )

    if len(b_candidates) > 1 and module_name:
        exact_b = [k for k in b_candidates if module_name in k]
        if len(exact_b) == 1:
            b_candidates = exact_b
        elif len(exact_b) > 1:
            raise ContractValidationError(
                f"Ambiguous lora_B candidate keys for module '{module_name}': {exact_b}",
                agent_fix_hint="Ensure adapter keys uniquely identify target modules.",
            )

    a_key = a_candidates[0] if a_candidates else None
    b_key = b_candidates[0] if b_candidates else None

    if a_key is None or b_key is None:
        raise ContractValidationError(
            f"Cannot find lora_A [{rank}, {in_features}] and "
            f"lora_B [{out_features}, {rank}] in safetensors state dict "
            f"(module_name={module_name}). Available keys: {list(state_dict.keys())}",
            agent_fix_hint=(
                "Ensure the .safetensors file contains 'lora_A'/'lora_B' "
                "or 'lora_down'/'lora_up' tensors with correct shapes matching target module."
            ),
        )

    return state_dict[a_key].float(), state_dict[b_key].float()
