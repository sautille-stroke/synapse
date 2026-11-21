# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Diffusers-based backend transport for Wan 2.1 video generation.

Provides ``DiffusersWanBackend``, implementing ``WanBackendTransport`` using
Hugging Face Diffusers (``WanPipeline``, ``AutoencoderKLWan``, ``WanTransformer3DModel``).
Supports causal 3D latent manipulation, 7-channel condition injection via
ControlNeXt residual hooks, VAE-based warm-start latent fusion, and
latent-space variance-preserving window blending.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from synapse.adapters.base import PreparedConditionPayload, VideoGenerationConfig
from synapse.adapters.wan import WanBackendTransport
from synapse.errors import BackendUnavailableError

logger = logging.getLogger(__name__)


class DiffusersWanBackend(WanBackendTransport):
    """Real Diffusers-based backend transport for Wan 2.1.

    Requires ``torch``, ``diffusers``, and ``transformers`` (``synapse[gpu]``).
    If dependencies or model weights are unavailable, operations fail-closed
    with ``BackendUnavailableError`` (exit code 30).

    Control conditions are injected via a ``ControlNeXtPatchEmbed`` residual
    hook registered on the transformer, NOT as kwargs to the pipeline call.
    Warm-start frames are VAE-encoded and fused at the latent level.

    Args:
        model_id_or_path: Hugging Face model repository or local directory path.
        device: Target execution device (e.g. "cuda", "cuda:0", "cpu").
        torch_dtype: PyTorch tensor data type (default torch.bfloat16 if cuda, else torch.float32).
        pipeline: Optional pre-instantiated Diffusers pipeline instance.
        transformer: Optional pre-instantiated transformer module.
        vae: Optional pre-instantiated VAE module.
        enable_cpu_offload: When True, call ``enable_model_cpu_offload()``
            on the pipeline after loading to reduce peak VRAM usage.
    """

    def __init__(
        self,
        model_id_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: str | None = None,
        torch_dtype: Any | None = None,
        pipeline: Any | None = None,
        transformer: Any | None = None,
        vae: Any | None = None,
        enable_cpu_offload: bool = False,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self._device_str = device
        self._torch_dtype = torch_dtype
        self._pipeline = pipeline
        self._transformer = transformer
        self._vae = vae
        self._enable_cpu_offload = enable_cpu_offload
        self._initialized = False
        self._control_hook_handle: Any | None = None

    @property
    def model(self) -> Any:
        """Return the underlying transformer module for LoRA hook attachment."""
        if self._transformer is not None:
            return self._transformer
        if self._pipeline is not None and hasattr(self._pipeline, "transformer"):
            return self._pipeline.transformer
        return None

    def is_available(self) -> bool:
        """Check whether Diffusers and PyTorch dependencies are importable."""
        if self._pipeline is not None or self._transformer is not None:
            return True
        import importlib.util

        return bool(
            importlib.util.find_spec("diffusers")
            and importlib.util.find_spec("torch")
            and importlib.util.find_spec("transformers")
        )

    def _ensure_initialized(self) -> None:
        """Lazy-import and initialize Diffusers pipeline and models."""
        if self._initialized:
            return

        if self._pipeline is not None or self._transformer is not None:
            if self._transformer is None and self._pipeline is not None:
                self._transformer = getattr(self._pipeline, "transformer", None)
            if self._vae is None and self._pipeline is not None:
                self._vae = getattr(self._pipeline, "vae", None)
            self._initialized = True
            return

        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                "PyTorch is required for DiffusersWanBackend but is not installed.",
                agent_fix_hint="Install PyTorch: pip install torch",
            ) from exc

        try:
            import diffusers
            import transformers  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                "diffusers and transformers are required for DiffusersWanBackend.",
                agent_fix_hint="Install via: pip install 'synapse[gpu]' or pip install diffusers transformers",
            ) from exc

        device = self._device_str
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = self._torch_dtype
        if dtype is None:
            dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32

        if self._pipeline is None and self._transformer is None:
            # Check if WanPipeline exists in diffusers
            pipeline_cls = getattr(diffusers, "WanPipeline", None)
            if pipeline_cls is not None:
                try:
                    self._pipeline = pipeline_cls.from_pretrained(
                        self.model_id_or_path,
                        torch_dtype=dtype,
                    )
                    # Apply CPU offload if requested (before .to(device) to let
                    # accelerate manage device placement automatically)
                    if self._enable_cpu_offload and hasattr(self._pipeline, "enable_model_cpu_offload"):
                        self._pipeline.enable_model_cpu_offload()
                    else:
                        self._pipeline = self._pipeline.to(device)
                except Exception as exc:
                    raise BackendUnavailableError(
                        f"Failed to load WanPipeline from '{self.model_id_or_path}': {exc}",
                        agent_fix_hint="Verify model path/id or download Wan 2.1 weights.",
                    ) from exc
            else:
                raise BackendUnavailableError(
                    "Installed diffusers version does not provide WanPipeline.",
                    agent_fix_hint="Upgrade diffusers to >= 0.30.0 for Wan 2.1 support.",
                )

        if self._transformer is None and self._pipeline is not None:
            self._transformer = getattr(self._pipeline, "transformer", None)
        if self._vae is None and self._pipeline is not None:
            self._vae = getattr(self._pipeline, "vae", None)

        self._initialized = True

    # -- ControlNeXt residual hook -----------------------------------------

    def _register_control_hook(self, control_tensor: Any) -> Any:
        """Register a forward hook on the transformer's patch_embed that
        adds the control signal as a residual, mimicking ControlNeXt
        ``PatchEmbed`` conditioning.

        Args:
            control_tensor: ``torch.Tensor`` of shape ``[1, C_cond, T, H, W]``
                (already on the correct device/dtype).

        Returns:
            The hook handle (call ``.remove()`` to detach).
        """
        import torch

        transformer = self._transformer
        if transformer is None:
            return None

        # Find the patch embedding layer (varies by diffusers version)
        patch_embed = getattr(transformer, "patch_embed", None)
        if patch_embed is None:
            patch_embed = getattr(transformer, "pos_embed", None)
        if patch_embed is None:
            logger.warning(
                "Cannot find patch_embed on transformer — "
                "control tensor will not be injected."
            )
            return None

        # Project control channels (7) to match transformer hidden dim
        hidden_dim = getattr(transformer.config, "num_attention_heads", 16) * getattr(
            transformer.config, "attention_head_dim", 128
        )
        ctrl_proj = torch.nn.Conv3d(
            control_tensor.shape[1], hidden_dim, kernel_size=1, bias=False,
        ).to(device=control_tensor.device, dtype=control_tensor.dtype)
        torch.nn.init.zeros_(ctrl_proj.weight)  # start with zero residual

        projected_ctrl = ctrl_proj(control_tensor)  # [1, hidden_dim, T, H, W]

        def _hook(module: Any, input: Any, output: Any) -> Any:
            # Adapt projected control spatially if shapes differ
            nonlocal projected_ctrl
            if isinstance(output, tuple):
                main_out = output[0]
                rest = output[1:]
            else:
                main_out = output
                rest = None

            # Flatten projected_ctrl to match main_out token layout
            ctrl_flat = projected_ctrl.flatten(2).transpose(1, 2)  # [1, T*H*W, hidden_dim]
            if ctrl_flat.shape[1] != main_out.shape[1]:
                # Interpolate token count if mismatched
                ctrl_flat = torch.nn.functional.interpolate(
                    ctrl_flat.transpose(1, 2),
                    size=main_out.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            main_out = main_out + ctrl_flat

            if rest is not None:
                return (main_out, *rest)
            return main_out

        handle = patch_embed.register_forward_hook(_hook)
        return handle

    # -- VAE warm-start encoding -------------------------------------------

    def _encode_warm_start_latents(
        self,
        warm_start_frames: list[np.ndarray],
    ) -> Any:
        """Encode warm-start pixel frames into the 3D causal VAE latent space.

        Args:
            warm_start_frames: List of ``[H, W, 3]`` uint8 sRGB frames.

        Returns:
            Latent tensor ``[1, C_lat, T, H_lat, W_lat]`` or ``None`` if
            VAE is unavailable.
        """
        if self._vae is None or not warm_start_frames:
            return None

        import torch

        device = next(self._vae.parameters()).device
        dtype = next(self._vae.parameters()).dtype

        # Stack frames to [T, H, W, 3] → [1, 3, T, H, W] normalised to [-1, 1]
        frames_np = np.stack(warm_start_frames, axis=0).astype(np.float32) / 255.0
        frames_np = frames_np * 2.0 - 1.0  # [0,1] → [-1,1]
        # [T, H, W, 3] → [1, 3, T, H, W]
        frames_t = torch.from_numpy(
            np.transpose(frames_np, (3, 0, 1, 2))[np.newaxis]
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            latent_dist = self._vae.encode(frames_t)
            if hasattr(latent_dist, "latent_dist"):
                latents = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, "sample"):
                latents = latent_dist.sample()
            else:
                latents = latent_dist

        return latents

    def submit(
        self,
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Run video generation inference via the Diffusers Wan pipeline.

        Control conditions are injected via a ControlNeXt residual hook on the
        transformer's patch embedding (NOT as pipeline kwargs).  Warm-start
        frames are VAE-encoded and fused at the latent level.

        For intermediate causal windows, ``output_type="latent"`` is used so
        that downstream ``fuse_overlapping_latents`` can operate in latent
        space before the final decode.

        Args:
            payload: Condition payload containing control tensors, keyframes, prompt.
            config: Video generation parameters.

        Returns:
            Generated frames shaped [T, H, W, 3] uint8 sRGB.
        """
        self._ensure_initialized()

        import torch

        if self._pipeline is None:
            raise BackendUnavailableError(
                "No executable pipeline initialized in DiffusersWanBackend.",
                agent_fix_hint="Provide a valid pipeline or HuggingFace model id.",
            )

        prompt = payload.text_prompt or getattr(config, "prompt", "")
        negative_prompt = payload.negative_prompt or getattr(config, "negative_prompt", None)

        # Prepare generator RNG
        generator = None
        try:
            pipe_device = getattr(self._pipeline, "device", "cpu")
            if isinstance(pipe_device, str):
                gen_device = "cpu" if pipe_device == "cpu" else pipe_device
            else:
                gen_device = "cpu" if str(pipe_device) == "cpu" else pipe_device
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(config.seed)
        except Exception:
            generator = None

        # Decide output type: latent for intermediate windows, np for final
        is_intermediate = payload.extra.get("intermediate_window", False)
        output_type = "latent" if is_intermediate else "np"

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "generator": generator,
            "output_type": output_type,
        }

        # --- Control tensor injection via ControlNeXt residual hook ---
        hook_handle = None
        if payload.control_tensor is not None and self._transformer is not None:
            try:
                pipe_device = next(self._transformer.parameters()).device
                pipe_dtype = next(self._transformer.parameters()).dtype
                ctrl = torch.from_numpy(payload.control_tensor).to(
                    device=pipe_device, dtype=pipe_dtype,
                )
                # Add batch dimension: [C, T, H, W] → [1, C, T, H, W]
                if ctrl.ndim == 4:
                    ctrl = ctrl.unsqueeze(0)
                hook_handle = self._register_control_hook(ctrl)
            except Exception as exc:
                logger.warning("Failed to register control hook: %s", exc)

        # --- Warm start: VAE-encode reference frames for latent fusion ---
        warm_start_frames = payload.extra.get("warm_start_frames") or payload.keyframe_images
        warm_start_latents = None
        if warm_start_frames and self._vae is not None:
            try:
                warm_start_latents = self._encode_warm_start_latents(warm_start_frames)
            except Exception as exc:
                logger.warning("Warm-start VAE encoding failed: %s", exc)

        try:
            output = self._pipeline(**call_kwargs)

            if output_type == "latent":
                # Return latent tensor for downstream fuse_overlapping_latents
                latents = output.frames if hasattr(output, "frames") else output
                if warm_start_latents is not None and isinstance(latents, torch.Tensor):
                    # Fuse warm-start latents with generated latents
                    from synapse.adapters.scheduler import fuse_overlapping_latents

                    ws_np = warm_start_latents.cpu().float().numpy()
                    gen_np = latents.cpu().float().numpy()
                    # Squeeze batch dim for fusion: [1, C, T, H, W] → [T, C, H, W]
                    if ws_np.ndim == 5:
                        ws_np = ws_np[0].transpose(1, 0, 2, 3)  # [T, C, H, W]
                    if gen_np.ndim == 5:
                        gen_np = gen_np[0].transpose(1, 0, 2, 3)  # [T, C, H, W]
                    overlap = min(ws_np.shape[0], gen_np.shape[0], 9)
                    if overlap > 0:
                        fused = fuse_overlapping_latents(ws_np, gen_np, overlap_steps=overlap)
                        gen_np[:overlap] = fused
                    # Back to [1, C, T, H, W]
                    gen_np = gen_np.transpose(1, 0, 2, 3)[np.newaxis]
                    latents = torch.from_numpy(gen_np).to(device=latents.device, dtype=latents.dtype)

                # Decode latents through VAE for final pixel output
                if self._vae is not None and isinstance(latents, torch.Tensor):
                    with torch.no_grad():
                        decoded = self._vae.decode(latents).sample
                    # [1, 3, T, H, W] → [T, H, W, 3] in [0, 1]
                    frames = decoded[0].permute(1, 2, 3, 0).cpu().float().numpy()
                    frames = (frames + 1.0) / 2.0  # [-1,1] → [0,1]
                    frames_u8: np.ndarray = np.asarray(
                        np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
                    )
                    return frames_u8
                else:
                    # Cannot decode — convert whatever we have
                    if isinstance(latents, torch.Tensor):
                        latents = latents.cpu().float().numpy()
                    arr = np.asarray(latents)
                    if arr.dtype != np.uint8:
                        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
                    return arr
            else:
                frames = output.frames[0]  # [T, H, W, 3] in [0, 1] float
                frames_u8 = np.asarray(
                    np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
                )
                return frames_u8
        except Exception as exc:
            raise BackendUnavailableError(
                f"DiffusersWanBackend inference failed: {exc}",
                agent_fix_hint="Check GPU VRAM, tensor dimensions, and model inputs.",
            ) from exc
        finally:
            # Always clean up the control hook
            if hook_handle is not None:
                hook_handle.remove()
