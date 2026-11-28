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
from pathlib import Path
from typing import Any

import numpy as np

from synapse.adapters.base import PreparedConditionPayload, VideoGenerationConfig
from synapse.adapters.wan import WanBackendTransport
from synapse.errors import AssetNotFoundError, BackendUnavailableError, SynapseError

logger = logging.getLogger(__name__)


def _build_controlnext_patchify_stem(
    in_channels: int,
    hidden_dim: int,
    spatial_stride_h: int = 16,
    spatial_stride_w: int = 16,
    temporal_stride: int = 4,
) -> Any:
    """Create a strided ControlNeXtPatchifyStem ``nn.Module`` lazily.

    Compresses pixel-space ``[B, C_cond, T, H, W]`` conditions by causal 4x
    temporal binning (isolating frame 0) and ``(spatial_stride_h, spatial_stride_w)``
    strided patchify projection directly into ``[B, hidden_dim, tau, H//sh, W//sw]``,
    reducing activation memory by ~1024x compared to full-resolution ``kernel_size=1``.
    """
    import torch

    class ControlNeXtPatchifyStem(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.hidden_dim = hidden_dim
            self.temporal_stride = max(1, temporal_stride)
            self.spatial_stride_h = max(1, spatial_stride_h)
            self.spatial_stride_w = max(1, spatial_stride_w)
            self.proj = torch.nn.Conv3d(
                in_channels,
                hidden_dim,
                kernel_size=(1, self.spatial_stride_h, self.spatial_stride_w),
                stride=(1, self.spatial_stride_h, self.spatial_stride_w),
                bias=False,
            )

        def forward(self, x: Any) -> Any:
            # Causal temporal compression: frame 0 isolated, rest binned by temporal_stride
            _, _, t, h, w = x.shape
            if t > 1 and self.temporal_stride > 1:
                tau = (t - 1) // self.temporal_stride + 1
                chunks = [x[:, :, 0:1]]
                for i in range(1, tau):
                    s = 1 + (i - 1) * self.temporal_stride
                    e = min(s + self.temporal_stride, t)
                    chunks.append(x[:, :, s:e].mean(dim=2, keepdim=True))
                x = torch.cat(chunks, dim=2)
            # Pad spatial dims if not divisible by stride
            pad_h = (self.spatial_stride_h - (h % self.spatial_stride_h)) % self.spatial_stride_h
            pad_w = (self.spatial_stride_w - (w % self.spatial_stride_w)) % self.spatial_stride_w
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            return self.proj(x)

    return ControlNeXtPatchifyStem()


def _get_vae_latent_stats(vae: Any, device: Any, dtype: Any) -> tuple[Any, Any]:
    """Extract AutoencoderKLWan ``latents_mean`` and ``latents_std`` as ``[1, C, 1, 1, 1]``."""
    import torch

    cfg = getattr(vae, "config", None)
    if cfg is None:
        return None, None
    mean_raw = getattr(cfg, "latents_mean", None)
    std_raw = getattr(cfg, "latents_std", None)
    if mean_raw is None or std_raw is None:
        return None, None
    mean_t = torch.as_tensor(mean_raw, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std_t = torch.as_tensor(std_raw, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean_t, std_t


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
        control_weights_path: str | Path | None = None,
        control_scale: float = 1.0,
        allow_untrained_control: bool = False,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self._device_str = device
        self._torch_dtype = torch_dtype
        self._pipeline = pipeline
        self._transformer = transformer
        self._vae = vae
        self._enable_cpu_offload = enable_cpu_offload
        self._control_weights_path = (
            Path(control_weights_path) if control_weights_path is not None else None
        )
        self._control_scale = float(control_scale)
        self._allow_untrained_control = bool(allow_untrained_control)
        self._initialized = False
        self._control_hook_handle: Any | None = None
        self._control_stem: Any | None = None

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

        # Project control channels (7) via strided ControlNeXtPatchifyStem
        # (stride=(4, 16, 16) clamped to input dims) to avoid full-res OOM.
        hidden_dim = getattr(transformer.config, "num_attention_heads", 16) * getattr(
            transformer.config, "attention_head_dim", 128
        )
        _, in_ch, _, h_in, w_in = control_tensor.shape
        sh = min(16, int(h_in))
        sw = min(16, int(w_in))

        if self._control_weights_path is None and not self._allow_untrained_control:
            raise BackendUnavailableError(
                "ControlNeXt weights are required for 7-channel FGPP control injection "
                "(refusing to silently execute a zero-initialized no-op hook).",
                agent_fix_hint=(
                    "Pass control_weights_path=<path/to/controlnext.safetensors> to "
                    "DiffusersWanBackend, or set allow_untrained_control=True for synthetic tests."
                ),
            )

        stem = _build_controlnext_patchify_stem(
            in_channels=int(in_ch),
            hidden_dim=int(hidden_dim),
            spatial_stride_h=sh,
            spatial_stride_w=sw,
            temporal_stride=4,
        ).to(device=control_tensor.device, dtype=control_tensor.dtype)

        if self._control_weights_path is not None:
            if not self._control_weights_path.is_file():
                raise AssetNotFoundError(
                    f"ControlNeXt weights file not found: {self._control_weights_path}",
                    agent_fix_hint=(
                        "Verify control_weights_path points to an existing .safetensors or .pt file."
                    ),
                )
            if self._control_weights_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(self._control_weights_path), device=str(control_tensor.device))
            else:
                state_dict = torch.load(
                    str(self._control_weights_path),
                    map_location=control_tensor.device,
                    weights_only=True,
                )
            stem.load_state_dict(state_dict, strict=True)
        else:
            # Deterministic non-zero small initialization for explicit allow_untrained_control tests
            gen = torch.Generator(device="cpu").manual_seed(42)
            with torch.no_grad():
                w_init = torch.randn(stem.proj.weight.shape, generator=gen, dtype=torch.float32) * 0.02
                stem.proj.weight.copy_(w_init.to(device=control_tensor.device, dtype=control_tensor.dtype))

        self._control_stem = stem
        with torch.no_grad():
            projected_ctrl = stem(control_tensor) * self._control_scale  # [1, hidden_dim, tau, H', W']

        def _hook(module: Any, inputs: Any, output: Any) -> Any:
            _ = (module, inputs)
            # Adapt projected control spatially & batch-wise (CFG B=2 broadcast)
            nonlocal projected_ctrl
            if isinstance(output, tuple):
                main_out = output[0]
                rest = output[1:]
            else:
                main_out = output
                rest = None

            # Flatten projected_ctrl to match main_out token layout: [B_c, L_c, hidden_dim]
            ctrl_flat = projected_ctrl.flatten(2).transpose(1, 2)
            if ctrl_flat.shape[1] != main_out.shape[1]:
                # Interpolate token count if mismatched
                ctrl_flat = torch.nn.functional.interpolate(
                    ctrl_flat.transpose(1, 2),
                    size=main_out.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            # CFG batch broadcast (e.g. main_out has B=2 for uncond+cond while ctrl has B=1)
            if ctrl_flat.shape[0] == 1 and main_out.shape[0] > 1:
                ctrl_flat = ctrl_flat.expand(main_out.shape[0], -1, -1)

            ctrl_flat = ctrl_flat.to(device=main_out.device, dtype=main_out.dtype)
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

            # Normalize into AutoencoderKLWan's standardized latent space
            lat_mean, lat_std = _get_vae_latent_stats(self._vae, latents.device, latents.dtype)
            if lat_mean is not None and lat_std is not None:
                latents = (latents - lat_mean) / lat_std

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
            except SynapseError:
                raise
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
                        decode_in = latents
                        lat_mean, lat_std = _get_vae_latent_stats(
                            self._vae, decode_in.device, decode_in.dtype
                        )
                        if lat_mean is not None and lat_std is not None:
                            decode_in = decode_in * lat_std + lat_mean
                        decoded = self._vae.decode(decode_in).sample
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
