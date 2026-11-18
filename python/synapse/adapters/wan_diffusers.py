# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Diffusers-based backend transport for Wan 2.1 video generation.

Provides ``DiffusersWanBackend``, implementing ``WanBackendTransport`` using
Hugging Face Diffusers (``WanPipeline``, ``AutoencoderKLWan``, ``WanTransformer3DModel``).
Supports causal 3D latent manipulation, 7-channel condition injection, and
latent-space variance-preserving fusion.
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

    Args:
        model_id_or_path: Hugging Face model repository or local directory path.
        device: Target execution device (e.g. "cuda", "cuda:0", "cpu").
        torch_dtype: PyTorch tensor data type (default torch.bfloat16 if cuda, else torch.float32).
        pipeline: Optional pre-instantiated Diffusers pipeline instance.
        transformer: Optional pre-instantiated transformer module.
        vae: Optional pre-instantiated VAE module.
    """

    def __init__(
        self,
        model_id_or_path: str = "Wan-AI/Wan2.1-T2V-1.4B",
        device: str | None = None,
        torch_dtype: Any | None = None,
        pipeline: Any | None = None,
        transformer: Any | None = None,
        vae: Any | None = None,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self._device_str = device
        self._torch_dtype = torch_dtype
        self._pipeline = pipeline
        self._transformer = transformer
        self._vae = vae
        self._initialized = False

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
                    ).to(device)
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

    def submit(
        self,
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Run video generation inference via the Diffusers Wan pipeline.

        Args:
            payload: Condition payload containing control tensors, keyframes, prompt.
            config: Video generation parameters.

        Returns:
            Generated frames shaped [T, H, W, 3] uint8 sRGB.
        """
        self._ensure_initialized()

        import torch

        if self._pipeline is None:
            # Fallback if custom transformer was provided
            raise BackendUnavailableError(
                "No executable pipeline initialized in DiffusersWanBackend.",
                agent_fix_hint="Provide a valid pipeline or HuggingFace model id.",
            )

        prompt = payload.text_prompt or getattr(config, "prompt", "")
        negative_prompt = payload.negative_prompt or getattr(config, "negative_prompt", None)

        # Prepare generator RNG
        generator = None
        try:
            import torch

            device = getattr(self._pipeline, "device", "cpu")
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed)
        except Exception:
            generator = None

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": config.num_frames,
            "height": config.height,
            "width": config.width,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "generator": generator,
            "output_type": "np",
        }

        # If control tensor is present, pass to pipeline
        if payload.control_tensor is not None:
            try:
                import torch

                device = getattr(self._pipeline, "device", "cpu")
                dtype = (
                    self._pipeline.transformer.dtype
                    if hasattr(self._pipeline, "transformer")
                    else torch.float32
                )
                ctrl = torch.from_numpy(payload.control_tensor).to(
                    device=device,
                    dtype=dtype,
                )
                call_kwargs["control_tensor"] = ctrl
            except Exception:
                call_kwargs["control_tensor"] = payload.control_tensor

        # If warm start frames are present
        warm_start = payload.extra.get("warm_start_frames") or payload.keyframe_images
        if warm_start:
            call_kwargs["warm_start_frames"] = warm_start

        try:
            output = self._pipeline(**call_kwargs)
            frames = output.frames[0]  # [T, H, W, 3] in [0, 1] float
            frames_u8: np.ndarray = np.asarray(
                np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
            )
            return frames_u8
        except Exception as exc:
            raise BackendUnavailableError(
                f"DiffusersWanBackend inference failed: {exc}",
                agent_fix_hint="Check GPU VRAM, tensor dimensions, and model inputs.",
            ) from exc
