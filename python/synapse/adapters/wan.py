"""Tier-1 Whitebox Wan 2.1 video model adapter.

Implements the ``WanAdapter`` — a whitebox adapter that converts FGPP
multi-channel sequences into ControlNeXt-compatible condition tensors
(camera-space normals, log-normalised depth, soft-knee inverse-tonemapped
radiance) and submits them to a configurable backend transport for Wan 2.1
video generation.

Transport backends:
* ``FoundaryHttpTransport`` — ships payloads to a ``foundary`` container
  over HTTP with W3C ``traceparent`` header support.
* ``LocalCallableBackend`` — synchronous in-process callable, perfect
  for unit-test injection.

LoRA manager integration is stubbed for M5 PR-16.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from synapse.adapters.base import (
    AdapterTier,
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoModelAdapter,
    VideoResult,
    register_adapter,
)
from synapse.errors import BackendUnavailableError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLOW_BIAS_GAMMA_MAX: float = 2.5
"""Maximum allowed value for ``gamma_flow_bias``."""

_RADIANCE_NORM_MAX: float = 12.0
"""Maximum output of ``soft_knee_inverse_tonemap`` (strictly < 12)."""

_BACKGROUND_NORMAL: np.ndarray = np.array(
    [0.0, 0.0, 1.0], dtype=np.float32,
)
"""Default normal for background pixels (zero-magnitude world normals)."""

# ---------------------------------------------------------------------------
# Backend transport ABCs & implementations
# ---------------------------------------------------------------------------


class WanBackendTransport(ABC):
    """Abstract transport for submitting generation payloads.

    Subclasses must implement ``submit`` (which runs the actual
    inference) and ``is_available`` (health/readiness probe).
    """

    @abstractmethod
    def submit(
        self,
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Submit a prepared payload and return generated frames.

        Args:
            payload: Condition payload from ``WanAdapter.prepare_conditions``.
            config: Generation configuration.

        Returns:
            ``[T, H, W, 3]`` uint8 sRGB frames.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend can accept work."""
        ...


class FoundaryHttpTransport(WanBackendTransport):
    """HTTP transport for the ``foundary`` container runtime.

    Serialises the condition payload as JSON and POSTs to the Wan 2.1
    inference endpoint, carrying a W3C ``traceparent`` header for
    distributed tracing.

    Raises ``BackendUnavailableError`` (exit_code=30) on timeout or
    connection-refused errors.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8090/v1/generate",
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s

    # -- WanBackendTransport -------------------------------------------------

    def submit(
        self,
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """POST the payload to the inference endpoint.

        Args:
            payload: Condition payload.
            config: Generation configuration.

        Returns:
            ``[T, H, W, 3]`` uint8 sRGB frames.

        Raises:
            BackendUnavailableError: On timeout, connection refused, or
                non-200 HTTP status.
        """
        import json
        import urllib.error
        import urllib.request

        body = json.dumps({
            "text_prompt": payload.text_prompt,
            "negative_prompt": payload.negative_prompt,
            "seed": config.seed,
            "fps": config.fps,
            "width": config.width,
            "height": config.height,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "num_frames": config.num_frames,
            "extra": config.extra,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "traceparent": _generate_traceparent(),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_s
            ) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                frames = np.array(
                    resp_data["frames"], dtype=np.uint8,
                )
                return frames
        except urllib.error.URLError as exc:
            raise BackendUnavailableError(
                f"Cannot reach Wan 2.1 endpoint at "
                f"{self.endpoint_url}: {exc}",
                agent_fix_hint=(
                    "Ensure the foundary container is running and "
                    "accessible. Restart with: "
                    "`docker compose up -d foundary`"
                ),
            ) from exc
        except TimeoutError as exc:
            raise BackendUnavailableError(
                f"Timeout ({self.timeout_s}s) contacting "
                f"{self.endpoint_url}",
                agent_fix_hint=(
                    "Increase timeout_s or check GPU load on the "
                    "foundary node."
                ),
            ) from exc

    def is_available(self) -> bool:
        """Probe the endpoint with a lightweight GET."""
        import urllib.error
        import urllib.request

        try:
            urllib.request.urlopen(
                self.endpoint_url, timeout=5.0,
            )
            return True
        except Exception:  # noqa: BLE001
            return False


class LocalCallableBackend(WanBackendTransport):
    """In-process callable backend for testing and local pipelines.

    Wraps any callable with signature
    ``(payload, config) -> np.ndarray`` where the return is
    ``[T, H, W, 3]`` uint8.
    """

    def __init__(
        self,
        pipeline_fn: Any,  # Callable[[PreparedConditionPayload, ...], ndarray]
    ) -> None:
        self._fn = pipeline_fn

    def submit(
        self,
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Invoke the wrapped callable directly.

        Args:
            payload: Condition payload.
            config: Generation configuration.

        Returns:
            ``[T, H, W, 3]`` uint8 sRGB frames.
        """
        return self._fn(payload, config)

    def is_available(self) -> bool:
        """Local backend is always available."""
        return True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _world_normal_to_camera(
    normals_world: np.ndarray,
    view_matrix: np.ndarray,
) -> np.ndarray:
    """Rotate world-space normals to camera space.

    Args:
        normals_world: ``[H, W, 3]`` float32 world-space normals.
        view_matrix: ``4x4`` row-major world-to-camera matrix.

    Returns:
        ``[H, W, 3]`` float32 camera-space normals.  Background pixels
        (zero-magnitude world normal) are mapped to ``[0, 0, 1]``.
    """
    h, w, _ = normals_world.shape
    normals = normals_world.astype(np.float64)

    # Detect background: magnitude near zero
    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    is_bg = mag < 1e-6  # [H, W, 1]

    # Extract 3x3 rotation from view matrix
    R = np.asarray(view_matrix[:3, :3], dtype=np.float64)

    # Rotate: result[i,j] = R @ normals[i,j]
    flat = normals.reshape(-1, 3)  # [H*W, 3]
    rotated = (R @ flat.T).T  # [H*W, 3]
    rotated = rotated.reshape(h, w, 3)

    # Re-normalise rotated normals
    rot_mag = np.linalg.norm(rotated, axis=-1, keepdims=True)
    rot_mag = np.where(rot_mag < 1e-9, 1.0, rot_mag)
    rotated = rotated / rot_mag

    # Replace background pixels with [0, 0, 1]
    bg_normal = _BACKGROUND_NORMAL.astype(np.float64)
    is_bg_3 = np.broadcast_to(is_bg, rotated.shape)
    result = np.where(is_bg_3, bg_normal, rotated)

    return result.astype(np.float32)


def _build_control_tensor(
    depth_norm: np.ndarray,
    cam_normals: np.ndarray,
    radiance_norm: np.ndarray,
) -> np.ndarray:
    """Stack condition channels into a single control tensor.

    Args:
        depth_norm: ``[H, W]`` float32 log-normalised depth in [0, 1].
        cam_normals: ``[H, W, 3]`` float32 camera-space normals.
        radiance_norm: ``[H, W, 3]`` float32 normalised radiance in
            [0, 1].

    Returns:
        ``[C, H, W]`` float32 where ``C = 1 + 3 + 3 = 7``.
    """
    # depth: [H, W] -> [1, H, W]
    d = depth_norm[np.newaxis, :, :]
    # normals: [H, W, 3] -> [3, H, W]
    n = np.transpose(cam_normals, (2, 0, 1))
    # radiance: [H, W, 3] -> [3, H, W]
    r = np.transpose(radiance_norm, (2, 0, 1))
    return np.concatenate([d, n, r], axis=0).astype(np.float32)


def _generate_traceparent() -> str:
    """Generate a minimal W3C ``traceparent`` header value.

    Uses ``os.urandom`` for the trace-id and parent-id segments.

    Returns:
        A ``traceparent`` string like
        ``00-<32hex>-<16hex>-01``.
    """
    import os

    trace_id = os.urandom(16).hex()
    parent_id = os.urandom(8).hex()
    return f"00-{trace_id}-{parent_id}-01"


# ---------------------------------------------------------------------------
# WanAdapter
# ---------------------------------------------------------------------------


@register_adapter
class WanAdapter(VideoModelAdapter):
    """Tier-1 whitebox adapter for Wan 2.1 (ControlNeXt-conditioned).

    Converts FGPP multi-channel sequences into ControlNeXt-compatible
    condition tensors (camera-space normals, log-normalised depth,
    soft-knee inverse-tonemapped radiance) and submits them to a
    configurable ``WanBackendTransport``.

    Args:
        backend: Transport backend for inference submission.  Defaults
            to ``None`` (must be set before calling ``generate``).
        gamma_flow_bias: Soft flow bias gamma for latent optical flow.
            Clamped to ``[0.0, 2.5]``.  ``0.0`` means no bias.
        lora_manager: Placeholder for M5 LoRA slot — unused in this PR.
    """

    def __init__(
        self,
        backend: WanBackendTransport | None = None,
        gamma_flow_bias: float = 0.0,
        lora_manager: Any | None = None,
    ) -> None:
        self._backend = backend
        self._gamma_flow_bias = float(
            np.clip(gamma_flow_bias, 0.0, _FLOW_BIAS_GAMMA_MAX)
        )
        self._lora_manager = lora_manager  # M5 PR-16 stub

    # -- properties ----------------------------------------------------------

    @property
    def adapter_id(self) -> str:
        return "wan2.1"

    @property
    def tier(self) -> AdapterTier:
        return "whitebox"

    @property
    def gamma_flow_bias(self) -> float:
        """Current latent flow bias gamma value."""
        return self._gamma_flow_bias

    # -- public interface ----------------------------------------------------

    def prepare_conditions(
        self,
        fgpp_sequence: Any,  # FGPPSequence
        config: VideoGenerationConfig,
    ) -> PreparedConditionPayload:
        """Transform an FGPP sequence into ControlNeXt condition tensors.

        Processing pipeline per frame:
        1. **Depth** — ``CameraLinearDepth.Z`` is log-normalised to
           [0, 1] via ``linear_to_log_normalized_depth``.
        2. **Normals** — ``WorldNormal.XYZ`` rotated from world to
           camera space using the frame's ``view_matrix[:3,:3]``.
           Background pixels (zero normal) → ``[0, 0, 1]``.
        3. **Radiance** — ``DirectRadiance.RGB`` passed through
           ``soft_knee_inverse_tonemap`` then normalised to [0, 1].
        4. Per-frame control tensors are stacked into
           ``[C_cond, T, H, W]``.

        Args:
            fgpp_sequence: Source ``FGPPSequence``.
            config: Generation configuration.

        Returns:
            ``PreparedConditionPayload`` with ``control_tensor`` shaped
            ``[C_cond, T, H, W]`` (C_cond = 7: 1 depth + 3 normal +
            3 radiance).
        """
        from synapse.fgpp.schema import linear_to_log_normalized_depth
        from synapse.style.aces_color import soft_knee_inverse_tonemap

        n_frames = len(fgpp_sequence.frames)
        if n_frames == 0:
            return PreparedConditionPayload()

        first = fgpp_sequence.frames[0]
        h, w = first.height, first.width

        per_frame_tensors: list[np.ndarray] = []

        for frame in fgpp_sequence.frames:
            # 1. Depth — log-normalised
            depth_raw = frame.depth.astype(np.float32)
            depth_norm = linear_to_log_normalized_depth(
                depth_raw,
                z_near_m=frame.camera.z_near,
                z_far_m=frame.camera.z_far,
            ).astype(np.float32)

            # 2. Normals — world → camera
            if frame.world_normal is not None:
                wn = frame.world_normal.astype(np.float32)
            else:
                wn = np.zeros((h, w, 3), dtype=np.float32)
            cam_normals = _world_normal_to_camera(
                wn, frame.camera.view_matrix,
            )

            # 3. Radiance — soft-knee inverse tonemap + normalise
            if frame.direct_radiance is not None:
                rad = frame.direct_radiance.astype(np.float64)
            else:
                rad = np.zeros((h, w, 3), dtype=np.float64)
            rad_tonemapped = soft_knee_inverse_tonemap(rad)
            # Normalise to [0, 1] (max output is < 12.0)
            rad_norm = np.clip(
                rad_tonemapped / _RADIANCE_NORM_MAX, 0.0, 1.0,
            ).astype(np.float32)

            # 4. Stack into [C, H, W]
            control = _build_control_tensor(
                depth_norm, cam_normals, rad_norm,
            )
            per_frame_tensors.append(control)

        # Stack into [T, C, H, W] then transpose to [C, T, H, W]
        stacked = np.stack(per_frame_tensors, axis=0)  # [T, C, H, W]
        condition_tensor = np.transpose(
            stacked, (1, 0, 2, 3),
        )  # [C, T, H, W]

        # Build metadata for flow bias / occlusion
        flow_meta: dict[str, Any] = {
            "flow_bias_gamma": self._gamma_flow_bias,
            "occlusion_mask": None,
        }

        return PreparedConditionPayload(
            control_tensor=condition_tensor,
            extra=flow_meta,
        )

    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> VideoResult:
        """Generate video frames via the configured backend transport.

        Args:
            conditions: Output of ``prepare_conditions()``.
            config: Generation configuration.

        Returns:
            ``VideoResult`` with sRGB uint8 frames.

        Raises:
            RuntimeError: If no backend transport has been configured.
        """
        if self._backend is None:
            raise RuntimeError(
                "WanAdapter has no backend transport configured. "
                "Pass a WanBackendTransport to the constructor."
            )

        raw_frames = self._backend.submit(conditions, config)

        return VideoResult(
            frames_srgb_u8=raw_frames,
            fps=config.fps,
            seed=config.seed,
            adapter_id=self.adapter_id,
        )
