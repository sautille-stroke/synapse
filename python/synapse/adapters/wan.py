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

LoRA manager integration wired in PR-42: ``MultiLoRAManager`` is lazily
constructed when ``lora_manifests`` are provided, and applied/restored
around each ``generate()`` call when Cryptomatte IDs are present.
"""

from __future__ import annotations

import base64
import contextlib
import io
from abc import ABC, abstractmethod
from typing import Any, cast

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
        traceparent: str | None = None,
    ) -> None:
        if endpoint_url:
            from synapse.bridge.http_security import validate_safe_url

            validate_safe_url(endpoint_url)
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s
        self._upstream_traceparent = traceparent

    # -- WanBackendTransport -------------------------------------------------

    def submit(
        self,
        payload: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """POST the payload to the inference endpoint.

        Fully serialises ``control_tensor`` (compressed binary),
        ``keyframe_images``, and ``payload.extra``, and forwards the
        upstream ``traceparent`` from ``TraceContext`` if provided.

        Args:
            payload: Condition payload.
            config: Generation configuration.

        Returns:
            ``[T, H, W, 3]`` uint8 sRGB frames.

        Raises:
            BackendUnavailableError: On timeout, connection refused, or
                non-200 HTTP status.
        """
        import http.client
        import json
        import urllib.error
        import urllib.request

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url

        validate_safe_url(self.endpoint_url)

        # Serialize control_tensor as compressed base64 numpy binary
        control_tensor_b64: str | None = None
        if payload.control_tensor is not None:
            buf = io.BytesIO()
            np.save(buf, payload.control_tensor)
            control_tensor_b64 = base64.b64encode(
                buf.getvalue()
            ).decode("ascii")

        # Serialize keyframe images as base64 list
        keyframe_images_b64: list[str] = []
        for kf in payload.keyframe_images:
            buf = io.BytesIO()
            np.save(buf, kf)
            keyframe_images_b64.append(
                base64.b64encode(buf.getvalue()).decode("ascii")
            )

        body = json.dumps({
            "text_prompt": payload.text_prompt,
            "negative_prompt": payload.negative_prompt,
            "control_tensor": control_tensor_b64,
            "keyframe_images": keyframe_images_b64,
            "seed": config.seed,
            "fps": config.fps,
            "width": config.width,
            "height": config.height,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "num_frames": config.num_frames,
            "extra": {**config.extra, **payload.extra},
        }).encode("utf-8")

        # Forward upstream traceparent or generate a new one
        traceparent = (
            self._upstream_traceparent
            if self._upstream_traceparent
            else _generate_traceparent()
        )

        req = urllib.request.Request(
            self.endpoint_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "traceparent": traceparent,
            },
            method="POST",
        )

        try:
            with safe_urlopen(
                req, timeout=self.timeout_s
            ) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                if isinstance(resp_data, dict) and resp_data.get("status") == "error":
                    raise BackendUnavailableError(
                        f"Wan 2.1 endpoint reported error: {resp_data.get('error', 'unknown')}",
                        agent_fix_hint="Check foundary container logs and GPU health.",
                    )
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
        except (TimeoutError, http.client.HTTPException, json.JSONDecodeError) as exc:
            raise BackendUnavailableError(
                f"Error or timeout ({self.timeout_s}s) contacting "
                f"{self.endpoint_url}: {exc}",
                agent_fix_hint=(
                    "Increase timeout_s or check GPU load on the "
                    "foundary node."
                ),
            ) from exc

    def is_available(self) -> bool:
        """Probe the endpoint with /healthz or lightweight check."""
        import urllib.error
        import urllib.parse
        import urllib.request

        from synapse.bridge.http_security import safe_urlopen, validate_safe_url

        try:
            validate_safe_url(self.endpoint_url)
        except Exception:
            return False

        parsed = urllib.parse.urlparse(self.endpoint_url)
        health_url = urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, "/healthz", "", "", "")
        )
        for url in (health_url, self.endpoint_url):
            try:
                with safe_urlopen(url, timeout=5.0) as resp:
                    if resp.status in (200, 204):
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code in (200, 204, 405):
                    return True
            except Exception:
                continue
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
        return cast(np.ndarray, self._fn(payload, config))

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


def _encode_depth(
    depth_raw: np.ndarray,
    z_near: float,
    z_far: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-normalise raw depth preserving float64 until quantisation.

    Args:
        depth_raw: ``[H, W]`` raw linear depth.
        z_near: Near plane in metres.
        z_far: Far plane in metres.

    Returns:
        Tuple of ``(depth_norm_f64, depth_norm_f32)`` where the f64
        version is the authoritative precision and f32 is for tensor
        construction.
    """
    from synapse.fgpp.schema import linear_to_log_normalized_depth

    depth_raw_f64 = depth_raw.astype(np.float64)
    depth_norm_f64 = linear_to_log_normalized_depth(
        depth_raw_f64, z_near_m=z_near, z_far_m=z_far,
    )
    depth_norm_f32 = depth_norm_f64.astype(np.float32)
    return depth_norm_f64, depth_norm_f32


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
        lora_manifests: Optional list of ``(LoRAManifest, scale)`` tuples.
            When provided, a ``MultiLoRAManager`` is constructed lazily
            and applied around each ``generate()`` call when Cryptomatte
            IDs are available on the FGPP conditions payload.
    """

    def __init__(
        self,
        backend: WanBackendTransport | None = None,
        gamma_flow_bias: float = 0.0,
        lora_manifests: list[Any] | None = None,
        *,
        test_synthetic: bool = False,
    ) -> None:
        self._backend = backend
        self._gamma_flow_bias = float(
            np.clip(gamma_flow_bias, 0.0, _FLOW_BIAS_GAMMA_MAX)
        )
        self._lora_manager: Any | None = None
        if lora_manifests is not None:
            from synapse.adapters.lora.manager import MultiLoRAManager  # lazy
            self._lora_manager = MultiLoRAManager(
                lora_bundles=lora_manifests,
                test_synthetic=test_synthetic,
            )

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

        Real ``cryptomatte_id`` from the FGPP sequence is passed
        through to the payload for downstream spatial masking.

        Args:
            fgpp_sequence: Source ``FGPPSequence``.
            config: Generation configuration.

        Returns:
            ``PreparedConditionPayload`` with ``control_tensor`` shaped
            ``[C_cond, T, H, W]`` (C_cond = 7: 1 depth + 3 normal +
            3 radiance).
        """
        from synapse.style.aces_color import soft_knee_inverse_tonemap

        n_frames = len(fgpp_sequence.frames)
        if n_frames == 0:
            return PreparedConditionPayload()

        first = fgpp_sequence.frames[0]
        h, w = first.height, first.width

        per_frame_tensors: list[np.ndarray] = []
        cryptomatte_maps: list[np.ndarray] = []

        for frame in fgpp_sequence.frames:
            # 1. Depth — log-normalised, preserving float64 precision
            depth_norm_f64, depth_norm = _encode_depth(
                frame.depth, frame.camera.z_near, frame.camera.z_far,
            )

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

            # Collect Cryptomatte IDs for passthrough
            if getattr(frame, "cryptomatte_id", None) is not None:
                cryptomatte_maps.append(frame.cryptomatte_id)

        # Stack into [T, C, H, W] then transpose to [C, T, H, W]
        stacked = np.stack(per_frame_tensors, axis=0)  # [T, C, H, W]
        condition_tensor = np.transpose(
            stacked, (1, 0, 2, 3),
        )  # [C, T, H, W]

        # 48-frame physical anchor indexing (RFC 0044 §3.2)
        physical_anchor_indices = [
            i for i, f in enumerate(fgpp_sequence.frames)
            if i % 48 == 0 or getattr(f, "is_physical_anchor", False)
        ]

        # Build metadata for flow bias / occlusion / anchors
        flow_meta: dict[str, Any] = {
            "flow_bias_gamma": self._gamma_flow_bias,
            "occlusion_mask": None,
            "physical_anchor_indices": physical_anchor_indices,
            "anchor_interval": 48,
        }

        # Build Cryptomatte passthrough if available
        cryptomatte_id_tensor: np.ndarray | None = None
        if len(cryptomatte_maps) == n_frames:
            cryptomatte_id_tensor = np.stack(
                cryptomatte_maps, axis=0,
            )  # [T, H, W]

        return PreparedConditionPayload(
            control_tensor=condition_tensor,
            cryptomatte_id=cryptomatte_id_tensor,
            extra=flow_meta,
        )

    def reanchor_physical_frame(
        self,
        frame: Any,  # FGPPFrame
        frame_index: int = 0,
    ) -> np.ndarray:
        """Extract ground-truth physical conditioning slice for an anchor frame.

        Re-anchors depth, camera normals, and radiance directly from the physical
        FGPP ground-truth frame, eliminating drift.

        Preserves ``depth_norm_f64`` as ``float64`` until the final
        quantisation/serialisation boundary.

        Args:
            frame: Source FGPPFrame with depth, normals, and radiance.
            frame_index: Frame index in the sequence.

        Returns:
            ``[7, H, W]`` float32 ControlNeXt condition tensor.
        """
        from synapse.style.aces_color import soft_knee_inverse_tonemap

        h, w = frame.height, frame.width

        # 1. Depth (float64 calculation -> float32 tensor)
        depth_norm_f64, depth_norm = _encode_depth(
            frame.depth, frame.camera.z_near, frame.camera.z_far,
        )

        # 2. Camera Normals
        if frame.world_normal is not None:
            wn = frame.world_normal.astype(np.float32)
        else:
            wn = np.zeros((h, w, 3), dtype=np.float32)
        cam_normals = _world_normal_to_camera(wn, frame.camera.view_matrix)

        # 3. Radiance
        if frame.direct_radiance is not None:
            rad = frame.direct_radiance.astype(np.float64)
        else:
            rad = np.zeros((h, w, 3), dtype=np.float64)
        rad_tonemapped = soft_knee_inverse_tonemap(rad)
        rad_norm = np.clip(
            rad_tonemapped / _RADIANCE_NORM_MAX, 0.0, 1.0,
        ).astype(np.float32)

        return _build_control_tensor(depth_norm, cam_normals, rad_norm)

    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
        model: Any | None = None,
        fgpp_frames: list[Any] | None = None,
        check_non_degenerate: bool = True,
    ) -> VideoResult:
        """Generate video frames via the configured backend transport.

        When a ``MultiLoRAManager`` is active (i.e. ``lora_manifests``
        were provided at construction time) **and** the supplied FGPP
        frames contain ``cryptomatte_id`` data, the manager's spatial
        masks are built and weights are applied to *model* before
        inference.  ``restore_base_weights`` is called in a ``finally``
        block to guarantee cleanup even on errors.

        For each anchor frame in the sequence, ``reanchor_physical_frame``
        is called to inject ground-truth conditioning.

        Args:
            conditions: Output of ``prepare_conditions()``.
            config: Generation configuration.
            model: Optional ``torch.nn.Module`` for LoRA weight injection.
                Required when ``_lora_manager`` is active and frames have
                Cryptomatte IDs.
            fgpp_frames: Optional list of ``FGPPFrame`` objects from which
                Cryptomatte entity ID maps are extracted for spatial masking.

        Returns:
            ``VideoResult`` with sRGB uint8 frames.

        Raises:
            RuntimeError: If no backend transport has been configured.
        """
        if self._backend is None:
            raise BackendUnavailableError(
                "WanAdapter has no backend transport configured. "
                "Pass a WanBackendTransport to the constructor.",
                agent_fix_hint="Pass a valid WanBackendTransport to the constructor.",
            )

        # Defensively copy control_tensor before re-anchoring to avoid in-place mutation
        working_control = conditions.control_tensor
        if working_control is not None:
            working_control = working_control.copy()

        # Apply reanchoring for physical anchor frames
        if fgpp_frames and conditions.extra.get("physical_anchor_indices"):
            for anchor_idx in conditions.extra["physical_anchor_indices"]:
                if anchor_idx < len(fgpp_frames):
                    reanchored = self.reanchor_physical_frame(
                        fgpp_frames[anchor_idx],
                        frame_index=anchor_idx,
                    )
                    if working_control is not None and anchor_idx < working_control.shape[1]:
                        working_control[:, anchor_idx] = reanchored

        conditions_to_use = PreparedConditionPayload(
            control_tensor=working_control,
            keyframe_images=list(conditions.keyframe_images),
            text_prompt=conditions.text_prompt,
            negative_prompt=conditions.negative_prompt,
            cryptomatte_id=conditions.cryptomatte_id,
            extra=dict(conditions.extra),
        )

        # Compute latent flow operator if gamma_flow_bias > 0
        if self.gamma_flow_bias > 0:
            mv = conditions_to_use.extra.get("motion_vectors")
            depth = conditions_to_use.extra.get("depth")
            if (
                mv is not None
                and depth is not None
                and isinstance(mv, np.ndarray)
                and isinstance(depth, np.ndarray)
            ):
                from synapse.adapters.scheduler import compute_latent_flow_operator

                try:
                    flow_op = compute_latent_flow_operator(mv, depth)
                    flow_meta = conditions_to_use.extra.setdefault("flow_meta", {})
                    flow_meta["occlusion_mask"] = flow_op
                except Exception:
                    pass

        # Determine whether LoRA application is needed this call
        _lora_applied = False
        cryptomatte_ids: np.ndarray | None = None
        target_entity_ids: set[int] | None = None

        if self._lora_manager is not None and model is not None:
            if fgpp_frames:
                crypt_maps = [
                    f.cryptomatte_id
                    for f in fgpp_frames
                    if getattr(f, "cryptomatte_id", None) is not None
                ]
                if crypt_maps:
                    cryptomatte_ids = np.stack(crypt_maps, axis=0)
            elif conditions_to_use.cryptomatte_id is not None:
                cryptomatte_ids = conditions_to_use.cryptomatte_id

            if cryptomatte_ids is not None:
                target_ids_extra = conditions_to_use.extra.get("target_entity_ids")
                if target_ids_extra is not None:
                    target_entity_ids = set(target_ids_extra)
                else:
                    target_entity_ids = {
                        int(v) for v in np.unique(cryptomatte_ids) if int(v) != 0
                    }
                self._lora_manager.build_spatial_lora_masks(
                    cryptomatte_ids, target_entity_ids
                )
                self._lora_manager.apply_merged_weights(model)
                _lora_applied = True

        total_frames = config.num_frames
        if working_control is not None:
            total_frames = max(total_frames, working_control.shape[1])
        if fgpp_frames:
            total_frames = max(total_frames, len(fgpp_frames))

        window_size = int(config.extra.get("window_frames", 97))

        try:
            hook_ctx = (
                self._lora_manager.manage_spatial_hooks(
                    model,
                    cryptomatte_ids=cryptomatte_ids,
                    target_entity_ids=target_entity_ids,
                )
                if (self._lora_manager is not None and model is not None)
                else contextlib.nullcontext()
            )

            with hook_ctx:
                if total_frames > window_size:
                    from synapse.adapters.scheduler import (
                        CausalWindowPlan,
                        ConditionScheduler,
                        DiscontinuityDetector,
                        fuse_overlapping_latents,
                    )

                    stride = int(config.extra.get("stride_frames", 64))
                    overlap = int(config.extra.get("overlap_frames", 33))
                    plan = CausalWindowPlan(
                        window_frames=window_size,
                        stride_frames=stride,
                        overlap_frames=overlap,
                        total_frames=total_frames,
                    )
                    detector = DiscontinuityDetector()
                    scheduler = ConditionScheduler(plan=plan, detector=detector)

                    class _SeqWrapper:
                        def __init__(self, f_list: list[Any]) -> None:
                            self.frames = f_list

                    scheduled_windows = scheduler.schedule(
                        _SeqWrapper(fgpp_frames or [])
                    )

                    if len(scheduled_windows) > 1:
                        accumulated_frames: np.ndarray | None = None
                        prev_window_tail: list[np.ndarray] = []

                        for w_idx, sw in enumerate(scheduled_windows):
                            w = sw.window
                            w_num_frames = w.end - w.start
                            sub_control = (
                                working_control[:, w.start : w.end]
                                if working_control is not None
                                else None
                            )
                            sub_crypt = (
                                conditions_to_use.cryptomatte_id[w.start : w.end]
                                if conditions_to_use.cryptomatte_id is not None
                                else None
                            )

                            sub_cfg = VideoGenerationConfig(
                                seed=config.seed + w_idx * 17,
                                fps=config.fps,
                                width=config.width,
                                height=config.height,
                                num_frames=w_num_frames,
                                guidance_scale=config.guidance_scale,
                                num_inference_steps=config.num_inference_steps,
                                extra=dict(config.extra),
                            )
                            sub_payload = PreparedConditionPayload(
                                control_tensor=sub_control,
                                keyframe_images=list(prev_window_tail[-3:]) if prev_window_tail else [],
                                text_prompt=conditions_to_use.text_prompt,
                                negative_prompt=conditions_to_use.negative_prompt,
                                cryptomatte_id=sub_crypt,
                                extra=dict(conditions_to_use.extra),
                            )
                            if prev_window_tail:
                                sub_payload.extra["warm_start_frames"] = list(prev_window_tail[-3:])

                            sub_result = self._backend.submit(sub_payload, sub_cfg)
                            prev_window_tail = [sub_result[idx] for idx in range(len(sub_result))]

                            if accumulated_frames is None:
                                accumulated_frames = sub_result
                            else:
                                actual_overlap = min(overlap, len(accumulated_frames), len(sub_result))
                                if actual_overlap > 0:
                                    prev_tail = accumulated_frames[-actual_overlap:].astype(np.float32)
                                    curr_head = sub_result[:actual_overlap].astype(np.float32)
                                    fused = fuse_overlapping_latents(
                                        prev_tail,
                                        curr_head,
                                        overlap_steps=actual_overlap,
                                    )
                                    fused_u8 = np.clip(np.rint(fused), 0, 255).astype(np.uint8)
                                    accumulated_frames = np.concatenate(
                                        [
                                            accumulated_frames[:-actual_overlap],
                                            fused_u8,
                                            sub_result[actual_overlap:],
                                        ],
                                        axis=0,
                                    )
                                else:
                                    accumulated_frames = np.concatenate(
                                        [accumulated_frames, sub_result], axis=0,
                                    )

                        raw_frames = (
                            accumulated_frames[:total_frames]
                            if accumulated_frames is not None
                            else np.zeros((0, config.height, config.width, 3), dtype=np.uint8)
                        )
                    else:
                        raw_frames = self._backend.submit(conditions_to_use, config)
                else:
                    raw_frames = self._backend.submit(conditions_to_use, config)
        finally:
            if _lora_applied and self._lora_manager is not None:
                self._lora_manager.restore_base_weights(model)

        if check_non_degenerate:
            from synapse.metrics.image_quality import validate_output_non_degenerate

            validate_output_non_degenerate(raw_frames)

        return VideoResult(
            frames_srgb_u8=raw_frames,
            fps=config.fps,
            seed=config.seed,
            adapter_id=self.adapter_id,
        )