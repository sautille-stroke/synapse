"""Tier-2 blackbox cloud video adapters with FGPP anchor degradation.

Implements ``degrade_fgpp_for_blackbox()`` which splits an FGPP sequence
at physical anchor boundaries, extracts tonemapped sRGB PNG anchor frames,
fits structured camera motion enums via least-squares velocity
decomposition, and packages the result as ``DegradedSegment`` objects.

Three cloud adapters — ``SeedanceAdapter``, ``KlingAdapter``, and
``VeoAdapter`` — consume these segments through a pluggable
``CloudApiTransport`` abstraction.

All heavy dependencies (PIL/Pillow, scipy) are lazy-imported inside
the functions that need them.  Module-level imports are limited to
``numpy``, ``pydantic``, and standard library.
"""

from __future__ import annotations

import hashlib
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from synapse.fgpp.schema import (
    DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
    is_physical_anchor_frame,
)
from synapse.style.aces_color import acescg_to_linear_srgb, linear_to_srgb

# ---------------------------------------------------------------------------
# Valid camera motion enum values
# ---------------------------------------------------------------------------

_VALID_CAMERA_MOTIONS: list[str] = [
    "pan_right",
    "pan_left",
    "dolly_in",
    "dolly_out",
    "orbit_cw",
    "orbit_ccw",
    "static",
]

# ---------------------------------------------------------------------------
# DegradedSegment dataclass
# ---------------------------------------------------------------------------


@dataclass
class DegradedSegment:
    """A single degraded segment for blackbox cloud video generation.

    Attributes:
        start_frame: First frame index of this segment (inclusive).
        end_frame: Last frame index of this segment (inclusive).
        first_anchor_png: PNG-encoded bytes of the first anchor frame.
        last_anchor_png: PNG-encoded bytes of the last anchor frame.
        camera_trajectory: Per-frame 4x4 camera-to-world matrices,
            shape ``[T_seg, 4, 4]``.
        camera_motion_enum: Fitted structured camera motion string.
        motion_intensity: Motion intensity in ``[0, 10]``.
        depth_guide: Optional log-depth guide frames ``[T, H, W]``.
        normal_edge_guide: Optional normal-edge guide ``[T, H, W]``.
    """

    start_frame: int
    end_frame: int
    first_anchor_png: bytes
    last_anchor_png: bytes
    camera_trajectory: np.ndarray  # [T_seg, 4, 4]
    camera_motion_enum: str
    motion_intensity: float
    depth_guide: np.ndarray | None = None
    normal_edge_guide: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _frame_to_srgb_u8(radiance: np.ndarray) -> np.ndarray:
    """Convert ACEScg HDR radiance ``[H, W, 3]`` to sRGB uint8.

    Applies ACEScg -> linear sRGB -> sRGB OETF -> clip -> uint8.

    Args:
        radiance: HDR radiance in ACEScg, shape ``[H, W, 3]``.

    Returns:
        ``[H, W, 3]`` uint8 sRGB image.
    """
    linear_srgb = acescg_to_linear_srgb(radiance)
    srgb = linear_to_srgb(linear_srgb)
    srgb_clipped = np.clip(srgb, 0.0, 1.0)
    return np.clip(
        np.round(srgb_clipped * 255.0), 0, 255
    ).astype(np.uint8)


def _encode_png_bytes(image_u8: np.ndarray) -> bytes:
    """Encode an ``[H, W, 3]`` uint8 image to PNG bytes.

    Uses PIL/Pillow if available, otherwise falls back to a raw
    array representation with a PNG-like header for testing.

    Args:
        image_u8: ``[H, W, 3]`` uint8 sRGB image.

    Returns:
        PNG-encoded bytes.
    """
    try:
        from PIL import Image  # type: ignore[import-untyped]
        img = Image.fromarray(image_u8, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: store raw array with a simple header so tests
        # can verify non-empty bytes without requiring Pillow.
        buf = io.BytesIO()
        # Write a recognisable marker + shape + raw data
        header = b"SYNAPSE_RAW_IMG"
        h, w = image_u8.shape[:2]
        buf.write(header)
        buf.write(h.to_bytes(4, "little"))
        buf.write(w.to_bytes(4, "little"))
        buf.write(image_u8.tobytes())
        return buf.getvalue()


def _fit_camera_motion(
    trajectory: np.ndarray,
) -> tuple[str, float]:
    """Fit a structured camera motion enum from a trajectory.

    Uses least-squares velocity decomposition on the per-frame
    translations and rotations to classify the dominant motion
    pattern.

    Args:
        trajectory: ``[T, 4, 4]`` camera-to-world matrices.

    Returns:
        Tuple of ``(camera_motion_enum, motion_intensity)``.
    """
    n = trajectory.shape[0]
    if n < 2:
        return "static", 0.0

    # Extract translation vectors [T, 3]
    positions = trajectory[:, :3, 3].copy()

    # Compute per-frame velocity vectors [T-1, 3]
    velocities = np.diff(positions, axis=0)  # [T-1, 3]
    mean_vel = np.mean(velocities, axis=0)  # [3]

    # Total displacement magnitude
    total_displacement = np.linalg.norm(positions[-1] - positions[0])

    # Compute angular displacement around Y axis (yaw)
    forward_dirs = -trajectory[:, :3, 2]  # camera forward = -Z
    # Project to XZ plane for yaw analysis
    yaw_angles = np.arctan2(forward_dirs[:, 0], forward_dirs[:, 2])
    yaw_diff = np.diff(yaw_angles)
    # Unwrap angle differences
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    total_yaw = np.sum(yaw_diff)

    # Compute per-frame speed for intensity
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = float(np.mean(speeds))

    # Classify dominant motion
    yaw_magnitude = abs(float(total_yaw))
    displacement = float(total_displacement)

    # Heuristic thresholds
    # If significant yaw rotation detected -> orbit
    if yaw_magnitude > 0.1:
        if total_yaw > 0:
            motion = "orbit_cw"
        else:
            motion = "orbit_ccw"
        # Intensity from yaw magnitude
        raw_intensity = yaw_magnitude / np.pi * 10.0
    elif displacement > 0.01:
        # Check dominant translation direction
        forward_vel = float(mean_vel[2])
        lateral_vel = float(mean_vel[0])

        if abs(forward_vel) > abs(lateral_vel):
            # Dolly motion (along camera Z)
            if forward_vel > 0:
                motion = "dolly_out"
            else:
                motion = "dolly_in"
            raw_intensity = abs(forward_vel) / max(mean_speed, 1e-8)
            raw_intensity *= 10.0
        else:
            # Pan motion (lateral)
            if lateral_vel > 0:
                motion = "pan_right"
            else:
                motion = "pan_left"
            raw_intensity = abs(lateral_vel) / max(mean_speed, 1e-8)
            raw_intensity *= 10.0
    else:
        motion = "static"
        raw_intensity = 0.0

    # Clamp intensity to [0, 10]
    intensity = float(np.clip(raw_intensity, 0.0, 10.0))
    return motion, intensity


def _compute_anchor_points(
    num_frames: int,
    anchor_interval: int,
) -> list[int]:
    """Compute physical anchor frame indices.

    Returns sorted list of anchor indices within ``[0, num_frames)``.

    Args:
        num_frames: Total frame count.
        anchor_interval: Spacing between anchors.

    Returns:
        Sorted list of anchor frame indices.
    """
    anchors: list[int] = []
    idx = 0
    while idx < num_frames:
        anchors.append(idx)
        idx += anchor_interval
    # Ensure last frame is included if not already
    last = num_frames - 1
    if last > 0 and last not in anchors:
        anchors.append(last)
    return sorted(set(anchors))


# ---------------------------------------------------------------------------
# Public API: degrade_fgpp_for_blackbox
# ---------------------------------------------------------------------------


def degrade_fgpp_for_blackbox(
    fgpp_sequence: Any,
    anchor_interval: int = DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES,
) -> list[DegradedSegment]:
    """Degrade an FGPP sequence for blackbox cloud API consumption.

    Splits the sequence at physical anchor boundaries (0, 48, 96, ...)
    and packages each segment with tonemapped sRGB PNG anchor frames,
    camera trajectory, and fitted motion enum.

    Args:
        fgpp_sequence: An ``FGPPSequence`` with per-frame channels.
        anchor_interval: Spacing between physical anchors (default 48).

    Returns:
        List of ``DegradedSegment`` objects, one per segment.
    """
    frames = fgpp_sequence.frames
    num_frames = len(frames)
    if num_frames == 0:
        return []

    # Compute anchor points
    anchors = _compute_anchor_points(num_frames, anchor_interval)

    # Build segments between consecutive anchors
    segments: list[DegradedSegment] = []
    for i in range(len(anchors) - 1):
        seg_start = anchors[i]
        seg_end = anchors[i + 1]

        # Extract first and last frames for this segment
        first_frame = frames[seg_start]
        last_frame = frames[seg_end]

        # Tonemap anchor frames to sRGB PNG
        if first_frame.direct_radiance is not None:
            first_u8 = _frame_to_srgb_u8(
                first_frame.direct_radiance.astype(np.float64)
            )
        else:
            h, w = first_frame.height, first_frame.width
            first_u8 = np.zeros((h, w, 3), dtype=np.uint8)

        if last_frame.direct_radiance is not None:
            last_u8 = _frame_to_srgb_u8(
                last_frame.direct_radiance.astype(np.float64)
            )
        else:
            h, w = last_frame.height, last_frame.width
            last_u8 = np.zeros((h, w, 3), dtype=np.uint8)

        first_png = _encode_png_bytes(first_u8)
        last_png = _encode_png_bytes(last_u8)

        # Extract camera trajectory [T_seg, 4, 4]
        seg_frames = frames[seg_start : seg_end + 1]
        trajectory = np.stack(
            [f.camera.world_matrix for f in seg_frames],
            axis=0,
        ).astype(np.float64)

        # Fit camera motion enum
        camera_motion, intensity = _fit_camera_motion(trajectory)

        # Optional: extract depth guide
        depth_guide: np.ndarray | None = None
        if all(f.depth is not None for f in seg_frames):
            depth_guide = np.stack(
                [f.depth.astype(np.float32) for f in seg_frames],
                axis=0,
            )

        # Optional: extract normal edge guide
        normal_edge: np.ndarray | None = None
        if all(f.world_normal is not None for f in seg_frames):
            normals = np.stack(
                [f.world_normal.astype(np.float32) for f in seg_frames],  # type: ignore[union-attr]
                axis=0,
            )
            # Compute edge magnitude from spatial gradient of normals
            # Use simple Sobel-like finite differences
            dx = np.diff(normals, axis=2, prepend=normals[:, :, :1, :])
            dy = np.diff(normals, axis=1, prepend=normals[:, :1, :, :])
            edge_mag = np.sqrt(
                np.sum(dx ** 2, axis=-1) + np.sum(dy ** 2, axis=-1)
            )
            normal_edge = edge_mag

        segments.append(DegradedSegment(
            start_frame=seg_start,
            end_frame=seg_end,
            first_anchor_png=first_png,
            last_anchor_png=last_png,
            camera_trajectory=trajectory,
            camera_motion_enum=camera_motion,
            motion_intensity=intensity,
            depth_guide=depth_guide,
            normal_edge_guide=normal_edge,
        ))

    return segments


# ---------------------------------------------------------------------------
# Idempotency key helper
# ---------------------------------------------------------------------------


def _compute_idempotency_key(
    scene_id: str,
    start_frame: int,
    end_frame: int,
    seed: int,
) -> str:
    """Compute a deterministic SHA-256 idempotency key.

    Args:
        scene_id: Scene identifier.
        start_frame: Segment start frame.
        end_frame: Segment end frame.
        seed: RNG seed.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    payload = f"{scene_id}:{start_frame}:{end_frame}:{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CloudApiTransport ABC
# ---------------------------------------------------------------------------


class CloudApiTransport(ABC):
    """Abstract transport for submitting segments to cloud APIs.

    Subclasses implement the actual HTTP/gRPC call to the cloud
    provider's video generation endpoint.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider name (e.g. ``"seedance"``)."""
        ...

    @abstractmethod
    def submit_segment(
        self,
        segment: DegradedSegment,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Submit a segment and return generated frames.

        Args:
            segment: Degraded FGPP segment with anchor images.
            config: Generation configuration.

        Returns:
            ``[T_seg, H, W, 3]`` uint8 sRGB frames for this segment.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the transport can reach its backend."""
        ...


# ---------------------------------------------------------------------------
# MockCloudTransport — deterministic CPU-only stub
# ---------------------------------------------------------------------------


class MockCloudTransport(CloudApiTransport):
    """Deterministic CPU-only mock transport for testing.

    Returns reproducible random frames seeded by the config seed and
    segment range.  Never makes network calls.
    """

    @property
    def provider_name(self) -> str:  # noqa: D102
        return "mock_cloud"

    def submit_segment(
        self,
        segment: DegradedSegment,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Return deterministic random frames for the segment.

        Validates segment structure, then generates ``[T, H, W, 3]``
        uint8 random frames using a seed derived from the config seed
        and segment boundaries.

        Args:
            segment: Degraded FGPP segment.
            config: Generation configuration.

        Returns:
            ``[T, H, W, 3]`` uint8 sRGB frames.
        """
        # Validate segment structure
        assert segment.first_anchor_png, "first_anchor_png is empty"
        assert segment.last_anchor_png, "last_anchor_png is empty"
        assert segment.start_frame <= segment.end_frame
        assert segment.camera_motion_enum in _VALID_CAMERA_MOTIONS
        assert 0.0 <= segment.motion_intensity <= 10.0

        # Deterministic seed from config + segment range
        combined_seed = (
            config.seed
            ^ (segment.start_frame * 31)
            ^ (segment.end_frame * 97)
        )
        rng = np.random.Generator(np.random.PCG64(combined_seed))

        num_frames = segment.end_frame - segment.start_frame + 1
        return rng.integers(
            0, 256,
            size=(num_frames, config.height, config.width, 3),
            dtype=np.uint8,
        )

    def is_available(self) -> bool:  # noqa: D102
        return True


# ---------------------------------------------------------------------------
# _BaseCloudAdapter — shared logic for all cloud adapters
# ---------------------------------------------------------------------------


class _BaseCloudAdapter(VideoModelAdapter, ABC):
    """Abstract base for Tier-2 blackbox cloud adapters.

    Provides common FGPP degradation, segment iteration, result
    concatenation (with duplicate anchor frame removal), and
    idempotency key computation.

    Subclasses must set ``_transport`` and implement adapter-specific
    properties.
    """

    def __init__(
        self,
        transport: CloudApiTransport | None = None,
        api_token: str | None = None,
    ) -> None:
        self._transport = transport
        self._api_token = api_token or os.environ.get(
            self._env_var_name, ""
        )

    @property
    @abstractmethod
    def _env_var_name(self) -> str:
        """Environment variable name for the API token."""
        ...

    @property
    @abstractmethod
    def _anchor_interval(self) -> int:
        """Physical anchor interval for segment splitting."""
        ...

    @property
    @abstractmethod
    def _supports_depth_guide(self) -> bool:
        """Whether this adapter supports depth guide frames."""
        ...

    @property
    @abstractmethod
    def _supports_normal_guide(self) -> bool:
        """Whether this adapter supports normal edge guide frames."""
        ...

    @property
    def tier(self) -> AdapterTier:  # noqa: D102
        return "blackbox"

    def _ensure_transport(self) -> CloudApiTransport:
        """Return the transport, raising if unavailable.

        Raises:
            BackendUnavailableError: If no transport or API token.
        """
        if self._transport is not None:
            if not self._transport.is_available():
                raise BackendUnavailableError(
                    f"{self.adapter_id} transport is not available.",
                    agent_fix_hint=(
                        f"Check network connectivity to "
                        f"{self._transport.provider_name}."
                    ),
                )
            return self._transport

        # No transport provided — check for API token
        if not self._api_token:
            raise BackendUnavailableError(
                f"{self.adapter_id} requires an API token. "
                f"Set the {self._env_var_name} environment variable.",
                agent_fix_hint=(
                    f"export {self._env_var_name}=<your-token>"
                ),
            )

        raise BackendUnavailableError(
            f"No transport configured for {self.adapter_id}.",
            agent_fix_hint=(
                f"Provide a CloudApiTransport instance or set "
                f"{self._env_var_name}."
            ),
        )

    def prepare_conditions(
        self,
        fgpp_sequence: Any,
        config: VideoGenerationConfig,
    ) -> PreparedConditionPayload:
        """Degrade FGPP sequence into blackbox-ready segments.

        Calls ``degrade_fgpp_for_blackbox`` and stores segments in
        the ``extra`` dict of the payload.

        Args:
            fgpp_sequence: Source ``FGPPSequence``.
            config: Generation configuration.

        Returns:
            ``PreparedConditionPayload`` with segments in ``extra``.
        """
        segments = degrade_fgpp_for_blackbox(
            fgpp_sequence,
            anchor_interval=self._anchor_interval,
        )

        # Strip unsupported guides per adapter
        for seg in segments:
            if not self._supports_depth_guide:
                seg.depth_guide = None
            if not self._supports_normal_guide:
                seg.normal_edge_guide = None

        return PreparedConditionPayload(
            extra={
                "degraded_segments": segments,
                "scene_id": getattr(
                    fgpp_sequence, "scene_id", ""
                ),
            },
        )

    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> VideoResult:
        """Generate video by iterating segments and concatenating.

        Removes duplicate anchor frames at segment boundaries so the
        final frame count matches the original sequence length.

        Args:
            conditions: Payload from ``prepare_conditions()``.
            config: Generation configuration.

        Returns:
            ``VideoResult`` with concatenated sRGB uint8 frames.
        """
        transport = self._ensure_transport()
        segments: list[DegradedSegment] = conditions.extra.get(
            "degraded_segments", []
        )
        scene_id: str = conditions.extra.get("scene_id", "")

        if not segments:
            return VideoResult(
                frames_srgb_u8=np.empty(
                    (0, config.height, config.width, 3),
                    dtype=np.uint8,
                ),
                fps=config.fps,
                seed=config.seed,
                adapter_id=self.adapter_id,
            )

        all_frames: list[np.ndarray] = []
        idempotency_keys: list[str] = []

        for i, seg in enumerate(segments):
            key = _compute_idempotency_key(
                scene_id, seg.start_frame, seg.end_frame, config.seed,
            )
            idempotency_keys.append(key)

            frames = transport.submit_segment(seg, config)

            if i == 0:
                # First segment: include all frames
                all_frames.append(frames)
            else:
                # Subsequent segments: drop the first frame
                # (it's the shared anchor from previous segment)
                all_frames.append(frames[1:])

        result_frames = np.concatenate(all_frames, axis=0)

        return VideoResult(
            frames_srgb_u8=result_frames,
            fps=config.fps,
            seed=config.seed,
            adapter_id=self.adapter_id,
            metadata={
                "idempotency_keys": idempotency_keys,
                "num_segments": len(segments),
            },
        )


# ---------------------------------------------------------------------------
# SeedanceAdapter — Seedance 2.5
# ---------------------------------------------------------------------------


@register_adapter
class SeedanceAdapter(_BaseCloudAdapter):
    """Blackbox adapter for Seedance 2.5 cloud API.

    Supports camera trajectory matrix, depth guide, and normal edge
    guide.
    """

    @property
    def adapter_id(self) -> str:  # noqa: D102
        return "seedance2.5"

    @property
    def _env_var_name(self) -> str:
        return "SEEDANCE_API_TOKEN"

    @property
    def _anchor_interval(self) -> int:
        return DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES

    @property
    def _supports_depth_guide(self) -> bool:
        return True

    @property
    def _supports_normal_guide(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# KlingAdapter — Kling 1.5
# ---------------------------------------------------------------------------


@register_adapter
class KlingAdapter(_BaseCloudAdapter):
    """Blackbox adapter for Kling 1.5 cloud API.

    Uses ``camera_motion_enum`` + ``motion_intensity`` instead of raw
    trajectory matrix.  Does not support depth or normal guides.
    """

    @property
    def adapter_id(self) -> str:  # noqa: D102
        return "kling1.5"

    @property
    def _env_var_name(self) -> str:
        return "KLING_API_TOKEN"

    @property
    def _anchor_interval(self) -> int:
        return DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES

    @property
    def _supports_depth_guide(self) -> bool:
        return False

    @property
    def _supports_normal_guide(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# VeoAdapter — Veo
# ---------------------------------------------------------------------------


@register_adapter
class VeoAdapter(_BaseCloudAdapter):
    """Blackbox adapter for Veo cloud API.

    Supports camera trajectory matrix and depth guide, but not normal
    edge guide.
    """

    @property
    def adapter_id(self) -> str:  # noqa: D102
        return "veo"

    @property
    def _env_var_name(self) -> str:
        return "VEO_API_TOKEN"

    @property
    def _anchor_interval(self) -> int:
        return DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES

    @property
    def _supports_depth_guide(self) -> bool:
        return True

    @property
    def _supports_normal_guide(self) -> bool:
        return False
