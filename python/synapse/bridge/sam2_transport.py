"""SAM 2.1 Video Object Segmentation (VOS) transport.

Bridges Meta's SAM 2.1 real-time video segmentation model into
synapse's instance-level masking pipeline.  SAM 2.1 provides:

- **Video Object Segmentation (VOS)**: propagate instance masks
  across a video sequence given initial prompts (points, boxes,
  or masks on the first frame).
- **Real-time 44fps inference**: streaming memory-bank architecture
  with occlusion-aware mask refinement.
- **Automatic re-initialization**: at scene cuts the tracker
  automatically resets its memory bank.

Transport backends:

- :class:`MockSAM2Backend` — deterministic CPU-only mock producing
  synthetic mask sequences for zero-GPU testing.
- :class:`FoundarySAM2Backend` — routes to the foundary container's
  ``sam2`` service endpoint.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.errors import BackendUnavailableError, ContractValidationError

# ---------------------------------------------------------------------------
# Prompt types
# ---------------------------------------------------------------------------


class SAM2PromptType(enum.Enum):
    """Type of initial segmentation prompt.

    Attributes:
        POINT: Positive/negative point clicks on the first frame.
        BOX: Bounding box on the first frame.
        MASK: Binary mask on the first frame.
        AUTO: Automatic mode — detect all instances.
    """

    POINT = "point"
    BOX = "box"
    MASK = "mask"
    AUTO = "auto"


@dataclass
class SAM2Prompt:
    """Initial segmentation prompt for SAM 2.1.

    Attributes:
        prompt_type: Type of prompt.
        points: ``[N, 2]`` float32 (x, y) coordinates for POINT prompts.
        point_labels: ``[N]`` int32 labels (1=positive, 0=negative).
        box: ``[4]`` float32 (x1, y1, x2, y2) for BOX prompts.
        mask: ``[H, W]`` bool for MASK prompts.
        instance_id: Unique identifier for the tracked instance.
    """

    prompt_type: SAM2PromptType = SAM2PromptType.AUTO
    points: np.ndarray | None = None
    point_labels: np.ndarray | None = None
    box: np.ndarray | None = None
    mask: np.ndarray | None = None
    instance_id: int = 0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SAM2InstanceMask:
    """Per-frame instance segmentation mask.

    Attributes:
        instance_id: Unique instance identifier.
        mask: ``[H, W]`` bool mask for this instance.
        confidence: Per-pixel mean confidence in [0, 1].
        is_occluded: Whether the instance is considered occluded.
    """

    instance_id: int
    mask: np.ndarray  # [H, W] bool
    confidence: float = 1.0
    is_occluded: bool = False


@dataclass
class SAM2FrameResult:
    """Segmentation result for a single frame.

    Attributes:
        frame_index: 0-indexed frame position.
        instances: List of instance masks detected in this frame.
    """

    frame_index: int
    instances: list[SAM2InstanceMask] = field(default_factory=list)

    @property
    def combined_mask(self) -> np.ndarray | None:
        """Union of all instance masks as a single bool array."""
        if not self.instances:
            return None
        combined = self.instances[0].mask.copy()
        for inst in self.instances[1:]:
            combined |= inst.mask
        return combined


@dataclass
class SAM2SequenceResult:
    """Full video segmentation result.

    Attributes:
        frames: Per-frame segmentation results.
        num_instances: Total unique instances tracked.
        scene_cuts: Frame indices where scene cuts were detected
            and the tracker was reinitialized.
        metadata: Backend metadata.
    """

    frames: list[SAM2FrameResult] = field(default_factory=list)
    num_instances: int = 0
    scene_cuts: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transport ABC
# ---------------------------------------------------------------------------


class SAM2Transport(ABC):
    """Abstract transport for SAM 2.1 Video Object Segmentation."""

    @abstractmethod
    def segment_video(
        self,
        frames: np.ndarray,
        prompts: list[SAM2Prompt] | None = None,
    ) -> SAM2SequenceResult:
        """Segment objects across a video sequence.

        Args:
            frames: ``[T, H, W, 3]`` uint8 sRGB video frames.
            prompts: Optional initial prompts. If ``None``, uses
                automatic instance detection on the first frame.

        Returns:
            :class:`SAM2SequenceResult` with per-frame masks.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend is ready."""
        ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockSAM2Backend(SAM2Transport):
    """Deterministic CPU-only mock for SAM 2.1.

    Generates synthetic elliptical masks that drift over time to
    simulate object tracking.
    """

    def segment_video(
        self,
        frames: np.ndarray,
        prompts: list[SAM2Prompt] | None = None,
    ) -> SAM2SequenceResult:
        """Generate synthetic tracked masks.

        Creates one instance per prompt (or one default instance in
        AUTO mode) with an elliptical mask that shifts slightly each
        frame to simulate motion tracking.

        Args:
            frames: ``[T, H, W, 3]`` uint8 sRGB.
            prompts: Optional prompts.

        Returns:
            Deterministic :class:`SAM2SequenceResult`.
        """
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ContractValidationError(
                f"Expected [T, H, W, 3] uint8 video, got shape {frames.shape}.",
                agent_fix_hint="Provide video as [T, H, W, 3] uint8 array.",
            )

        t, h, w, _ = frames.shape
        num_instances = len(prompts) if prompts else 1

        frame_results = []
        for fi in range(t):
            instances = []
            for inst_id in range(num_instances):
                # Elliptical mask with slight per-frame drift
                mask = _generate_elliptical_mask(
                    h, w,
                    cx=w // 2 + inst_id * (w // 4) + fi * 2,
                    cy=h // 2 + fi,
                    rx=w // 6,
                    ry=h // 4,
                )
                instances.append(SAM2InstanceMask(
                    instance_id=inst_id,
                    mask=mask,
                    confidence=0.95 - fi * 0.001,  # Slight decay
                    is_occluded=False,
                ))
            frame_results.append(SAM2FrameResult(
                frame_index=fi,
                instances=instances,
            ))

        return SAM2SequenceResult(
            frames=frame_results,
            num_instances=num_instances,
            scene_cuts=[],
            metadata={
                "backend": "mock",
                "model": "sam2.1_hiera_large",
                "fps_estimate": 44.0,
            },
        )

    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True


def _generate_elliptical_mask(
    h: int, w: int,
    cx: int, cy: int,
    rx: int, ry: int,
) -> np.ndarray:
    """Generate a filled elliptical boolean mask.

    Args:
        h, w: Image dimensions.
        cx, cy: Ellipse centre coordinates.
        rx, ry: Ellipse semi-axes (horizontal, vertical).

    Returns:
        ``[H, W]`` bool mask.
    """
    yy, xx = np.mgrid[:h, :w]
    # Clamp centre to image bounds for safety
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    rx = max(1, rx)
    ry = max(1, ry)
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return np.asarray(ellipse <= 1.0)


# ---------------------------------------------------------------------------
# Foundary-backed transport
# ---------------------------------------------------------------------------


class FoundarySAM2Backend(SAM2Transport):
    """SAM 2.1 transport via the foundary container service.

    Args:
        foundary_url: Base URL of the foundary container.
    """

    def __init__(self, foundary_url: str = "") -> None:
        self._url = foundary_url

    def segment_video(
        self,
        frames: np.ndarray,
        prompts: list[SAM2Prompt] | None = None,
    ) -> SAM2SequenceResult:
        """Submit segmentation job to foundary.

        Raises:
            BackendUnavailableError: Always — not yet implemented.
        """
        raise BackendUnavailableError(
            "FoundarySAM2Backend.segment_video not yet "
            "connected to live endpoint.",
            agent_fix_hint=(
                "Use MockSAM2Backend for testing, or "
                "configure foundary_url to point at a running "
                "sam2 service."
            ),
        )

    def is_available(self) -> bool:
        """Return ``False`` — live endpoint not yet connected."""
        return False


# ---------------------------------------------------------------------------
# Cryptomatte fusion
# ---------------------------------------------------------------------------


def fuse_sam2_cryptomatte(
    sam2_mask: np.ndarray,
    cryptomatte_id: np.ndarray,
    target_instance_ids: list[int] | None = None,
    dilation_px: int = 2,
) -> np.ndarray:
    """Fuse SAM2 VOS mask with Cryptomatte.ID for robust segmentation.

    Computes: ``M_fused = Dilate(M_SAM2 ∪ M_Cryptomatte_target)``

    When both SAM2 and Cryptomatte agree on instance boundaries, the
    result is tighter.  Where they disagree, the union + dilation
    provides conservative coverage.

    Args:
        sam2_mask: ``[H, W]`` bool SAM2 instance mask.
        cryptomatte_id: ``[H, W]`` uint32 Cryptomatte instance IDs
            from the FGPP ``Cryptomatte.ID`` channel.
        target_instance_ids: List of Cryptomatte instance IDs to
            include.  If ``None``, uses all non-zero IDs.
        dilation_px: Isotropic dilation radius in pixels.

    Returns:
        ``[H, W]`` bool fused mask.
    """
    sam2_bool = np.asarray(sam2_mask, dtype=bool)
    crypto = np.asarray(cryptomatte_id, dtype=np.uint32)

    if sam2_bool.shape != crypto.shape:
        raise ContractValidationError(
            f"SAM2 mask shape {sam2_bool.shape} != Cryptomatte shape {crypto.shape}.",
            agent_fix_hint="Ensure both masks have the same spatial resolution.",
        )

    # Build Cryptomatte target mask
    if target_instance_ids is not None:
        crypto_mask = np.isin(crypto, target_instance_ids)
    else:
        crypto_mask = crypto > 0

    # Union
    fused = sam2_bool | crypto_mask

    # Simple box dilation (no scipy dependency)
    if dilation_px > 0:
        fused = _box_dilate(fused, dilation_px)

    return fused


def _box_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Simple box dilation using cumulative sums (no scipy).

    Args:
        mask: ``[H, W]`` bool mask.
        radius: Dilation radius.

    Returns:
        Dilated ``[H, W]`` bool mask.
    """
    if radius <= 0:
        return mask

    h, w = mask.shape
    padded = np.pad(mask.astype(np.float32), radius, mode="constant")
    # Cumulative sum for fast box filter
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    r = radius
    # Compute box sum
    result = (
        cs[2 * r:, 2 * r:]
        - cs[:h, 2 * r:]
        - cs[2 * r:, :w]
        + cs[:h, :w]
    )
    return result > 0


def compute_mask_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        mask_a: ``[H, W]`` bool mask.
        mask_b: ``[H, W]`` bool mask.

    Returns:
        IoU score in [0, 1].
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = float(np.sum(a & b))
    union = float(np.sum(a | b))
    if union < 1.0:
        return 0.0
    return intersection / union


__all__ = [
    "FoundarySAM2Backend",
    "MockSAM2Backend",
    "SAM2FrameResult",
    "SAM2InstanceMask",
    "SAM2Prompt",
    "SAM2PromptType",
    "SAM2SequenceResult",
    "SAM2Transport",
    "compute_mask_iou",
    "fuse_sam2_cryptomatte",
]
