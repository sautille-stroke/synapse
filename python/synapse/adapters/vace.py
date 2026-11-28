"""Wan2.1-VACE unified video generation and editing adapter.

Implements the ``VaceAdapter`` — a whitebox adapter built on the Wan2.1-VACE
unified framework that consolidates Reference-to-Video (R2V),
Video-to-Video (V2V), and Masked Video-to-Video (MV2V) into a single
Video Condition Unit (VCU) interface.

The VCU unifies heterogeneous conditioning inputs — text prompts (T),
reference frame sequences (F), and spatiotemporal masks (M) — into a
single uniform condition tensor V = [T; F; M] that is consumed by the
VACE DiT backbone.

This adapter co-exists with the original ``WanAdapter`` (7-channel
ControlNeXt conditioning) and can be selected via the
``ConditionMode`` enum.

Transport backends reuse the existing ``WanBackendTransport`` ABC.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from synapse.adapters.base import (
    AdapterTier,
    PreparedConditionPayload,
    VideoGenerationConfig,
    VideoModelAdapter,
    VideoResult,
    register_adapter,
)
from synapse.errors import BackendUnavailableError, ContractValidationError

if TYPE_CHECKING:
    from synapse.fgpp.frame import FGPPSequence

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConditionMode(enum.Enum):
    """Conditioning mode selector.

    Attributes:
        FGPP_7CH: Legacy 7-channel ControlNeXt conditioning (WanAdapter).
        VCU_UNIFIED: VACE Video Condition Unit unified interface.
    """

    FGPP_7CH = "fgpp_7ch"
    VCU_UNIFIED = "vcu_unified"


class VaceTaskMode(enum.Enum):
    """VACE task mode selector.

    Attributes:
        REFERENCE_TO_VIDEO: Reference-Anything — guide generation from
            reference characters, styles, or objects.
        VIDEO_TO_VIDEO: Full-frame video editing (style transfer,
            colorisation, structural alteration).
        MASKED_VIDEO_TO_VIDEO: Spatiotemporal masked editing
            (inpainting, outpainting, object swapping).
    """

    REFERENCE_TO_VIDEO = "r2v"
    VIDEO_TO_VIDEO = "v2v"
    MASKED_VIDEO_TO_VIDEO = "mv2v"


# ---------------------------------------------------------------------------
# VCU Payload
# ---------------------------------------------------------------------------


@dataclass
class VCUPayload:
    """Video Condition Unit payload: V = [T; F; M].

    Encodes the unified condition tensor that VACE consumes, fusing
    text embeddings, reference frame features, and spatiotemporal masks
    into a single conditioning signal.

    Attributes:
        text_embedding: Text condition vector ``[D_text]`` float32.
            Extracted from CLIP/T5 text encoder.
        reference_frames: Optional reference frame sequence
            ``[N_ref, H, W, 3]`` uint8 sRGB.  Used in R2V mode.
        spatiotemporal_mask: Optional binary mask ``[T, H, W]`` float32
            in [0, 1].  Defines editable regions in MV2V mode.
        task_mode: Which VACE task to execute.
        fgpp_control_tensor: Optional FGPP-derived 7-channel control
            ``[7, T, H, W]`` float32 for hybrid conditioning.
    """

    text_embedding: np.ndarray | None = None  # [D_text] float32
    reference_frames: np.ndarray | None = None  # [N_ref, H, W, 3] uint8
    spatiotemporal_mask: np.ndarray | None = None  # [T, H, W] float32
    task_mode: VaceTaskMode = VaceTaskMode.REFERENCE_TO_VIDEO
    fgpp_control_tensor: np.ndarray | None = None  # [7, T, H, W] float32


# ---------------------------------------------------------------------------
# VACE Backend Transport ABC
# ---------------------------------------------------------------------------


class VaceBackendTransport(ABC):
    """Abstract transport for VACE inference submission.

    Subclasses must implement ``submit`` (which runs the actual
    inference) and ``is_available`` (health/readiness probe).
    """

    @abstractmethod
    def submit(
        self,
        vcu_payload: VCUPayload,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Submit a VCU payload and return generated/edited frames.

        Args:
            vcu_payload: The unified VCU condition payload.
            conditions: Standard condition payload (text, keyframes, etc.).
            config: Generation configuration.

        Returns:
            ``[T, H, W, 3]`` uint8 sRGB frames.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend can accept work."""
        ...


class MockVaceBackend(VaceBackendTransport):
    """Deterministic CPU-only VACE backend for testing.

    Generates a gradient pattern modulated by task mode for
    deterministic output validation.
    """

    def submit(
        self,
        vcu_payload: VCUPayload,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> np.ndarray:
        """Generate deterministic test frames.

        The output encodes task_mode in the blue channel intensity:
        R2V=64, V2V=128, MV2V=192.

        Args:
            vcu_payload: VCU payload (task_mode determines pattern).
            conditions: Standard conditions.
            config: Generation config.

        Returns:
            ``[T, H, W, 3]`` uint8 deterministic frames.
        """
        rng = np.random.RandomState(config.seed)

        t, h, w = config.num_frames, config.height, config.width
        frames = np.zeros((t, h, w, 3), dtype=np.uint8)

        # Encode task mode in blue channel
        mode_intensity: dict[VaceTaskMode, int] = {
            VaceTaskMode.REFERENCE_TO_VIDEO: 64,
            VaceTaskMode.VIDEO_TO_VIDEO: 128,
            VaceTaskMode.MASKED_VIDEO_TO_VIDEO: 192,
        }
        blue_val = mode_intensity.get(vcu_payload.task_mode, 128)

        for i in range(t):
            # Gradient in R, noise in G, mode in B
            grad = np.linspace(0, 255, w, dtype=np.uint8)
            frames[i, :, :, 0] = np.broadcast_to(grad, (h, w))
            frames[i, :, :, 1] = rng.randint(0, 128, (h, w), dtype=np.uint8)
            frames[i, :, :, 2] = blue_val

            # Apply mask modulation in MV2V mode
            if (
                vcu_payload.task_mode == VaceTaskMode.MASKED_VIDEO_TO_VIDEO
                and vcu_payload.spatiotemporal_mask is not None
                and i < vcu_payload.spatiotemporal_mask.shape[0]
            ):
                mask_slice = vcu_payload.spatiotemporal_mask[i]  # [H, W]
                # Darken unmasked regions
                mask_u8 = (mask_slice * 255).astype(np.uint8)
                mask_h, mask_w = mask_u8.shape[:2]
                crop_h, crop_w = min(mask_h, h), min(mask_w, w)
                frames[i, :crop_h, :crop_w, 0] = (
                    frames[i, :crop_h, :crop_w, 0].astype(np.float32)
                    * mask_u8[:crop_h, :crop_w].astype(np.float32)
                    / 255.0
                ).astype(np.uint8)

        return frames

    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True


# ---------------------------------------------------------------------------
# VaceAdapter
# ---------------------------------------------------------------------------


def _build_vcu_reference_tensor(
    reference_images: list[np.ndarray],
) -> np.ndarray:
    """Stack reference images into a single tensor.

    Args:
        reference_images: List of ``[H, W, 3]`` uint8 images.

    Returns:
        ``[N_ref, H, W, 3]`` uint8 tensor.

    Raises:
        ContractValidationError: If images have inconsistent shapes.
    """
    if not reference_images:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    shapes = {img.shape[:2] for img in reference_images}
    if len(shapes) > 1:
        raise ContractValidationError(
            f"Reference images have inconsistent shapes: {shapes}",
            agent_fix_hint="Resize all reference images to the same resolution.",
        )

    return np.stack(reference_images, axis=0)


def _build_spatiotemporal_mask(
    cryptomatte_ids: np.ndarray | None,
    target_entity_ids: list[int] | None = None,
) -> np.ndarray | None:
    """Build a spatiotemporal binary mask from Cryptomatte IDs.

    Args:
        cryptomatte_ids: ``[T, H, W]`` uint32 entity IDs.
        target_entity_ids: Entity IDs to include in the mask.
            If ``None``, all non-zero IDs are included.

    Returns:
        ``[T, H, W]`` float32 mask in [0, 1], or ``None`` if no IDs.
    """
    if cryptomatte_ids is None:
        return None

    if target_entity_ids is not None:
        mask = np.zeros_like(cryptomatte_ids, dtype=np.float32)
        for eid in target_entity_ids:
            mask[cryptomatte_ids == eid] = 1.0
    else:
        mask = (cryptomatte_ids > 0).astype(np.float32)

    return mask


@register_adapter
class VaceAdapter(VideoModelAdapter):
    """Tier-1 whitebox adapter for Wan 2.1-VACE unified framework.

    Consolidates R2V, V2V, and MV2V tasks into a single VCU interface,
    while optionally consuming FGPP 7-channel control tensors for
    hybrid conditioning.

    Args:
        backend: VACE transport backend for inference.
        task_mode: Default task mode (R2V, V2V, MV2V).
        condition_mode: Whether to use pure VCU or hybrid FGPP+VCU.
        reference_images: Optional list of reference images for R2V.
        target_entity_ids: Optional entity IDs for MV2V mask generation.
    """

    ADAPTER_ID = "vace"

    def __init__(
        self,
        backend: VaceBackendTransport | None = None,
        task_mode: VaceTaskMode = VaceTaskMode.REFERENCE_TO_VIDEO,
        condition_mode: ConditionMode = ConditionMode.VCU_UNIFIED,
        reference_images: list[np.ndarray] | None = None,
        target_entity_ids: list[int] | None = None,
    ) -> None:
        self._backend = backend
        self._task_mode = task_mode
        self._condition_mode = condition_mode
        self._reference_images = reference_images or []
        self._target_entity_ids = target_entity_ids

    # -- properties ----------------------------------------------------------

    @property
    def adapter_id(self) -> str:
        return self.ADAPTER_ID

    @property
    def tier(self) -> AdapterTier:
        return "whitebox"

    @property
    def task_mode(self) -> VaceTaskMode:
        """Current VACE task mode."""
        return self._task_mode

    @task_mode.setter
    def task_mode(self, mode: VaceTaskMode) -> None:
        self._task_mode = mode

    @property
    def condition_mode(self) -> ConditionMode:
        """Current conditioning mode."""
        return self._condition_mode

    # -- VCU construction ----------------------------------------------------

    def build_vcu_payload(
        self,
        fgpp_sequence: FGPPSequence,
        config: VideoGenerationConfig,
    ) -> VCUPayload:
        """Construct the unified VCU payload from FGPP data.

        Pipeline:
        1. Extract text prompt → text embedding placeholder (actual
           embedding done by backend).
        2. Stack reference frames if R2V mode.
        3. Build spatiotemporal mask from Cryptomatte if MV2V mode.
        4. Optionally build FGPP 7-channel control tensor for hybrid
           conditioning.

        Args:
            fgpp_sequence: Source FGPP sequence.
            config: Generation configuration.

        Returns:
            ``VCUPayload`` ready for backend submission.
        """
        n_frames = len(fgpp_sequence.frames)

        # 1. Text embedding placeholder (backend handles actual encoding)
        text_emb = None

        # 2. Reference frames (R2V mode)
        ref_tensor: np.ndarray | None = None
        if self._task_mode == VaceTaskMode.REFERENCE_TO_VIDEO and self._reference_images:
            ref_tensor = _build_vcu_reference_tensor(self._reference_images)

        # 3. Spatiotemporal mask (MV2V mode)
        st_mask: np.ndarray | None = None
        if self._task_mode == VaceTaskMode.MASKED_VIDEO_TO_VIDEO and n_frames > 0:
            # Collect cryptomatte IDs
            crypto_maps: list[np.ndarray] = []
            for frame in fgpp_sequence.frames:
                if frame.cryptomatte_id is not None:
                    crypto_maps.append(frame.cryptomatte_id)

            if len(crypto_maps) == n_frames:
                crypto_stack = np.stack(crypto_maps, axis=0)  # [T, H, W]
                st_mask = _build_spatiotemporal_mask(
                    crypto_stack,
                    self._target_entity_ids,
                )

        # 4. Optional FGPP 7-channel control tensor for hybrid mode
        fgpp_control: np.ndarray | None = None
        if self._condition_mode == ConditionMode.VCU_UNIFIED and n_frames > 0:
            # Build 7ch control via WanAdapter's internal method
            from synapse.adapters.wan import WanAdapter

            wan = WanAdapter()
            payload = wan.prepare_conditions(fgpp_sequence, config)
            fgpp_control = payload.control_tensor

        return VCUPayload(
            text_embedding=text_emb,
            reference_frames=ref_tensor,
            spatiotemporal_mask=st_mask,
            task_mode=self._task_mode,
            fgpp_control_tensor=fgpp_control,
        )

    # -- VideoModelAdapter interface -----------------------------------------

    def prepare_conditions(
        self,
        fgpp_sequence: FGPPSequence,
        config: VideoGenerationConfig,
    ) -> PreparedConditionPayload:
        """Transform an FGPP sequence into VACE-ready conditions.

        Builds a ``VCUPayload`` and embeds it in the standard
        ``PreparedConditionPayload.extra["vcu_payload"]`` for
        downstream consumption by ``generate()``.

        Args:
            fgpp_sequence: Source FGPP sequence.
            config: Generation configuration.

        Returns:
            ``PreparedConditionPayload`` with VCU data in ``extra``.
        """
        vcu = self.build_vcu_payload(fgpp_sequence, config)

        # Build Cryptomatte passthrough
        n_frames = len(fgpp_sequence.frames)
        crypto_id: np.ndarray | None = None
        crypto_maps: list[np.ndarray] = [
            f.cryptomatte_id
            for f in fgpp_sequence.frames
            if f.cryptomatte_id is not None
        ]
        if len(crypto_maps) == n_frames and n_frames > 0:
            crypto_id = np.stack(crypto_maps, axis=0)

        return PreparedConditionPayload(
            control_tensor=vcu.fgpp_control_tensor,
            cryptomatte_id=crypto_id,
            fgpp_frames=list(fgpp_sequence.frames),
            extra={
                "vcu_payload": vcu,
                "condition_mode": self._condition_mode.value,
                "task_mode": self._task_mode.value,
            },
        )

    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> VideoResult:
        """Generate or edit video via the VACE backend.

        Extracts the ``VCUPayload`` from ``conditions.extra`` and
        submits to the configured backend transport.

        Args:
            conditions: Output of ``prepare_conditions()``.
            config: Generation configuration.

        Returns:
            ``VideoResult`` with sRGB uint8 frames.

        Raises:
            BackendUnavailableError: If no backend is configured.
        """
        if self._backend is None:
            raise BackendUnavailableError(
                "VaceAdapter has no backend transport configured.",
                agent_fix_hint="Pass a VaceBackendTransport to the constructor.",
            )

        vcu = conditions.extra.get("vcu_payload")
        if vcu is None:
            # Reconstruct VCU from conditions if not present
            vcu = VCUPayload(
                task_mode=self._task_mode,
                fgpp_control_tensor=conditions.control_tensor,
            )

        frames = self._backend.submit(vcu, conditions, config)

        return VideoResult(
            frames_srgb_u8=frames,
            fps=config.fps,
            seed=config.seed,
            adapter_id=self.adapter_id,
            metadata={
                "task_mode": vcu.task_mode.value,
                "condition_mode": self._condition_mode.value,
                "has_reference": vcu.reference_frames is not None,
                "has_mask": vcu.spatiotemporal_mask is not None,
            },
        )
