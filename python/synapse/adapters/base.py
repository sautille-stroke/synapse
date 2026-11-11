"""Video model adapter abstract base class and global registry.

Provides the ``VideoModelAdapter`` ABC that every video-generation backend
must implement, along with a lightweight decorator-based plugin registry
(``register_adapter`` / ``create_adapter`` / ``list_adapters``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

AdapterTier = Literal["whitebox", "blackbox", "mock"]
"""Classification of adapter capabilities.

* ``whitebox``  — full access to model internals (latents, attention maps).
* ``blackbox``  — input/output only (e.g. hosted API).
* ``mock``      — deterministic CPU-only stub for testing.
"""

# ---------------------------------------------------------------------------
# Configuration & result data classes
# ---------------------------------------------------------------------------


@dataclass
class VideoGenerationConfig:
    """Configuration for a single video generation call.

    Attributes:
        seed: RNG seed for reproducible generation.
        fps: Target frames per second.
        width: Output width in pixels.
        height: Output height in pixels.
        guidance_scale: Classifier-free guidance weight.
        num_inference_steps: Number of denoising steps.
        num_frames: Number of frames to generate.
        extra: Backend-specific overrides passed through verbatim.
    """

    seed: int = 42
    fps: float = 24.0
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_frames: int = 16
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedConditionPayload:
    """Prepared condition tensors ready for adapter consumption.

    Attributes:
        control_tensor: Optional control signal ``[C, T, H, W]`` float32.
        text_prompt: Positive text prompt.
        negative_prompt: Negative text prompt.
        keyframe_images: Optional list of ``[H, W, 3]`` uint8 keyframes.
        base_color: Optional ``[T, H, W, 3]`` float32 base-colour sequence.
        world_normal: Optional ``[T, H, W, 3]`` float32 world-normal sequence.
        direct_radiance: Optional ``[T, H, W, 3]`` float32 radiance sequence.
        depth: Optional ``[T, H, W]`` float32 depth sequence.
        cryptomatte_id: Optional ``[T, H, W]`` uint32 entity-ID sequence.
        extra: Backend-specific auxiliary data.
    """

    control_tensor: np.ndarray | None = None
    text_prompt: str = ""
    negative_prompt: str = ""
    keyframe_images: list[np.ndarray] = field(default_factory=list)

    # FGPP-derived per-frame channels (set by ``prepare_conditions``)
    base_color: np.ndarray | None = None  # [T, H, W, 3] float32
    world_normal: np.ndarray | None = None  # [T, H, W, 3] float32
    direct_radiance: np.ndarray | None = None  # [T, H, W, 3] float32
    depth: np.ndarray | None = None  # [T, H, W] float32
    cryptomatte_id: np.ndarray | None = None  # [T, H, W] uint32

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoResult:
    """Result of a video generation call.

    Attributes:
        frames_srgb_u8: Output video ``[T, H, W, 3]`` uint8 sRGB.
        fps: Frames per second of the generated video.
        seed: RNG seed that was used.
        adapter_id: Identifier of the adapter that produced this result.
        metadata: Arbitrary metadata (defect log, timing, etc.).
    """

    frames_srgb_u8: np.ndarray  # [T, H, W, 3] uint8
    fps: float
    seed: int
    adapter_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class VideoModelAdapter(ABC):
    """Abstract base class for video generation model adapters.

    Subclasses must implement:
    * ``adapter_id`` — unique string identifier (used as registry key).
    * ``tier`` — one of ``"whitebox"``, ``"blackbox"``, ``"mock"``.
    * ``prepare_conditions`` — extract / transform FGPP data into a
      ``PreparedConditionPayload``.
    * ``generate`` — run the generative model and return a ``VideoResult``.
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique identifier for this adapter (e.g. ``"wan2.1"``)."""
        ...

    @property
    @abstractmethod
    def tier(self) -> AdapterTier:
        """Capability tier of this adapter."""
        ...

    @abstractmethod
    def prepare_conditions(
        self,
        fgpp_sequence: Any,  # synapse.fgpp.frame.FGPPSequence
        config: VideoGenerationConfig,
    ) -> PreparedConditionPayload:
        """Transform an FGPP sequence into adapter-ready condition tensors.

        Args:
            fgpp_sequence: Source ``FGPPSequence`` with per-frame channels.
            config: Generation configuration (resolution, seed, etc.).

        Returns:
            A ``PreparedConditionPayload`` ready for ``generate()``.
        """
        ...

    @abstractmethod
    def generate(
        self,
        conditions: PreparedConditionPayload,
        config: VideoGenerationConfig,
    ) -> VideoResult:
        """Generate video frames from the prepared conditions.

        Args:
            conditions: Output of ``prepare_conditions()``.
            config: Generation configuration.

        Returns:
            A ``VideoResult`` containing sRGB uint8 frames.
        """
        ...


# ---------------------------------------------------------------------------
# Global plugin registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[VideoModelAdapter]] = {}


def register_adapter(
    cls: type[VideoModelAdapter],
) -> type[VideoModelAdapter]:
    """Class decorator that registers a ``VideoModelAdapter`` subclass.

    The adapter is keyed by the ``adapter_id`` property value on a
    temporary throwaway instance (constructed with no arguments).

    Args:
        cls: The adapter class to register.

    Returns:
        The same class, unmodified.

    Raises:
        TypeError: If *cls* is not a ``VideoModelAdapter`` subclass.
        ValueError: If an adapter with the same ID is already registered.
    """
    if not (isinstance(cls, type) and issubclass(cls, VideoModelAdapter)):
        raise TypeError(
            f"register_adapter expects a VideoModelAdapter subclass, "
            f"got {cls!r}"
        )

    # Instantiate with no args to read the adapter_id property.
    instance = cls()
    aid = instance.adapter_id

    if aid in _ADAPTER_REGISTRY:
        existing = _ADAPTER_REGISTRY[aid]
        if existing is not cls:
            raise ValueError(
                f"Adapter ID {aid!r} already registered by "
                f"{existing.__qualname__}; cannot re-register with "
                f"{cls.__qualname__}"
            )
    _ADAPTER_REGISTRY[aid] = cls
    return cls


def create_adapter(name: str, **kwargs: Any) -> VideoModelAdapter:
    """Instantiate a registered adapter by its ``adapter_id``.

    Args:
        name: Adapter identifier (e.g. ``"mock"``).
        **kwargs: Forwarded to the adapter constructor.

    Returns:
        A newly constructed ``VideoModelAdapter`` instance.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown adapter {name!r}. "
            f"Available adapters: {available}"
        )
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """Return a sorted list of registered adapter identifiers."""
    return sorted(_ADAPTER_REGISTRY.keys())
