"""NVIDIA DiffusionRenderer neural inverse/forward rendering transport.

Bridges the NVIDIA DiffusionRenderer (CVPR 2025) / Cosmos-Transfer1-
DiffusionRenderer pipeline into synapse's inverse rendering stack.

DiffusionRenderer performs bidirectional G-buffer ↔ photorealistic image
translation using a video diffusion backbone:

- **Inverse mode**: Input video → G-buffers (depth, normals, albedo,
  roughness, metallic).
- **Forward mode**: G-buffers + lighting → relit photorealistic video.

This module maps DiffusionRenderer's G-buffer schema to/from synapse's
FGPP 15-channel wire format, enabling dual-track inverse rendering
(classical ``synapse.recon.delight`` SH path vs neural DiffusionRenderer
path) with cross-validation.

Transport backends:

- :class:`MockDiffusionRendererBackend` — deterministic CPU-only mock
  for zero-GPU regression testing.
- :class:`FoundaryDiffusionRendererBackend` — routes to the foundary
  container's ``diffusion_renderer`` service endpoint.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.errors import BackendUnavailableError, ContractValidationError

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DiffRendererMode(enum.Enum):
    """DiffusionRenderer operation mode.

    Attributes:
        INVERSE: Input video → G-buffer decomposition.
        FORWARD: G-buffers + lighting → relit video.
    """

    INVERSE = "inverse"
    FORWARD = "forward"


# ---------------------------------------------------------------------------
# G-buffer result container
# ---------------------------------------------------------------------------


@dataclass
class GBufferResult:
    """G-buffer decomposition result from DiffusionRenderer inverse mode.

    All spatial arrays share the same ``[T, H, W, ...]`` layout where
    T is the number of frames.

    Attributes:
        depth: ``[T, H, W]`` float32 linear depth in metres.
        normals: ``[T, H, W, 3]`` float32 world-space unit normals.
        albedo: ``[T, H, W, 3]`` float32 linear RGB base colour.
        roughness: ``[T, H, W]`` float32 perceptual roughness in [0, 1].
        metallic: ``[T, H, W]`` float32 metalness in [0, 1].
        metadata: Arbitrary backend metadata (timings, model version, etc.).
    """

    depth: np.ndarray  # [T, H, W] float32
    normals: np.ndarray  # [T, H, W, 3] float32
    albedo: np.ndarray  # [T, H, W, 3] float32
    roughness: np.ndarray  # [T, H, W] float32
    metallic: np.ndarray  # [T, H, W] float32
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Relit result container
# ---------------------------------------------------------------------------


@dataclass
class RelitResult:
    """Relit video result from DiffusionRenderer forward mode.

    Attributes:
        frames_srgb_u8: ``[T, H, W, 3]`` uint8 sRGB relit frames.
        metadata: Arbitrary backend metadata.
    """

    frames_srgb_u8: np.ndarray  # [T, H, W, 3] uint8
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FGPP ↔ G-buffer mapping
# ---------------------------------------------------------------------------


def gbuffer_to_fgpp_channels(gbuf: GBufferResult) -> dict[str, np.ndarray]:
    """Map a DiffusionRenderer G-buffer to FGPP-compatible channels.

    Performs the following conversions:

    - ``depth`` → FGPP ``CameraLinearDepth.Z`` (identity, metres)
    - ``normals`` → FGPP ``WorldNormal.XYZ`` (identity, unit vectors)
    - ``albedo`` → FGPP ``BaseColor.RGB`` (linear RGB)
    - ``roughness`` + ``metallic`` → FGPP ``RoughnessMetallic.RG``

    Args:
        gbuf: DiffusionRenderer G-buffer result.

    Returns:
        Dict mapping FGPP channel names to numpy arrays.

    Raises:
        ContractValidationError: If array shapes are inconsistent.
    """
    t = gbuf.depth.shape[0]
    h, w = gbuf.depth.shape[1], gbuf.depth.shape[2]

    # Validate spatial consistency
    expected_3d = (t, h, w)
    expected_4d = (t, h, w, 3)

    if gbuf.normals.shape != expected_4d:
        raise ContractValidationError(
            f"normals shape {gbuf.normals.shape} != expected {expected_4d}",
            agent_fix_hint="Ensure DiffusionRenderer outputs consistent spatial dims.",
        )
    if gbuf.albedo.shape != expected_4d:
        raise ContractValidationError(
            f"albedo shape {gbuf.albedo.shape} != expected {expected_4d}",
            agent_fix_hint="Ensure DiffusionRenderer outputs consistent spatial dims.",
        )
    if gbuf.roughness.shape != expected_3d:
        raise ContractValidationError(
            f"roughness shape {gbuf.roughness.shape} != expected {expected_3d}",
            agent_fix_hint="Ensure DiffusionRenderer outputs consistent spatial dims.",
        )
    if gbuf.metallic.shape != expected_3d:
        raise ContractValidationError(
            f"metallic shape {gbuf.metallic.shape} != expected {expected_3d}",
            agent_fix_hint="Ensure DiffusionRenderer outputs consistent spatial dims.",
        )

    # Pack roughness + metallic → [T, H, W, 2]
    roughness_metallic = np.stack(
        [gbuf.roughness, gbuf.metallic], axis=-1,
    ).astype(np.float32)

    return {
        "depth": gbuf.depth.astype(np.float32),
        "world_normal": gbuf.normals.astype(np.float32),
        "base_color": gbuf.albedo.astype(np.float32),
        "roughness_metallic": roughness_metallic,
    }


def fgpp_channels_to_gbuffer_inputs(
    depth: np.ndarray,
    world_normal: np.ndarray,
    base_color: np.ndarray,
    roughness_metallic: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map FGPP channels to DiffusionRenderer forward-mode inputs.

    This is the reverse of :func:`gbuffer_to_fgpp_channels`, preparing
    the G-buffer tensor for the neural forward renderer.

    Args:
        depth: ``[T, H, W]`` float32 linear depth.
        world_normal: ``[T, H, W, 3]`` float32 normals.
        base_color: ``[T, H, W, 3]`` float32 linear RGB.
        roughness_metallic: ``[T, H, W, 2]`` float32 roughness+metallic.

    Returns:
        Dict with ``depth``, ``normals``, ``albedo``, ``roughness``,
        ``metallic`` arrays ready for the forward renderer.
    """
    return {
        "depth": depth.astype(np.float32),
        "normals": world_normal.astype(np.float32),
        "albedo": base_color.astype(np.float32),
        "roughness": roughness_metallic[..., 0].astype(np.float32),
        "metallic": roughness_metallic[..., 1].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Transport ABC
# ---------------------------------------------------------------------------


class DiffusionRendererTransport(ABC):
    """Abstract transport for DiffusionRenderer inference.

    Subclasses implement the inverse and forward rendering pipelines.
    """

    @abstractmethod
    def inverse_render(
        self,
        frames_srgb_u8: np.ndarray,
    ) -> GBufferResult:
        """Decompose input video frames into G-buffers.

        Args:
            frames_srgb_u8: ``[T, H, W, 3]`` uint8 sRGB input video.

        Returns:
            :class:`GBufferResult` with per-pixel material decomposition.
        """
        ...

    @abstractmethod
    def forward_render(
        self,
        gbuffer: GBufferResult,
        environment_map: np.ndarray | None = None,
    ) -> RelitResult:
        """Render G-buffers into photorealistic video under new lighting.

        Args:
            gbuffer: G-buffer decomposition to render.
            environment_map: Optional ``[H_env, W_env, 3]`` float32
                HDR environment map.  If ``None``, uses a default
                uniform white lighting.

        Returns:
            :class:`RelitResult` with relit sRGB frames.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend is ready to accept work."""
        ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockDiffusionRendererBackend(DiffusionRendererTransport):
    """Deterministic CPU-only mock for DiffusionRenderer.

    Inverse mode produces synthetic G-buffers from the input video's
    luminance and gradients.  Forward mode applies a simple Lambertian
    shading to the G-buffers.
    """

    def inverse_render(
        self,
        frames_srgb_u8: np.ndarray,
    ) -> GBufferResult:
        """Generate synthetic G-buffers from input video luminance.

        - ``depth``: normalised mean luminance per pixel.
        - ``normals``: constant [0, 0, 1] (front-facing).
        - ``albedo``: input sRGB / 255 (approximate linear).
        - ``roughness``: constant 0.5.
        - ``metallic``: constant 0.0.

        Args:
            frames_srgb_u8: ``[T, H, W, 3]`` uint8 sRGB.

        Returns:
            Deterministic :class:`GBufferResult`.
        """
        t, h, w, _ = frames_srgb_u8.shape
        f32 = frames_srgb_u8.astype(np.float32) / 255.0

        # Depth from mean luminance (synthetic)
        depth = np.mean(f32, axis=-1)  # [T, H, W]

        # Front-facing normals
        normals = np.zeros((t, h, w, 3), dtype=np.float32)
        normals[..., 2] = 1.0  # z = 1 (towards camera)

        # Approximate linear albedo
        albedo = f32

        # Uniform material properties
        roughness = np.full((t, h, w), 0.5, dtype=np.float32)
        metallic = np.zeros((t, h, w), dtype=np.float32)

        return GBufferResult(
            depth=depth,
            normals=normals,
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            metadata={"backend": "mock", "mode": "inverse"},
        )

    def forward_render(
        self,
        gbuffer: GBufferResult,
        environment_map: np.ndarray | None = None,
    ) -> RelitResult:
        """Apply simple Lambertian shading to G-buffers.

        Uses ``n · l`` with a fixed directional light at [0.577, 0.577, 0.577].

        Args:
            gbuffer: G-buffer to render.
            environment_map: Ignored in mock mode.

        Returns:
            Deterministic :class:`RelitResult`.
        """
        normals = gbuffer.normals  # [T, H, W, 3]
        albedo = gbuffer.albedo  # [T, H, W, 3]

        # Fixed directional light (normalised [1, 1, 1])
        light_dir = np.array([0.57735, 0.57735, 0.57735], dtype=np.float32)
        n_dot_l = np.sum(normals * light_dir, axis=-1, keepdims=True)
        n_dot_l = np.clip(n_dot_l, 0.0, 1.0)

        # Lambertian: albedo * max(n·l, 0) + ambient
        ambient = 0.1
        lit = albedo * (n_dot_l + ambient)
        lit = np.clip(lit, 0.0, 1.0)

        # Convert to uint8 sRGB (approximate gamma)
        gamma = np.power(lit, 1.0 / 2.2)
        frames_u8 = (gamma * 255.0).astype(np.uint8)

        return RelitResult(
            frames_srgb_u8=frames_u8,
            metadata={"backend": "mock", "mode": "forward"},
        )

    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True


# ---------------------------------------------------------------------------
# Foundary-backed transport
# ---------------------------------------------------------------------------


class FoundaryDiffusionRendererBackend(DiffusionRendererTransport):
    """DiffusionRenderer transport via the foundary container service.

    Submits inference requests to the ``diffusion_renderer`` service
    endpoint registered in the foundary container.

    Args:
        foundary_url: Base URL of the foundary container
            (e.g., ``http://localhost:8080``).
    """

    def __init__(self, foundary_url: str = "") -> None:
        self._url = foundary_url

    def inverse_render(
        self,
        frames_srgb_u8: np.ndarray,
    ) -> GBufferResult:
        """Submit inverse rendering job to foundary.

        Raises:
            BackendUnavailableError: Always — not yet implemented.
        """
        raise BackendUnavailableError(
            "FoundaryDiffusionRendererBackend.inverse_render not yet "
            "connected to live endpoint.",
            agent_fix_hint=(
                "Use MockDiffusionRendererBackend for testing, or "
                "configure foundary_url to point at a running "
                "diffusion_renderer service."
            ),
        )

    def forward_render(
        self,
        gbuffer: GBufferResult,
        environment_map: np.ndarray | None = None,
    ) -> RelitResult:
        """Submit forward rendering job to foundary.

        Raises:
            BackendUnavailableError: Always — not yet implemented.
        """
        raise BackendUnavailableError(
            "FoundaryDiffusionRendererBackend.forward_render not yet "
            "connected to live endpoint.",
            agent_fix_hint=(
                "Use MockDiffusionRendererBackend for testing, or "
                "configure foundary_url to point at a running "
                "diffusion_renderer service."
            ),
        )

    def is_available(self) -> bool:
        """Return ``False`` — live endpoint not yet connected."""
        return False


# ---------------------------------------------------------------------------
# Dual-track inverse rendering comparator
# ---------------------------------------------------------------------------


@dataclass
class DualTrackComparisonResult:
    """Result of comparing classical vs neural inverse rendering paths.

    Attributes:
        albedo_psnr: PSNR between classical delight albedo and
            DiffusionRenderer albedo (dB).
        normal_cosine_sim: Mean cosine similarity of normals
            between both paths.
        depth_rmse: Root mean square error of depth estimates.
        agreement_score: Composite agreement score in [0, 1].
        metadata: Per-channel comparison details.
    """

    albedo_psnr: float = 0.0
    normal_cosine_sim: float = 0.0
    depth_rmse: float = 0.0
    agreement_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def compare_inverse_rendering_tracks(
    classical_albedo: np.ndarray,
    classical_normals: np.ndarray,
    classical_depth: np.ndarray,
    neural_gbuffer: GBufferResult,
) -> DualTrackComparisonResult:
    """Compare classical SH delight path vs DiffusionRenderer neural path.

    Computes per-channel metrics to quantify agreement between the two
    inverse rendering approaches for quality assurance.

    Args:
        classical_albedo: ``[T, H, W, 3]`` float32 from ``synapse.recon.delight``.
        classical_normals: ``[T, H, W, 3]`` float32 classical normals.
        classical_depth: ``[T, H, W]`` float32 classical depth.
        neural_gbuffer: DiffusionRenderer G-buffer result.

    Returns:
        :class:`DualTrackComparisonResult` with per-channel metrics.
    """
    # Albedo PSNR
    mse_albedo = float(np.mean(
        (classical_albedo.astype(np.float64) - neural_gbuffer.albedo.astype(np.float64)) ** 2,
    ))
    if mse_albedo < 1e-12:
        albedo_psnr = 100.0  # Perfect match
    else:
        albedo_psnr = float(10.0 * np.log10(1.0 / mse_albedo))

    # Normal cosine similarity
    cn = classical_normals.astype(np.float64).reshape(-1, 3)
    nn = neural_gbuffer.normals.astype(np.float64).reshape(-1, 3)
    cn_norm = np.linalg.norm(cn, axis=-1, keepdims=True)
    nn_norm = np.linalg.norm(nn, axis=-1, keepdims=True)
    cn_norm = np.maximum(cn_norm, 1e-8)
    nn_norm = np.maximum(nn_norm, 1e-8)
    cos_sim = np.sum((cn / cn_norm) * (nn / nn_norm), axis=-1)
    normal_cosine_sim = float(np.mean(np.clip(cos_sim, -1.0, 1.0)))

    # Depth RMSE
    depth_diff = classical_depth.astype(np.float64) - neural_gbuffer.depth.astype(np.float64)
    depth_rmse = float(np.sqrt(np.mean(depth_diff ** 2)))

    # Composite agreement: weighted average
    # Higher PSNR → better, higher cosine → better, lower RMSE → better
    psnr_score = min(albedo_psnr / 40.0, 1.0)  # 40 dB = perfect
    cos_score = max(normal_cosine_sim, 0.0)
    rmse_score = max(1.0 - depth_rmse / 10.0, 0.0)  # 10m RMSE = 0
    agreement = 0.4 * psnr_score + 0.4 * cos_score + 0.2 * rmse_score

    return DualTrackComparisonResult(
        albedo_psnr=albedo_psnr,
        normal_cosine_sim=normal_cosine_sim,
        depth_rmse=depth_rmse,
        agreement_score=float(np.clip(agreement, 0.0, 1.0)),
        metadata={
            "mse_albedo": mse_albedo,
            "psnr_score": psnr_score,
            "cos_score": cos_score,
            "rmse_score": rmse_score,
        },
    )


__all__ = [
    "DiffRendererMode",
    "DiffusionRendererTransport",
    "DualTrackComparisonResult",
    "FoundaryDiffusionRendererBackend",
    "GBufferResult",
    "MockDiffusionRendererBackend",
    "RelitResult",
    "compare_inverse_rendering_tracks",
    "fgpp_channels_to_gbuffer_inputs",
    "gbuffer_to_fgpp_channels",
]
