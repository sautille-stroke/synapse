"""Causal window scheduler, discontinuity detector, and latent flow pooling.

Implements the causal sliding-window partitioning strategy for long-sequence
video generation with 3D causal VAE warm-start, per-frame discontinuity
scoring, overlap fusion with variance-preserving sigmoid annealing, and
motion-vector-based latent-space flow operator computation.

All heavy dependencies (scipy, torch) are lazy-imported inside the
functions that need them.  Module-level imports are limited to ``numpy``
and standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# -----------------------------------------------------------------------
# WindowSlice — describes a single causal generation window
# -----------------------------------------------------------------------


@dataclass
class WindowSlice:
    """Contiguous frame slice for a single causal generation window.

    Attributes:
        start: First frame index (inclusive).
        end: Last frame index (exclusive).
        causal_warmup_prefix_indices: Frame indices fed as warm-start
            context to the 3D causal VAE for windows k >= 1.
    """

    start: int
    end: int
    causal_warmup_prefix_indices: list[int] = field(
        default_factory=list,
    )

    @property
    def length(self) -> int:
        """Number of frames in this window."""
        return self.end - self.start


# -----------------------------------------------------------------------
# CausalWindowPlan — partitions a long sequence into overlapping windows
# -----------------------------------------------------------------------


@dataclass
class CausalWindowPlan:
    """Partition plan for causal sliding-window video generation.

    Default values correspond to tau=25 latent steps at temporal
    compression factor 4, with 9-step latent overlap.

    Attributes:
        window_frames: Frames per generation window (tau=25 latent).
        stride_frames: Non-overlapping stride between windows
            (delta_tau=16 latent steps).
        overlap_frames: Overlapping frames between consecutive windows
            (tau_overlap=9 latent steps).
        total_frames: Total number of frames in the sequence.
    """

    window_frames: int = 97
    stride_frames: int = 64
    overlap_frames: int = 33
    total_frames: int = 97

    def compute_windows(self) -> list[WindowSlice]:
        """Compute the list of causal generation windows.

        For window k >= 1, the causal warm-up prefix consists of 3
        frames immediately preceding the window start:
        ``[stride*k - 3, stride*k - 2, stride*k - 1]``.

        The last window may be shorter than ``window_frames`` if
        ``total_frames`` does not align evenly.

        Returns:
            Ordered list of ``WindowSlice`` instances covering the
            entire sequence.
        """
        if self.total_frames <= 0:
            return []

        windows: list[WindowSlice] = []
        k = 0
        while True:
            start = k * self.stride_frames
            if start >= self.total_frames:
                break
            end = min(start + self.window_frames, self.total_frames)

            # Causal warm-up prefix for k >= 1
            warmup: list[int] = []
            if k >= 1:
                base = self.stride_frames * k
                warmup = [base - 3, base - 2, base - 1]

            windows.append(WindowSlice(
                start=start,
                end=end,
                causal_warmup_prefix_indices=warmup,
            ))

            # If this window already reaches the end, stop
            if end >= self.total_frames:
                break
            k += 1

        return windows


# -----------------------------------------------------------------------
# ReanchorRequest — emitted when a discontinuity is detected
# -----------------------------------------------------------------------


@dataclass
class ReanchorRequest:
    """Request to re-anchor conditioning at a discontinuity.

    Attributes:
        frame_index: Frame index where kappa exceeded the threshold.
        subframe_window: Tuple ``(left, right)`` relative to
            ``frame_index`` defining the re-anchor context window.
        kappa: Discontinuity score at this frame.
    """

    frame_index: int
    subframe_window: tuple[int, int] = (-2, 2)
    kappa: float = 0.0


# -----------------------------------------------------------------------
# DiscontinuityDetector
# -----------------------------------------------------------------------


class DiscontinuityDetector:
    """Per-frame discontinuity index scorer.

    The composite discontinuity index is::

        kappa(t) = 0.4 * norm_motion_jump(t)
                 + 0.4 * normal_cosine_drop(t)
                 + 0.2 * (1 - mask_iou(t))

    When ``kappa(t) > threshold`` (default 0.65), the frame is marked
    as a dynamic anchor requiring re-conditioning.

    Parameters:
        threshold: kappa value above which a frame is flagged.
    """

    def __init__(self, threshold: float = 0.65) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Current discontinuity threshold."""
        return self._threshold

    # -- component scorers -----------------------------------------------

    @staticmethod
    def norm_motion_jump(
        mv_prev: np.ndarray,
        mv_curr: np.ndarray,
    ) -> float:
        """Normalised magnitude of motion-vector change.

        Args:
            mv_prev: ``[H, W, 2]`` motion vectors at frame t-1.
            mv_curr: ``[H, W, 2]`` motion vectors at frame t.

        Returns:
            Scalar in ``[0, 1]`` — 0 means no change, 1 means large
            jump.
        """
        diff = mv_curr.astype(np.float64) - mv_prev.astype(np.float64)
        mag = np.sqrt(np.sum(diff * diff, axis=-1))  # [H, W]
        mean_mag = float(np.mean(mag))
        # Normalise: sigmoid-like saturation at ~20 px jump
        return float(min(mean_mag / 20.0, 1.0))

    @staticmethod
    def normal_cosine_drop(
        normal_prev: np.ndarray,
        normal_curr: np.ndarray,
        fg_mask: np.ndarray,
    ) -> float:
        """One minus mean cosine similarity of normals in foreground.

        Args:
            normal_prev: ``[H, W, 3]`` world normals at frame t-1.
            normal_curr: ``[H, W, 3]`` world normals at frame t.
            fg_mask: ``[H, W]`` boolean foreground mask.

        Returns:
            Scalar in ``[0, 2]`` (clamped to ``[0, 1]`` downstream).
        """
        if not np.any(fg_mask):
            return 0.0
        n0 = normal_prev[fg_mask].astype(np.float64)
        n1 = normal_curr[fg_mask].astype(np.float64)
        # Per-pixel cosine similarity
        dot = np.sum(n0 * n1, axis=-1)
        mag0 = np.sqrt(np.sum(n0 * n0, axis=-1))
        mag1 = np.sqrt(np.sum(n1 * n1, axis=-1))
        denom = mag0 * mag1
        safe = denom > 1e-8
        cos_sim = np.where(safe, dot / denom, 1.0)
        mean_cos = float(np.mean(cos_sim))
        return float(np.clip(1.0 - mean_cos, 0.0, 1.0))

    @staticmethod
    def mask_iou(
        crypto_prev: np.ndarray,
        crypto_curr: np.ndarray,
    ) -> float:
        """Intersection over union of foreground masks.

        Foreground is defined as ``cryptomatte > 0``.

        Args:
            crypto_prev: ``[H, W]`` uint32 cryptomatte at frame t-1.
            crypto_curr: ``[H, W]`` uint32 cryptomatte at frame t.

        Returns:
            IoU in ``[0, 1]``.
        """
        fg_prev = crypto_prev > 0
        fg_curr = crypto_curr > 0
        intersection = np.count_nonzero(fg_prev & fg_curr)
        union = np.count_nonzero(fg_prev | fg_curr)
        if union == 0:
            return 1.0  # both empty — no discontinuity
        return float(int(intersection) / int(union))

    def compute_kappa(
        self,
        mv_prev: np.ndarray,
        mv_curr: np.ndarray,
        normal_prev: np.ndarray,
        normal_curr: np.ndarray,
        crypto_prev: np.ndarray,
        crypto_curr: np.ndarray,
    ) -> float:
        """Compute composite discontinuity index kappa(t).

        Args:
            mv_prev: ``[H, W, 2]`` motion vectors at frame t-1.
            mv_curr: ``[H, W, 2]`` motion vectors at frame t.
            normal_prev: ``[H, W, 3]`` world normals at frame t-1.
            normal_curr: ``[H, W, 3]`` world normals at frame t.
            crypto_prev: ``[H, W]`` uint32 cryptomatte at frame t-1.
            crypto_curr: ``[H, W]`` uint32 cryptomatte at frame t.

        Returns:
            Composite kappa score in ``[0, 1]``.
        """
        fg_mask = crypto_curr > 0

        nmj = self.norm_motion_jump(mv_prev, mv_curr)
        ncd = self.normal_cosine_drop(
            normal_prev, normal_curr, fg_mask,
        )
        iou = self.mask_iou(crypto_prev, crypto_curr)

        kappa = 0.4 * nmj + 0.4 * ncd + 0.2 * (1.0 - iou)
        return float(np.clip(kappa, 0.0, 1.0))

    def detect(
        self,
        fgpp_sequence: Any,
    ) -> list[ReanchorRequest]:
        """Scan an FGPP sequence for discontinuity frames.

        Iterates consecutive frame pairs, computes kappa, and emits
        a ``ReanchorRequest`` for every frame where kappa exceeds the
        threshold.

        Args:
            fgpp_sequence: ``FGPPSequence`` with ``motion_vector``,
                ``world_normal``, and ``cryptomatte_id`` channels.

        Returns:
            List of ``ReanchorRequest`` for flagged frames.
        """
        requests: list[ReanchorRequest] = []
        frames = fgpp_sequence.frames
        n = len(frames)
        if n < 2:
            return requests

        for t in range(1, n):
            prev_f = frames[t - 1]
            curr_f = frames[t]

            mv_prev = _safe_channel(
                prev_f.motion_vector, (prev_f.height, prev_f.width, 2),
            )
            mv_curr = _safe_channel(
                curr_f.motion_vector, (curr_f.height, curr_f.width, 2),
            )
            normal_prev = _safe_channel(
                prev_f.world_normal,
                (prev_f.height, prev_f.width, 3),
            )
            normal_curr = _safe_channel(
                curr_f.world_normal,
                (curr_f.height, curr_f.width, 3),
            )
            crypto_prev = _safe_channel_1d(
                prev_f.cryptomatte_id,
                (prev_f.height, prev_f.width),
                np.uint32,
            )
            crypto_curr = _safe_channel_1d(
                curr_f.cryptomatte_id,
                (curr_f.height, curr_f.width),
                np.uint32,
            )

            kappa = self.compute_kappa(
                mv_prev, mv_curr,
                normal_prev, normal_curr,
                crypto_prev, crypto_curr,
            )

            if kappa > self._threshold:
                requests.append(ReanchorRequest(
                    frame_index=t,
                    subframe_window=(-2, 2),
                    kappa=kappa,
                ))

        return requests


# -----------------------------------------------------------------------
# ScheduledWindow — a window annotated with scheduling metadata
# -----------------------------------------------------------------------


@dataclass
class ScheduledWindow:
    """A generation window with scheduling metadata.

    Attributes:
        window: The underlying ``WindowSlice``.
        window_index: Zero-based window index.
        needs_reanchor: Whether this window contains at least one
            discontinuity requiring re-anchoring.
        reanchor_requests: List of ``ReanchorRequest`` within the
            window bounds.
    """

    window: WindowSlice
    window_index: int = 0
    needs_reanchor: bool = False
    reanchor_requests: list[ReanchorRequest] = field(
        default_factory=list,
    )


# -----------------------------------------------------------------------
# ConditionScheduler — combines window plan and discontinuity detection
# -----------------------------------------------------------------------


class ConditionScheduler:
    """Orchestrator that combines window planning with discontinuity
    detection.

    Args:
        plan: The causal window partition plan.
        detector: Discontinuity detector instance.
    """

    def __init__(
        self,
        plan: CausalWindowPlan,
        detector: DiscontinuityDetector,
    ) -> None:
        self._plan = plan
        self._detector = detector

    @property
    def plan(self) -> CausalWindowPlan:
        """The underlying window plan."""
        return self._plan

    @property
    def detector(self) -> DiscontinuityDetector:
        """The discontinuity detector."""
        return self._detector

    def schedule(
        self,
        fgpp_sequence: Any,
    ) -> list[ScheduledWindow]:
        """Compute scheduled windows with reanchor annotations.

        Args:
            fgpp_sequence: ``FGPPSequence`` to analyse.

        Returns:
            List of ``ScheduledWindow`` instances, one per generation
            window.
        """
        windows = self._plan.compute_windows()
        reanchors = self._detector.detect(fgpp_sequence)

        # Index reanchor requests by frame
        reanchor_by_frame: dict[int, ReanchorRequest] = {
            r.frame_index: r for r in reanchors
        }

        scheduled: list[ScheduledWindow] = []
        for idx, w in enumerate(windows):
            # Collect reanchors that fall within this window
            window_reanchors: list[ReanchorRequest] = []
            for frame_idx in range(w.start, w.end):
                if frame_idx in reanchor_by_frame:
                    window_reanchors.append(
                        reanchor_by_frame[frame_idx],
                    )

            scheduled.append(ScheduledWindow(
                window=w,
                window_index=idx,
                needs_reanchor=len(window_reanchors) > 0,
                reanchor_requests=window_reanchors,
            ))

        return scheduled


# -----------------------------------------------------------------------
# fuse_overlapping_latents — variance-preserving sigmoid annealing
# -----------------------------------------------------------------------


def fuse_overlapping_latents(
    z_prev: np.ndarray,
    z_curr: np.ndarray,
    overlap_steps: int = 9,
    lam: float = 8.0,
) -> np.ndarray:
    """Fuse overlapping latent tensors with variance-preserving sigmoid
    causal annealing.

    The blending weight at overlap step tau' is::

        alpha(0) = 0.0          (keep z_prev entirely)
        alpha(tau') = sigmoid(lam * (tau' / (overlap_steps - 1) - 0.5))
            for tau' = 1 .. overlap_steps - 1

    The fused output preserves unit variance::

        z_fused = ((1 - alpha) * z_prev + alpha * z_curr)
                / sqrt((1 - alpha)^2 + alpha^2)

    Args:
        z_prev: Latent tensor from the previous window; first
            dimension is the overlap time axis with size
            ``>= overlap_steps``.
        z_curr: Latent tensor from the current window; first dimension
            must be ``>= overlap_steps``.
        overlap_steps: Number of overlapping latent steps (default 9).
        lam: Steepness of the sigmoid transition (default 8.0).

    Returns:
        Fused latent tensor with the same shape as
        ``z_prev[:overlap_steps]``.

    Raises:
        ValueError: If either tensor has fewer steps than
            ``overlap_steps``.
    """
    if z_prev.shape[0] < overlap_steps:
        raise ValueError(
            f"z_prev has {z_prev.shape[0]} steps, "
            f"need >= {overlap_steps}"
        )
    if z_curr.shape[0] < overlap_steps:
        raise ValueError(
            f"z_curr has {z_curr.shape[0]} steps, "
            f"need >= {overlap_steps}"
        )

    zp = z_prev[:overlap_steps].astype(np.float64)
    zc = z_curr[:overlap_steps].astype(np.float64)

    # Build alpha schedule
    alphas = np.zeros(overlap_steps, dtype=np.float64)
    for tau in range(1, overlap_steps):
        x = lam * (tau / (overlap_steps - 1) - 0.5)
        alphas[tau] = 1.0 / (1.0 + math.exp(-x))

    # Reshape for broadcasting: [overlap_steps, 1, 1, ...]
    extra_dims = len(zp.shape) - 1
    shape = (overlap_steps,) + (1,) * extra_dims
    a = alphas.reshape(shape)

    blended = (1.0 - a) * zp + a * zc
    normaliser = np.sqrt((1.0 - a) ** 2 + a ** 2)
    # Avoid division by zero (should not happen with valid alpha)
    normaliser = np.maximum(normaliser, 1e-12)
    fused = blended / normaliser

    return fused.astype(z_prev.dtype)


# -----------------------------------------------------------------------
# compute_latent_flow_operator — motion-vector to latent displacement
# -----------------------------------------------------------------------


def compute_latent_flow_operator(
    motion_vectors: np.ndarray,
    depths: np.ndarray,
    temporal_compression: int = 4,
    spatial_block: int = 8,
    discontinuity_frames: list[int] | None = None,
) -> np.ndarray:
    """Compute latent-space displacement grid from pixel motion vectors.

    Pipeline:

    1. Validate that ``(T - 1) % temporal_compression == 0``.
    2. 4-frame forward Euler trajectory integration with occlusion mask
       (residual magnitude threshold).
    3. At discontinuity frames, reset the integrator.
    4. For each ``spatial_block x spatial_block`` spatial block, select
       the pixel with ``argmin(depth)`` and take its integrated
       displacement.
    5. Output: ``[tau, H // spatial_block, W // spatial_block, 2]``
       latent-space displacement grid where
       ``tau = (T - 1) // temporal_compression + 1``.

    Args:
        motion_vectors: ``[T, H, W, 2]`` float pixel-space MV.
        depths: ``[T, H, W]`` float depth.
        temporal_compression: Temporal compression factor (default 4).
        spatial_block: Spatial block size for pooling (default 8).
        discontinuity_frames: Optional list of frame indices at which
            to reset the integrator.

    Returns:
        ``[tau, Hb, Wb, 2]`` float32 latent displacement grid.

    Raises:
        ValueError: If ``(T - 1) % temporal_compression != 0``.
    """
    T, H, W, _ = motion_vectors.shape
    if (T - 1) % temporal_compression != 0:
        raise ValueError(
            f"(T-1) must be divisible by temporal_compression: "
            f"({T}-1) % {temporal_compression} = "
            f"{(T - 1) % temporal_compression} != 0"
        )

    disc_set = set(discontinuity_frames or [])
    occlusion_threshold = 5.0  # pixels — residual threshold

    tau = (T - 1) // temporal_compression + 1
    Hb = H // spatial_block
    Wb = W // spatial_block

    mv = motion_vectors.astype(np.float64)
    dep = depths.astype(np.float64)

    # Accumulated displacement per pixel: [H, W, 2]
    accum = np.zeros((H, W, 2), dtype=np.float64)

    output = np.zeros((tau, Hb, Wb, 2), dtype=np.float64)

    # Snapshot the accumulated displacement at the first frame
    output[0] = _pool_displacement(accum, dep[0], spatial_block, Hb, Wb)

    latent_idx = 1
    step_in_group = 0

    for t in range(1, T):
        # Reset at discontinuity frames
        if t in disc_set:
            accum = np.zeros((H, W, 2), dtype=np.float64)

        # Forward Euler step: accumulate motion vector
        step_mv = mv[t]  # [H, W, 2]

        # Occlusion mask: large residual means occluded, skip
        mag = np.sqrt(
            step_mv[..., 0] ** 2 + step_mv[..., 1] ** 2,
        )
        valid = mag < occlusion_threshold
        accum[valid] += step_mv[valid]

        step_in_group += 1

        if step_in_group == temporal_compression:
            # Emit a latent-space displacement sample
            output[latent_idx] = _pool_displacement(
                accum, dep[t], spatial_block, Hb, Wb,
            )
            latent_idx += 1
            step_in_group = 0

    return output.astype(np.float32)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _pool_displacement(
    accum: np.ndarray,
    depth_frame: np.ndarray,
    block: int,
    Hb: int,
    Wb: int,
) -> np.ndarray:
    """Pool pixel displacement via argmin-depth within spatial blocks.

    Args:
        accum: ``[H, W, 2]`` accumulated displacement.
        depth_frame: ``[H, W]`` depth for the current frame.
        block: Spatial block size.
        Hb: Number of vertical blocks.
        Wb: Number of horizontal blocks.

    Returns:
        ``[Hb, Wb, 2]`` pooled displacement.
    """
    result = np.zeros((Hb, Wb, 2), dtype=np.float64)
    for by in range(Hb):
        for bx in range(Wb):
            y0 = by * block
            y1 = y0 + block
            x0 = bx * block
            x1 = x0 + block

            depth_block = depth_frame[y0:y1, x0:x1]  # [block, block]
            flat_idx = int(np.argmin(depth_block))
            dy = flat_idx // block
            dx = flat_idx % block

            result[by, bx] = accum[y0 + dy, x0 + dx]

    return result


def _safe_channel(
    arr: np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return *arr* if not None, else zeros of *shape* float32."""
    if arr is not None:
        return arr.astype(np.float32)
    return np.zeros(shape, dtype=np.float32)


def _safe_channel_1d(
    arr: np.ndarray | None,
    shape: tuple[int, ...],
    dtype: Any = np.uint32,
) -> np.ndarray:
    """Return *arr* if not None, else zeros of *shape* with *dtype*."""
    if arr is not None:
        return np.asarray(arr, dtype=dtype)
    return np.zeros(shape, dtype=dtype)
