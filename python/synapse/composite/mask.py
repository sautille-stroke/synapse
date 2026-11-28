"""Motion-adaptive surgical compositing masks.

Implements the surgical mask equation from the compositing pipeline:

    M_surgical(t) = Dilate_v(t)( M_3D_new(t) ∪ M_SAM2_old(t) )

The mask is dilated isotropically and then directionally along the
dominant motion vector, producing a soft alpha with cosine feathering
for seamless blending.

All operations are vectorised NumPy.  ``scipy.ndimage`` is used for
morphological dilation when available, with a pure-NumPy fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synapse.render_core.morphology import (
    binary_dilation as _binary_dilation,
)
from synapse.render_core.morphology import (
    circular_structuring_element as _circular_structuring_element,
)
from synapse.render_core.morphology import (
    exact_euclidean_distance_transform,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _directional_structuring_element(
    angle_rad: float,
    length: int,
) -> np.ndarray:
    """Create a forward-biased line structuring element.

    The line extends from the centre primarily in the *angle_rad*
    direction, giving asymmetric (motion-forward) dilation.  A small
    backward extent (¼ of total length) is kept for seamless overlap.

    Returns a bool array large enough to contain the line, centred.
    """
    if length < 1:
        return np.ones((1, 1), dtype=bool)

    # Forward-biased: extend mostly in the positive direction.
    back = max(length // 4, 1)
    forward = length - back
    half = max(forward, back)
    size = 2 * half + 1
    se = np.zeros((size, size), dtype=bool)
    cx = cy = half

    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    for i in range(-back, forward + 1):
        x = int(round(cx + i * dx))
        y = int(round(cy + i * dy))
        if 0 <= x < size and 0 <= y < size:
            se[y, x] = True

    return se


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_surgical_mask(
    mask_3d_new: np.ndarray,
    mask_sam2_old: np.ndarray | None = None,
    motion_vectors: np.ndarray | None = None,
    r_base: int = 8,
    motion_scale: float = 1.5,
    max_motion_kernel: int = 32,
    feather_width: float = 6.0,
    mask_sam2_vos: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a motion-adaptive surgical compositing mask.

    Implements::

        M_surgical(t) = Dilate_v(t)( M_3D_new(t) ∪ M_SAM2_old(t) ∪ M_SAM2_VOS(t) )

    Args:
        mask_3d_new: ``[H, W]`` bool array — new 3D-rendered entity
            mask.
        mask_sam2_old: ``[H, W]`` bool array or ``None`` — SAM2-
            predicted mask of the old entity to be replaced.
        motion_vectors: ``[H, W, 2]`` float16/32 array or ``None``
            — screen-space motion vectors ``(dx, dy)`` per pixel.
        r_base: Isotropic dilation radius in pixels (default 8).
        motion_scale: Motion-vector-to-dilation scaling factor.
        max_motion_kernel: Maximum directional kernel length in
            pixels.
        feather_width: Cosine-falloff feather width in pixels.
        mask_sam2_vos: ``[H, W]`` bool array or ``None`` — SAM 2.1
            temporally propagated VOS mask fused with Cryptomatte.

    Returns:
        ``[H, W]`` float32 alpha mask in ``[0.0, 1.0]``.
    """
    # 1. Union of masks (preserve raw foreground union for undiluted motion estimation).
    raw_combined = mask_3d_new.astype(bool).copy()
    if mask_sam2_old is not None:
        raw_combined |= mask_sam2_old.astype(bool)
    if mask_sam2_vos is not None:
        raw_combined |= mask_sam2_vos.astype(bool)
    combined = raw_combined.copy()

    # 2. Isotropic dilation with circular structuring element.
    if r_base > 0:
        se_circle = _circular_structuring_element(r_base)
        combined = _binary_dilation(combined, se_circle)

    # 3. Directional (motion-adaptive) dilation using undiluted foreground motion.
    if motion_vectors is not None and np.any(raw_combined):
        mv = motion_vectors.astype(np.float32)
        # Replace NaN with 0 for safe mean computation.
        mv_safe = np.where(np.isfinite(mv), mv, 0.0)

        # Mean motion within the original pre-dilation foreground mask region.
        count = float(raw_combined.sum())
        if count > 0:
            mean_dx = float((mv_safe[..., 0] * raw_combined).sum() / count)
            mean_dy = float((mv_safe[..., 1] * raw_combined).sum() / count)
            mag = math.sqrt(mean_dx * mean_dx + mean_dy * mean_dy)

            kernel_len = int(
                min(
                    max_motion_kernel,
                    round(mag * motion_scale),
                )
            )

            if kernel_len >= 2:
                angle = math.atan2(mean_dy, mean_dx)
                se_line = _directional_structuring_element(angle, kernel_len)
                combined = _binary_dilation(combined, se_line)

    # 4. Soft alpha with cosine feathering.
    if feather_width <= 0 or not np.any(combined):
        return np.asarray(combined.astype(np.float32))

    dist = exact_euclidean_distance_transform(combined)
    alpha = np.ones_like(dist)

    # Feather zone: outside the hard mask but within feather_width.
    feather_mask = (~combined) & (dist <= feather_width)
    alpha[feather_mask] = 0.5 * (1.0 + np.cos(np.pi * dist[feather_mask] / feather_width))
    alpha[~combined & ~feather_mask] = 0.0
    alpha[combined] = 1.0

    return alpha.astype(np.float32)


def apply_surgical_mask(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Alpha-composite *foreground* over *background* using *alpha*.

    Args:
        background: ``[H, W, C]`` image array.
        foreground: ``[H, W, C]`` image array (same shape & dtype as
            *background*).
        alpha: ``[H, W]`` float32 blending mask in ``[0, 1]``.

    Returns:
        Composited ``[H, W, C]`` image with the same dtype as
        *background*.
    """
    src_dtype = background.dtype
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)
    a = alpha[..., np.newaxis].astype(np.float32)

    out = fg * a + bg * (1.0 - a)

    if np.issubdtype(src_dtype, np.integer):
        out = np.clip(out, 0, np.iinfo(src_dtype).max)
    return out.astype(src_dtype)


def generate_surgical_mask_sequence(
    frames_srgb_u8: np.ndarray,
    masks_3d_new: np.ndarray,
    *,
    masks_sam2_old: np.ndarray | None = None,
    cryptomatte_ids: np.ndarray | None = None,
    motion_vectors: np.ndarray | None = None,
    sam2_transport: Any | None = None,
    prompts_by_segment: dict[int, list[Any]] | None = None,
    cut_frames: list[int] | None = None,
    target_instance_ids: set[int] | None = None,
    dilation_px: int = 4,
    r_base: int = 8,
    motion_scale: float = 1.5,
    max_motion_kernel: int = 32,
    feather_width: float = 6.0,
) -> tuple[np.ndarray, int]:
    """Generate surgical alpha masks over a sequence with SAM 2.1 VOS + cut resets.

    Splits ``frames_srgb_u8`` at sorted ``cut_frames`` so that
    ``sam2_transport.segment_video`` restarts cleanly at every scene cut, fuses
    per-frame VOS masks with ``cryptomatte_ids`` via ``fuse_sam2_cryptomatte``,
    and computes :func:`generate_surgical_mask` per frame.

    Args:
        frames_srgb_u8: ``[T, H, W, 3]`` uint8 sRGB input video.
        masks_3d_new: ``[T, H, W]`` bool rendered entity masks.
        masks_sam2_old: Optional ``[T, H, W]`` bool legacy SAM2 masks.
        cryptomatte_ids: Optional ``[T, H, W]`` uint32 Cryptomatte IDs.
        motion_vectors: Optional ``[T, H, W, 2]`` screen-space motion vectors.
        sam2_transport: Optional :class:`~synapse.bridge.sam2_transport.SAM2Transport`.
        prompts_by_segment: Optional prompts dict keyed by segment start frame.
        cut_frames: Optional frame indices marking shot cuts where SAM2 VOS
            tracking must restart cleanly.
        target_instance_ids: Optional entity/instance filter passed to Cryptomatte.
        dilation_px: Fusion dilation radius passed to ``fuse_sam2_cryptomatte``.
        r_base: Isotropic surgical dilation radius in pixels.
        motion_scale: Directional motion scaling factor.
        max_motion_kernel: Max directional dilation kernel length.
        feather_width: Cosine-falloff edge feathering width.

    Returns:
        Tuple ``(alpha_sequence, segment_count)`` where ``alpha_sequence`` is
        ``[T, H, W]`` float32 in ``[0.0, 1.0]`` and ``segment_count`` records
        how many shot segments SAM2 ran on (0 if ``sam2_transport`` is ``None``).
    """
    frames_arr = np.asarray(frames_srgb_u8)
    m3d = np.asarray(masks_3d_new, dtype=bool)
    if m3d.ndim != 3:
        raise ValueError(f"Expected 3-D masks_3d_new [T, H, W], got shape {m3d.shape}.")
    t_len, h, w = m3d.shape
    if t_len == 0:
        return np.empty((0, h, w), dtype=np.float32), 0

    valid_cuts = sorted({c for c in (cut_frames or []) if 0 < c < t_len})
    boundaries = [0, *valid_cuts, t_len]

    vos_fused = np.zeros((t_len, h, w), dtype=bool)
    segment_count = 0

    if sam2_transport is not None:
        from synapse.bridge.sam2_transport import fuse_sam2_cryptomatte

        for seg_idx in range(len(boundaries) - 1):
            seg_start = boundaries[seg_idx]
            seg_end = boundaries[seg_idx + 1]
            seg_video = frames_arr[seg_start:seg_end]
            seg_prompts = (
                prompts_by_segment.get(seg_start)
                if prompts_by_segment is not None
                else None
            )
            seg_result = sam2_transport.segment_video(seg_video, prompts=seg_prompts)
            segment_count += 1
            for local_t, f_res in enumerate(seg_result.frames):
                global_t = seg_start + local_t
                if global_t >= seg_end:
                    break
                sam2_combined = f_res.combined_mask
                if cryptomatte_ids is not None:
                    vos_fused[global_t] = fuse_sam2_cryptomatte(
                        sam2_combined,
                        cryptomatte_ids[global_t],
                        target_instance_ids=(
                            sorted(target_instance_ids)
                            if target_instance_ids is not None
                            else None
                        ),
                        dilation_px=dilation_px,
                    )
                else:
                    vos_fused[global_t] = sam2_combined
    elif cryptomatte_ids is not None:
        if target_instance_ids:
            vos_fused = np.isin(cryptomatte_ids, list(target_instance_ids))
        else:
            vos_fused = cryptomatte_ids > 0

    alphas = np.empty((t_len, h, w), dtype=np.float32)
    for t in range(t_len):
        m_old = masks_sam2_old[t] if masks_sam2_old is not None else None
        mv = motion_vectors[t] if motion_vectors is not None else None
        m_vos = vos_fused[t] if (sam2_transport is not None or cryptomatte_ids is not None) else None
        alphas[t] = generate_surgical_mask(
            m3d[t],
            mask_sam2_old=m_old,
            motion_vectors=mv,
            r_base=r_base,
            motion_scale=motion_scale,
            max_motion_kernel=max_motion_kernel,
            feather_width=feather_width,
            mask_sam2_vos=m_vos,
        )
    return alphas, segment_count
