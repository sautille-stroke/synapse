# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""ScalpelStitcher — ACEScg 16 px boundary-aware sequence stitching.

Implements the RFC 0061 scalpel stitcher that composites locally refined
anatomical patches back into the full-frame video sequence with seamless
16 px boundary blending via:

  1. Inverse affine transform (``M⁻¹``) to map the refined crop back.
  2. Surgical mask generation for the refined region.
  3. Border-ring LAB colour matching at the stitch boundary.
  4. Optical increments overlay (if FGPP is provided).

All operations are pure NumPy on float32 ACEScg tensors.  ``cv2`` is
lazily imported for ``warpAffine`` when an explicit inverse affine matrix
is provided; otherwise a pure-NumPy bilinear sampler is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synapse.composite.blend import (
    border_ring_lab_match,
    laplacian_pyramid_blend,
)
from synapse.composite.mask import generate_surgical_mask
from synapse.refine.types import RefinementResult
from synapse.style.aces_color import (
    acescg_to_linear_srgb,
    linear_to_srgb,
    rgb_to_cielab,
    srgb_to_linear,
)


@dataclass
class StitchConfig:
    """Configuration for scalpel stitching.

    Attributes:
        border_ring_px: Width of the border blending ring (default 16).
        ema_alpha: Temporal EMA blending factor (default 0.85).
        max_flow_magnitude: Flow magnitude above which EMA alpha is
            reduced (default 15.0 px).
    """

    border_ring_px: int = 16
    ema_alpha: float = 0.85
    max_flow_magnitude: float = 15.0


@dataclass
class StitchReport:
    """Diagnostic report from a stitching pass.

    Attributes:
        num_frames: Number of frames processed.
        num_patches_applied: Total patches applied across all frames.
        mean_border_psnr_db: Mean PSNR at border ring (dB).
        mean_border_ciede2000: Mean CIE ΔE₀₀ at border ring.
    """

    num_frames: int = 0
    num_patches_applied: int = 0
    mean_border_psnr_db: float = 0.0
    mean_border_ciede2000: float = 0.0


class ScalpelStitcher:
    """Stitch locally refined patches back into the source video.

    Operates frame-by-frame, applying each
    :class:`~synapse.refine.types.RefinementResult` back into the
    original frame with boundary-aware blending.

    The stitching pipeline for each patch:

    1. **Inverse affine warp** — map the refined crop back to full-frame
       coordinates using the explicit ``inverse_affine_2x3`` matrix when
       provided, or a scale/translate computed from ``bbox_xyxy``.
    2. **Surgical mask** — ``generate_surgical_mask`` from
       ``synapse.composite.mask`` produces a motion-adaptive feathered
       alpha mask.
    3. **Border-ring LAB colour match** — ``border_ring_lab_match`` from
       ``synapse.composite.blend`` matches the patch boundary colours to
       the background in CIE LAB.
    4. **Laplacian pyramid blend** — ``laplacian_pyramid_blend`` fuses
       the colour-matched patch into the background using multi-scale
       blending.
    5. **Temporal EMA warp** — when ``motion_vectors`` are supplied, the
       previous frame's composited result is backward-warped by the flow
       field and EMA-smoothed with the current composite to suppress
       temporal boundary flicker.

    Args:
        config: Stitching configuration.
    """

    def __init__(self, config: StitchConfig | None = None) -> None:
        self.config = config or StitchConfig()

    def stitch_sequence(
        self,
        frames_acescg: np.ndarray,
        refinements: list[list[RefinementResult]],
        motion_vectors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, StitchReport]:
        """Stitch refined patches into the source sequence.

        Args:
            frames_acescg: ``(T, H, W, 3)`` float32 ACEScg source.
            refinements: Per-frame list of
                :class:`RefinementResult` objects.  ``refinements[t]``
                contains all patches for frame *t*.
            motion_vectors: ``(T, H, W, 2)`` optional flow field for
                temporal warp-aware blending.

        Returns:
            Tuple of stitched ``(T, H, W, 3)`` float32 ACEScg frames
            and a :class:`StitchReport`.
        """
        t_count, h, w, c = frames_acescg.shape
        result = frames_acescg.copy()

        total_patches = 0
        psnr_accum = 0.0
        ciede_accum = 0.0
        measured = 0

        ring = self.config.border_ring_px
        prev_composite: np.ndarray | None = None

        for t in range(t_count):
            patches = refinements[t] if t < len(refinements) else []

            for patch in patches:
                # -------------------------------------------------------
                # 1. Inverse affine warp — place the refined crop into a
                #    full-frame canvas at the correct position.
                # -------------------------------------------------------
                fg_full, fg_mask_bool = _inverse_affine_paste(
                    patch, h, w,
                )

                # -------------------------------------------------------
                # 2. Surgical mask — motion-adaptive feathered alpha.
                # -------------------------------------------------------
                mv_frame = (
                    motion_vectors[t] if motion_vectors is not None else None
                )
                surgical_alpha = generate_surgical_mask(
                    mask_3d_new=fg_mask_bool,
                    mask_sam2_old=None,
                    motion_vectors=mv_frame,
                    r_base=ring // 2,
                    feather_width=float(ring),
                )

                # -------------------------------------------------------
                # 3. Border-ring LAB colour match on sRGB proxies.
                # -------------------------------------------------------
                fg_srgb = _acescg_to_srgb_f32(fg_full)
                bg_srgb = _acescg_to_srgb_f32(result[t])
                matched_srgb = border_ring_lab_match(
                    fg_srgb, bg_srgb, fg_mask_bool,
                    ring_width_px=ring,
                )

                # -------------------------------------------------------
                # 4. Laplacian pyramid blend in sRGB then convert back.
                # -------------------------------------------------------
                blended_srgb = laplacian_pyramid_blend(
                    matched_srgb, bg_srgb, surgical_alpha,
                    num_levels=4,
                )
                result[t] = _srgb_to_acescg_f32(blended_srgb)

                # -------------------------------------------------------
                # Measure border quality with real CIE ΔE₀₀.
                # -------------------------------------------------------
                border_mask = _border_ring_mask(h, w, ring, fg_mask_bool)
                if np.any(border_mask):
                    psnr_val = _border_psnr(
                        result[t], frames_acescg[t], border_mask,
                    )
                    ciede_val = _border_ciede2000(
                        result[t], frames_acescg[t], border_mask,
                    )
                    psnr_accum += psnr_val
                    ciede_accum += ciede_val
                    measured += 1

                total_patches += 1

            # -----------------------------------------------------------
            # 5. Temporal EMA warp — smooth with previous composite.
            # -----------------------------------------------------------
            if (
                t > 0
                and prev_composite is not None
                and motion_vectors is not None
            ):
                mv_t = motion_vectors[t]
                result[t] = _temporal_ema_blend(
                    current=result[t],
                    prev_composite=prev_composite,
                    flow=mv_t,
                    ema_alpha=self.config.ema_alpha,
                    max_flow_magnitude=self.config.max_flow_magnitude,
                )

            prev_composite = result[t].copy()

        report = StitchReport(
            num_frames=t_count,
            num_patches_applied=total_patches,
            mean_border_psnr_db=(
                psnr_accum / measured if measured > 0 else 0.0
            ),
            mean_border_ciede2000=(
                ciede_accum / measured if measured > 0 else 0.0
            ),
        )

        return result, report


# ---------------------------------------------------------------------------
# Internal helpers — inverse affine paste
# ---------------------------------------------------------------------------


def _compute_inverse_affine_from_bbox(
    bbox_xyxy: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> np.ndarray:
    """Compute a 2×3 inverse affine matrix from ``bbox_xyxy``.

    Maps from **patch** coordinates ``(patch_w, patch_h)`` back to
    full-frame coordinates defined by the bounding box.

    Returns:
        ``(2, 3)`` float64 affine matrix suitable for ``warpAffine``
        with ``WARP_INVERSE_MAP`` (i.e. it maps *dst → src* when used
        as the forward map from patch to frame).
    """
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float64)
    bw = x2 - x1
    bh = y2 - y1
    sx = bw / max(float(patch_w), 1.0)
    sy = bh / max(float(patch_h), 1.0)
    # [sx  0  tx]
    # [0  sy  ty]
    return np.array([
        [sx, 0.0, x1],
        [0.0, sy, y1],
    ], dtype=np.float64)


def _inverse_affine_paste(
    patch: RefinementResult,
    frame_h: int,
    frame_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a refined patch into full-frame coordinates via inverse
    affine sampling.

    Returns:
        ``(fg_full, fg_mask_bool)`` where ``fg_full`` is
        ``(H, W, 3)`` float32 ACEScg and ``fg_mask_bool`` is
        ``(H, W)`` bool indicating covered pixels.
    """
    ph, pw = patch.refined_acescg.shape[:2]
    refined = patch.refined_acescg.astype(np.float32)

    # Determine the affine: patch coords → frame coords.
    if patch.inverse_affine_2x3 is not None:
        M = patch.inverse_affine_2x3.astype(np.float64)
    else:
        M = _compute_inverse_affine_from_bbox(patch.bbox_xyxy, ph, pw)

    # Try cv2.warpAffine for sub-pixel accuracy; fall back to pure
    # NumPy bilinear sampling when cv2 is unavailable.
    fg_full, fg_mask = _warp_affine_bilinear(
        refined, M, frame_h, frame_w,
    )
    return fg_full, fg_mask


def _warp_affine_bilinear(
    src: np.ndarray,
    M_patch_to_frame: np.ndarray,
    dst_h: int,
    dst_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp *src* from patch space into frame space via bilinear
    backward sampling.

    Attempts ``cv2.warpAffine`` first, falling back to pure-NumPy.

    Args:
        src: ``(ph, pw, 3)`` float32 source patch.
        M_patch_to_frame: ``(2, 3)`` affine mapping patch → frame.
        dst_h, dst_w: Output frame dimensions.

    Returns:
        ``(dst_h, dst_w, 3)`` float32 warped image and
        ``(dst_h, dst_w)`` bool mask of covered pixels.
    """
    try:
        import cv2

        # cv2.warpAffine with WARP_INVERSE_MAP expects a matrix that
        # maps *dst* → *src*.  We have *src* (patch) → *dst* (frame),
        # so we need the inverse.
        M_2x3 = M_patch_to_frame[:2, :]
        # Build 3x3 and invert.
        M_3x3 = np.eye(3, dtype=np.float64)
        M_3x3[:2, :] = M_2x3
        M_inv = np.linalg.inv(M_3x3)[:2, :]

        dst = cv2.warpAffine(
            src, M_inv.astype(np.float32), (dst_w, dst_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0),
        )
        # Build coverage mask: warp a ones image the same way.
        ones_src = np.ones(src.shape[:2], dtype=np.float32)
        mask_f = cv2.warpAffine(
            ones_src, M_inv.astype(np.float32), (dst_w, dst_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return dst.astype(np.float32), mask_f > 0.5
    except ImportError:
        pass

    # Pure-NumPy backward-sampling fallback.
    return _warp_affine_numpy(src, M_patch_to_frame, dst_h, dst_w)


def _warp_affine_numpy(
    src: np.ndarray,
    M_patch_to_frame: np.ndarray,
    dst_h: int,
    dst_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy bilinear backward warp."""
    ph, pw, c = src.shape
    M_3x3 = np.eye(3, dtype=np.float64)
    M_3x3[:2, :] = M_patch_to_frame[:2, :]
    M_inv = np.linalg.inv(M_3x3)

    # Build destination grid.
    yy, xx = np.mgrid[:dst_h, :dst_w]
    dst_coords = np.stack(
        [xx.ravel().astype(np.float64),
         yy.ravel().astype(np.float64),
         np.ones(dst_h * dst_w, dtype=np.float64)],
        axis=0,
    )  # (3, N)
    src_coords = M_inv @ dst_coords  # (3, N)
    sx = src_coords[0].reshape(dst_h, dst_w)
    sy = src_coords[1].reshape(dst_h, dst_w)

    # Bilinear sampling from src.
    valid = (sx >= 0) & (sx <= pw - 1) & (sy >= 0) & (sy <= ph - 1)
    sx_c = np.clip(sx, 0, pw - 1)
    sy_c = np.clip(sy, 0, ph - 1)

    x0 = np.floor(sx_c).astype(np.intp)
    y0 = np.floor(sy_c).astype(np.intp)
    x1 = np.minimum(x0 + 1, pw - 1)
    y1 = np.minimum(y0 + 1, ph - 1)
    xf = (sx_c - x0).astype(np.float32)
    yf = (sy_c - y0).astype(np.float32)

    out = np.zeros((dst_h, dst_w, c), dtype=np.float32)
    for ch in range(c):
        tl = src[y0, x0, ch]
        tr = src[y0, x1, ch]
        bl = src[y1, x0, ch]
        br = src[y1, x1, ch]
        top = tl * (1.0 - xf) + tr * xf
        bot = bl * (1.0 - xf) + br * xf
        out[..., ch] = top * (1.0 - yf) + bot * yf

    out[~valid] = 0.0
    return out, valid


# ---------------------------------------------------------------------------
# Internal helpers — colour space conversion utilities
# ---------------------------------------------------------------------------


def _acescg_to_srgb_f32(acescg: np.ndarray) -> np.ndarray:
    """Convert ACEScg float32 → sRGB float32 in [0, 1]."""
    linear_srgb = acescg_to_linear_srgb(
        acescg.astype(np.float64),
    )
    srgb = linear_to_srgb(np.clip(linear_srgb, 0.0, None))
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def _srgb_to_acescg_f32(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB float32 [0, 1] → ACEScg float32."""
    from synapse.style.aces_color import linear_srgb_to_acescg

    linear = srgb_to_linear(np.clip(srgb.astype(np.float64), 0.0, 1.0))
    acescg = linear_srgb_to_acescg(linear)
    return acescg.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers — border metrics
# ---------------------------------------------------------------------------


def _border_ring_mask(
    h: int,
    w: int,
    ring: int,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Bool mask for the border ring around the foreground mask.

    Returns pixels that are within ``ring`` pixels of the foreground
    boundary but outside the foreground interior.
    """
    from synapse.composite.blend import (
        _binary_dilation,
        _binary_erosion,
        _circular_structuring_element,
    )

    se = _circular_structuring_element(ring)
    dilated = _binary_dilation(fg_mask, se)
    eroded = _binary_erosion(fg_mask, se)
    return dilated & (~eroded)


def _border_psnr(
    result_frame: np.ndarray,
    source_frame: np.ndarray,
    border_mask: np.ndarray,
) -> float:
    """Compute PSNR at the border ring between result and source."""
    diff = (
        result_frame.astype(np.float64)
        - source_frame.astype(np.float64)
    )
    mse = float(np.mean(diff[border_mask] ** 2))
    if mse <= 1e-10:
        return 100.0
    return float(min(10.0 * np.log10(1.0 / mse), 100.0))


def _border_ciede2000(
    result_frame: np.ndarray,
    source_frame: np.ndarray,
    border_mask: np.ndarray,
) -> float:
    """Compute mean CIE ΔE₀₀ at the border ring.

    Converts both frames ACEScg → linear sRGB → CIE LAB, then uses
    ``ciede2000_delta_e`` from ``synapse.style.aces_color``.
    """
    from synapse.style.aces_color import ciede2000_delta_e

    result_linear = acescg_to_linear_srgb(
        result_frame.astype(np.float64),
    )
    source_linear = acescg_to_linear_srgb(
        source_frame.astype(np.float64),
    )
    lab_result = rgb_to_cielab(np.clip(result_linear, 0.0, None))
    lab_source = rgb_to_cielab(np.clip(source_linear, 0.0, None))

    de = ciede2000_delta_e(lab_result, lab_source)
    return float(np.mean(de[border_mask]))


# ---------------------------------------------------------------------------
# Internal helpers — temporal EMA with backward warp
# ---------------------------------------------------------------------------


def _backward_warp_bilinear(
    img: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    """Backward-warp *img* by *flow* using bilinear interpolation.

    For each pixel ``(x, y)`` in the output, sample from
    ``img[y + flow_y, x + flow_x]``.

    Args:
        img: ``(H, W, C)`` float32/float64 image.
        flow: ``(H, W, 2)`` float32 flow field ``(dx, dy)``.

    Returns:
        Warped ``(H, W, C)`` image with the same dtype.
    """
    h, w, c = img.shape
    src = img.astype(np.float64)
    fl = flow.astype(np.float64)

    yy, xx = np.mgrid[:h, :w]
    sx = xx.astype(np.float64) + fl[..., 0]
    sy = yy.astype(np.float64) + fl[..., 1]

    # Clamp to valid range.
    sx_c = np.clip(sx, 0, w - 1)
    sy_c = np.clip(sy, 0, h - 1)

    x0 = np.floor(sx_c).astype(np.intp)
    y0 = np.floor(sy_c).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    xf = (sx_c - x0).astype(np.float64)
    yf = (sy_c - y0).astype(np.float64)

    out = np.empty_like(src)
    for ch in range(c):
        tl = src[y0, x0, ch]
        tr = src[y0, x1, ch]
        bl = src[y1, x0, ch]
        br = src[y1, x1, ch]
        top = tl * (1.0 - xf) + tr * xf
        bot = bl * (1.0 - xf) + br * xf
        out[..., ch] = top * (1.0 - yf) + bot * yf

    return out.astype(img.dtype)


def _temporal_ema_blend(
    current: np.ndarray,
    prev_composite: np.ndarray,
    flow: np.ndarray,
    ema_alpha: float,
    max_flow_magnitude: float,
) -> np.ndarray:
    """Blend *current* with motion-warped *prev_composite* via EMA.

    When the mean flow magnitude exceeds ``max_flow_magnitude``, the
    EMA alpha is adaptively reduced to prevent ghosting.

    Args:
        current: ``(H, W, 3)`` float32 current composite.
        prev_composite: ``(H, W, 3)`` float32 previous composite.
        flow: ``(H, W, 2)`` float32 motion vectors.
        ema_alpha: Base EMA alpha (weight on current frame).
        max_flow_magnitude: Threshold for adaptive alpha reduction.

    Returns:
        Blended ``(H, W, 3)`` float32 frame.
    """
    # Warp previous composite by flow field.
    warped_prev = _backward_warp_bilinear(prev_composite, flow)

    # Compute effective alpha — decay when flow is large.
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_mag = float(np.mean(flow_mag))
    effective_alpha = ema_alpha
    if mean_mag > max_flow_magnitude:
        effective_alpha = max(
            0.30,
            ema_alpha * (max_flow_magnitude / max(mean_mag, 1e-6)),
        )

    # EMA blend.
    blended = (
        effective_alpha * current.astype(np.float64)
        + (1.0 - effective_alpha) * warped_prev.astype(np.float64)
    )
    return blended.astype(np.float32)
