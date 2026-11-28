"""Composite blending — border-ring LAB matching, Laplacian pyramid
blend, surgical sequence compositing, and optical increments overlay.

Implements four compositing stages for seamless 3D re-rendering
integration:

1. **Border-ring LAB match** — Affine colour transfer at mask
   boundaries using CIE LAB statistics.
2. **Laplacian pyramid blend** — Multi-scale alpha blending for
   artefact-free compositing without hard seams.
3. **Surgical sequence** — Per-frame compositing pipeline with
   temporal EMA smoothing to suppress boundary flicker.
4. **Optical increments** — Physics-based shadow and caustics
   overlay in ACEScg colour space with guided-filter alignment.

All operations are vectorised NumPy; ``scipy.ndimage`` is used
for Gaussian blur when available, with a pure-NumPy fallback.
"""

from __future__ import annotations

import numpy as np

from synapse.render_core.morphology import (
    binary_dilation,
    binary_erosion,
    circular_structuring_element,
)
from synapse.render_core.pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    downsample2x,
    gaussian_kernel_1d,
    reconstruct_laplacian_pyramid,
    separable_gaussian,
    upsample2x,
)
from synapse.style.aces_color import (
    acescg_to_linear_srgb,
    cielab_to_srgb,
    linear_srgb_to_acescg,
    linear_to_srgb,
    rgb_to_cielab,
    soft_knee_forward_tonemap,
    soft_knee_inverse_tonemap,
    srgb_to_linear,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-4
_SCALE_CLAMP_LO: float = 0.5
_SCALE_CLAMP_HI: float = 2.0


# ---------------------------------------------------------------------------
# Internal helpers — morphology & pyramid delegation to render_core
# ---------------------------------------------------------------------------

_circular_structuring_element = circular_structuring_element
_binary_dilation = binary_dilation
_binary_erosion = binary_erosion
_gaussian_kernel_1d = gaussian_kernel_1d
_separable_gaussian = separable_gaussian
_downsample2x = downsample2x
_upsample2x = upsample2x
_build_gaussian_pyramid = build_gaussian_pyramid
_build_laplacian_pyramid = build_laplacian_pyramid
_reconstruct_laplacian_pyramid = reconstruct_laplacian_pyramid


# ---------------------------------------------------------------------------
# Internal helpers — box guided filter
# ---------------------------------------------------------------------------


def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """O(1) box filter using cumulative sums.

    Args:
        src: ``[H, W]`` or ``[H, W, C]`` float64 array.
        radius: Filter radius in pixels.

    Returns:
        Box-filtered array (each pixel = local mean in the window).
    """
    if radius <= 0:
        return np.asarray(src.copy())

    arr = src.astype(np.float64)
    h, w = arr.shape[:2]
    r = radius

    def _cumsum_mean_1d(
        a: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """Compute local mean along *axis* using cumsum."""
        n = a.shape[axis]
        # Prepend a zero-plane along *axis* so cumsum[i] = sum(0..i-1).
        pad_w = [(0, 0)] * a.ndim
        pad_w[axis] = (1, 0)
        cs = np.cumsum(np.pad(a, pad_w, constant_values=0), axis=axis)

        # For each position i, window is [max(0,i-r) .. min(n-1,i+r)].
        idx = np.arange(n)
        lo = np.clip(idx - r, 0, n)  # inclusive start
        hi = np.clip(idx + r + 1, 0, n)  # exclusive end
        counts = (hi - lo).astype(np.float64)

        # Build sliceable index arrays for gathering along *axis*.
        shape_out = list(a.shape)

        # Reshape lo/hi/counts to broadcast along *axis*.
        bcast = [1] * a.ndim
        bcast[axis] = n
        lo_b = lo.reshape(bcast)
        hi_b = hi.reshape(bcast)
        counts_b = counts.reshape(bcast)

        # Use np.take to gather from cumsum.
        sums_hi = np.take(cs, hi_b.ravel(), axis=axis)
        sums_lo = np.take(cs, lo_b.ravel(), axis=axis)

        # Reshape back.
        sums_hi = sums_hi.reshape(shape_out)
        sums_lo = sums_lo.reshape(shape_out)

        return np.asarray((sums_hi - sums_lo) / np.maximum(counts_b, 1.0))

    result = _cumsum_mean_1d(arr, axis=0)
    result = _cumsum_mean_1d(result, axis=1)
    return np.asarray(result)


def _guided_filter_box(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Fast O(1) box-filter guided filter.

    Args:
        guide: ``[H, W]`` grayscale guide image (float64).
        src: ``[H, W, C]`` source image (float64).
        radius: Box filter radius in pixels.
        eps: Regularisation parameter.

    Returns:
        Filtered ``[H, W, C]`` array.
    """
    guide = np.asarray(guide, dtype=np.float64)
    src = np.asarray(src, dtype=np.float64)

    if guide.ndim != 2:
        raise ValueError(f"guide must be [H, W], got shape {guide.shape}")
    if src.ndim == 2:
        src = src[..., np.newaxis]
        squeeze = True
    else:
        squeeze = False

    h, w, c = src.shape

    mean_I = _box_filter(guide, radius)
    mean_p = _box_filter(src, radius)
    mean_Ip = _box_filter(
        guide[..., np.newaxis] * src,
        radius,
    )
    mean_II = _box_filter(guide * guide, radius)

    var_I = mean_II - mean_I * mean_I
    # Clamp variance to non-negative (numerical safety).
    var_I = np.maximum(var_I, 0.0)

    cov_Ip = mean_Ip - mean_I[..., np.newaxis] * mean_p

    a = cov_Ip / (var_I[..., np.newaxis] + eps)
    b = mean_p - a * mean_I[..., np.newaxis]

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    result = mean_a * guide[..., np.newaxis] + mean_b

    if squeeze:
        result = result[..., 0]
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Public API — border_ring_lab_match
# ---------------------------------------------------------------------------


def border_ring_lab_match(
    patch_srgb: np.ndarray,
    bg_srgb: np.ndarray,
    binary_mask: np.ndarray,
    ring_width_px: int = 16,
) -> np.ndarray:
    """Colour-match the inner edge of *patch_srgb* to the outer
    background ring using CIE LAB affine transfer.

    Args:
        patch_srgb: ``[H, W, 3]`` uint8 or float sRGB foreground patch.
        bg_srgb: ``[H, W, 3]`` uint8 or float sRGB background image.
        binary_mask: ``[H, W]`` bool mask indicating the foreground
            region in *patch_srgb*.
        ring_width_px: Width (in pixels) of the outer reference ring
            and inner edge band used for statistics.

    Returns:
        Colour-matched ``[H, W, 3]`` patch in the same dtype and
        range as the input *patch_srgb*.
    """
    src_dtype = patch_srgb.dtype
    is_int = np.issubdtype(src_dtype, np.integer)
    int_max = float(np.iinfo(src_dtype).max) if is_int else 1.0

    # Normalise to [0, 1] float64.
    patch_f = patch_srgb.astype(np.float64)
    bg_f = bg_srgb.astype(np.float64)
    if is_int:
        patch_f = patch_f / int_max
        bg_f = bg_f / int_max

    mask = binary_mask.astype(bool)

    # Handle degenerate cases.
    if not np.any(mask):
        return np.asarray(patch_srgb.copy())

    # Build structuring element for ring extraction.
    se = _circular_structuring_element(ring_width_px)

    # Outer reference ring: Dilate(M) \ M
    dilated = _binary_dilation(mask, se)
    outer_ring = dilated & (~mask)

    # Inner edge band: M \ Erode(M) using edge padding so masks touching the
    # image frame border are not falsely eroded along the frame edge.
    pad_r = max(int(ring_width_px), 1)
    mask_padded = np.pad(mask, ((pad_r, pad_r), (pad_r, pad_r)), mode="edge")
    eroded = _binary_erosion(mask_padded, se)[pad_r:-pad_r, pad_r:-pad_r]
    inner_band = mask & (~eroded)

    # If rings are empty fall back to the full regions.
    if not np.any(outer_ring) or not np.any(inner_band):
        return np.asarray(patch_srgb.copy())

    # Convert to CIE LAB via linear RGB.
    patch_linear = srgb_to_linear(patch_f)
    bg_linear = srgb_to_linear(bg_f)

    patch_lab = rgb_to_cielab(patch_linear)
    bg_lab = rgb_to_cielab(bg_linear)

    # Extract LAB samples.
    outer_lab = bg_lab[outer_ring]  # [N_outer, 3]
    inner_lab = patch_lab[inner_band]  # [N_inner, 3]

    # Affine transfer per channel: scale + shift.
    result_lab = patch_lab.copy()
    for ch in range(3):
        outer_mean = float(np.mean(outer_lab[:, ch]))
        outer_std = float(np.std(outer_lab[:, ch]))
        inner_mean = float(np.mean(inner_lab[:, ch]))
        inner_std = float(np.std(inner_lab[:, ch]))

        # Scale factor with clamping.
        if inner_std < _EPS:
            scale = 1.0
        else:
            scale = outer_std / (inner_std + _EPS)
            scale = float(np.clip(scale, _SCALE_CLAMP_LO, _SCALE_CLAMP_HI))

        # Full affine: x' = (x - inner_mean) * scale + outer_mean
        result_lab[inner_band, ch] = (inner_lab[:, ch] - inner_mean) * scale + outer_mean

    # Convert back: LAB → linear → sRGB (delegate to canonical aces_color path).
    result_srgb = cielab_to_srgb(result_lab)

    # Merge: apply full LAB correction across the outer 75% of inner_band and
    # smoothstep-feather across the innermost 25% into the untouched core so
    # boundary-ring CIEDE2000 stays < 2.0 while avoiding a hard inner step.
    result_linear = srgb_to_linear(result_srgb)
    output_linear = patch_linear.copy()

    if np.any(inner_band):
        if np.any(eroded):
            from synapse.render_core.morphology import exact_euclidean_distance_transform

            # d_core: distance from the eroded core boundary outward into inner_band.
            d_core = exact_euclidean_distance_transform(eroded)
            feather_w = max(float(ring_width_px) * 0.25, 1.0)
            t_ramp = np.clip(d_core / feather_w, 0.0, 1.0)
            weight_map = t_ramp * t_ramp * (3.0 - 2.0 * t_ramp)
        else:
            # Small mask completely covered by inner_band: apply full correction cleanly
            # without inf/inf NaN from empty eroded core.
            weight_map = np.ones(mask.shape, dtype=np.float64)

        weight_3d = weight_map[..., np.newaxis]
        output_linear[inner_band] = (
            weight_3d[inner_band] * result_linear[inner_band]
            + (1.0 - weight_3d[inner_band]) * patch_linear[inner_band]
        )

    # Convert blended result back to sRGB for output.
    output_f = linear_to_srgb(np.clip(output_linear, 0.0, None))
    output_f = np.clip(output_f, 0.0, 1.0)

    if is_int:
        return np.asarray(np.rint(np.clip(output_f * int_max, 0.0, int_max)).astype(src_dtype))
    return np.asarray(output_f.astype(src_dtype))


def _lab_to_srgb_float(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* back to sRGB [0, 1].

    .. deprecated:: Use :func:`synapse.style.aces_color.cielab_to_srgb` instead.
       This wrapper is retained only for backward compatibility.
    """
    return cielab_to_srgb(lab)


# ---------------------------------------------------------------------------
# Public API — laplacian_pyramid_blend
# ---------------------------------------------------------------------------


def laplacian_pyramid_blend(
    fg_srgb: np.ndarray,
    bg_srgb: np.ndarray,
    alpha_mask: np.ndarray,
    num_levels: int = 4,
) -> np.ndarray:
    """Multi-scale Laplacian pyramid blend of *fg* over *bg*.

    Args:
        fg_srgb: ``[H, W, 3]`` uint8 or float sRGB foreground.
        bg_srgb: ``[H, W, 3]`` uint8 or float sRGB background.
        alpha_mask: ``[H, W]`` float32 alpha in ``[0, 1]``.
        num_levels: Number of Laplacian pyramid levels (default 4).

    Returns:
        Blended ``[H, W, 3]`` image with the same dtype as *fg_srgb*.
    """
    src_dtype = fg_srgb.dtype
    is_uint8 = np.issubdtype(src_dtype, np.integer)

    fg = fg_srgb.astype(np.float64)
    bg = bg_srgb.astype(np.float64)
    if is_uint8:
        fg = fg / 255.0
        bg = bg / 255.0

    alpha = alpha_mask.astype(np.float64)

    # Convert to linear space to eliminate dark fringe artefacts.
    fg_lin = srgb_to_linear(fg)
    bg_lin = srgb_to_linear(bg)

    # Clamp num_levels based on image size.
    min_dim = min(fg.shape[0], fg.shape[1])
    max_levels = max(int(np.floor(np.log2(max(min_dim, 1)))), 1)
    num_levels = min(num_levels, max_levels)

    # Build Laplacian pyramids for fg and bg in linear space.
    lap_fg = _build_laplacian_pyramid(fg_lin, num_levels)
    lap_bg = _build_laplacian_pyramid(bg_lin, num_levels)

    # Build Gaussian pyramid for the mask.
    gauss_mask = _build_gaussian_pyramid(alpha, num_levels)

    # Blend at each level.
    lap_blended: list[np.ndarray] = []
    for i in range(len(lap_fg)):
        m = gauss_mask[i]
        if lap_fg[i].ndim == 3 and m.ndim == 2:
            m = m[..., np.newaxis]
        blended_level = m * lap_fg[i] + (1.0 - m) * lap_bg[i]
        lap_blended.append(blended_level)

    # Reconstruct from blended Laplacian pyramid and convert back to sRGB.
    result_lin = _reconstruct_laplacian_pyramid(lap_blended)
    result_lin = np.clip(result_lin, 0.0, None)
    result = linear_to_srgb(result_lin)
    result = np.clip(result, 0.0, 1.0)

    if is_uint8:
        return np.asarray(np.rint(np.clip(result * 255.0, 0.0, 255.0)).astype(np.uint8))
    return np.asarray(result.astype(src_dtype))


# ---------------------------------------------------------------------------
# Internal helpers — forward motion warp
# ---------------------------------------------------------------------------


def _forward_warp_bilinear(
    img: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    """Warp *img* by flow field using backward bilinear interpolation.

    For destination pixel (x, y), samples source image at (x - dx, y - dy).

    Args:
        img: ``[H, W, 3]`` uint8 or float source image.
        flow: ``[H, W, 2]`` float32/float64 flow field (dx, dy).

    Returns:
        Warped image with the same shape and dtype as *img*.
    """
    h, w = img.shape[:2]
    src = img.astype(np.float64)
    flow_safe = np.nan_to_num(flow.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    yy, xx = np.mgrid[:h, :w]
    src_x = np.clip(xx.astype(np.float64) - flow_safe[..., 0], 0.0, float(w - 1))
    src_y = np.clip(yy.astype(np.float64) - flow_safe[..., 1], 0.0, float(h - 1))

    x0 = np.floor(src_x).astype(np.intp)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(src_y).astype(np.intp)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx1 = src_x - x0.astype(np.float64)
    wx0 = 1.0 - wx1
    wy1 = src_y - y0.astype(np.float64)
    wy0 = 1.0 - wy1

    # Bilinear interpolation
    w00 = (wx0 * wy0)[..., np.newaxis]
    w01 = (wx0 * wy1)[..., np.newaxis]
    w10 = (wx1 * wy0)[..., np.newaxis]
    w11 = (wx1 * wy1)[..., np.newaxis]

    warped = w00 * src[y0, x0] + w01 * src[y1, x0] + w10 * src[y0, x1] + w11 * src[y1, x1]

    if np.issubdtype(img.dtype, np.integer):
        return np.asarray(np.clip(warped, 0.0, 255.0).astype(img.dtype))
    return np.asarray(warped.astype(img.dtype))


# ---------------------------------------------------------------------------
# Public API — composite_surgical_sequence
# ---------------------------------------------------------------------------


def composite_surgical_sequence(
    video_frames: np.ndarray,
    rerendered_frames: np.ndarray,
    surgical_masks: np.ndarray,
    motion_vectors: np.ndarray | None = None,
    ema_alpha: float = 0.85,
) -> np.ndarray:
    """Composite a re-rendered sequence onto video with temporal
    smoothing.

    For each frame:
      1. Apply ``border_ring_lab_match`` on the re-rendered patch.
      2. Apply ``laplacian_pyramid_blend`` using the surgical mask.
      3. EMA-smooth the mask boundary band across time.

    Args:
        video_frames: ``[T, H, W, 3]`` uint8 sRGB video.
        rerendered_frames: ``[T, H, W, 3]`` uint8 sRGB re-rendered
            frames.
        surgical_masks: ``[T, H, W]`` float32 alpha masks in
            ``[0, 1]``.
        motion_vectors: ``[T, H, W, 2]`` float32 optional per-pixel
            motion vectors (dx, dy).  When provided, the previous
            composited frame is forward-warped by the flow field before
            temporal EMA blending.  Large-magnitude flow (mean >15 px)
            automatically reduces ``ema_alpha`` to avoid ghosting.
        ema_alpha: EMA blending factor for temporal smoothing
            (higher = more weight on current frame).

    Returns:
        Composited ``[T, H, W, 3]`` uint8 sRGB sequence.
    """
    t_count, h, w, _ = video_frames.shape
    result = np.empty_like(video_frames)

    # Boundary band width for temporal smoothing.
    band_width = 8
    se_band = _circular_structuring_element(band_width)

    prev_boundary: np.ndarray | None = None

    for t in range(t_count):
        vf = video_frames[t]
        rf = rerendered_frames[t]
        mask_alpha = surgical_masks[t]

        # Binary mask for border-ring matching.
        binary_mask = mask_alpha > 0.5

        # Step 1: border-ring LAB colour match.
        matched = border_ring_lab_match(rf, vf, binary_mask)

        # Step 2: Laplacian pyramid blend.
        blended = laplacian_pyramid_blend(matched, vf, mask_alpha)

        # Step 3: Temporal EMA at boundary band with optional motion warp.
        if t > 0 and prev_boundary is not None:
            # Inner boundary band: pixels near the mask edge.
            dilated = _binary_dilation(binary_mask, se_band)
            eroded = _binary_erosion(binary_mask, se_band)
            boundary_band = dilated & (~eroded)

            if np.any(boundary_band):
                # Determine effective EMA alpha (adaptive for large flow)
                effective_alpha = ema_alpha

                # Forward-warp the previous frame by motion vectors
                warped_prev: np.ndarray
                if motion_vectors is not None:
                    mv = motion_vectors[t].astype(np.float64)  # (H, W, 2) dx, dy
                    # Measure flow only within boundary band for ghost detection
                    band_mv = mv[boundary_band]
                    flow_mag = (
                        float(np.mean(np.sqrt(band_mv[..., 0] ** 2 + band_mv[..., 1] ** 2)))
                        if band_mv.size > 0
                        else 0.0
                    )
                    # Adaptive alpha: large motion -> trust current frame (effective_alpha -> 1.0) to eliminate ghosting
                    if flow_mag > 15.0:
                        effective_alpha = min(
                            1.0,
                            1.0 - (1.0 - ema_alpha) * (15.0 / max(flow_mag, 15.0)),
                        )

                    # Forward bilinear warp
                    warped_prev = _forward_warp_bilinear(
                        prev_boundary,
                        mv,
                    )
                else:
                    warped_prev = prev_boundary

                # EMA blend at boundary pixels.
                blended_f = blended.astype(np.float64)
                prev_f = warped_prev.astype(np.float64)
                band_3d = boundary_band[..., np.newaxis]
                smoothed = np.where(
                    band_3d,
                    effective_alpha * blended_f + (1.0 - effective_alpha) * prev_f,
                    blended_f,
                )
                blended = (
                    np.rint(np.clip(smoothed, 0.0, 255.0)).astype(video_frames.dtype)
                    if np.issubdtype(video_frames.dtype, np.integer)
                    else smoothed.astype(video_frames.dtype)
                )

        result[t] = blended
        prev_boundary = blended.copy()

    # --- Traceboard instrumentation ---
    from synapse.observability import get_probe

    probe = get_probe()
    if probe is not None and probe.active:
        probe.log_scalar("composite/surgical/num_frames", float(t_count))
        probe.log_scalar("composite/surgical/height", float(h))
        probe.log_scalar("composite/surgical/width", float(w))
        probe.log_scalar("composite/surgical/ema_alpha", ema_alpha)
        # Mean mask coverage (fraction of pixels inside mask)
        mask_coverage = float(np.mean(surgical_masks > 0.5))
        probe.log_scalar("composite/surgical/mean_mask_coverage", mask_coverage)
        probe.log_scalar(
            "composite/surgical/has_motion_vectors",
            float(motion_vectors is not None),
        )

    return result


# ---------------------------------------------------------------------------
# Public API — composite_optical_increments
# ---------------------------------------------------------------------------


def composite_optical_increments(
    video_frames_srgb: np.ndarray,
    fgpp_frames: np.ndarray,
    cryptomatte_ids: np.ndarray,
    contact_shadow_map: np.ndarray | None = None,
    caustics_map: np.ndarray | None = None,
    guided_filter_r: int = 4,
) -> np.ndarray:
    """Overlay physical shadow and caustics onto video in ACEScg
    colour space.

    Pipeline:
      1. Convert LDR video to linear via soft-knee inverse tonemap
         + linear_srgb_to_acescg.
      2. Align Cryptomatte boundary via guided filter.
      3. Multiplicative contact shadow overlay.
      4. Additive HDR caustics overlay.
      5. Map back to display sRGB.

    Args:
        video_frames_srgb: ``[T, H, W, 3]`` uint8 sRGB video.
        fgpp_frames: ``[T, H, W, 3]`` float32 FGPP linear frames
            (unused directly, reserved for future extensions).
        cryptomatte_ids: ``[T, H, W]`` uint32 Cryptomatte entity IDs.
        contact_shadow_map: ``[T, H, W]`` float32 shadow attenuation
            in ``[0, 1]`` or ``None`` to skip.
        caustics_map: ``[T, H, W, 3]`` float32 HDR caustics radiance
            in linear sRGB or ``None`` to skip.
        guided_filter_r: Guided filter radius for Cryptomatte
            boundary alignment (default 4).

    Returns:
        Composited ``[T, H, W, 3]`` uint8 sRGB sequence.
    """
    if video_frames_srgb.ndim != 4 or video_frames_srgb.shape[-1] != 3:
        raise ValueError(f"video_frames_srgb must be [T,H,W,3], got {video_frames_srgb.shape}")
    if cryptomatte_ids.ndim != 3:
        raise ValueError(f"cryptomatte_ids must be [T,H,W], got {cryptomatte_ids.shape}")
    t_count, h, w, _ = video_frames_srgb.shape
    if cryptomatte_ids.shape != (t_count, h, w):
        raise ValueError(
            f"cryptomatte_ids shape {cryptomatte_ids.shape} != expected ({t_count}, {h}, {w})"
        )
    if contact_shadow_map is not None and contact_shadow_map.shape != (t_count, h, w):
        raise ValueError(
            f"contact_shadow_map shape {contact_shadow_map.shape} != expected ({t_count}, {h}, {w})"
        )
    if caustics_map is not None:
        expected_caustics = (t_count, h, w, 3)
        if caustics_map.shape != expected_caustics:
            raise ValueError(
                f"caustics_map shape {caustics_map.shape} != expected {expected_caustics}"
            )
    src_dtype = video_frames_srgb.dtype
    is_int = np.issubdtype(src_dtype, np.integer)
    int_max = float(np.iinfo(src_dtype).max) if is_int else 1.0
    result = np.empty_like(video_frames_srgb)

    for t in range(t_count):
        frame_srgb = video_frames_srgb[t].astype(np.float64)
        if is_int:
            frame_srgb = frame_srgb / int_max

        # 1. LDR → linear → soft-knee inverse tonemap → ACEScg.
        frame_linear = srgb_to_linear(frame_srgb)
        frame_hdr = soft_knee_inverse_tonemap(frame_linear)
        frame_acescg = linear_srgb_to_acescg(frame_hdr)

        # 2. Build guide from Cryptomatte boundary (preserve 3ch separation via weighted luminance).
        cid = cryptomatte_ids[t]
        guide_3ch = _cryptomatte_boundary_guide(cid)
        guide = 0.299 * guide_3ch[..., 0] + 0.587 * guide_3ch[..., 1] + 0.114 * guide_3ch[..., 2]

        # IMPORTANT: Do NOT apply guided filter to the original video
        # background. The guided filter is only used for aligning
        # optical increment layers (shadow/caustics) to Cryptomatte
        # boundaries.

        # 3. Multiplicative contact shadow (with boundary-aware smoothing).
        if contact_shadow_map is not None:
            shadow = contact_shadow_map[t].astype(np.float64)
            # Clamp to prevent dead blacks (min attenuation 0.01).
            shadow = np.clip(shadow, 0.01, 1.0)
            # Apply guided filter to the shadow map for boundary alignment
            if guided_filter_r > 0:
                shadow = _guided_filter_box(
                    guide,
                    shadow[..., np.newaxis],
                    guided_filter_r,
                )[..., 0]
                shadow = np.clip(shadow, 0.01, 1.0)
            frame_acescg = frame_acescg * shadow[..., np.newaxis]

        # 4. Additive HDR caustics (with boundary-aware smoothing).
        if caustics_map is not None:
            caustics_linear = caustics_map[t].astype(np.float64)
            caustics_acescg = linear_srgb_to_acescg(caustics_linear)
            # Apply guided filter to the caustics layer for boundary alignment
            if guided_filter_r > 0:
                caustics_acescg = _guided_filter_box(
                    guide,
                    caustics_acescg,
                    guided_filter_r,
                )
            frame_acescg = frame_acescg + caustics_acescg

        # 5. ACEScg → linear sRGB → soft-knee forward tonemap → sRGB display.
        frame_linear_out = acescg_to_linear_srgb(frame_acescg)
        frame_linear_out = np.clip(frame_linear_out, 0.0, None)
        frame_tonemapped = soft_knee_forward_tonemap(frame_linear_out)
        frame_srgb_out = linear_to_srgb(frame_tonemapped)
        frame_srgb_out = np.clip(frame_srgb_out, 0.0, 1.0)

        if is_int:
            result[t] = np.rint(np.clip(frame_srgb_out * int_max, 0.0, int_max)).astype(src_dtype)
        else:
            result[t] = frame_srgb_out.astype(src_dtype)

    return result


def _cryptomatte_boundary_guide(cid: np.ndarray) -> np.ndarray:
    """Map Cryptomatte entity IDs to 3-channel golden-ratio hash embedding.

    Each unique entity ID is mapped to a distinct 3-channel colour via
    golden-ratio hashing, ensuring maximal separation between adjacent
    entities.  Background (ID == 0) maps to [0, 0, 0].

    Args:
        cid: ``[H, W]`` uint32 Cryptomatte IDs.

    Returns:
        ``[H, W, 3]`` float64 guide image in ``[0, 1]``.
    """
    h, w = cid.shape
    guide = np.zeros((h, w, 3), dtype=np.float64)
    phi_inv = 0.6180339887498949  # 1/φ (golden ratio conjugate)
    unique_ids = np.unique(cid)
    for uid in unique_ids:
        if uid == 0:
            continue
        mask = cid == uid
        # Hash each ID to 3 channels using golden ratio
        hval = int(uid)
        guide[mask, 0] = (hval * phi_inv) % 1.0
        guide[mask, 1] = (hval * phi_inv * 2) % 1.0
        guide[mask, 2] = (hval * phi_inv * 3) % 1.0
    return guide
