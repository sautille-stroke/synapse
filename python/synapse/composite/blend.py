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

from synapse.style.aces_color import (
    acescg_to_linear_srgb,
    linear_srgb_to_acescg,
    linear_to_srgb,
    rgb_to_cielab,
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
# Internal helpers — morphology
# ---------------------------------------------------------------------------


def _circular_structuring_element(radius: int) -> np.ndarray:
    """Create a circular (disk) structuring element of given radius.

    Returns a bool array of shape ``(2*radius+1, 2*radius+1)``.
    """
    d = 2 * radius + 1
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _binary_dilation(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Dilate a bool mask with a structuring element.

    Uses ``scipy.ndimage.binary_dilation`` when available, otherwise
    falls back to a pure-NumPy pad-shift-or implementation.
    """
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, structure=se).astype(bool)
    except ImportError:
        pass

    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    padded = np.pad(mask, ((ph, ph), (pw, pw)), constant_values=False)
    result = np.zeros_like(padded)
    ys, xs = np.nonzero(se)
    for dy, dx in zip(ys, xs):
        oy = dy - ph
        ox = dx - pw
        shifted = np.roll(np.roll(padded, oy, axis=0), ox, axis=1)
        result |= shifted
    return result[ph:ph + mask.shape[0], pw:pw + mask.shape[1]]


def _binary_erosion(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Erode a bool mask with a structuring element.

    Uses ``scipy.ndimage.binary_erosion`` when available, otherwise
    falls back to dilation of the inverse.
    """
    try:
        from scipy.ndimage import binary_erosion
        return binary_erosion(mask, structure=se).astype(bool)
    except ImportError:
        pass

    # Erosion(A, B) = complement(Dilation(complement(A), B))
    return ~_binary_dilation(~mask, se)


# ---------------------------------------------------------------------------
# Internal helpers — Gaussian convolution
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create a normalised 1-D Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(sigma, _EPS)) ** 2)
    return k / k.sum()


def _separable_gaussian(
    img: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply separable Gaussian blur to a float64 image.

    Uses ``scipy.ndimage.gaussian_filter`` when available, otherwise
    falls back to a manual separable convolution via cumsum tricks.

    Args:
        img: ``[H, W]`` or ``[H, W, C]`` float64 array.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred array with the same shape and dtype.
    """
    if sigma < 0.5:
        return img.copy()

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(
            img, sigma=sigma, mode="reflect",
        ).astype(np.float64)
    except ImportError:
        pass

    # Pure NumPy separable convolution.
    radius = max(int(round(3.0 * sigma)), 1)
    kernel = _gaussian_kernel_1d(sigma, radius)

    def _convolve_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        pad_widths = [(0, 0)] * arr.ndim
        pad_widths[axis] = (radius, radius)
        padded = np.pad(arr, pad_widths, mode="reflect")
        shape = list(padded.shape)
        n = shape[axis]
        out_n = n - 2 * radius
        result = np.zeros_like(arr)
        for i, w in enumerate(kernel):
            slices_src = [slice(None)] * arr.ndim
            slices_src[axis] = slice(i, i + out_n)
            result += w * padded[tuple(slices_src)]
        return result

    out = _convolve_axis(img, axis=0)
    out = _convolve_axis(out, axis=1)
    return out


def _downsample2x(img: np.ndarray) -> np.ndarray:
    """Downsample by 2x with Gaussian pre-filter."""
    blurred = _separable_gaussian(img, sigma=1.0)
    return blurred[::2, ::2]


def _upsample2x(img: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Upsample by 2x with bilinear interpolation to *target_shape*."""
    th, tw = target_shape[:2]
    h, w = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros((th, tw), dtype=img.dtype)
    else:
        out = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)

    # Simple bilinear upsample via index mapping.
    row_idx = np.arange(th, dtype=np.float64) * (h - 1) / max(th - 1, 1)
    col_idx = np.arange(tw, dtype=np.float64) * (w - 1) / max(tw - 1, 1)

    r0 = np.floor(row_idx).astype(np.intp)
    r1 = np.minimum(r0 + 1, h - 1)
    rf = (row_idx - r0).astype(np.float64)

    c0 = np.floor(col_idx).astype(np.intp)
    c1 = np.minimum(c0 + 1, w - 1)
    cf = (col_idx - c0).astype(np.float64)

    if img.ndim == 2:
        # [th] x [tw] outer product interpolation
        for i in range(th):
            row_frac = rf[i]
            top = (1.0 - row_frac) * img[r0[i], :] + row_frac * img[r1[i], :]
            out[i, :] = (
                (1.0 - cf) * top[c0] + cf * top[c1]
            )
    else:
        for i in range(th):
            row_frac = rf[i]
            top = (
                (1.0 - row_frac) * img[r0[i], :, :]
                + row_frac * img[r1[i], :, :]
            )
            out[i, :, :] = (
                (1.0 - cf)[:, np.newaxis] * top[c0, :]
                + cf[:, np.newaxis] * top[c1, :]
            )
    return out


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
        return src.copy()

    arr = src.astype(np.float64)
    h, w = arr.shape[:2]
    r = radius

    def _cumsum_mean_1d(
        a: np.ndarray, axis: int,
    ) -> np.ndarray:
        """Compute local mean along *axis* using cumsum."""
        n = a.shape[axis]
        # Prepend a zero-plane along *axis* so cumsum[i] = sum(0..i-1).
        pad_w = [(0, 0)] * a.ndim
        pad_w[axis] = (1, 0)
        cs = np.cumsum(np.pad(a, pad_w, constant_values=0), axis=axis)

        # For each position i, window is [max(0,i-r) .. min(n-1,i+r)].
        idx = np.arange(n)
        lo = np.clip(idx - r, 0, n)      # inclusive start
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

        return (sums_hi - sums_lo) / np.maximum(counts_b, 1.0)

    result = _cumsum_mean_1d(arr, axis=0)
    result = _cumsum_mean_1d(result, axis=1)
    return result


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
        raise ValueError(
            f"guide must be [H, W], got shape {guide.shape}"
        )
    if src.ndim == 2:
        src = src[..., np.newaxis]
        squeeze = True
    else:
        squeeze = False

    h, w, c = src.shape

    mean_I = _box_filter(guide, radius)
    mean_p = _box_filter(src, radius)
    mean_Ip = _box_filter(
        guide[..., np.newaxis] * src, radius,
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
    return result


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
    is_uint8 = np.issubdtype(src_dtype, np.integer)

    # Normalise to [0, 1] float64.
    patch_f = patch_srgb.astype(np.float64)
    bg_f = bg_srgb.astype(np.float64)
    if is_uint8:
        patch_f = patch_f / 255.0
        bg_f = bg_f / 255.0

    mask = binary_mask.astype(bool)

    # Handle degenerate cases.
    if not np.any(mask):
        return patch_srgb.copy()

    # Build structuring element for ring extraction.
    se = _circular_structuring_element(ring_width_px)

    # Outer reference ring: Dilate(M) \ M
    dilated = _binary_dilation(mask, se)
    outer_ring = dilated & (~mask)

    # Inner edge band: M \ Erode(M)
    eroded = _binary_erosion(mask, se)
    inner_band = mask & (~eroded)

    # If rings are empty fall back to the full regions.
    if not np.any(outer_ring) or not np.any(inner_band):
        return patch_srgb.copy()

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

        # Apply only within the inner band (gradually blending into
        # the mask interior to avoid a secondary seam).
        # Full affine: x' = (x - inner_mean) * scale + outer_mean
        result_lab[inner_band, ch] = (
            (inner_lab[:, ch] - inner_mean) * scale + outer_mean
        )

    # Convert back: LAB → linear → sRGB.
    # Inverse of rgb_to_cielab is not provided in aces_color, so we
    # do it inline.
    result_srgb = _lab_to_srgb_float(result_lab)

    # Merge: only modify the inner band pixels.
    output_f = patch_f.copy()
    output_f[inner_band] = result_srgb[inner_band]

    if is_uint8:
        output_f = np.clip(output_f * 255.0, 0.0, 255.0)
        return output_f.astype(np.uint8)
    return np.clip(output_f, 0.0, 1.0).astype(src_dtype)


def _lab_to_srgb_float(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* back to sRGB [0, 1].

    Inverse of ``rgb_to_cielab(srgb_to_linear(x))``.
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # D65 white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    x = np.where(fx > delta, fx ** 3, 3.0 * delta ** 2 * (fx - 4.0 / 29.0))
    y = np.where(fy > delta, fy ** 3, 3.0 * delta ** 2 * (fy - 4.0 / 29.0))
    z = np.where(fz > delta, fz ** 3, 3.0 * delta ** 2 * (fz - 4.0 / 29.0))

    x *= Xn
    y *= Yn
    z *= Zn

    xyz = np.stack([x, y, z], axis=-1)

    # XYZ → linear sRGB  (inverse of _M_SRGB_TO_XYZ)
    M_XYZ_TO_SRGB = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float64)

    linear_rgb = xyz @ M_XYZ_TO_SRGB.T
    linear_rgb = np.clip(linear_rgb, 0.0, None)

    srgb = linear_to_srgb(linear_rgb)
    return np.clip(srgb, 0.0, 1.0)


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

    # Clamp num_levels based on image size.
    min_dim = min(fg.shape[0], fg.shape[1])
    max_levels = max(int(np.floor(np.log2(max(min_dim, 1)))), 1)
    num_levels = min(num_levels, max_levels)

    # Build Laplacian pyramids for fg and bg.
    lap_fg = _build_laplacian_pyramid(fg, num_levels)
    lap_bg = _build_laplacian_pyramid(bg, num_levels)

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

    # Reconstruct from blended Laplacian pyramid.
    result = _reconstruct_laplacian_pyramid(lap_blended)
    result = np.clip(result, 0.0, 1.0)

    if is_uint8:
        return (result * 255.0).astype(np.uint8)
    return result.astype(src_dtype)


def _build_gaussian_pyramid(
    img: np.ndarray,
    num_levels: int,
) -> list[np.ndarray]:
    """Build a Gaussian pyramid with *num_levels* + 1 entries."""
    pyramid = [img.copy()]
    current = img.copy()
    for _ in range(num_levels):
        current = _downsample2x(current)
        pyramid.append(current)
    return pyramid


def _build_laplacian_pyramid(
    img: np.ndarray,
    num_levels: int,
) -> list[np.ndarray]:
    """Build a Laplacian pyramid with *num_levels* + 1 entries.

    The last entry is the low-pass residual (Gaussian level).
    """
    gauss = _build_gaussian_pyramid(img, num_levels)
    laplacian: list[np.ndarray] = []
    for i in range(num_levels):
        upsampled = _upsample2x(gauss[i + 1], gauss[i].shape)
        laplacian.append(gauss[i] - upsampled)
    laplacian.append(gauss[-1])  # residual
    return laplacian


def _reconstruct_laplacian_pyramid(
    pyramid: list[np.ndarray],
) -> np.ndarray:
    """Reconstruct an image from its Laplacian pyramid."""
    current = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        upsampled = _upsample2x(current, pyramid[i].shape)
        current = pyramid[i] + upsampled
    return current


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
            motion vectors. Currently used for future motion-guided
            EMA (reserved).
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

        # Step 3: Temporal EMA at boundary band.
        if t > 0 and prev_boundary is not None:
            # Inner boundary band: pixels near the mask edge.
            dilated = _binary_dilation(binary_mask, se_band)
            eroded = _binary_erosion(binary_mask, se_band)
            boundary_band = dilated & (~eroded)

            if np.any(boundary_band):
                # EMA blend at boundary pixels.
                blended_f = blended.astype(np.float64)
                prev_f = prev_boundary.astype(np.float64)
                band_3d = boundary_band[..., np.newaxis]
                smoothed = np.where(
                    band_3d,
                    ema_alpha * blended_f + (1.0 - ema_alpha) * prev_f,
                    blended_f,
                )
                blended = np.clip(smoothed, 0.0, 255.0).astype(np.uint8)

        result[t] = blended
        prev_boundary = blended.copy()

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
    t_count, h, w, _ = video_frames_srgb.shape
    result = np.empty_like(video_frames_srgb)

    for t in range(t_count):
        frame_srgb = video_frames_srgb[t].astype(np.float64) / 255.0

        # 1. LDR → linear → soft-knee inverse tonemap → ACEScg.
        frame_linear = srgb_to_linear(frame_srgb)
        frame_hdr = soft_knee_inverse_tonemap(frame_linear)
        frame_acescg = linear_srgb_to_acescg(frame_hdr)

        # 2. Build guide from Cryptomatte boundary.
        cid = cryptomatte_ids[t]
        guide = _cryptomatte_boundary_guide(cid)

        # Apply guided filter to align the frame to Cryptomatte
        # boundaries.
        if guided_filter_r > 0:
            frame_acescg = _guided_filter_box(
                guide, frame_acescg, guided_filter_r,
            )

        # 3. Multiplicative contact shadow.
        if contact_shadow_map is not None:
            shadow = contact_shadow_map[t].astype(np.float64)
            # Clamp to prevent dead blacks (min attenuation 0.01).
            shadow = np.clip(shadow, 0.01, 1.0)
            frame_acescg = frame_acescg * shadow[..., np.newaxis]

        # 4. Additive HDR caustics.
        if caustics_map is not None:
            caustics_linear = caustics_map[t].astype(np.float64)
            caustics_acescg = linear_srgb_to_acescg(caustics_linear)
            frame_acescg = frame_acescg + caustics_acescg

        # 5. ACEScg → linear sRGB → sRGB display.
        frame_linear_out = acescg_to_linear_srgb(frame_acescg)
        frame_linear_out = np.clip(frame_linear_out, 0.0, None)
        frame_srgb_out = linear_to_srgb(frame_linear_out)
        frame_srgb_out = np.clip(frame_srgb_out, 0.0, 1.0)

        result[t] = (frame_srgb_out * 255.0).astype(np.uint8)

    return result


def _cryptomatte_boundary_guide(cid: np.ndarray) -> np.ndarray:
    """Extract Cryptomatte boundary as a grayscale guide image.

    Boundary pixels (where adjacent IDs differ) are set to 1.0,
    interior pixels to the normalised Cryptomatte ID as a grayscale
    hint.

    Args:
        cid: ``[H, W]`` uint32 Cryptomatte IDs.

    Returns:
        ``[H, W]`` float64 guide image in ``[0, 1]``.
    """
    h, w = cid.shape
    boundary = np.zeros((h, w), dtype=bool)
    if w > 1:
        boundary[:, :-1] |= cid[:, :-1] != cid[:, 1:]
        boundary[:, 1:] |= cid[:, :-1] != cid[:, 1:]
    if h > 1:
        boundary[:-1, :] |= cid[:-1, :] != cid[1:, :]
        boundary[1:, :] |= cid[:-1, :] != cid[1:, :]

    # Normalised grayscale from IDs.
    cid_f = cid.astype(np.float64)
    max_id = cid_f.max()
    if max_id > 0:
        guide = cid_f / max_id
    else:
        guide = np.zeros((h, w), dtype=np.float64)

    # Override boundary pixels.
    guide[boundary] = 1.0
    return guide
