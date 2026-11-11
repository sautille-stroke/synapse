"""Style extraction from sRGB video frames and FGPP sequences.

Two extraction modes:

* **Optical-only** — estimates colour temperature, exposure, and grain
  from sRGB pixel statistics alone.
* **FGPP-intrinsic** — additionally recovers key-light direction and
  ambient ratio via least-squares on FGPP base_color / world_normal.
"""

from __future__ import annotations

import numpy as np

from synapse.fgpp.frame import FGPPSequence
from synapse.style.aces_color import srgb_to_linear
from synapse.style.descriptor import StyleDescriptor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# sRGB → XYZ matrix (Rec. 709 primaries, D65) for chromaticity estimation
_M_LINEAR_TO_XYZ: np.ndarray = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


def _estimate_cct_mccamy(rgb_linear: np.ndarray) -> float:
    """Estimate correlated colour temperature via McCamy approximation.

    Args:
        rgb_linear: Linear sRGB image, shape (H, W, 3).

    Returns:
        Estimated CCT in kelvin, clamped to [2000, 20000].
    """
    # Mean colour
    mean_rgb = np.mean(rgb_linear, axis=(0, 1))  # (3,)
    mean_rgb = np.clip(mean_rgb, 1e-10, None)

    # Convert to XYZ then xy chromaticity
    xyz = mean_rgb @ _M_LINEAR_TO_XYZ.T
    s = xyz.sum()
    if s < 1e-12:
        return 6500.0
    x = xyz[0] / s
    y = xyz[1] / s

    # McCamy's formula
    n = (x - 0.3320) / (0.1858 - y + 1e-12)
    cct = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
    return float(np.clip(cct, 2000.0, 20000.0))


def _estimate_exposure_ev(rgb_linear: np.ndarray) -> float:
    """Estimate exposure compensation in EV relative to 18% grey.

    Args:
        rgb_linear: Linear sRGB image, shape (H, W, 3).

    Returns:
        Exposure offset in EV.
    """
    # Luminance (Rec. 709 coefficients)
    lum = (
        0.2126 * rgb_linear[..., 0]
        + 0.7152 * rgb_linear[..., 1]
        + 0.0722 * rgb_linear[..., 2]
    )
    lum_positive = lum[lum > 1e-10]
    if lum_positive.size == 0:
        return 0.0
    median_log_lum = float(np.median(np.log2(lum_positive)))
    # 18% grey ≈ 0.18 → log2(0.18) ≈ -2.474
    return median_log_lum - np.log2(0.18)


def _estimate_grain_sigma(rgb_linear: np.ndarray) -> float:
    """Estimate film grain / noise sigma via Laplacian residual.

    Uses a 3×3 Laplacian kernel on luminance, then takes the robust
    MAD-based sigma estimate.

    Args:
        rgb_linear: Linear sRGB image, shape (H, W, 3).

    Returns:
        Estimated grain sigma (>= 0).
    """
    lum = (
        0.2126 * rgb_linear[..., 0]
        + 0.7152 * rgb_linear[..., 1]
        + 0.0722 * rgb_linear[..., 2]
    )
    if lum.shape[0] < 3 or lum.shape[1] < 3:
        return 0.0

    # Laplacian via finite differences (no opencv dependency)
    # L(i,j) = -4*p(i,j) + p(i-1,j) + p(i+1,j) + p(i,j-1) + p(i,j+1)
    lap = (
        -4.0 * lum[1:-1, 1:-1]
        + lum[:-2, 1:-1]
        + lum[2:, 1:-1]
        + lum[1:-1, :-2]
        + lum[1:-1, 2:]
    )
    # Robust sigma via MAD (median absolute deviation)
    # For Laplacian of Gaussian noise: sigma ≈ MAD / 0.6745 / sqrt(20)
    # where sqrt(20) accounts for the Laplacian kernel gain
    mad = float(np.median(np.abs(lap - np.median(lap))))
    sigma = mad / 0.6745 / np.sqrt(20.0)
    return max(float(sigma), 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_style_optical(frames_srgb: np.ndarray) -> StyleDescriptor:
    """Extract visual style parameters from sRGB video frames (optical only).

    Estimates colour temperature, exposure, contrast, saturation, and
    grain using per-frame statistics aggregated via cross-frame median.

    Args:
        frames_srgb: sRGB video frames, shape (T, H, W, 3), dtype uint8.

    Returns:
        Populated ``StyleDescriptor`` with optical estimates.
    """
    frames = np.asarray(frames_srgb, dtype=np.float64) / 255.0
    T = frames.shape[0]

    cct_list: list[float] = []
    ev_list: list[float] = []
    grain_list: list[float] = []

    for t in range(T):
        frame_linear = srgb_to_linear(frames[t])
        cct_list.append(_estimate_cct_mccamy(frame_linear))
        ev_list.append(_estimate_exposure_ev(frame_linear))
        grain_list.append(_estimate_grain_sigma(frame_linear))

    return StyleDescriptor(
        color_temp_kelvin=float(np.median(cct_list)),
        exposure_ev=float(np.median(ev_list)),
        grain_sigma=float(np.median(grain_list)),
    )


def extract_style_fgpp_intrinsic(
    frames_srgb: np.ndarray,
    fgpp_seq: FGPPSequence,
) -> StyleDescriptor:
    """Extract style using sRGB video frames combined with FGPP data.

    Recovers key-light direction and ambient ratio by solving a
    least-squares lighting model on FGPP base_color and world_normal
    channels.

    Args:
        frames_srgb: sRGB video frames, shape (T, H, W, 3), dtype uint8.
        fgpp_seq: Corresponding FGPP sequence with base_color,
            world_normal, and optionally cryptomatte_id / roughness.

    Returns:
        ``StyleDescriptor`` enriched with lighting parameters.
    """
    # Start with optical estimates
    desc = extract_style_optical(frames_srgb)
    frames_f = np.asarray(frames_srgb, dtype=np.float64) / 255.0

    T = min(frames_f.shape[0], fgpp_seq.num_frames)

    light_dirs: list[np.ndarray] = []
    ambient_ratios: list[float] = []

    for t in range(T):
        fgpp_frame = fgpp_seq.frames[t]

        # Need base_color and world_normal
        if fgpp_frame.base_color is None or fgpp_frame.world_normal is None:
            continue

        base_color = np.asarray(fgpp_frame.base_color, dtype=np.float64)
        normals = np.asarray(fgpp_frame.world_normal, dtype=np.float64)

        # Build foreground mask
        if fgpp_frame.cryptomatte_id is not None:
            fg_mask = fgpp_frame.cryptomatte_id > 0
        else:
            fg_mask = np.ones(
                (fgpp_frame.height, fgpp_frame.width), dtype=bool,
            )

        # Optional roughness filter (diffuse-dominated: roughness > 0.3)
        if fgpp_frame.roughness_metallic is not None:
            rm = np.asarray(
                fgpp_frame.roughness_metallic, dtype=np.float64,
            )
            fg_mask = fg_mask & (rm[..., 0] > 0.3)

        # Flatten valid pixels
        valid = fg_mask.ravel()
        if valid.sum() < 10:
            continue

        # Observed radiance (sRGB frame → linear)
        frame_linear = srgb_to_linear(frames_f[t])
        obs_lum = (
            0.2126 * frame_linear[..., 0]
            + 0.7152 * frame_linear[..., 1]
            + 0.0722 * frame_linear[..., 2]
        ).ravel()[valid]

        albedo_lum = (
            0.2126 * base_color[..., 0]
            + 0.7152 * base_color[..., 1]
            + 0.0722 * base_color[..., 2]
        ).ravel()[valid]
        albedo_lum = np.clip(albedo_lum, 1e-6, None)

        # Normalise: observed / albedo ≈ ambient + dot(n, light_dir)
        ratio = obs_lum / albedo_lum

        nx = normals[..., 0].ravel()[valid]
        ny = normals[..., 1].ravel()[valid]
        nz = normals[..., 2].ravel()[valid]

        # Design matrix: [1, nx, ny, nz]
        A = np.column_stack([np.ones_like(nx), nx, ny, nz])

        # Solve least squares: ratio ≈ A @ [a0, a1, a2, a3]
        result, _, _, _ = np.linalg.lstsq(A, ratio, rcond=None)
        a0 = result[0]
        light_vec = result[1:4]
        light_norm = float(np.linalg.norm(light_vec))

        if light_norm > 1e-6:
            light_dir = light_vec / light_norm
            light_dirs.append(light_dir)
            amb = float(a0 / (a0 + light_norm)) if (a0 + light_norm) > 0 else 0.2
            ambient_ratios.append(np.clip(amb, 0.0, 1.0))

    # Cross-frame median aggregation
    if light_dirs:
        median_dir = np.median(np.array(light_dirs), axis=0)
        norm = float(np.linalg.norm(median_dir))
        if norm > 1e-8:
            median_dir = median_dir / norm
        desc.key_light_dir_world = median_dir.tolist()

    if ambient_ratios:
        desc.ambient_ratio = float(np.median(ambient_ratios))

    return desc
