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
    return float(median_log_lum - np.log2(0.18))


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


def _estimate_contrast(rgb_linear: np.ndarray) -> float:
    """Estimate scene contrast from linear luminance.

    Measures standard deviation of log2 luminance relative to a reference standard (1.5 EV).
    For uniform or near-flat images, returns 1.0.
    """
    lum = (
        0.2126 * rgb_linear[..., 0]
        + 0.7152 * rgb_linear[..., 1]
        + 0.0722 * rgb_linear[..., 2]
    )
    pos = lum[lum > 1e-6]
    if pos.size < 10:
        return 1.0
    std = float(np.std(np.log2(pos)))
    if std < 1e-4:
        return 1.0
    contrast = std / 1.5
    return float(np.clip(contrast, 0.1, 5.0))


def _estimate_saturation(rgb_linear: np.ndarray) -> float:
    """Estimate scene colour saturation.

    Measures mean HSV saturation relative to reference natural image saturation (0.35).
    """
    max_c = np.max(rgb_linear, axis=-1)
    min_c = np.min(rgb_linear, axis=-1)
    denom = max_c + 1e-6
    sat = (max_c - min_c) / denom
    valid = max_c > 1e-4
    if not np.any(valid):
        return 0.0
    mean_sat = float(np.mean(sat[valid]))
    rel_sat = mean_sat / 0.35
    return float(np.clip(rel_sat, 0.0, 5.0))


def _solve_half_cosine_light_irls(
    normals: np.ndarray,
    ratio: np.ndarray,
    max_iter: int = 6,
) -> tuple[float, np.ndarray, float]:
    """Solve for key light vector and ambient ratio using IRLS.

    Models observed ratio y ≈ a0 + max(0, dot(n, L_vec)).
    Uses Huber loss weights to reject specular, shadow, and occlusion outliers.

    Args:
        normals: [M, 3] float64 unit surface normals.
        ratio: [M] float64 observed radiance / albedo ratio.
        max_iter: Maximum IRLS iterations.

    Returns:
        tuple of (ambient_ratio, unit_light_dir, light_norm).
    """
    m = len(ratio)
    if m < 4:
        return 0.2, np.array([0.0, 1.0, 0.0], dtype=np.float64), 0.0

    nx = normals[:, 0]
    ny = normals[:, 1]
    nz = normals[:, 2]
    A = np.column_stack([np.ones(m, dtype=np.float64), nx, ny, nz])

    # Initial OLS fit
    beta, _, _, _ = np.linalg.lstsq(A, ratio, rcond=None)

    for _ in range(max_iter):
        a0 = beta[0]
        L_vec = beta[1:4]
        dot_nl = nx * L_vec[0] + ny * L_vec[1] + nz * L_vec[2]
        pred = a0 + np.maximum(0.0, dot_nl)
        residual = ratio - pred

        med_res = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med_res)))
        scale = 1.4826 * mad + 1e-6

        # Huber residual centered at 0.0 (not median) for symmetric
        # treatment of backlit faces.
        u = np.abs(residual) / scale
        huber_w = np.where(u <= 1.345, 1.0, 1.345 / np.maximum(u, 1e-6))
        # Zero out normal Jacobian for backlit faces (dot_nl <= 0)
        # so they cannot steer the light direction estimate.
        facing_w = np.where(dot_nl <= 0.0, 0.0, 1.0)
        w = np.clip(huber_w * facing_w, 1e-4, 1.0)

        w_sqrt = np.sqrt(w)[:, np.newaxis]
        A_w = A * w_sqrt
        y_w = ratio * w_sqrt[:, 0]
        beta_new, _, _, _ = np.linalg.lstsq(A_w, y_w, rcond=None)
        if np.linalg.norm(beta_new - beta) < 1e-5:
            beta = beta_new
            break
        beta = beta_new

    a0_final = max(0.0, float(beta[0]))
    L_vec_final = beta[1:4]
    L_norm = float(np.linalg.norm(L_vec_final))
    if L_norm > 1e-6:
        unit_dir = L_vec_final / L_norm
        amb = a0_final / (a0_final + L_norm) if (a0_final + L_norm) > 0.0 else 0.2
        amb_ratio = float(np.clip(amb, 0.0, 1.0))
    else:
        unit_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        amb_ratio = 0.2

    return amb_ratio, unit_dir, L_norm


def _frechet_mean_s2(
    directions: list[np.ndarray],
    max_iter: int = 25,
) -> np.ndarray:
    """Compute the Fréchet (Karcher) mean on the unit 2-sphere S^2.

    Uses Riemannian gradient descent with logarithmic and exponential maps on S^2.

    Args:
        directions: List of [3] unit vectors.
        max_iter: Maximum iterations.

    Returns:
        [3] float64 unit vector representing the spherical center of mass.
    """
    if not directions:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if len(directions) == 1:
        d = directions[0].astype(np.float64)
        d_norm = float(np.linalg.norm(d))
        return d / d_norm if d_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

    unit_dirs: list[np.ndarray] = []
    for d_vec in directions:
        d_arr = np.asarray(d_vec, dtype=np.float64)
        arr_norm = float(np.linalg.norm(d_arr))
        if arr_norm > 1e-8:
            unit_dirs.append(d_arr / arr_norm)

    if not unit_dirs:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    sum_vec = np.sum(unit_dirs, axis=0)
    sum_norm = float(np.linalg.norm(sum_vec))
    mu = sum_vec / sum_norm if sum_norm > 1e-8 else unit_dirs[0].copy()

    for _ in range(max_iter):
        tangents: list[np.ndarray] = []
        for x in unit_dirs:
            cos_theta = float(np.clip(np.dot(mu, x), -1.0, 1.0))
            theta = float(np.arccos(cos_theta))
            if theta < 1e-7:
                tangents.append(np.zeros(3, dtype=np.float64))
            else:
                sin_theta = float(np.sin(theta))
                v = (theta / max(sin_theta, 1e-12)) * (x - mu * cos_theta)
                tangents.append(v)

        mean_tangent = np.mean(tangents, axis=0)
        step = float(np.linalg.norm(mean_tangent))
        if step < 1e-7:
            break

        v_unit = mean_tangent / step
        mu = np.cos(step) * mu + np.sin(step) * v_unit
        mu_norm = float(np.linalg.norm(mu))
        if mu_norm > 1e-12:
            mu /= mu_norm

    return mu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_style_optical(frames_srgb: np.ndarray) -> StyleDescriptor:
    """Extract visual style parameters from sRGB video frames (optical only).

    Estimates colour temperature, exposure, contrast, saturation, and
    grain using per-frame statistics aggregated via cross-frame median.

    Args:
        frames_srgb: sRGB video frames, shape (T, H, W, 3), dtype uint8
            or float32/float64 (already in [0,1] range).

    Returns:
        Populated ``StyleDescriptor`` with optical estimates.
    """
    frames = np.asarray(frames_srgb, dtype=np.float64)
    # Only divide by 255.0 for integer (uint8) input; float HDR input
    # is already in [0, 1] or wider range and must NOT be re-normalised.
    if np.issubdtype(frames_srgb.dtype, np.integer):
        frames = frames / 255.0
    T = frames.shape[0]

    cct_list: list[float] = []
    ev_list: list[float] = []
    grain_list: list[float] = []
    contrast_list: list[float] = []
    sat_list: list[float] = []

    for t in range(T):
        frame_linear = srgb_to_linear(frames[t])
        cct_list.append(_estimate_cct_mccamy(frame_linear))
        ev_list.append(_estimate_exposure_ev(frame_linear))
        grain_list.append(_estimate_grain_sigma(frame_linear))
        contrast_list.append(_estimate_contrast(frame_linear))
        sat_list.append(_estimate_saturation(frame_linear))

    return StyleDescriptor(
        color_temp_kelvin=float(np.median(cct_list)),
        exposure_ev=float(np.median(ev_list)),
        grain_sigma=float(np.median(grain_list)),
        contrast=float(np.median(contrast_list)),
        saturation=float(np.median(sat_list)),
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
    frames_f = np.asarray(frames_srgb, dtype=np.float64)
    # Only divide by 255.0 for integer (uint8) input; float HDR input
    # is already in [0, 1] or wider range.
    if np.issubdtype(frames_srgb.dtype, np.integer):
        frames_f = frames_f / 255.0

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

        # Filter backfacing and invalid normal pixels.
        # Only include pixels where (normal · -view_dir) > 0.05
        # to ensure the normal faces toward the camera.
        view_dir = fgpp_frame.camera.forward().astype(np.float64)
        neg_view_dir = -view_dir  # direction from surface toward camera
        # dot(normal, -view_dir) for each pixel
        ndotv = (
            normals[..., 0] * neg_view_dir[0]
            + normals[..., 1] * neg_view_dir[1]
            + normals[..., 2] * neg_view_dir[2]
        )
        # Also filter normals with near-zero magnitude (invalid)
        normal_len = np.sqrt(
            normals[..., 0] ** 2
            + normals[..., 1] ** 2
            + normals[..., 2] ** 2
        )
        fg_mask = fg_mask & (ndotv > 0.05) & (normal_len > 0.1)

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
        valid_normals = np.column_stack([nx, ny, nz])

        # Solve half-cosine lighting model via IRLS
        amb_ratio, light_dir, light_norm = _solve_half_cosine_light_irls(
            valid_normals, ratio
        )

        if light_norm > 1e-6:
            light_dirs.append(light_dir)
            ambient_ratios.append(amb_ratio)

    # Cross-frame spherical Fréchet mean aggregation
    if light_dirs:
        mean_dir = _frechet_mean_s2(light_dirs)
        desc.key_light_dir_world = mean_dir.tolist()

    if ambient_ratios:
        desc.ambient_ratio = float(np.median(ambient_ratios))

    return desc
