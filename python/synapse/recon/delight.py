"""Intrinsic material de-lighting and albedo decomposition.

Removes cast shadows, directional key lighting, and ambient occlusion from
neural video frames using Spherical Harmonics (SH) irradiance estimation
and edge-preserving Retinex reflectance recovery.

.. note:: **synapse-original implementation**

    Despite being listed in the cross-repo dedup audit (Issue #34, area ⑥),
    ``@aether//spectra`` covers spectral rendering (wavelength sampling,
    CIE 1931, blackbody, RGB-to-Spectrum) but has **no** Spherical Harmonics
    evaluation.  This SH de-lighting module is inverse-rendering-specific
    and has no upstream equivalent.  Once ``fulgur`` ships SH evaluation
    utilities, the ``evaluate_spherical_harmonics_basis`` function should
    delegate to the native implementation.
"""

from __future__ import annotations

import numpy as np

from synapse.recon.types import CameraExtrinsics, DelightResult, MeshGeometry


def evaluate_spherical_harmonics_basis(normals: np.ndarray) -> np.ndarray:
    """Evaluate 2nd-order (9-term) real Spherical Harmonics basis functions.

    Args:
        normals: Unit surface normal vectors, shape (..., 3), (nx, ny, nz).

    Returns:
        Basis matrix of shape (..., 9).
    """
    n = np.asarray(normals, dtype=np.float64)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    n = n / norm

    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]

    # Ramamoorthi & Hanrahan 2001 representation
    c0 = 0.282095
    c1 = 0.488603
    c2 = 1.092548
    c3 = 0.315392
    c4 = 0.546274

    b0 = np.full_like(x, c0)
    b1 = c1 * y
    b2 = c1 * z
    b3 = c1 * x
    b4 = c2 * (x * y)
    b5 = c2 * (y * z)
    b6 = c3 * (3.0 * z * z - 1.0)
    b7 = c2 * (x * z)
    b8 = c4 * (x * x - y * y)

    basis = np.stack([b0, b1, b2, b3, b4, b5, b6, b7, b8], axis=-1)
    return basis


def _spatial_smooth(img: np.ndarray, radius: int = 3) -> np.ndarray:
    """Perform spatial smoothing for illumination estimation."""
    try:
        from scipy.ndimage import uniform_filter

        size = (2 * radius + 1, 2 * radius + 1, 1)
        return np.asarray(uniform_filter(img, size=size))
    except ImportError:
        # Pure numpy separable box filter
        res = np.copy(img)
        h, w, c = img.shape
        # Horizontal pass
        temp = np.zeros_like(img)
        for i in range(w):
            w_start = max(0, i - radius)
            w_end = min(w, i + radius + 1)
            temp[:, i, :] = np.mean(res[:, w_start:w_end, :], axis=1)
        # Vertical pass
        for j in range(h):
            h_start = max(0, j - radius)
            h_end = min(h, j + radius + 1)
            res[j, :, :] = np.mean(temp[h_start:h_end, :, :], axis=0)
        return res


def _normalize_normals(nrm: np.ndarray) -> np.ndarray:
    """Normalize a normal map to unit length, auto-detecting tangent-space encoding.

    If normals are in [0, 1] tangent-space encoding (X, Y centered around 0.5,
    Z > 0.4), remap to [-1, 1] world-space before normalizing.

    Args:
        nrm: Normal map, shape (..., 3).

    Returns:
        Unit-length normal vectors, same shape.
    """
    nrm_vec = np.copy(nrm)
    if (np.min(nrm) >= 0.0 and np.max(nrm) <= 1.0
            and np.mean(nrm[..., :2]) > 0.35 and np.mean(nrm[..., 2]) > 0.6):
        nrm_vec = nrm * 2.0 - 1.0

    lens = np.linalg.norm(nrm_vec, axis=-1, keepdims=True)
    lens[lens < 1e-12] = 1.0
    result: np.ndarray = nrm_vec / lens
    return result


def _build_chromaticity_mask(
    radiance: np.ndarray,
    *,
    lum_floor: float = 0.01,
    chroma_consistency_radius: int = 2,
) -> np.ndarray:
    """Build a boolean mask of pixels with locally consistent chromaticity.

    High-chromaticity-consistency regions have stable hue ratios, making them
    reliable for SH irradiance fitting (shadows change luminance but not hue
    for diffuse Lambertian surfaces).

    Args:
        radiance: Linear radiance image, shape (H, W, 3), float.
        lum_floor: Minimum luminance to consider a pixel valid.
        chroma_consistency_radius: Neighborhood radius for chromaticity
            variance computation.

    Returns:
        Boolean mask of shape (H, W).
    """
    lum = 0.2126 * radiance[..., 0] + 0.7152 * radiance[..., 1] + 0.0722 * radiance[..., 2]
    valid = lum > lum_floor

    # Chromaticity = channel / luminance
    safe_lum = np.clip(lum, lum_floor, None)[..., np.newaxis]
    chroma = radiance / safe_lum  # (H, W, 3)

    # Local variance of chromaticity (box filter)
    h, w = radiance.shape[:2]
    r = chroma_consistency_radius
    chroma_var = np.zeros((h, w), dtype=np.float64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = np.roll(np.roll(chroma, dy, axis=0), dx, axis=1)
            chroma_var += np.sum((shifted - chroma) ** 2, axis=-1)
    n_neighbors = (2 * r + 1) ** 2
    chroma_var /= max(n_neighbors, 1)

    # Adaptive threshold: median + 1.5 * MAD
    flat_var = chroma_var[valid]
    if len(flat_var) == 0:
        return valid
    med = float(np.median(flat_var))
    mad = float(np.median(np.abs(flat_var - med)))
    threshold = med + 1.5 * max(mad, 1e-6)

    return valid & (chroma_var < threshold)


def _solve_sh_irradiance(
    normals_flat: np.ndarray,
    luminance_flat: np.ndarray,
    mask_flat: np.ndarray,
    *,
    max_samples: int = 4096,
    reg: float = 1e-3,
) -> np.ndarray:
    """Solve 2nd-order SH irradiance coefficients from masked pixels.

    Args:
        normals_flat: Flat normal vectors, shape (N, 3).
        luminance_flat: Per-pixel luminance, shape (N,).
        mask_flat: Boolean mask of valid pixels, shape (N,).
        max_samples: Max pixels to subsample for the least-squares solve.
        reg: Tikhonov regularization weight.

    Returns:
        SH coefficients, shape (9,).
    """
    valid_idx = np.where(mask_flat)[0]
    if len(valid_idx) < 9:
        # Not enough samples — return uniform ambient
        coeffs = np.zeros(9, dtype=np.float64)
        coeffs[0] = float(np.mean(luminance_flat)) / 0.282095 if len(luminance_flat) > 0 else 1.0
        return coeffs

    # Subsample for efficiency
    step = max(1, len(valid_idx) // max_samples)
    sel = valid_idx[::step]

    basis = evaluate_spherical_harmonics_basis(normals_flat[sel])  # (M, 9)
    target = luminance_flat[sel]  # (M,)

    lhs = basis.T @ basis + reg * np.eye(9, dtype=np.float64)
    rhs = basis.T @ target
    return np.linalg.solve(lhs, rhs)


def delight(
    albedo: np.ndarray,
    normal: np.ndarray,
    radiance: np.ndarray,
) -> np.ndarray:
    """Remove baked directional lighting and shadows from radiance image.

    Combines 2nd-order Spherical Harmonics ambient irradiance estimation
    with spatial Retinex illumination decomposition to factor out shading
    while preserving high-frequency surface texture.

    Args:
        albedo: Prior diffuse albedo or initial base-color, shape (H, W, 3).
        normal: World or view normal map, shape (H, W, 3).
        radiance: Direct lit radiance image, shape (H, W, 3).

    Returns:
        De-lit diffuse albedo image with shadows and shading factored out,
        shape (H, W, 3), matching input dtype.
    """
    is_uint8 = albedo.dtype == np.uint8 or radiance.dtype == np.uint8

    alb = np.asarray(albedo, dtype=np.float64)
    nrm = np.asarray(normal, dtype=np.float64)
    rad = np.asarray(radiance, dtype=np.float64)

    if is_uint8:
        alb = alb / 255.0
        rad = rad / 255.0

    # Ensure shape (H, W, 3)
    if alb.ndim != 3 or alb.shape[2] != 3:
        return albedo

    h, w, c = alb.shape

    nrm_vec = _normalize_normals(nrm)

    # 1. Spherical Harmonics irradiance regression for directional lighting
    flat_nrm = nrm_vec.reshape(-1, 3)
    flat_rad = rad.reshape(-1, 3)
    flat_alb = alb.reshape(-1, 3)

    sample_step = max(1, (h * w) // 2048)
    sub_nrm = flat_nrm[::sample_step]
    sub_rad = flat_rad[::sample_step]
    sub_alb = flat_alb[::sample_step]

    basis_sub = evaluate_spherical_harmonics_basis(sub_nrm)
    target_e = sub_rad / np.clip(sub_alb, 0.05, None)

    reg = 1e-3
    lhs = basis_sub.T @ basis_sub + reg * np.eye(9, dtype=np.float64)
    rhs = basis_sub.T @ target_e
    sh_coeffs = np.linalg.solve(lhs, rhs)

    basis_full = evaluate_spherical_harmonics_basis(flat_nrm)
    sh_irradiance = (basis_full @ sh_coeffs).reshape(h, w, c)
    sh_irradiance = np.clip(sh_irradiance, 0.1, 5.0)

    # 2. Retinex spatial illumination estimation for cast shadows & occlusion
    observed_illumination = rad / np.clip(alb, 0.05, None)
    smooth_illumination = _spatial_smooth(observed_illumination, radius=max(2, min(h, w) // 8))
    smooth_illumination = np.clip(smooth_illumination, 0.1, 5.0)

    # 3. Combined total shading factor
    total_shading = 0.5 * sh_irradiance + 0.5 * smooth_illumination
    total_shading = np.clip(total_shading, 0.1, 5.0)

    # Recover clean albedo: A = Radiance / Shading
    recovered_albedo = rad / total_shading

    # Retinex scale calibration to match prior reference luminance
    target_lum = np.mean(alb, axis=(0, 1), keepdims=True)
    current_lum = np.mean(recovered_albedo, axis=(0, 1), keepdims=True)
    scale_factor = target_lum / np.clip(current_lum, 1e-4, None)
    scale_factor = np.clip(scale_factor, 0.5, 2.0)
    recovered_albedo = recovered_albedo * scale_factor

    if is_uint8:
        final_albedo = np.clip(recovered_albedo, 0.0, 1.0)
        return np.asarray((final_albedo * 255.0).astype(np.uint8))

    # HDR: preserve values > 1.0
    final_albedo = np.clip(recovered_albedo, 0.0, None)
    return np.asarray(final_albedo, dtype=np.float32)


def _pseudo_normal_map(image: np.ndarray) -> np.ndarray:
    """Generate a pseudo-normal map from image intensity gradients.

    Args:
        image: RGB image, shape (H, W, 3), float.

    Returns:
        Normal map, shape (H, W, 3), unit-length.
    """
    gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    gy, gx = np.gradient(gray)
    gz = np.ones_like(gray) * 0.5
    norm_len = np.sqrt(gx * gx + gy * gy + gz * gz + 1e-8)
    return np.stack([-gx / norm_len, -gy / norm_len, gz / norm_len], axis=-1)


def decompose_intrinsic(
    radiance: np.ndarray,
    normal: np.ndarray | None = None,
) -> DelightResult:
    """Decompose lit radiance into intrinsic Albedo, Shading, and Specular.

    Uses pixel normals and luminance under a local high-chromaticity
    consistency mask to solve 2nd-order SH irradiance coefficients ``c_SH``.
    Estimates global smooth shading field ``E(x) = clip(Y(n)^T c_SH, eps,
    E_max)``.  Decomposes intrinsic albedo ``rho_hat = I / E``.

    Only uint8 inputs are divided by 255.0; float32/float64 HDR values
    above 1.0 are preserved (not erroneously scaled down).

    Args:
        radiance: Input lit RGB image, shape (H, W, 3).
        normal: Optional surface normal map, shape (H, W, 3).

    Returns:
        DelightResult containing albedo, shading, specular, and SH parameters.
    """
    is_uint8 = np.asarray(radiance).dtype == np.uint8
    rad = np.asarray(radiance, dtype=np.float64)
    if is_uint8:
        rad = rad / 255.0

    # Do NOT divide float32/float64 HDR by 255.0 — preserve linear range.

    h, w, _ = rad.shape

    if normal is None:
        norm_map = _pseudo_normal_map(rad)
    else:
        norm_map = np.asarray(normal, dtype=np.float64)

    nrm_vec = _normalize_normals(norm_map)

    # --- SH irradiance solve under chromaticity consistency mask ---
    # Build chromaticity consistency mask
    chroma_mask = _build_chromaticity_mask(rad)

    flat_nrm = nrm_vec.reshape(-1, 3)
    flat_mask = chroma_mask.reshape(-1)

    # Step 1: Compute luminance
    lum = 0.2126 * rad[..., 0] + 0.7152 * rad[..., 1] + 0.0722 * rad[..., 2]
    flat_lum = lum.reshape(-1)

    # Step 2: Estimate shading field E(n) using SH fit on luminance.
    # Under constant-albedo assumption, L(x) = rho_L * E(n(x)),
    # so E ∝ L. We normalize by median luminance to get E_hat ≈ E / E_mean.
    median_lum = float(np.median(flat_lum[flat_mask]) if np.any(flat_mask) else np.median(flat_lum))
    median_lum = max(median_lum, 1e-6)
    normalized_lum = flat_lum / median_lum

    # Fit SH to normalized luminance (captures the shading pattern)
    sh_lum_coeffs = _solve_sh_irradiance(flat_nrm, normalized_lum, flat_mask)

    # Evaluate SH shading field
    basis_full = evaluate_spherical_harmonics_basis(flat_nrm)
    sh_shading_normalized = (basis_full @ sh_lum_coeffs).reshape(h, w)

    # Scale back to absolute irradiance
    eps = 0.02
    e_max = 20.0
    sh_shading = np.clip(sh_shading_normalized * median_lum, eps, e_max)

    # Step 3: Retinex spatial smoothing for residual shading (cast shadows,
    # sharp terminators) that SH band-limiting cannot capture.
    # Compute residual luminance ratio after SH division
    albedo_sh_lum = lum / sh_shading  # initial SH-based albedo luminance
    # Smooth the initial albedo estimate to separate texture from residual shading
    albedo_sh_smooth = _spatial_smooth(
        albedo_sh_lum[..., np.newaxis], radius=max(2, min(h, w) // 8),
    )[..., 0]
    albedo_sh_smooth = np.clip(albedo_sh_smooth, 1e-4, None)
    # Residual shading = smooth(albedo_sh) / albedo_sh → inverted residual
    # Better: total shading = lum / smooth_albedo
    total_shading_lum = lum / albedo_sh_smooth
    total_shading_lum = np.clip(total_shading_lum, eps, e_max)

    # Expand to 3 channels (achromatic shading)
    shading_map = np.stack([total_shading_lum] * 3, axis=-1)

    # Compute per-channel SH coefficients for the ambient_sh output
    sh_rgb_coeffs = np.zeros((9, 3), dtype=np.float64)
    for ch in range(3):
        ch_lum = rad[..., ch]
        ch_albedo = ch_lum / total_shading_lum
        ch_albedo_smooth = _spatial_smooth(
            ch_albedo[..., np.newaxis], radius=max(2, min(h, w) // 8),
        )[..., 0]
        ch_albedo_smooth = np.clip(ch_albedo_smooth, 1e-4, None)
        ch_shading = ch_lum / ch_albedo_smooth
        sh_rgb_coeffs[:, ch] = _solve_sh_irradiance(
            flat_nrm, ch_shading.reshape(-1), flat_mask, max_samples=2048,
        )

    # Decompose intrinsic albedo: rho_hat = I / E
    albedo_map = rad / shading_map

    # For LDR content, clip albedo to [0, 1]; for HDR, allow > 1.0
    if is_uint8:
        albedo_map = np.clip(albedo_map, 0.0, 1.0)
    else:
        albedo_map = np.clip(albedo_map, 0.0, None)
    albedo_map = albedo_map.astype(np.float32)

    # Normalize shading for output (keep relative scale)
    shading_out = np.clip(shading_map, 0.0, e_max).astype(np.float32)

    # Specular residual: non-negative difference between observed and relit
    relit = albedo_map.astype(np.float64) * shading_map
    specular_map = np.maximum(0.0, rad - relit).astype(np.float32)

    # Compute delta_e_residual: re-light and compare to original (CIEDE2000)
    relit_clamped = np.clip(relit, 0.0, 1.0)
    rad_clamped = np.clip(rad, 0.0, 1.0)
    lab_relit = _rgb_to_lab(relit_clamped, is_linear=not is_uint8)
    lab_orig = _rgb_to_lab(rad_clamped, is_linear=not is_uint8)
    delta_e = float(np.mean(_delta_e_00(lab_relit, lab_orig)))

    return DelightResult(
        albedo=albedo_map,
        shading=shading_out,
        specular=specular_map,
        ambient_sh=sh_rgb_coeffs,
        delta_e_residual=delta_e,
    )


def _rgb_to_lab(rgb: np.ndarray, is_linear: bool = False) -> np.ndarray:
    """Convert RGB [0,1] to CIELAB (pure numpy, no external deps).

    Uses D65 illuminant. Accepts (..., 3) input.

    Args:
        rgb: Color array, values in [0, 1].
        is_linear: If True, input is already linear (skips sRGB expansion).

    Returns:
        Array of shape (..., 3) with (L*, a*, b*).
    """
    if is_linear:
        lin = np.asarray(rgb, dtype=np.float64)
    else:
        # sRGB → linear
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # linear sRGB → XYZ (D65)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # D65 reference white
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx = x / xn
    fy = y / yn
    fz = z / zn

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    fx = np.where(fx > eps, np.cbrt(fx), (kappa * fx + 16.0) / 116.0)
    fy = np.where(fy > eps, np.cbrt(fy), (kappa * fy + 16.0) / 116.0)
    fz = np.where(fz > eps, np.cbrt(fz), (kappa * fz + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return np.stack([L, a_star, b_star], axis=-1)


def _delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 Delta-E distance (Euclidean in Lab space)."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def _delta_e_00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE Delta-E 2000 distance (CIEDE2000).

    Simplified but faithful implementation of the CIEDE2000 formula.
    Accepts (..., 3) shaped arrays of (L*, a*, b*).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Step 1: Compute C' and h'
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    C_bar_7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    # Step 2: Compute delta L', delta C', delta H'
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.where(
        C1p * C2p == 0.0,
        0.0,
        np.where(
            np.abs(h2p - h1p) <= 180.0,
            h2p - h1p,
            np.where(h2p - h1p > 180.0, h2p - h1p - 360.0, h2p - h1p + 360.0),
        ),
    )
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    # Step 3: Weighting functions
    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    hp_bar = np.where(
        C1p * C2p == 0.0,
        h1p + h2p,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            (h1p + h2p) / 2.0,
            np.where(h1p + h2p < 360.0, (h1p + h2p + 360.0) / 2.0, (h1p + h2p - 360.0) / 2.0),
        ),
    )

    T = (1.0
         - 0.17 * np.cos(np.radians(hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0)))

    SL = 1.0 + 0.015 * (Lp_bar - 50.0)**2 / np.sqrt(20.0 + (Lp_bar - 50.0)**2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T

    Cp_bar_7 = Cp_bar**7
    RT = (-2.0 * np.sqrt(Cp_bar_7 / (Cp_bar_7 + 25.0**7))
          * np.sin(np.radians(60.0 * np.exp(-((hp_bar - 275.0) / 25.0)**2))))

    dE: np.ndarray = np.sqrt(
        (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return dE


def _project_mesh_to_view(
    vertices: np.ndarray,
    normals: np.ndarray,
    camera: CameraExtrinsics,
    h: int,
    w: int,
    faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project mesh into a camera view with Z-buffer occlusion testing.

    Args:
        vertices: Mesh vertices, shape (V, 3).
        normals: Per-vertex normals, shape (V, 3).
        camera: Camera extrinsics with .transform (4x4 world-to-cam or cam-to-world).
        h: Image height.
        w: Image width.
        faces: Optional mesh face indices (F, 3). If provided, rasterizes triangles
            with Z-buffering; otherwise scatters vertices.

    Returns:
        normal_map: Projected normal map, shape (H, W, 3).
        weight_map: cos_theta_v visibility weight, shape (H, W, 1).
        valid_mask: Boolean mask of pixels with valid projection, shape (H, W).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    nrms = np.asarray(normals, dtype=np.float64)

    # Transform to camera space
    T = np.asarray(camera.transform, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    cam_pos = np.asarray(camera.position, dtype=np.float64)

    # Transform vertices to camera space: p_cam = R @ p_world + t
    verts_cam = (R @ verts.T).T + t

    # Simple pinhole projection
    fx = fy = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    z = verts_cam[:, 2]

    # Project to pixel coordinates
    px = fx * verts_cam[:, 0] / np.clip(z, 1e-4, None) + cx
    py = fy * verts_cam[:, 1] / np.clip(z, 1e-4, None) + cy

    # Compute view direction for cos_theta_v weighting
    view_dirs = verts - cam_pos[np.newaxis, :]
    view_dist = np.linalg.norm(view_dirs, axis=-1, keepdims=True)
    view_dist[view_dist < 1e-8] = 1.0
    view_dirs = view_dirs / view_dist

    cos_theta = np.sum(nrms * (-view_dirs), axis=-1)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    # Transform normals to camera space
    nrms_cam = (R @ nrms.T).T

    normal_map = np.zeros((h, w, 3), dtype=np.float64)
    weight_map = np.zeros((h, w, 1), dtype=np.float64)
    depth_buf = np.full((h, w), np.inf, dtype=np.float64)

    if faces is not None and len(faces) > 0:
        f_arr = np.asarray(faces, dtype=np.int64)
        for tri in f_arr:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            z0, z1, z2 = float(z[i0]), float(z[i1]), float(z[i2])
            if z0 <= 0.01 or z1 <= 0.01 or z2 <= 0.01:
                continue

            x0, y0 = px[i0], py[i0]
            x1, y1 = px[i1], py[i1]
            x2, y2 = px[i2], py[i2]

            min_x = max(int(np.floor(min(x0, x1, x2))), 0)
            max_x = min(int(np.ceil(max(x0, x1, x2))), w - 1)
            min_y = max(int(np.floor(min(y0, y1, y2))), 0)
            max_y = min(int(np.ceil(max(y0, y1, y2))), h - 1)
            if min_x > max_x or min_y > max_y:
                continue

            det = float((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
            if abs(det) < 1e-8:
                continue

            for py_i in range(min_y, max_y + 1):
                for px_i in range(min_x, max_x + 1):
                    w0 = ((x1 - px_i) * (y2 - py_i) - (y1 - py_i) * (x2 - px_i)) / det
                    w1 = ((x2 - px_i) * (y0 - py_i) - (y2 - py_i) * (x0 - px_i)) / det
                    w2 = 1.0 - w0 - w1
                    if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                        zi = float(w0 * z0 + w1 * z1 + w2 * z2)
                        if zi < depth_buf[py_i, px_i]:
                            depth_buf[py_i, px_i] = zi
                            normal_map[py_i, px_i] = (
                                w0 * nrms_cam[i0] + w1 * nrms_cam[i1] + w2 * nrms_cam[i2]
                            )
                            weight_map[py_i, px_i, 0] = max(
                                0.0,
                                float(w0 * cos_theta[i0] + w1 * cos_theta[i1] + w2 * cos_theta[i2]),
                            )
    else:
        # Vertex scatter fallback
        in_front = z > 0.01
        px_int = px.astype(np.int32)
        py_int = py.astype(np.int32)
        valid = in_front & (px_int >= 0) & (px_int < w) & (py_int >= 0) & (py_int < h) & (cos_theta > 0.05)
        indices = np.where(valid)[0]

        for idx in indices:
            xi, yi, zi = int(px_int[idx]), int(py_int[idx]), float(z[idx])
            if zi < depth_buf[yi, xi]:
                depth_buf[yi, xi] = zi
                normal_map[yi, xi] = nrms_cam[idx]
                weight_map[yi, xi, 0] = cos_theta[idx]

    valid_mask = weight_map[..., 0] > 0.0
    nrm_lens = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    nrm_lens[nrm_lens < 1e-12] = 1.0
    normal_map = normal_map / nrm_lens

    return normal_map, weight_map, valid_mask


def delight_multiview_albedo(
    frames: np.ndarray,
    cameras: list[CameraExtrinsics],
    mesh: MeshGeometry,
    shading_model: str = "sh2",
) -> DelightResult:
    """Multi-view albedo de-lighting via inverse rendering.

    Projects each frame onto the mesh using camera extrinsics, solves for
    per-texel albedo by removing SH-estimated shading with view-weight
    (cos_theta_v visibility) fusion, then computes the CIEDE2000 Delta-E
    residual between the re-lit composite and the original observations.

    Args:
        frames: RGB video frames, shape ``(N, H, W, 3)``, float32 or uint8.
        cameras: Per-view camera extrinsics (list of
            :class:`~synapse.recon.types.CameraExtrinsics`).
        mesh: A :class:`~synapse.recon.types.MeshGeometry` instance.
        shading_model: ``"sh2"`` for 2nd-order (9-term) SH.

    Returns:
        :class:`DelightResult` with aggregated albedo, shading, specular,
        ambient SH coefficients, and ``delta_e_residual``.
    """
    is_uint8 = np.asarray(frames).dtype == np.uint8
    frames_arr = np.asarray(frames, dtype=np.float64)
    if is_uint8:
        frames_arr = frames_arr / 255.0

    # Do NOT divide float32/float64 HDR by 255.0 — preserve linear range.

    n_views = frames_arr.shape[0]
    h, w = frames_arr.shape[1], frames_arr.shape[2]

    # Compute mesh vertex normals if not present
    mesh_normals = getattr(mesh, "normals", None)
    if mesh_normals is None or (hasattr(mesh_normals, "__len__") and len(mesh_normals) == 0):
        from synapse.recon.geometry import compute_normals as _cn

        mesh_normals = _cn(mesh.vertices, mesh.faces)

    mesh_verts = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_nrms = np.asarray(mesh_normals, dtype=np.float64)

    # --- Per-view SH solve and weighted albedo accumulation ---
    albedo_accum = np.zeros((h, w, 3), dtype=np.float64)
    shading_accum = np.zeros((h, w, 3), dtype=np.float64)
    weight_accum = np.zeros((h, w, 1), dtype=np.float64)

    per_view_sh_coeffs = []

    for vi in range(n_views):
        frame_rgb = frames_arr[vi]
        cam = cameras[vi] if vi < len(cameras) else cameras[-1]

        # Project mesh normals into this view
        proj_normal, proj_weight, proj_valid = _project_mesh_to_view(
            mesh_verts, mesh_nrms, cam, h, w, faces=getattr(mesh, "faces", None),
        )

        # Use projected normals where available, fall back to pseudo-normals
        has_projection = np.sum(proj_valid) > (h * w * 0.01)  # >1% coverage
        if has_projection:
            normal_map = proj_normal
            view_weight = proj_weight
        else:
            normal_map = _pseudo_normal_map(frame_rgb)
            view_weight = np.ones((h, w, 1), dtype=np.float64)

        nrm_vec = _normalize_normals(normal_map)

        # Build chromaticity mask for SH fitting
        chroma_mask = _build_chromaticity_mask(frame_rgb)

        # Solve SH irradiance using normalized luminance (not raw radiance)
        flat_nrm = nrm_vec.reshape(-1, 3)
        flat_mask = chroma_mask.reshape(-1)

        # Compute luminance
        view_lum = (0.2126 * frame_rgb[..., 0] + 0.7152 * frame_rgb[..., 1]
                    + 0.0722 * frame_rgb[..., 2])
        flat_view_lum = view_lum.reshape(-1)
        median_view_lum = float(
            np.median(flat_view_lum[flat_mask]) if np.any(flat_mask)
            else np.median(flat_view_lum)
        )
        median_view_lum = max(median_view_lum, 1e-6)
        normalized_view_lum = flat_view_lum / median_view_lum

        # Fit SH to normalized luminance
        sh_lum_coeffs_view = _solve_sh_irradiance(
            flat_nrm, normalized_view_lum, flat_mask, max_samples=2048,
        )

        # SH coefficients stored as per-channel (broadcast to 3ch for output)
        sh_coeffs_view = np.column_stack([
            sh_lum_coeffs_view * median_view_lum,
            sh_lum_coeffs_view * median_view_lum,
            sh_lum_coeffs_view * median_view_lum,
        ])
        per_view_sh_coeffs.append(sh_coeffs_view)

        # Evaluate SH shading field for this view
        basis_full = evaluate_spherical_harmonics_basis(flat_nrm)
        sh_shading_scalar = (basis_full @ sh_lum_coeffs_view).reshape(h, w)
        sh_shading_abs = sh_shading_scalar * median_view_lum

        eps = 0.02
        e_max = 20.0
        sh_shading_clamped = np.clip(sh_shading_abs, eps, e_max)
        view_shading = np.stack([sh_shading_clamped] * 3, axis=-1)

        # Decompose albedo: rho = I / E
        view_albedo = frame_rgb / view_shading

        # Weighted accumulation (cos_theta_v from mesh projection)
        w_map = view_weight  # (H, W, 1)
        albedo_accum += view_albedo * w_map
        shading_accum += view_shading * w_map
        weight_accum += w_map

    # Weighted average
    safe_weight = np.clip(weight_accum, 1e-8, None)
    avg_albedo = albedo_accum / safe_weight

    if is_uint8:
        avg_albedo = np.clip(avg_albedo, 0.0, 1.0)
    else:
        avg_albedo = np.clip(avg_albedo, 0.0, None)
    avg_albedo = avg_albedo.astype(np.float32)

    avg_shading = np.clip(shading_accum / safe_weight, 0.0, 20.0).astype(np.float32)

    # Specular residual (mean across views)
    mean_frame = np.mean(frames_arr, axis=0)
    relit = avg_albedo.astype(np.float64) * avg_shading.astype(np.float64)
    specular = np.maximum(0.0, mean_frame - relit).astype(np.float32)

    # Average SH coefficients across views
    if per_view_sh_coeffs:
        ambient_sh = np.mean(per_view_sh_coeffs, axis=0)
    else:
        ambient_sh = np.zeros((9, 3), dtype=np.float64)

    # Compute CIEDE2000 delta_e_residual between re-lit and original
    relit_clamped = np.clip(relit, 0.0, 1.0)
    mean_frame_clamped = np.clip(mean_frame, 0.0, 1.0)
    lab_relit = _rgb_to_lab(relit_clamped)
    lab_orig = _rgb_to_lab(mean_frame_clamped)
    delta_e = float(np.mean(_delta_e_00(lab_relit, lab_orig)))

    return DelightResult(
        albedo=avg_albedo,
        shading=avg_shading,
        specular=specular,
        ambient_sh=ambient_sh,
        delta_e_residual=delta_e,
    )
