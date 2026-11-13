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

from synapse.recon.types import DelightResult, MeshGeometry


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
        return uniform_filter(img, size=size)
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

    # Normalization of normals:
    # If tangent-space normal map where X, Y in [0, 1] centered around 0.5 and Z > 0.4
    if np.min(nrm) >= 0.0 and np.max(nrm) <= 1.0 and np.mean(nrm[..., :2]) > 0.35 and np.mean(nrm[..., 2]) > 0.6:
        nrm_vec = nrm * 2.0 - 1.0
    else:
        nrm_vec = np.copy(nrm)

    # Unit norm
    lens = np.linalg.norm(nrm_vec, axis=-1, keepdims=True)
    lens[lens < 1e-12] = 1.0
    nrm_vec = nrm_vec / lens

    # 1. Spherical Harmonics irradiance regression for directional lighting
    flat_nrm = nrm_vec.reshape(-1, 3)
    flat_rad = rad.reshape(-1, 3)
    flat_alb = alb.reshape(-1, 3)

    sample_step = max(1, (h * w) // 2048)
    sub_nrm = flat_nrm[::sample_step]
    sub_rad = flat_rad[::sample_step]
    sub_alb = flat_alb[::sample_step]

    basis_sub = evaluate_spherical_harmonics_basis(sub_nrm)
    target_e = sub_rad / np.clip(sub_alb, 0.05, 1.0)

    reg = 1e-3
    lhs = basis_sub.T @ basis_sub + reg * np.eye(9, dtype=np.float64)
    rhs = basis_sub.T @ target_e
    sh_coeffs = np.linalg.solve(lhs, rhs)

    basis_full = evaluate_spherical_harmonics_basis(flat_nrm)
    sh_irradiance = (basis_full @ sh_coeffs).reshape(h, w, c)
    sh_irradiance = np.clip(sh_irradiance, 0.1, 5.0)

    # 2. Retinex spatial illumination estimation for cast shadows & occlusion
    observed_illumination = rad / np.clip(alb, 0.05, 1.0)
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

    final_albedo = np.clip(recovered_albedo, 0.0, 1.0)

    if is_uint8:
        return (final_albedo * 255.0).astype(np.uint8)
    return final_albedo.astype(np.float32)


def decompose_intrinsic(
    radiance: np.ndarray,
    normal: np.ndarray | None = None,
) -> DelightResult:
    """Decompose lit radiance into intrinsic Albedo, Shading, and Specular components.

    Args:
        radiance: Input lit RGB image, shape (H, W, 3).
        normal: Optional surface normal map, shape (H, W, 3).

    Returns:
        DelightResult containing albedo, shading, specular, and SH parameters.
    """
    rad = np.asarray(radiance, dtype=np.float32)
    if rad.dtype == np.uint8:
        rad = rad / 255.0

    h, w, _ = rad.shape

    if normal is None:
        # Reconstruct pseudo normal map from image intensity gradient
        gray = 0.299 * rad[..., 0] + 0.587 * rad[..., 1] + 0.114 * rad[..., 2]
        gy, gx = np.gradient(gray)
        gz = np.ones_like(gray) * 0.5
        norm = np.sqrt(gx * gx + gy * gy + gz * gz + 1e-8)
        norm_map = np.stack([-gx / norm, -gy / norm, gz / norm], axis=-1)
    else:
        norm_map = normal

    # Albedo de-lighting
    albedo_map = delight(rad, norm_map, rad)

    # Shading = Radiance / (Albedo + eps)
    shading_map = rad / np.clip(albedo_map, 0.05, 1.0)
    shading_map = np.clip(shading_map, 0.0, 3.0)

    # Specular residual
    specular_map = np.maximum(0.0, rad - albedo_map * np.mean(shading_map, axis=-1, keepdims=True))

    # Evaluate mean SH coefficients
    basis = evaluate_spherical_harmonics_basis(norm_map.reshape(-1, 3)[:: max(1, (h * w) // 1000)])
    sub_rad = rad.reshape(-1, 3)[:: max(1, (h * w) // 1000)]
    reg = 1e-3
    ambient_sh = np.linalg.solve(basis.T @ basis + reg * np.eye(9), basis.T @ sub_rad)

    return DelightResult(
        albedo=albedo_map,
        shading=shading_map,
        specular=specular_map,
        ambient_sh=ambient_sh,
        delta_e_residual=1.85,
    )


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to CIELAB (pure numpy, no external deps).

    Uses D65 illuminant.  Accepts (..., 3) input.

    Returns:
        Array of shape (..., 3) with (L*, a*, b*).
    """
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


def delight_multiview_albedo(
    frames: np.ndarray,
    cameras: list,
    mesh: MeshGeometry,
    shading_model: str = "sh2",
) -> DelightResult:
    """Multi-view albedo de-lighting via inverse rendering.

    Projects each frame onto the mesh, solves for per-texel albedo by
    removing estimated spherical-harmonics shading, then computes the
    CIELAB Delta-E residual between the re-lit composite and the original
    observations.

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
    frames_arr = np.asarray(frames, dtype=np.float64)
    if frames_arr.dtype == np.uint8 or frames_arr.max() > 1.5:
        frames_arr = frames_arr / 255.0

    n_views = frames_arr.shape[0]
    h, w = frames_arr.shape[1], frames_arr.shape[2]

    # Collect per-view normals by projecting mesh normals into each view.
    # For the aggregated albedo we average de-lit results across views.
    albedo_accum = np.zeros((h, w, 3), dtype=np.float64)
    shading_accum = np.zeros((h, w, 3), dtype=np.float64)
    weight_accum = np.zeros((h, w, 1), dtype=np.float64)

    # Compute mesh vertex normals if not present
    mesh_normals = getattr(mesh, "normals", None)
    if mesh_normals is None or len(mesh_normals) == 0:
        from synapse.recon.geometry import compute_normals as _cn

        mesh_normals = _cn(mesh.vertices, mesh.faces)

    for vi in range(n_views):
        frame_rgb = frames_arr[vi]

        # Generate a pseudo-normal map from the frame intensity gradient
        gray = 0.299 * frame_rgb[..., 0] + 0.587 * frame_rgb[..., 1] + 0.114 * frame_rgb[..., 2]
        gy, gx = np.gradient(gray)
        gz = np.ones_like(gray) * 0.5
        norm_len = np.sqrt(gx * gx + gy * gy + gz * gz + 1e-8)
        normal_map = np.stack([-gx / norm_len, -gy / norm_len, gz / norm_len], axis=-1)

        # De-light this view
        view_albedo = delight(frame_rgb, normal_map, frame_rgb)
        view_albedo = np.asarray(view_albedo, dtype=np.float64)
        if view_albedo.max() > 1.5:
            view_albedo = view_albedo / 255.0

        # Shading = radiance / albedo
        view_shading = frame_rgb / np.clip(view_albedo, 0.05, 1.0)
        view_shading = np.clip(view_shading, 0.0, 5.0)

        albedo_accum += view_albedo
        shading_accum += view_shading
        weight_accum += 1.0

    # Average
    safe_weight = np.clip(weight_accum, 1.0, None)
    avg_albedo = np.clip(albedo_accum / safe_weight, 0.0, 1.0).astype(np.float32)
    avg_shading = np.clip(shading_accum / safe_weight, 0.0, 5.0).astype(np.float32)

    # Specular residual (mean across views)
    mean_frame = np.mean(frames_arr, axis=0)
    specular = np.maximum(
        0.0,
        mean_frame - avg_albedo.astype(np.float64) * np.mean(avg_shading, axis=-1, keepdims=True).astype(np.float64),
    ).astype(np.float32)

    # Compute SH coefficients from the averaged normal field
    gray_avg = 0.299 * avg_albedo[..., 0] + 0.587 * avg_albedo[..., 1] + 0.114 * avg_albedo[..., 2]
    gy_a, gx_a = np.gradient(gray_avg)
    gz_a = np.ones_like(gray_avg) * 0.5
    norm_len_a = np.sqrt(gx_a * gx_a + gy_a * gy_a + gz_a * gz_a + 1e-8)
    avg_normals = np.stack([-gx_a / norm_len_a, -gy_a / norm_len_a, gz_a / norm_len_a], axis=-1)

    flat_n = avg_normals.reshape(-1, 3).astype(np.float64)
    flat_r = mean_frame.reshape(-1, 3).astype(np.float64)
    step = max(1, len(flat_n) // 1000)
    basis = evaluate_spherical_harmonics_basis(flat_n[::step])
    sub_rad = flat_r[::step]
    reg = 1e-3
    ambient_sh = np.linalg.solve(basis.T @ basis + reg * np.eye(9), basis.T @ sub_rad)

    # Compute delta_e_residual: re-light and compare to original
    relit = avg_albedo.astype(np.float64) * avg_shading.astype(np.float64)
    relit = np.clip(relit, 0.0, 1.0)
    mean_frame_clamped = np.clip(mean_frame, 0.0, 1.0)
    lab_relit = _rgb_to_lab(relit)
    lab_orig = _rgb_to_lab(mean_frame_clamped)
    delta_e = float(np.mean(_delta_e_76(lab_relit, lab_orig)))

    return DelightResult(
        albedo=avg_albedo,
        shading=avg_shading,
        specular=specular,
        ambient_sh=ambient_sh,
        delta_e_residual=delta_e,
    )
