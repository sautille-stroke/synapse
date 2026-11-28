"""Training losses for LoRA fine-tuning.

Provides differentiable PyTorch loss functions that preserve the full
``autograd`` gradient chain for back-propagation:

- ``FlowMatchingLoss``: Rectified flow velocity field MSE
- ``EdgePreservingLoss``: Sobel-gradient edge consistency
- ``NormalReconstructionLoss``: Cosine angular distance on normals
- ``DiffusionDPOLoss``: Flow-matching DPO preference loss (with reference model)

Legacy numpy-only helpers are retained as module-level functions for
backward-compatible CPU evaluation (used by existing tests and offline
metrics).

Heavy dependencies are lazy-imported inside classes/functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synapse.observability import get_probe

# ===================================================================
# Differentiable PyTorch loss classes
# ===================================================================


def _align_mask_for_loss(mask: Any, target: Any, torch_mod: Any) -> Any:
    """Broadcast or pool *mask* into a float32 tensor in [0, 1] matching *target*."""
    m = torch_mod.as_tensor(mask, dtype=torch_mod.float32, device=target.device)
    m = m.clamp(0.0, 1.0)
    if m.shape == target.shape:
        return m

    if m.ndim == target.ndim - 1:
        if (
            target.ndim >= 3
            and m.shape[0] == target.shape[0]
            and m.shape[1:] == target.shape[2:]
        ):
            m = m.unsqueeze(1)
        elif m.shape == target.shape[1:]:
            m = m.unsqueeze(0)
        else:
            m = m.unsqueeze(1)

    while m.ndim < target.ndim:
        m = m.unsqueeze(0)

    if m.ndim == 5 and m.shape[2:] != target.shape[2:]:
        m = torch_mod.nn.functional.adaptive_avg_pool3d(m, target.shape[2:])
    elif m.ndim == 4 and m.shape[2:] != target.shape[2:]:
        m = torch_mod.nn.functional.adaptive_avg_pool2d(m, target.shape[2:])
    elif m.ndim == 3 and m.shape[1:] != target.shape[1:]:
        m = torch_mod.nn.functional.adaptive_avg_pool2d(
            m.unsqueeze(1), target.shape[1:]
        ).squeeze(1)

    return m.clamp(0.0, 1.0)


class FlowMatchingLoss:
    """Rectified flow velocity field MSE (differentiable).

    Computes::

        E[||v_pred - (x_1 - x_0)||_2^2]

    All ops use pure ``torch`` tensor math — no ``np.asarray``
    graph breaks.
    """

    def __call__(
        self,
        v_pred: Any,
        x_1: Any,
        x_0: Any,
        mask: Any | None = None,
        background_weight: float = 0.0,
    ) -> Any:
        """Compute flow-matching MSE loss with optional spatial mask weighting.

        Args:
            v_pred: Predicted velocity field (torch.Tensor).
            x_1: Target sample (noise-free data), same shape.
            x_0: Source sample (pure noise), same shape.
            mask: Optional foreground spatial/spatiotemporal mask in ``[0, 1]``.
            background_weight: Weight multiplier assigned to background
                regions where ``mask == 0`` (default ``0.0``).

        Returns:
            Scalar torch.Tensor loss (retains grad).
        """
        import torch

        v_pred = torch.as_tensor(v_pred).float()
        x_1 = torch.as_tensor(x_1).float()
        x_0 = torch.as_tensor(x_0).float()
        target = x_1 - x_0
        diff_sq = (v_pred - target) ** 2
        if mask is None:
            return torch.mean(diff_sq)

        aligned = _align_mask_for_loss(mask, diff_sq, torch)
        bg_w = max(float(background_weight), 0.0)
        weights = (aligned * (1.0 - bg_w) + bg_w).expand_as(diff_sq)
        denom = weights.sum()
        weighted_mean = (diff_sq * weights).sum() / denom.clamp(min=1e-8)
        zero_guard = (diff_sq * 0.0).sum()
        return torch.where(denom > 1e-8, weighted_mean, zero_guard)


class EdgePreservingLoss:
    """Sobel-gradient edge alignment loss (differentiable).

    Computes 2D Sobel gradient magnitude on both inputs and returns
    the mean absolute difference. Fully differentiable through
    ``torch`` convolutions. Supports 2D [H, W], 3D [C, H, W], 4D [B, C, H, W],
    and 5D [B, C, T, H, W] tensors.
    """

    def __call__(
        self,
        z_pred: Any,
        depth_latent: Any,
    ) -> Any:
        """Compute edge-preserving loss.

        Args:
            z_pred: Predicted denoised latent ``[H, W]``, ``[C, H, W]``,
                ``[B, C, H, W]``, or ``[B, C, T, H, W]`` (torch.Tensor).
            depth_latent: Depth latent, same spatial/temporal shape.

        Returns:
            Scalar torch.Tensor loss (retains grad).
        """
        import torch
        import torch.nn.functional as torch_f

        z = torch.as_tensor(z_pred, dtype=torch.float32)
        d = torch.as_tensor(depth_latent, dtype=torch.float32)

        if z.ndim == 2:
            z_batch = z.unsqueeze(0).unsqueeze(0)
            d_batch = d.unsqueeze(0).unsqueeze(0)
        elif z.ndim == 3:
            z_batch = z.mean(dim=0, keepdim=True).unsqueeze(0)
            d_batch = d.mean(dim=0, keepdim=True).unsqueeze(0)
        elif z.ndim == 4:
            z_batch = z.mean(dim=1, keepdim=True)
            d_batch = d.mean(dim=1, keepdim=True)
        elif z.ndim == 5:
            b, c, t, h, w = z.shape
            z_flat = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            d_flat = d.permute(0, 2, 1, 3, 4).reshape(b * t, -1, h, w)
            z_batch = z_flat.mean(dim=1, keepdim=True)
            d_batch = d_flat.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor ndim {z.ndim} for EdgePreservingLoss")

        grad_z = self._sobel_magnitude_batch(z_batch, torch_f)
        grad_d = self._sobel_magnitude_batch(d_batch, torch_f)
        return torch.mean(torch.abs(grad_z - grad_d))

    @staticmethod
    def _sobel_magnitude_batch(
        x: Any,
        functional_mod: Any,
    ) -> Any:
        """Compute Sobel gradient magnitude on [N, 1, H, W] via conv2d."""
        import torch

        h, w = x.shape[-2], x.shape[-1]
        if h <= 2 or w <= 2:
            return torch.zeros_like(x)

        sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32,
                device=x.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32,
                device=x.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        x_pad = functional_mod.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = functional_mod.conv2d(x_pad, sobel_x, padding=0)
        gy = functional_mod.conv2d(x_pad, sobel_y, padding=0)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return mag

    @classmethod
    def _sobel_magnitude(
        cls,
        img: Any,
        functional_mod: Any,
    ) -> Any:
        """Backward-compatible helper for 2D inputs."""
        import torch

        img_t = torch.as_tensor(img, dtype=torch.float32)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
        elif img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        mag = cls._sobel_magnitude_batch(img_t, functional_mod)
        return mag.squeeze(0).squeeze(0)


_latent_geometry_probe_cls: type | None = None


def _get_latent_geometry_probe_class() -> type:
    """Return a ``torch.nn.Module`` LatentGeometryProbe class, created once."""
    global _latent_geometry_probe_cls
    if _latent_geometry_probe_cls is not None:
        return _latent_geometry_probe_cls

    import torch

    class _LatentGeometryProbeImpl(torch.nn.Module):
        """Lightweight 1x1x1 convolutional probe projecting 16-channel DiT latent
        velocity to geometric fields (1-ch depth or 3-ch normal).

        Subclasses ``nn.Module`` so it can be registered with AdamW and
        participates in ``named_parameters()`` / ``state_dict()``."""

        def __init__(self, in_channels: int = 16, out_channels: int = 3) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self._conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

        def forward(self, z: Any) -> Any:
            return self._conv(z)

    _latent_geometry_probe_cls = _LatentGeometryProbeImpl
    return _latent_geometry_probe_cls


class LatentGeometryProbe:
    """Factory wrapper for the lazy-imported ``nn.Module`` LatentGeometryProbe.

    Calling ``LatentGeometryProbe(...)`` returns an ``nn.Module`` instance
    so it can be registered with an optimizer, while keeping ``torch``
    lazy-imported."""

    def __new__(cls, in_channels: int = 16, out_channels: int = 3) -> Any:
        klass = _get_latent_geometry_probe_class()
        return klass(in_channels, out_channels)


class NormalReconstructionLoss:
    """Cosine angular distance on normals (differentiable).

    Computes::

        L_normal = mean(1 - <n_pred, n_gt>) over masked region.
    """

    def __call__(
        self,
        n_pred: Any,
        n_gt: Any,
        mask: Any | None = None,
    ) -> Any:
        """Compute normal reconstruction loss.

        Args:
            n_pred: Predicted normal field ``[H, W, 3]`` or ``[3, H, W]``.
            n_gt: Ground-truth normal field, same shape.
            mask: Optional ``[H, W]`` foreground mask.

        Returns:
            Scalar torch.Tensor loss (retains grad).
        """
        import torch

        n_p = torch.as_tensor(n_pred, dtype=torch.float32)
        n_g = torch.as_tensor(n_gt, dtype=torch.float32)

        # Handle various tensor layouts: [3, H, W], [B, 3, H, W], [B, C, T, H, W]
        is_5d = n_p.ndim == 5
        if is_5d:
            # Video: [B, C, T, H, W] → slice [:3] BEFORE permute/reshape so C > 3 never crashes
            if n_p.shape[1] > 3:
                n_p = n_p[:, :3]
            if n_g.ndim == 5 and n_g.shape[1] > 3:
                n_g = n_g[:, :3]
            b, c_p, t, h, w = n_p.shape
            n_p = n_p.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c_p)
            n_g = n_g.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, n_g.shape[1])
        elif n_p.ndim == 4:
            # Batch: [B, C, H, W] → [B, H, W, C]
            n_p = n_p.permute(0, 2, 3, 1)
            n_g = n_g.permute(0, 2, 3, 1)
        elif n_p.ndim == 3 and n_p.shape[0] == 3:
            # If channel-first [3, H, W], transpose to [H, W, 3]
            n_p = n_p.permute(1, 2, 0)
        if n_g.ndim == 3 and n_g.shape[0] == 3:
            n_g = n_g.permute(1, 2, 0)

        # Ensure only 3-channel normals are used
        if n_p.shape[-1] > 3:
            n_p = n_p[..., :3]
        if n_g.shape[-1] > 3:
            n_g = n_g[..., :3]

        # L2-normalise
        eps = 1e-8
        n_p = n_p / (torch.linalg.norm(n_p, dim=-1, keepdim=True) + eps)
        n_g = n_g / (torch.linalg.norm(n_g, dim=-1, keepdim=True) + eps)

        cos_sim = torch.sum(n_p * n_g, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        loss_map = 1.0 - cos_sim

        if mask is not None:
            fg = torch.as_tensor(mask, dtype=torch.float32, device=loss_map.device) > 0
            if is_5d:
                if fg.ndim == 5 and fg.shape[1] == 1:
                    fg = fg.squeeze(1)
                if fg.ndim == 4:
                    fg = fg.reshape(-1, fg.shape[-2], fg.shape[-1])
            elif fg.ndim == 4 and fg.shape[1] == 1 and loss_map.ndim == 3:
                fg = fg.squeeze(1)
            if fg.any():
                return torch.mean(loss_map[fg])
            return (n_p * 0.0).sum()

        return torch.mean(loss_map)


class DiffusionDPOLoss:
    """Flow-matching DPO preference loss (differentiable).

    Computes the Diffusion-DPO objective (Wallace et al., CVPR 2024)::

        -log(sigmoid(-beta * ((err_theta_w - err_ref_w)
                              - (err_theta_l - err_ref_l))))

    Fully differentiable through ``torch.nn.functional.softplus``.
    """

    def __init__(self, beta: float = 500.0) -> None:
        self.beta = beta

    def __call__(
        self,
        v_theta_w: Any,
        v_theta_l: Any,
        v_ref_w: Any,
        v_ref_l: Any,
        target_w: Any,
        target_l: Any,
        step: int | None = None,
    ) -> Any:
        """Compute DPO loss.

        Args:
            v_theta_w: Policy prediction on chosen sample.
            v_theta_l: Policy prediction on rejected sample.
            v_ref_w: Reference prediction on chosen sample.
            v_ref_l: Reference prediction on rejected sample.
            target_w: Ground-truth velocity for chosen.
            target_l: Ground-truth velocity for rejected.

        Returns:
            Scalar torch.Tensor loss (retains grad).
        """
        import torch

        v_tw = torch.as_tensor(v_theta_w).float()
        v_tl = torch.as_tensor(v_theta_l).float()
        v_rw = torch.as_tensor(v_ref_w).float()
        v_rl = torch.as_tensor(v_ref_l).float()
        tw = torch.as_tensor(target_w).float()
        tl = torch.as_tensor(target_l).float()

        # Per-sample reduction over non-batch dimensions
        if v_tw.ndim > 1:
            dims = tuple(range(1, v_tw.ndim))
            err_theta_w = torch.mean((v_tw - tw) ** 2, dim=dims)
            err_theta_l = torch.mean((v_tl - tl) ** 2, dim=dims)
            err_ref_w = torch.mean((v_rw - tw) ** 2, dim=dims)
            err_ref_l = torch.mean((v_rl - tl) ** 2, dim=dims)
        else:
            err_theta_w = torch.mean((v_tw - tw) ** 2)
            err_theta_l = torch.mean((v_tl - tl) ** 2)
            err_ref_w = torch.mean((v_rw - tw) ** 2)
            err_ref_l = torch.mean((v_rl - tl) ** 2)

        diff_w = err_theta_w - err_ref_w
        diff_l = err_theta_l - err_ref_l
        logit = -self.beta * (diff_w - diff_l)

        losses = torch.nn.functional.softplus(-logit)
        result = torch.mean(losses)

        # Log DPO loss components to Traceboard
        probe = get_probe()
        if probe is not None and probe.active:
            probe.log_scalar(
                "train/dpo_preferred_err",
                float(torch.mean(err_theta_w).item()),
                step=step,
            )
            probe.log_scalar(
                "train/dpo_rejected_err",
                float(torch.mean(err_theta_l).item()),
                step=step,
            )

        return result


# ===================================================================
# Legacy numpy-only functions (backward-compatible CPU evaluation)
# ===================================================================


def flow_matching_loss(
    v_pred: np.ndarray,
    x_1: np.ndarray,
    x_0: np.ndarray,
    mask: np.ndarray | None = None,
    background_weight: float = 0.0,
) -> float:
    """Rectified flow velocity field mean squared error (numpy).

    Computes::

        E[w * ||v_pred - (x_1 - x_0)||_2^2]

    where ``w = 1`` inside *mask* and ``w = background_weight`` outside.
    """
    v_pred = np.asarray(v_pred, dtype=np.float64)
    x_1 = np.asarray(x_1, dtype=np.float64)
    x_0 = np.asarray(x_0, dtype=np.float64)

    target = x_1 - x_0
    diff_sq = (v_pred - target) ** 2
    if mask is None:
        return float(np.mean(diff_sq))

    m = np.clip(np.asarray(mask, dtype=np.float64), 0.0, 1.0)
    if m.ndim == diff_sq.ndim - 1:
        if diff_sq.ndim >= 3 and m.shape[0] == diff_sq.shape[0] and m.shape[1:] == diff_sq.shape[2:]:
            m = np.expand_dims(m, axis=1)
        elif m.shape == diff_sq.shape[1:]:
            m = np.expand_dims(m, axis=0)
    while m.ndim < diff_sq.ndim:
        m = np.expand_dims(m, axis=0)
    bg_w = max(float(background_weight), 0.0)
    weights = np.broadcast_to(m * (1.0 - bg_w) + bg_w, diff_sq.shape)
    denom = float(weights.sum())
    if denom <= 1e-8:
        return 0.0
    return float((diff_sq * weights).sum() / denom)


def latent_edge_alignment_loss(
    z_pred: np.ndarray,
    depth_latent: np.ndarray,
) -> float:
    """Sobel-gradient edge alignment between prediction and depth (numpy).

    Computes 2D Sobel gradient magnitude on both inputs and
    returns the mean absolute difference.

    Args:
        z_pred: Predicted denoised latent ``[H, W]`` or
            ``[C, H, W]`` (averaged across channels if 3D).
        depth_latent: Min-pool depth latent, same spatial shape.

    Returns:
        Scalar edge alignment loss value.
    """
    z = np.asarray(z_pred, dtype=np.float64)
    d = np.asarray(depth_latent, dtype=np.float64)

    # If 3D, average across channel dimension
    if z.ndim == 3:
        z = z.mean(axis=0)
    if d.ndim == 3:
        d = d.mean(axis=0)

    grad_z = _sobel_magnitude(z)
    grad_d = _sobel_magnitude(d)

    return float(np.mean(np.abs(grad_z - grad_d)))


def normal_reconstruction_loss(
    n_pred: np.ndarray,
    n_gt: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Cosine angular distance between predicted and ground-truth normals (numpy).

    Computes the per-pixel normal reconstruction loss::

        L_normal = mean(1 - <n_pred, n_gt>) over masked region.

    where both inputs are expected to be unit-length normal vectors
    from the FGPP ``WorldNormal.XYZ`` channels.

    If *mask* is provided, only foreground pixels (``mask > 0``) contribute
    to the average.  Handles both ``[H, W, C]`` and ``[C, H, W]`` layouts
    via auto-transpose.

    Locked hyperparameter HP-04 from PLAN-ALGO-TRAINING-DATA-PIPELINE.md.

    Args:
        n_pred: Predicted normal field, shape ``[H, W, 3]`` or
            ``[3, H, W]``.  Will be L2-normalised internally.
        n_gt: Ground-truth normal field from FGPP, same shape.
        mask: Optional foreground mask, shape ``[H, W]``.  When given,
            the loss is computed only over pixels where ``mask > 0``.

    Returns:
        Scalar cosine angular distance loss in ``[0, 2]``.
    """
    n_p = np.asarray(n_pred, dtype=np.float64)
    n_g = np.asarray(n_gt, dtype=np.float64)

    # If channel-first [3, H, W], transpose to [H, W, 3]
    if n_p.ndim == 3 and n_p.shape[0] == 3:
        n_p = np.transpose(n_p, (1, 2, 0))
    if n_g.ndim == 3 and n_g.shape[0] == 3:
        n_g = np.transpose(n_g, (1, 2, 0))

    # L2-normalise to unit vectors (avoid div-by-zero)
    eps = 1e-8
    n_p = n_p / (np.linalg.norm(n_p, axis=-1, keepdims=True) + eps)
    n_g = n_g / (np.linalg.norm(n_g, axis=-1, keepdims=True) + eps)

    # Cosine similarity per pixel, clamp to [-1, 1]
    cos_sim = np.sum(n_p * n_g, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Per-pixel loss
    loss_map = 1.0 - cos_sim

    # Apply foreground mask if provided
    if mask is not None:
        fg = np.asarray(mask, dtype=np.float64) > 0
        if fg.any():
            return float(np.mean(loss_map[fg]))
        return 0.0

    return float(np.mean(loss_map))


def diffusion_dpo_loss(
    v_theta_w: np.ndarray,
    v_theta_l: np.ndarray,
    v_ref_w: np.ndarray,
    v_ref_l: np.ndarray,
    beta: float = 500.0,
    target_w: np.ndarray | None = None,
    target_l: np.ndarray | None = None,
) -> float:
    """Flow-matching DPO preference loss (numpy, for offline evaluation).

    Computes the Diffusion-DPO objective (Wallace et al., CVPR 2024)::

        -log(sigmoid(-beta * ((err_theta_w - err_ref_w)
                              - (err_theta_l - err_ref_l))))

    where ``w`` = chosen (winner) and ``l`` = rejected (loser).
    When ``target_w`` and ``target_l`` are provided, ``err_theta_w = ||v_theta_w - target_w||^2``
    and ``err_ref_w = ||v_ref_w - target_w||^2`` (and analogously for ``l``),
    so that when ``v_theta == v_ref`` the loss is identically ``ln(2)``.
    When ``target_w`` / ``target_l`` are omitted, ``v_ref_w`` and ``v_ref_l``
    are treated as the ground-truth velocity targets ``v_target_w`` / ``v_target_l``
    (with ``err_ref_w = err_ref_l = 0``) for backward-compatible 4-argument calls.

    Args:
        v_theta_w: Policy model prediction on chosen sample.
        v_theta_l: Policy model prediction on rejected sample.
        v_ref_w: Reference model prediction on chosen sample (or target_w if target_w is None).
        v_ref_l: Reference model prediction on rejected sample (or target_l if target_l is None).
        beta: DPO temperature scaling factor.
        target_w: Ground-truth velocity ``x_1_w - x_0`` for chosen sample.
        target_l: Ground-truth velocity ``x_1_l - x_0`` for rejected sample.

    Returns:
        Scalar DPO loss value (non-negative).
    """
    v_tw = np.asarray(v_theta_w, dtype=np.float64)
    v_tl = np.asarray(v_theta_l, dtype=np.float64)
    v_rw = np.asarray(v_ref_w, dtype=np.float64)
    v_rl = np.asarray(v_ref_l, dtype=np.float64)

    if target_w is not None and target_l is not None:
        tw = np.asarray(target_w, dtype=np.float64)
        tl = np.asarray(target_l, dtype=np.float64)
        err_theta_w = float(np.mean((v_tw - tw) ** 2))
        err_theta_l = float(np.mean((v_tl - tl) ** 2))
        err_ref_w = float(np.mean((v_rw - tw) ** 2))
        err_ref_l = float(np.mean((v_rl - tl) ** 2))
        diff_w = err_theta_w - err_ref_w
        diff_l = err_theta_l - err_ref_l
        logit = -beta * (diff_w - diff_l)
    else:
        err_w = float(np.mean((v_rw - v_tw) ** 2))
        err_l = float(np.mean((v_rl - v_tl) ** 2))
        logit = beta * (err_l - err_w)

    # -log(sigmoid(logit)) = log(1 + exp(-logit))
    if logit < -20.0:
        return float(-logit)
    if logit > 20.0:
        return float(np.exp(-logit))
    return float(np.log(1.0 + np.exp(-logit)))


def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a 2D array."""
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    if w > 2 and h > 2:
        gx[1:-1, 1:-1] = (
            -img[:-2, :-2]
            + img[:-2, 2:]
            - 2 * img[1:-1, :-2]
            + 2 * img[1:-1, 2:]
            - img[2:, :-2]
            + img[2:, 2:]
        )
        gy[1:-1, 1:-1] = (
            -img[:-2, :-2]
            - 2 * img[:-2, 1:-1]
            - img[:-2, 2:]
            + img[2:, :-2]
            + 2 * img[2:, 1:-1]
            + img[2:, 2:]
        )

    return np.sqrt(gx * gx + gy * gy + 1e-8)  # type: ignore[no-any-return]
