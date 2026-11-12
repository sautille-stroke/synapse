"""Visualization and colormapping utilities for FGPP channels."""

from __future__ import annotations

import numpy as np


def colorize_depth(depth: np.ndarray, mask: np.ndarray, colormap: int | None = None) -> np.ndarray:
    """Map single-channel float depth into a vibrant RGB colormap.

    Background pixels (where mask == 0) are kept dark.
    """
    import cv2  # lazy: heavy native dependency, not needed at import time

    if colormap is None:
        colormap = cv2.COLORMAP_TURBO

    h, w = depth.shape
    fg_mask = mask > 0

    if not np.any(fg_mask):
        return np.zeros((h, w, 3), dtype=np.uint8)

    d_min = float(depth[fg_mask].min())
    d_max = float(depth[fg_mask].max())

    if abs(d_max - d_min) < 1e-6:
        norm = np.zeros_like(depth)
    else:
        norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)

    # Invert so near objects are warmer/brighter
    norm = 1.0 - norm
    u8 = (norm * 255.0).astype(np.uint8)

    color_bgr = cv2.applyColorMap(u8, colormap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Blank out background
    color_rgb[~fg_mask] = (15, 18, 24)
    return color_rgb


def colorize_normal(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map world-space normals [-1, 1]³ to RGB [0, 255]³."""
    fg_mask = mask > 0
    norm_u8 = np.clip((normal + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    norm_u8[~fg_mask] = (15, 18, 24)
    return norm_u8


def colorize_flow(flow: np.ndarray, mask: np.ndarray, max_flow: float = 10.0) -> np.ndarray:
    """Map 2D optical flow (dx, dy) to an HSV color wheel representation."""
    import cv2  # lazy: heavy native dependency, not needed at import time

    h, w, _ = flow.shape
    fg_mask = mask > 0

    dx = flow[:, :, 0]
    dy = flow[:, :, 1]

    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Hue: direction (0-179 for OpenCV)
    hsv[:, :, 0] = (ang * 0.5).astype(np.uint8)
    # Saturation: 255
    hsv[:, :, 1] = 255
    # Value: scaled by magnitude
    val_norm = np.clip(mag / max(1e-3, max_flow), 0.0, 1.0)
    hsv[:, :, 2] = (val_norm * 255.0).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~fg_mask] = (15, 18, 24)
    return rgb


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask into a high-contrast 3-channel visualization."""
    m = np.where(mask > 0, 240, 20).astype(np.uint8)
    return np.stack([m] * 3, axis=-1)


def make_contact_sheet(
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    normal_rgb: np.ndarray,
    flow_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    label: str = "",
) -> np.ndarray:
    """Combine 5 channels into a single side-by-side composite comparison strip."""
    channels = [rgb, depth_rgb, normal_rgb, flow_rgb, mask_rgb]
    sheet = np.concatenate(channels, axis=1)

    if label:
        import cv2  # lazy: heavy native dependency, not needed at import time

        # Overlay subtle title
        cv2.putText(
            sheet,
            label,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return sheet
