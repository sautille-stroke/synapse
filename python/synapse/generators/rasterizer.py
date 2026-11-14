"""Multi-channel software rasterizer for Synapse generators.

.. deprecated:: 1.0
    This module is a compatibility shim.  The canonical software rasterizer
    lives in :mod:`synapse.render_core.rasterizer`.  This module preserves the
    legacy ``SoftwareRasterizer(mesh, camera)`` interface while delegating
    the rasterization core to ``synapse.render_core.rasterizer``.

    ``generators.rasterizer`` was identified as an internal duplication of
    ``bench.rasterizer`` in the cross-repo dedup audit (354 vs 318 lines of
    nearly identical scanline rasterization).
    See https://github.com/sautille-stroke/synapse/issues/34

    Direct consumers should migrate to ``synapse.render_core.rasterizer``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from synapse.fgpp.frame import CameraParams
    from synapse.models.asset import ModelAsset

__all__ = [
    "RenderResult",
    "SoftwareRasterizer",
]

_DEPRECATION_MSG = (
    "synapse.generators.rasterizer is deprecated; use "
    "synapse.render_core.rasterizer instead.  "
    "See https://github.com/sautille-stroke/synapse/issues/34"
)


@dataclass
class RenderResult:
    """Multi-channel rendering output for a single frame.

    Compatibility wrapper — the canonical multi-channel output format is
    :class:`synapse.fgpp.frame.FGPPFrame` from ``synapse.bench.rasterizer``.
    """

    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] float32
    world_normal: np.ndarray  # [H, W, 3] float32 [-1, 1]
    motion_vector: np.ndarray  # [H, W, 2] float32 (dx, dy)
    mask: np.ndarray  # [H, W] uint8 (0 or 1)
    base_color: np.ndarray | None = None  # [H, W, 3] float32 [0, 1]


class SoftwareRasterizer:
    """Compatibility shim — delegates to :class:`synapse.render_core.rasterizer.SoftwareRasterizer`.

    Accepts ``ModelAsset`` objects and returns ``RenderResult``, converting
    internally to/from the render_core rasterizer's raw-array + ``FGPPFrame`` interface.

    .. deprecated:: 1.0
        Use ``synapse.render_core.rasterizer.SoftwareRasterizer`` directly.
    """

    def __init__(self, width: int = 512, height: int = 512):
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        from synapse.render_core.rasterizer import (
            SoftwareRasterizer as _BenchRasterizer,
        )

        self._delegate = _BenchRasterizer(width=width, height=height)
        self.width = width
        self.height = height

    def render(
        self,
        mesh: ModelAsset,
        camera: CameraParams,
        prev_camera: CameraParams | None = None,
    ) -> RenderResult:
        """Render a single frame yielding RGB, Depth, Normal, Motion Vector, and Mask.

        Unwraps the ``ModelAsset`` and delegates to
        ``synapse.bench.rasterizer.SoftwareRasterizer.render``.
        """
        fgpp_frame = self._delegate.render(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.normals,
            camera=camera,
            base_color=None,  # bench rasterizer uses uniform default
            entity_name=getattr(mesh, "name", "default"),
            prev_camera=prev_camera,
        )

        # Convert FGPPFrame → RenderResult for backward compatibility
        #
        # FGPPFrame stores HDR radiance as float16; RenderResult expects
        # LDR uint8 RGB.  Clamp to [0, 1] and quantize.
        h, w = self.height, self.width

        if fgpp_frame.direct_radiance is not None:
            radiance = np.asarray(fgpp_frame.direct_radiance, dtype=np.float32)
            rgb = np.clip(radiance * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        depth = fgpp_frame.depth if fgpp_frame.depth is not None else np.zeros((h, w), dtype=np.float32)

        if fgpp_frame.world_normal is not None:
            world_normal = np.asarray(fgpp_frame.world_normal, dtype=np.float32)
        else:
            world_normal = np.zeros((h, w, 3), dtype=np.float32)

        if fgpp_frame.motion_vector is not None:
            motion_vector = np.asarray(fgpp_frame.motion_vector, dtype=np.float32)
        else:
            motion_vector = np.zeros((h, w, 2), dtype=np.float32)

        if fgpp_frame.cryptomatte_id is not None:
            mask = (fgpp_frame.cryptomatte_id > 0).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        return RenderResult(
            rgb=rgb,
            depth=depth,
            world_normal=world_normal,
            motion_vector=motion_vector,
            mask=mask,
        )
