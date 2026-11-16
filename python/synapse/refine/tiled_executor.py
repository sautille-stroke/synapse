# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Tiled bypass executor for anatomical local refinement (RFC 0061).

Tiles are extracted from frames at ROI locations.  Tiles whose quality
score exceeds a health threshold are *bypassed* (skipped), avoiding
unnecessary refinement work.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from synapse.refine.types import RefinementROI


class TiledBypassExecutor:
    """Extracts tiles from frames, skips healthy ones, batches defective ones.

    A tile is considered *healthy* when its quality score (supplied by
    ``quality_fn`` or falling back to ``RefinementROI.confidence``) exceeds
    ``health_threshold``, meaning no refinement is needed.

    Args:
        tile_size: Side length of each square tile in pixels.
        health_threshold: Minimum quality / confidence for a tile to be
            considered healthy and bypassed.
    """

    def __init__(
        self,
        tile_size: int = 512,
        health_threshold: float = 0.8,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if not 0.0 <= health_threshold <= 1.0:
            raise ValueError(
                f"health_threshold must be in [0, 1], got {health_threshold}"
            )
        self._tile_size = tile_size
        self._health_threshold = health_threshold

    @property
    def tile_size(self) -> int:
        """Side length of tiles in pixels."""
        return self._tile_size

    @property
    def health_threshold(self) -> float:
        """Quality threshold above which a tile is bypassed."""
        return self._health_threshold

    def extract_and_filter(
        self,
        frames: np.ndarray,
        rois: list[RefinementROI],
        quality_fn: Callable[[np.ndarray], float] | None = None,
    ) -> tuple[list[RefinementROI], int]:
        """Filter ROIs, returning only those that need refinement.

        For each ROI the quality score is determined by calling
        ``quality_fn(roi.crop_acescg)`` when a quality function is provided,
        otherwise ``roi.confidence`` is used as a proxy.  ROIs with a score
        *strictly above* the health threshold are bypassed.

        Args:
            frames: Video tensor of shape ``(T, H, W, 3)`` in linear ACEScg.
                Currently used only for shape validation; per-ROI crops are
                already stored inside each :class:`RefinementROI`.
            rois: Detected ROIs to evaluate.
            quality_fn: Optional callable that scores an ACEScg crop and
                returns a float in ``[0, 1]``.

        Returns:
            ``(defective_rois, bypass_count)`` — the ROIs that *do* need
            refinement and the count of healthy ROIs that were bypassed.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"frames must have shape (T, H, W, 3), got {frames.shape}"
            )

        defective: list[RefinementROI] = []
        bypass_count = 0

        for roi in rois:
            crop = roi.crop_acescg
            h, w = crop.shape[:2]
            ts = self._tile_size

            if quality_fn is not None:
                if h > ts or w > ts:
                    tile_scores = []
                    for y0 in range(0, h, ts):
                        for x0 in range(0, w, ts):
                            y1 = min(y0 + ts, h)
                            x1 = min(x0 + ts, w)
                            tile_scores.append(quality_fn(crop[y0:y1, x0:x1]))
                    score = min(tile_scores) if tile_scores else 1.0
                else:
                    score = quality_fn(crop)
            else:
                score = roi.confidence

            if score > self._health_threshold:
                bypass_count += 1
            else:
                defective.append(roi)

        return defective, bypass_count

    def tile_and_execute(
        self,
        crop: np.ndarray,
        process_tile_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Process a large crop by tiling into tile_size chunks, executing process_tile_fn, and stitching."""
        h, w = crop.shape[:2]
        ts = self._tile_size
        if h <= ts and w <= ts:
            return process_tile_fn(crop)

        out = np.zeros_like(crop)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        stride = max(ts // 2, 1)
        for y0 in range(0, max(h - ts + 1, 1), stride):
            for x0 in range(0, max(w - ts + 1, 1), stride):
                y1 = min(y0 + ts, h)
                x1 = min(x0 + ts, w)
                tile = crop[y0:y1, x0:x1]
                processed = process_tile_fn(tile)
                out[y0:y1, x0:x1] += processed
                weight[y0:y1, x0:x1] += 1.0

        weight = np.maximum(weight, 1.0)
        return out / weight
