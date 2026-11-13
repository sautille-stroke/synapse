# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""ClosedLoopRunner — 5-step inverse rendering closed loop (RFC 0062).

Implements the full forward-inverse-scalpel cycle:

  1. **Forward Guide** — build conditioning channels and generate a target
     video via :class:`~synapse.recon.loop.ForwardGuide`.
  2. **Target Video** — produce the neural video from conditioning
     (mock / real adapter).
  3. **Inverse Optimisation** — extract material maps from the generated
     video with :class:`~synapse.recon.loop.AnalyticUVInverseOptimizer`.
  4. **Aligned 3D Render** — re-render from optimised materials to get a
     reference frame for the scalpel stitcher.
  5. **Scalpel Patching** — use
     :class:`~synapse.refine.stitcher.ScalpelStitcher` to blend the
     optimised render back onto the original video with sub-pixel
     border matching.

All five steps run on CPU using numpy; heavy backends (mitsuba, torch)
are injected lazily and optional.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapse.recon.loop.forward_guide import ForwardGuide
from synapse.recon.loop.mitsuba_optimizer import (
    AnalyticUVInverseOptimizer,
    InverseOptimConfig,
    InverseOptimResult,
)
from synapse.recon.loop.texture_extractor import (
    PBRTextureSet,
    extract_and_pack_pbr_textures,
)
from synapse.recon.loop.types import ForwardGuideResult, LoopConfig
from synapse.refine.stitcher import ScalpelStitcher, StitchConfig, StitchReport
from synapse.refine.types import AnatomicalRegion, FusionMethod, RefinementResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ClosedLoopResult:
    """Result of a full closed-loop inverse rendering cycle.

    Attributes:
        output_frames: Final composited frames ``(T, H, W, 3)`` float32.
        forward_guide_result: Output of the forward guide step.
        inverse_result: Analytic UV optimisation output.
        pbr_textures: Extracted PBR texture set (albedo, normals, etc.).
        stitch_report: Scalpel stitcher quality diagnostics.
        border_ring_ciede2000: ΔE₀₀ at the stitch boundary (must ≤ 2.0).
        elapsed_seconds: Total wall-clock time for all 5 steps.
        step_timings: Per-step wall-clock time in seconds.
    """

    output_frames: np.ndarray
    forward_guide_result: ForwardGuideResult
    inverse_result: InverseOptimResult
    pbr_textures: PBRTextureSet
    stitch_report: StitchReport
    border_ring_ciede2000: float = 0.0
    elapsed_seconds: float = 0.0
    step_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (excludes large arrays)."""
        return {
            "num_frames": int(self.output_frames.shape[0]),
            "forward_phase_valid": self.forward_guide_result.phase_valid,
            "forward_preroll_count": self.forward_guide_result.preroll_count,
            "inverse_psnr_before_db": float(
                self.inverse_result.psnr_before_db,
            ),
            "inverse_psnr_after_db": float(
                self.inverse_result.psnr_after_db,
            ),
            "inverse_ciede2000": float(self.inverse_result.ciede2000_after),
            "inverse_converged": self.inverse_result.converged,
            "inverse_iterations": self.inverse_result.iterations_run,
            "pbr_albedo_shape": list(self.pbr_textures.albedo.shape),
            "stitch_mean_border_psnr_db": float(
                self.stitch_report.mean_border_psnr_db,
            ),
            "stitch_mean_border_ciede2000": float(
                self.stitch_report.mean_border_ciede2000,
            ),
            "border_ring_ciede2000": self.border_ring_ciede2000,
            "elapsed_seconds": self.elapsed_seconds,
            "step_timings": self.step_timings,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ClosedLoopRunner:
    """End-to-end 5-step inverse rendering closed loop.

    Args:
        loop_config: Configuration for the forward guide VAE constraints.
        optim_config: Configuration for the analytic UV inverse optimiser.
        stitch_config: Configuration for the scalpel stitcher.
    """

    def __init__(
        self,
        loop_config: LoopConfig | None = None,
        optim_config: InverseOptimConfig | None = None,
        stitch_config: StitchConfig | None = None,
    ) -> None:
        self._loop_config = loop_config or LoopConfig()
        self._optim_config = optim_config or InverseOptimConfig()
        self._stitch_config = stitch_config or StitchConfig()

    def run(
        self,
        source_frames: np.ndarray,
        normals: np.ndarray,
        uv_maps: np.ndarray,
        conditioning: dict[str, np.ndarray] | None = None,
        adapter: Any = None,
        n_production_frames: int | None = None,
    ) -> ClosedLoopResult:
        """Execute all 5 closed-loop steps.

        Args:
            source_frames: Original video ``(T, H, W, 3)`` float32 ACEScg.
            normals: Per-pixel world normals ``(T, H, W, 3)`` float32.
            uv_maps: Per-pixel UV coordinates ``(T, H, W, 2)`` float32.
            conditioning: FGPP channel dict for forward guide.  When
                ``None``, synthetic channels are generated from the
                source frames.
            adapter: Optional video adapter for step 2.  ``None`` uses
                the CPU mock pass-through.
            n_production_frames: Number of production frames for forward
                guide.  Defaults to ``source_frames.shape[0]``.

        Returns:
            :class:`ClosedLoopResult` with all diagnostics.
        """
        t_total_start = time.monotonic()
        step_timings: dict[str, float] = {}
        t_count = source_frames.shape[0]
        if n_production_frames is None:
            n_production_frames = t_count

        # ---------------------------------------------------------------
        # Step 1: Forward Guide — build conditioning and generation plan
        # ---------------------------------------------------------------
        t0 = time.monotonic()
        guide = ForwardGuide(self._loop_config)

        if conditioning is None:
            conditioning = self._build_synthetic_conditioning(source_frames)

        forward_result = guide.generate(
            conditioning=conditioning,
            adapter=adapter,
            n_production_frames=n_production_frames,
        )
        step_timings["step1_forward_guide"] = time.monotonic() - t0

        # ---------------------------------------------------------------
        # Step 2: Target Video — the forward result IS the target video
        # (generated by mock or real adapter inside step 1)
        # ---------------------------------------------------------------
        t1 = time.monotonic()
        target_frames = forward_result.frames
        step_timings["step2_target_video"] = time.monotonic() - t1

        # ---------------------------------------------------------------
        # Step 3: Inverse Optimisation — extract material from target
        # ---------------------------------------------------------------
        t2 = time.monotonic()
        optimizer = AnalyticUVInverseOptimizer(self._optim_config)

        # Trim normals and uv_maps to match target frame count
        n_target = target_frames.shape[0]
        normals_trimmed = normals[:n_target]
        uv_trimmed = uv_maps[:n_target]

        inverse_result = optimizer.optimize_material_from_video(
            frames=target_frames,
            normals=normals_trimmed,
            uv_maps=uv_trimmed,
        )
        step_timings["step3_inverse_optim"] = time.monotonic() - t2

        # ---------------------------------------------------------------
        # Step 4: Aligned 3D Render — re-render with optimised albedo
        # ---------------------------------------------------------------
        t3 = time.monotonic()
        # Use the optimised albedo to reconstruct frames via UV lookup
        rerendered = self._rerender_from_albedo(
            albedo_map=inverse_result.albedo_map,
            uv_maps=uv_trimmed,
            normals=normals_trimmed,
        )
        step_timings["step4_aligned_render"] = time.monotonic() - t3

        # ---------------------------------------------------------------
        # Step 5: Scalpel Patching — stitch re-rendered onto source
        # ---------------------------------------------------------------
        t4 = time.monotonic()
        stitcher = ScalpelStitcher(self._stitch_config)

        # Build per-frame RefinementResults for the stitcher
        source_trimmed = source_frames[:n_target]
        refinements = self._build_refinements(
            rerendered, source_trimmed,
        )

        output_frames, stitch_report = stitcher.stitch_sequence(
            source_trimmed, refinements,
        )

        # Extract PBR textures
        pbr = extract_and_pack_pbr_textures(
            albedo_map=inverse_result.albedo_map,
            normal_map=normals_trimmed[0] if n_target > 0 else None,
        )
        step_timings["step5_scalpel_patch"] = time.monotonic() - t4

        elapsed = time.monotonic() - t_total_start

        return ClosedLoopResult(
            output_frames=output_frames,
            forward_guide_result=forward_result,
            inverse_result=inverse_result,
            pbr_textures=pbr,
            stitch_report=stitch_report,
            border_ring_ciede2000=float(
                stitch_report.mean_border_ciede2000,
            ),
            elapsed_seconds=elapsed,
            step_timings=step_timings,
        )

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _build_synthetic_conditioning(
        frames: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build minimal FGPP conditioning from source frames.

        Creates depth-from-luminance, flat normals, and an entity mask.
        Uses the first frame only (2D spatial) to avoid the
        ``ForwardGuide._spatial_dims`` treating the temporal axis as
        a spatial dimension.
        """
        # Use only the first frame for conditioning (H, W) arrays
        first = frames[0]  # (H, W, 3)

        # Pseudo-depth from luminance
        lum = np.mean(first, axis=-1)  # (H, W)
        depth = (1.0 - lum).astype(np.float32)  # darker = closer

        # Flat normals pointing +Z
        h, w = first.shape[:2]
        nx = np.zeros((h, w), dtype=np.float32)
        ny = np.zeros((h, w), dtype=np.float32)
        nz = np.ones((h, w), dtype=np.float32)

        return {
            "CameraLinearDepth.Z": depth,
            "WorldNormal.X": nx,
            "WorldNormal.Y": ny,
            "WorldNormal.Z": nz,
        }

    @staticmethod
    def _rerender_from_albedo(
        albedo_map: np.ndarray,
        uv_maps: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        """Re-render frames by sampling the albedo map at UV coordinates.

        Applies simple Lambertian shading: ``color = albedo * max(N·L, 0)``
        where L is the fixed light direction ``(0, 0, 1)``.

        Args:
            albedo_map: ``(H_tex, W_tex, 3)`` float32 albedo texture.
            uv_maps: ``(T, H, W, 2)`` float32 UV coordinates in [0, 1].
            normals: ``(T, H, W, 3)`` float32 world-space normals.

        Returns:
            ``(T, H, W, 3)`` float32 re-rendered frames.
        """
        t_count, h, w, _ = normals.shape
        tex_h, tex_w = albedo_map.shape[:2]

        result = np.zeros((t_count, h, w, 3), dtype=np.float32)

        for t in range(t_count):
            # Sample UVs — nearest-neighbour for speed
            u = np.clip(uv_maps[t, :, :, 0], 0.0, 1.0)
            v = np.clip(uv_maps[t, :, :, 1], 0.0, 1.0)
            ix = np.clip(
                (u * (tex_w - 1)).astype(np.int32), 0, tex_w - 1,
            )
            iy = np.clip(
                (v * (tex_h - 1)).astype(np.int32), 0, tex_h - 1,
            )

            sampled_albedo = albedo_map[iy, ix]  # (H, W, 3)

            # Lambertian: N·L where L = (0, 0, 1)
            n_dot_l = np.clip(normals[t, :, :, 2], 0.0, 1.0)
            shading = n_dot_l[..., np.newaxis]  # (H, W, 1)

            result[t] = sampled_albedo * shading

        return result

    @staticmethod
    def _build_refinements(
        rerendered: np.ndarray,
        source: np.ndarray,
    ) -> list[list[RefinementResult]]:
        """Build per-frame RefinementResults covering the entire frame.

        Each frame gets a single FULL_BODY region covering the whole image.
        """
        t_count, h, w, _ = source.shape
        refinements: list[list[RefinementResult]] = []

        for t in range(t_count):
            ref = RefinementResult(
                refined_acescg=rerendered[t],
                region=AnatomicalRegion.FULL_BODY,
                fusion_method=FusionMethod.LAPLACIAN_FREQUENCY,
                quality_score=1.0,
                frame_index=t,
                bbox_xyxy=np.array(
                    [0, 0, w, h], dtype=np.float32,
                ),
            )
            refinements.append([ref])

        return refinements
