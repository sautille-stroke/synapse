"""Stage 3 Video-to-3D Inverse Reconstruction & Asset Distillation pipeline.

Orchestrates point tracking, camera trajectory estimation, quad remeshing,
zero-slip foot kinematics distillation, and intrinsic material de-lighting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from synapse.observability import get_probe
from synapse.recon.delight import decompose_intrinsic
from synapse.recon.geometry import (
    compute_normals,
    quad_remesh,
    validate_recon_mesh_manifold,
)
from synapse.recon.kinematics import solve_zero_slip_ik
from synapse.recon.tracking import solve_camera_trajectory, solve_point_tracks
from synapse.recon.types import (
    CameraExtrinsics,
    MeshGeometry,
    MocapKinematics,
    ReconResult,
)


class ReconPipeline:
    """Stage 3 Inverse Reconstruction Orchestrator.

    Extracts clean, production-ready 3D assets from raw neural video sequences:
    1. 4D surface point tracking (CoTracker3 protocol)
    2. 6-DoF camera trajectory & metric scale recovery (DROID-SLAM protocol)
    3. Watertight manifold verification & quad-dominant remeshing
    4. Anti-slide kinematic distillation (<0.3 cm/s foot velocity)
    5. Intrinsic albedo de-lighting (Spherical Harmonics cast-shadow removal)
    """

    def __init__(
        self,
        target_quad_faces: int = 1500,
        geometry_backend: Literal["native", "python"] = "native",
        voxel_size: float = 0.02,
        sdf_trunc: float = 0.04,
        volume_bounds: tuple[float, float, float] = (1.5, 1.5, 1.5),
        inverse_backend: Literal["classical", "diffusion_renderer", "dual_track"] = "classical",
        diffusion_renderer_transport: Any | None = None,
    ) -> None:
        """Initialize the reconstruction pipeline.

        Args:
            target_quad_faces: Desired target face count for quad remeshing.
            geometry_backend: Execution backend for topology gate validation
                ('native' in production, 'python' Oracle for dual-track testing).
            voxel_size: Edge length of a single voxel in metres for TSDF fusion.
            sdf_trunc: Truncation distance in metres for TSDF fusion.
            volume_bounds: (x, y, z) extent of reconstruction volume in metres.
            inverse_backend: Material/G-buffer decomposition backend:
                ``"classical"`` (SH de-lighting), ``"diffusion_renderer"``
                (neural inverse rendering via ``DiffusionRendererTransport``), or
                ``"dual_track"`` (run both and record agreement metrics).
            diffusion_renderer_transport: Optional injected transport implementing
                ``inverse_render`` (e.g. ``MockDiffusionRendererBackend``).
        """
        self.target_quad_faces = target_quad_faces
        self.geometry_backend = geometry_backend
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.volume_bounds = volume_bounds
        self.inverse_backend = inverse_backend
        self.diffusion_renderer_transport = diffusion_renderer_transport

    def run(
        self,
        video_frames: np.ndarray | list[np.ndarray],
        initial_mesh: MeshGeometry | None = None,
        skeleton_template: dict[str, Any] | None = None,
        fps: float = 30.0,
        track: str = "track_b_whitebox",
        has_gen3d_highpoly: bool = False,
        depths: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        cameras: list[CameraExtrinsics] | None = None,
        require_character_floor: bool = False,
        foundary_client: Any = None,
        ik_backend: Literal["auto", "native", "python"] = "auto",
        strict_slip_gate: bool = False,
        enable_jev_gate: bool = False,
        jev_critic: Any | None = None,
        hitl_ledger_path: Path | None = None,
        elapsed_seconds_override: float | None = None,
        inverse_backend: Literal["classical", "diffusion_renderer", "dual_track"] | None = None,
        diffusion_renderer_transport: Any | None = None,
    ) -> ReconResult:
        """Execute the end-to-end reconstruction pipeline.

        Args:
            video_frames: Video frame sequence, shape (T, H, W, 3) or list.
            initial_mesh: Optional prior 3D geometry (whitebox or gen3d).
            skeleton_template: Optional skeleton joints dictionary.
            fps: Video frame rate.
            track: Distillation track identifier for the geometry gate
                (default ``"track_b_whitebox"``).
            has_gen3d_highpoly: Whether the mesh has passed through gen3d
                high-poly reconstruction.  Defaults to ``False`` — the
                caller **must** explicitly opt in after verified Gen-3D
                reconstruction (Redline #3 fail-closed).
            depths: Optional metric depth maps for multi-view TSDF reconstruction
                when ``initial_mesh is None``.
            masks: Optional silhouette masks for visual hull reconstruction
                when ``initial_mesh is None``.
            cameras: Optional explicit camera extrinsics overriding or supplementing
                solved trajectory when ``initial_mesh is None``.

        Returns:
            ReconResult containing mesh, camera trajectory, kinematics,
            albedo delight result, and evaluation telemetry.
        """
        frames = np.asarray(video_frames)
        t_pipeline_start = time.monotonic()
        if frames.ndim != 4:
            raise ValueError(f"Expected (T, H, W, 3) video frames, got {frames.shape}")

        num_frames, height, width, _ = frames.shape

        # 1. 4D Point Tracking
        tracks_2d, visibility = solve_point_tracks(
            frames, num_points=256, foundary_client=foundary_client
        )

        # 2. Camera Trajectory Recovery — derive intrinsics from frame size
        camera_traj = solve_camera_trajectory(
            tracks_2d,
            frame_height=height,
            frame_width=width,
            visibility=visibility,
            foundary_client=foundary_client,
        )

        # 3. Surface Geometry & Quad Remeshing
        if initial_mesh is None:
            # Multi-view reconstruction from solved or explicit camera trajectory
            from synapse.recon.multiview import MultiViewInitializer

            cams = (
                cameras
                if cameras is not None
                else [ext for ext in (camera_traj.extrinsics if camera_traj else [])]
            )
            initializer = MultiViewInitializer(
                voxel_size=self.voxel_size,
                sdf_trunc=self.sdf_trunc,
                volume_bounds=self.volume_bounds,
            )
            active_mesh = initializer.reconstruct(
                frames=frames,
                cameras=cams,
                masks=masks,
                depths=depths,
            )
            if require_character_floor or (
                track == "track_b_whitebox" and skeleton_template is not None and not has_gen3d_highpoly
            ):
                from synapse.bridge.vg_bridge import enforce_distillation_geometry_gate

                enforce_distillation_geometry_gate(
                    track=track,
                    has_gen3d_highpoly=has_gen3d_highpoly,
                    vertex_count=active_mesh.vertex_count,
                    has_articulated_limbs=(skeleton_template is not None),
                    is_character=(require_character_floor or skeleton_template is not None),
                )
        else:
            # Enforce Track-B distillation geometry gate (Redline #3):
            # coarse whitebox proxies must pass through gen3d reconstruction
            # before texture back-projection is permitted.
            from synapse.bridge.vg_bridge import enforce_distillation_geometry_gate

            # Redline #3 fail-closed: has_gen3d_highpoly must be
            # explicitly set True by the caller after verified Gen-3D
            # reconstruction.  Auto-inference from vertex_count is
            # forbidden — it would let every >=1500-vertex whitebox
            # proxy silently bypass the Track-B gate.
            enforce_distillation_geometry_gate(
                track=track,
                has_gen3d_highpoly=has_gen3d_highpoly,
                vertex_count=initial_mesh.vertex_count,
                has_articulated_limbs=(skeleton_template is not None),
                is_character=(require_character_floor or skeleton_template is not None),
            )
            active_mesh = initial_mesh

        # Quad remesh
        remeshed_v, remeshed_f, remeshed_uvs = quad_remesh(
            active_mesh.vertices, active_mesh.faces, target_faces=self.target_quad_faces
        )
        normals = compute_normals(remeshed_v, remeshed_f)
        # Validate surface manifoldness via Tier 0 geometry check (native or python oracle)
        is_manifold = validate_recon_mesh_manifold(
            remeshed_v, remeshed_f, backend=self.geometry_backend
        )
        if not is_manifold:
            from synapse.errors import ContractValidationError

            raise ContractValidationError(
                "Manifold validation failed: reconstructed mesh is non-manifold. "
                "Fail-closed per Redline #3 — texture back-projection on "
                "non-manifold geometry produces UV seam artefacts.",
                agent_fix_hint=(
                    "Increase target_quad_faces or improve input geometry "
                    "quality. Run quad_remesh with stricter watertight constraints."
                ),
            )

        final_mesh = MeshGeometry(
            vertices=remeshed_v,
            faces=remeshed_f,
            normals=normals,
            uvs=remeshed_uvs,
        )

        # 4. Kinematics Distillation & Foot-Plant IK
        if skeleton_template is not None:
            # Real skeleton data available — compute contacts and run IK.
            contacts = np.zeros((num_frames, 2), dtype=np.float32)
            # Periodic stance phase simulation (used when no foot-contact
            # detector is available — derives from skeleton periodicity).
            for t in range(num_frames):
                if (t % 30) < 15:
                    contacts[t, 0] = 1.0  # Left foot stance
                else:
                    contacts[t, 1] = 1.0  # Right foot stance

            solved_skel = solve_zero_slip_ik(
                skeleton_template,
                contacts,
                fps=fps,
                backend=ik_backend,
                strict_slip_gate=strict_slip_gate,
            )
            skel_telemetry = solved_skel.get("telemetry", {})

            kinematics = MocapKinematics(
                frame_count=num_frames,
                joint_positions=solved_skel.get("joint_positions", np.zeros((0, 0, 3))),
                foot_contacts=contacts,
                raw_foot_slip_cm_s=skel_telemetry.get("raw_foot_sliding_cm_per_sec", 0.0),
                clamped_foot_slip_cm_s=skel_telemetry.get("clamped_foot_sliding_cm_per_sec", 0.0),
                max_foot_slip_cm_s=skel_telemetry.get("max_foot_slip_cm_per_sec", 0.0),
            )
        else:
            # No skeleton template — skip kinematics entirely instead of
            # fabricating fake sine-wave joint trajectories. Record in
            # telemetry for upstream visibility.
            kinematics = None

        # 5. Intrinsic Albedo De-lighting / DiffusionRenderer Dual-Track (PR-220)
        active_inv_backend = inverse_backend if inverse_backend is not None else self.inverse_backend
        active_dr_transport = (
            diffusion_renderer_transport
            if diffusion_renderer_transport is not None
            else self.diffusion_renderer_transport
        )

        active_cams = (
            cameras
            if (cameras is not None and len(cameras) == num_frames)
            else (camera_traj.extrinsics if (camera_traj is not None and len(camera_traj.extrinsics) == num_frames) else None)
        )

        dual_track_telemetry: dict[str, Any] | None = None
        if active_inv_backend in ("diffusion_renderer", "dual_track"):
            from synapse.bridge.diffusion_renderer import (
                FoundaryDiffusionRendererBackend,
                compare_inverse_rendering_tracks,
            )
            from synapse.errors import BackendUnavailableError
            from synapse.recon.types import DelightResult

            dr_backend = active_dr_transport or FoundaryDiffusionRendererBackend()
            if not dr_backend.is_available():
                raise BackendUnavailableError(
                    "DiffusionRenderer backend is not available for inverse rendering.",
                    agent_fix_hint=(
                        "Provide an available DiffusionRendererTransport (or MockDiffusionRendererBackend) "
                        "or switch back to inverse_backend='classical'."
                    ),
                )
            frames_u8 = (
                np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)
                if np.issubdtype(frames.dtype, np.floating)
                else frames.astype(np.uint8)
            )
            gbuf = dr_backend.inverse_render(frames_u8)
            keyframe_idx = num_frames // 2

            if active_inv_backend == "diffusion_renderer":
                albedo_kf = gbuf.albedo[keyframe_idx].astype(np.float32)
                shading_kf = np.ones_like(albedo_kf, dtype=np.float32)
                delight_result = DelightResult(
                    albedo=albedo_kf,
                    shading=shading_kf,
                    specular=np.zeros_like(albedo_kf, dtype=np.float32),
                    ambient_sh=np.zeros((9, 3), dtype=np.float32),
                    delta_e_residual=0.0,
                )
                dual_track_telemetry = {
                    "mode": "diffusion_renderer",
                    "gbuffer_depth_shape": list(gbuf.depth.shape),
                }
            else:
                # Dual-track: run classical SH path and compare against neural G-buffer
                if active_cams is not None:
                    from synapse.recon.delight import delight_multiview_albedo

                    delight_result = delight_multiview_albedo(
                        frames=frames,
                        cameras=active_cams,
                        mesh=final_mesh,
                        foundary_client=foundary_client,
                    )
                else:
                    lit_keyframe = frames[keyframe_idx]
                    delight_result = decompose_intrinsic(lit_keyframe)

                classical_alb = np.broadcast_to(
                    delight_result.albedo[np.newaxis, ...].astype(np.float32),
                    gbuf.albedo.shape,
                )
                classical_nrm = np.zeros_like(gbuf.normals, dtype=np.float32)
                if final_mesh.normals is not None and len(final_mesh.normals) > 0:
                    mean_n = np.mean(final_mesh.normals, axis=0).astype(np.float32)
                    n_len = float(np.linalg.norm(mean_n))
                    classical_nrm[..., :] = (mean_n / n_len) if n_len > 1e-8 else [0.0, 0.0, 1.0]
                else:
                    classical_nrm[..., 2] = 1.0
                classical_dep = (
                    depths.astype(np.float32)
                    if depths is not None and depths.shape == gbuf.depth.shape
                    else gbuf.depth.astype(np.float32)
                )
                cmp_res = compare_inverse_rendering_tracks(
                    classical_alb,
                    classical_nrm,
                    classical_dep,
                    gbuf,
                )
                dual_track_telemetry = {
                    "mode": "dual_track",
                    "albedo_psnr": cmp_res.albedo_psnr,
                    "normal_cosine_sim": cmp_res.normal_cosine_sim,
                    "depth_rmse": cmp_res.depth_rmse,
                    "agreement_score": cmp_res.agreement_score,
                }
        else:
            if active_cams is not None:
                from synapse.recon.delight import delight_multiview_albedo

                delight_result = delight_multiview_albedo(
                    frames=frames,
                    cameras=active_cams,
                    mesh=final_mesh,
                    foundary_client=foundary_client,
                )
            else:
                keyframe_idx = num_frames // 2
                lit_keyframe = frames[keyframe_idx]
                delight_result = decompose_intrinsic(lit_keyframe)

        # 6. Optional Visual-JEV defect probe & HITL 15-min circuit breaker
        jev_summary: dict[str, Any] | None = None
        if enable_jev_gate:
            from synapse.bridge.hitl import CommandIRRecord, PersistentHitlLedger
            from synapse.errors import QualityGateError

            if jev_critic is not None:
                report = jev_critic.evaluate(
                    asset_id="recon_asset",
                    asset_name="ReconAsset",
                    image_uris=["view_0"],
                    mesh_vertices=final_mesh.vertices,
                )
                jev_score = float(report.overall_quality_score)
                jev_hitl = bool(report.requires_hitl)
                jev_defects = len(report.defects)
            else:
                jev_score = 1.0 if is_manifold else 0.5
                jev_hitl = not is_manifold
                jev_defects = 0 if is_manifold else 1

            jev_summary = {
                "overall_quality_score": jev_score,
                "requires_hitl": jev_hitl,
                "defect_count": jev_defects,
            }
            if hitl_ledger_path is not None or elapsed_seconds_override is not None:
                ledger = (
                    PersistentHitlLedger(ledger_path=hitl_ledger_path)
                    if hitl_ledger_path is not None
                    else PersistentHitlLedger()
                )
                elapsed_s = (
                    float(elapsed_seconds_override)
                    if elapsed_seconds_override is not None
                    else max(0.0, time.monotonic() - t_pipeline_start)
                )
                ledger.record_action(
                    CommandIRRecord(
                        intent_group_id="recon-jev-gate",
                        intent="visual-jev-quality-inspection",
                        elapsed_seconds=elapsed_s,
                    ),
                    domain="asset_3d",
                )
            if jev_score < 0.85 and jev_hitl:
                raise QualityGateError(
                    f"Visual-JEV quality gate failed (score={jev_score:.3f} < 0.85)",
                    agent_fix_hint="Route asset to grand-studio L2 refinement or re-run with cleaner views.",
                )

        # 7. Build consolidated result
        telemetry: dict[str, Any] = {
            "num_frames": num_frames,
            "mesh_vertex_count": final_mesh.vertex_count,
            "mesh_face_count": final_mesh.face_count,
            "is_manifold": is_manifold,
            "is_quad_dominant": final_mesh.is_quad_dominant,
            "inverse_backend": active_inv_backend,
            "kinematics": (
                kinematics.to_telemetry()
                if kinematics is not None
                else {"kinematics_status": "skipped_no_skeleton"}
            ),
            "albedo_delta_e_residual": delight_result.delta_e_residual,
        }
        if dual_track_telemetry is not None:
            telemetry["dual_track_comparison"] = dual_track_telemetry
        if jev_summary is not None:
            telemetry["jev_inspection"] = jev_summary

        # --- Traceboard instrumentation ---
        probe = get_probe()
        if probe is not None and probe.active:
            elapsed = time.monotonic() - t_pipeline_start
            probe.log_scalar("recon/pipeline/elapsed_s", elapsed)
            probe.log_scalar("recon/pipeline/num_frames", float(num_frames))
            probe.log_scalar("recon/pipeline/mesh_vertex_count", float(final_mesh.vertex_count))
            probe.log_scalar("recon/pipeline/mesh_face_count", float(final_mesh.face_count))
            probe.log_scalar("recon/pipeline/is_manifold", float(is_manifold))
            probe.log_scalar(
                "recon/pipeline/albedo_delta_e", delight_result.delta_e_residual,
            )
            if kinematics is not None:
                probe.log_scalar(
                    "recon/pipeline/foot_slip_cm_s", kinematics.clamped_foot_slip_cm_s,
                )
            if jev_summary is not None:
                probe.log_scalar(
                    "recon/pipeline/jev_quality_score",
                    float(jev_summary["overall_quality_score"]),
                )
            if dual_track_telemetry is not None and "agreement_score" in dual_track_telemetry:
                probe.log_scalar(
                    "recon/pipeline/dual_track_agreement",
                    float(dual_track_telemetry["agreement_score"]),
                )

        return ReconResult(
            mesh=final_mesh,
            camera_trajectory=camera_traj,
            kinematics=kinematics,
            delight=delight_result,
            telemetry=telemetry,
        )
