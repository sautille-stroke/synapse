# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Pipeline stage adapters for 3D reconstruction and inverse rendering.

Registers ``video_to_3d_invert``, ``inverse_render_loop``, and
``delight_albedo`` stages with the
:func:`~synapse.pipeline.bundle_worker.register_stage` decorator so
they are discoverable by the ``JobBundleWorker``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from synapse.pipeline.bundle_worker import BundleManifest, register_stage

logger = logging.getLogger(__name__)


@register_stage("video_to_3d_invert")
@register_stage("inverse_reconstruct")
def handle_video_to_3d_invert(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Execute the Video-to-3D inverse reconstruction stage (RFC 0060).

    Expects:
        - ``manifest.inputs["video_frames"]``: Path to ``(T,H,W,3)`` ``.npy``.
        - ``manifest.params["fgpp_dir"]``: Optional FGPP directory for re-ingestion.

    Returns:
        Dict with ``output_paths`` to the reconstruction outputs.
    """
    import time

    from synapse.bridge.hitl import CommandIRRecord, PersistentHitlLedger
    from synapse.recon import ReconPipeline
    from synapse.recon.delight import delight_multiview_albedo

    t_start = time.monotonic()

    params = dict(manifest.params)
    target_faces = int(params.get("target_faces", 1500))

    # Load FGPP conditioning if present
    fgpp_dir_str = params.get("fgpp_dir", "")
    fgpp_conditioning: dict[str, Any] = {}
    if fgpp_dir_str and Path(fgpp_dir_str).is_dir():
        try:
            from synapse.fgpp.io import import_fgpp

            for exr_file in sorted(Path(fgpp_dir_str).glob("*.exr")):
                try:
                    frame = import_fgpp(exr_file)
                    fgpp_conditioning[exr_file.stem] = frame
                except Exception as exc:
                    logger.warning("Skipping unreadable FGPP EXR '%s': %s", exr_file, exc)
                    continue
        except ImportError as exc:
            logger.debug("FGPP EXR import skipped (OpenEXR unavailable): %s", exc)

    # Load skeleton template — Redline #3 fail-closed
    skeleton_template = None
    skeleton_path_str = params.get("skeleton", "") or manifest.inputs.get("skeleton", "")
    if skeleton_path_str and Path(skeleton_path_str).is_file():
        skeleton_template = json.loads(Path(skeleton_path_str).read_text(encoding="utf-8"))

    # Load input frames via unified video I/O — no synthetic fallback
    video_path = manifest.inputs.get("video_frames", "")
    if not video_path or not Path(video_path).exists():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Video frames file not found: {video_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['video_frames'] pointing to "
                "a valid .npy/.npz, image directory, or .mp4/.mov video file."
            ),
        )
    from synapse.recon.io import export_recon_fgpp_sequence, load_video_frames

    frames, video_meta = load_video_frames(video_path)

    pipe = ReconPipeline(target_quad_faces=target_faces)
    result = pipe.run(frames, skeleton_template=skeleton_template, fps=video_meta.fps)

    output_paths: dict[str, str] = {}

    # Save mesh
    mesh_path = manifest.outputs.get(
        "mesh",
        str(Path("output") / "reconstructed_mesh.npy"),
    )
    p_mesh = Path(mesh_path)
    p_mesh.parent.mkdir(parents=True, exist_ok=True)
    if result.mesh is not None:
        # Save full mesh geometry: vertices, faces, and uvs (if available)
        npz_path = str(p_mesh).rsplit(".", 1)[0] + ".npz"
        if result.mesh.uvs is not None:
            np.savez(
                npz_path,
                vertices=result.mesh.vertices,
                faces=result.mesh.faces,
                uvs=result.mesh.uvs,
            )
        else:
            np.savez(
                npz_path,
                vertices=result.mesh.vertices,
                faces=result.mesh.faces,
            )
        output_paths["mesh"] = npz_path

    # Save trajectory
    traj_path = manifest.outputs.get(
        "trajectory",
        str(Path("output") / "camera_trajectory.json"),
    )
    p_traj = Path(traj_path)
    p_traj.parent.mkdir(parents=True, exist_ok=True)
    if result.camera_trajectory:
        with open(p_traj, "w", encoding="utf-8") as f:
            json.dump(result.camera_trajectory.to_dict(), f, indent=2)
        output_paths["trajectory"] = str(p_traj)

    # De-light albedo if geometry available
    if result.mesh is not None and result.mesh.normals is not None:
        albedo_res = delight_multiview_albedo(
            frames=frames,
            cameras=result.camera_trajectory.extrinsics if result.camera_trajectory else [],
            mesh=result.mesh,
        )
        albedo_path = manifest.outputs.get(
            "albedo",
            str(Path("output") / "albedo.npy"),
        )
        p_albedo = Path(albedo_path)
        p_albedo.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p_albedo), albedo_res.albedo)
        output_paths["albedo"] = str(p_albedo)

    fgpp_out_dir = params.get("fgpp_out", "") or manifest.outputs.get("fgpp_dir", "")
    if fgpp_out_dir and result.mesh is not None:
        exr_paths = export_recon_fgpp_sequence(
            frames=frames,
            camera_trajectory=result.camera_trajectory,
            mesh=result.mesh,
            output_dir=Path(fgpp_out_dir),
            fps=video_meta.fps,
        )
        output_paths["fgpp_dir"] = str(fgpp_out_dir)
        output_paths["fgpp_exr_count"] = str(len(exr_paths))

    # HITL circuit breaker — enforce 15-min ceiling via PersistentHitlLedger
    # (Redline #3: fail-closed persistent tracking across CLI invocations)
    elapsed = time.monotonic() - t_start
    ledger = PersistentHitlLedger()
    ledger.record_action(
        CommandIRRecord(
            intent_group_id=f"recon-stage-{manifest.job_id if hasattr(manifest, 'job_id') else 'default'}",
            intent="video-to-3d-inverse-reconstruction-stage",
            elapsed_seconds=elapsed,
        ),
        domain="asset_3d",
    )

    return {"output_paths": output_paths}


@register_stage("kinematic_distill")
def handle_kinematic_distill(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Execute kinematic distillation: solve zero-slip foot IK (RFC 0060).

    Expects:
        - ``manifest.inputs["skeleton"]``: Path to skeleton JSON.
        - ``manifest.inputs["foot_contacts"]``: Path to ``(T,2)`` ``.npy``.

    Returns:
        Dict with ``output_paths`` to the IK-solved skeleton data.
    """
    from synapse.recon.kinematics import solve_zero_slip_ik

    params = dict(manifest.params)
    fps = float(params.get("fps", 30.0))

    # Load skeleton template
    skeleton_path = manifest.inputs.get("skeleton", "")
    if skeleton_path and Path(skeleton_path).is_file():
        with open(skeleton_path, encoding="utf-8") as f:
            skeleton_data = json.loads(f.read())
    else:
        skeleton_data = dict(params.get("skeleton_template", {}))

    # Load foot contacts
    contacts_path = manifest.inputs.get("foot_contacts", "")
    if contacts_path and Path(contacts_path).is_file():
        contacts = np.load(contacts_path)
    else:
        # Minimal periodic stance phase contacts
        n_frames = int(params.get("num_frames", 60))
        contacts = np.zeros((n_frames, 2), dtype=np.float32)
        for t in range(n_frames):
            if (t % 30) < 15:
                contacts[t, 0] = 1.0
            else:
                contacts[t, 1] = 1.0

    ik_backend_raw = str(params.get("ik_backend", "auto"))
    ik_backend: Any = ik_backend_raw if ik_backend_raw in ("auto", "native", "python") else "auto"
    strict_slip_gate = bool(params.get("strict_slip_gate", False))
    solved = solve_zero_slip_ik(
        skeleton_data,
        contacts,
        fps=fps,
        backend=ik_backend,
        strict_slip_gate=strict_slip_gate,
    )

    output_paths: dict[str, str] = {}
    out_path = manifest.outputs.get(
        "ik_skeleton",
        str(Path("output") / "ik_solved_skeleton.json"),
    )
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    # Serialise — joint_positions are ndarray → convert to list
    serialisable = {}
    for k, v in solved.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        else:
            serialisable[k] = v
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=str)
    output_paths["ik_skeleton"] = str(p_out)

    return {"output_paths": output_paths}


@register_stage("inverse_render_loop")
def handle_inverse_render_loop(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Execute the 5-step inverse rendering closed loop (RFC 0062).

    Expects:
        - ``manifest.inputs["video_frames"]``: Path to ``(T,H,W,3)`` ``.npy``.
        - ``manifest.params["max_iterations"]``: Max optimisation iterations.
        - ``manifest.params["n_preroll"]``: Preroll frame count (0, 4, or 8).

    Returns:
        Dict with ``output_paths`` and ``diagnostics``.
    """
    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner
    from synapse.recon.loop.mitsuba_optimizer import InverseOptimConfig
    from synapse.recon.loop.types import LoopConfig

    params = dict(manifest.params)
    max_iters = int(params.get("max_iterations", 5))
    n_preroll = int(params.get("n_preroll", 0))

    # Load input frames via unified video I/O — no synthetic fallback
    video_path = manifest.inputs.get("video_frames", "")
    if not video_path or not Path(video_path).exists():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Video frames file not found: {video_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['video_frames'] pointing to "
                "a valid .npy/.npz, image directory, or .mp4/.mov video file."
            ),
        )
    from synapse.recon.io import load_video_frames

    raw_frames, _ = load_video_frames(video_path)
    frames = raw_frames.astype(np.float32)

    # Load normals — no synthetic fallback
    normals_path = manifest.inputs.get("normals", "")
    if not normals_path or not Path(normals_path).is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Normals file not found: {normals_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['normals'] pointing to "
                "a valid .npy file of shape (T, H, W, 3) float32."
            ),
        )
    normals = np.load(normals_path).astype(np.float32)

    # Load UV maps — no synthetic fallback
    uv_path = manifest.inputs.get("uv_maps", "")
    if not uv_path or not Path(uv_path).is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"UV maps file not found: {uv_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['uv_maps'] pointing to "
                "a valid .npy file of shape (T, H, W, 2) float32."
            ),
        )
    uv_maps = np.load(uv_path).astype(np.float32)

    loop_config = LoopConfig(
        n_preroll=n_preroll,
        max_iterations=max_iters,
    )
    optim_config = InverseOptimConfig(max_iterations=max_iters)

    runner = ClosedLoopRunner(
        loop_config=loop_config,
        optim_config=optim_config,
    )
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    output_paths: dict[str, str] = {}

    # Save output frames
    out_path = manifest.outputs.get(
        "output_frames",
        str(Path("output") / "closed_loop_frames.npy"),
    )
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p_out), result.output_frames)
    output_paths["output_frames"] = str(p_out)

    # Save diagnostics
    diag_path = manifest.outputs.get(
        "diagnostics",
        str(Path("output") / "closed_loop_diagnostics.json"),
    )
    p_diag = Path(diag_path)
    p_diag.parent.mkdir(parents=True, exist_ok=True)
    diag = result.to_dict()
    with open(p_diag, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, default=str)
    output_paths["diagnostics"] = str(p_diag)

    return {"output_paths": output_paths, "diagnostics": diag}


@register_stage("delight_albedo")
def handle_delight_albedo(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """De-light multiview frames to recover intrinsic albedo (ClosureType.DELIGHTING).

    Expects:
        - ``manifest.inputs["video_frames"]``: Path to ``(T,H,W,3)`` ``.npy``.
        - ``manifest.params["target_faces"]``: Optional target face count for
          mesh reconstruction (default 1500).

    Normals are reconstructed internally via :class:`ReconPipeline` rather
    than taken from an external input.

    Returns:
        Dict with ``output_paths`` to the delighted albedo.
    """
    from synapse.recon import ReconPipeline
    from synapse.recon.delight import delight_multiview_albedo

    # Load input frames — no synthetic fallback
    video_path = manifest.inputs.get("video_frames", "")
    if not video_path or not Path(video_path).is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Video frames file not found: {video_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['video_frames'] pointing to "
                "a valid .npy file of shape (T, H, W, 3)."
            ),
        )
    frames = np.load(video_path)

    # Reconstruct mesh for normals
    params = dict(manifest.params)
    target_faces = int(params.get("target_faces", 1500))
    pipe = ReconPipeline(target_quad_faces=target_faces)
    result = pipe.run(frames)

    if result.mesh is None or result.mesh.normals is None:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "Mesh reconstruction produced no normals — cannot delight.",
            agent_fix_hint=(
                "Check input video frames quality and ensure "
                "ReconPipeline can triangulate valid geometry."
            ),
        )

    albedo_res = delight_multiview_albedo(
        frames=frames,
        cameras=(result.camera_trajectory.extrinsics if result.camera_trajectory else []),
        mesh=result.mesh,
    )
    albedo_path = manifest.outputs.get(
        "albedo",
        str(Path("output") / "delighted_albedo.npy"),
    )
    p_albedo = Path(albedo_path)
    # Normalize suffix — np.save auto-appends .npy if missing, so ensure
    # output_paths matches the actual file on disk.
    if p_albedo.suffix != ".npy":
        p_albedo = p_albedo.with_suffix(".npy")
    p_albedo.parent.mkdir(parents=True, exist_ok=True)
    with open(p_albedo, "wb") as f_out:
        np.save(f_out, albedo_res.albedo)

    return {"output_paths": {"albedo": str(p_albedo)}}
