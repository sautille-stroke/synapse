# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Pipeline stage adapters for 3D reconstruction and inverse rendering.

Registers ``video_to_3d_invert`` and ``inverse_render_loop`` stages
with the :func:`~synapse.pipeline.bundle_worker.register_stage`
decorator so they are discoverable by the ``JobBundleWorker``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from synapse.pipeline.bundle_worker import BundleManifest, register_stage


@register_stage("video_to_3d_invert")
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
    from synapse.recon import ReconPipeline
    from synapse.recon.delight import delight_multiview_albedo

    params = dict(manifest.params)
    target_faces = int(params.get("target_faces", 1500))

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

    pipe = ReconPipeline(target_quad_faces=target_faces)
    result = pipe.run(frames)

    output_paths: dict[str, str] = {}

    # Save mesh
    mesh_path = manifest.outputs.get(
        "mesh", str(Path("output") / "reconstructed_mesh.npy"),
    )
    p_mesh = Path(mesh_path)
    p_mesh.parent.mkdir(parents=True, exist_ok=True)
    if result.mesh is not None:
        np.save(str(p_mesh), result.mesh.vertices)
        output_paths["mesh"] = str(p_mesh)

    # Save trajectory
    traj_path = manifest.outputs.get(
        "trajectory", str(Path("output") / "camera_trajectory.json"),
    )
    p_traj = Path(traj_path)
    p_traj.parent.mkdir(parents=True, exist_ok=True)
    if result.camera_trajectory:
        with open(p_traj, "w", encoding="utf-8") as f:
            json.dump(result.camera_trajectory.to_dict(), f, indent=2)
        output_paths["trajectory"] = str(p_traj)

    # De-light albedo if geometry available
    if result.mesh is not None and result.mesh.normals is not None:
        albedo = delight_multiview_albedo(
            frames=frames,
            cameras=[],
            mesh=result.mesh,
        )
        albedo_path = manifest.outputs.get(
            "albedo", str(Path("output") / "albedo.npy"),
        )
        p_albedo = Path(albedo_path)
        p_albedo.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p_albedo), albedo)
        output_paths["albedo"] = str(p_albedo)

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

    # Load input frames — no synthetic fallback
    video_path = manifest.inputs.get("video_frames", "")
    if not video_path or not Path(video_path).is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Video frames file not found: {video_path!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['video_frames'] pointing to "
                "a valid .npy file of shape (T, H, W, 3) float32."
            ),
        )
    frames = np.load(video_path).astype(np.float32)

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
