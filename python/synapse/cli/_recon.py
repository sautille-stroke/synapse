"""Recon subcommand handlers: invert-video, closed-loop."""

from __future__ import annotations

import logging
import time
from typing import Literal, cast

import numpy as np

from synapse.cli.common import (
    Any,
    AssetNotFoundError,
    ContractValidationError,
    ExitCode,
    Path,
    SynapseError,
    _run_with_json_mode,
    argparse,
    json,
    sys,
)

logger = logging.getLogger(__name__)


def _cmd_recon_invert_video(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run video-to-3D inverse reconstruction (RFC 0060)."""
    from synapse.bridge.hitl import CommandIRRecord, PersistentHitlLedger
    from synapse.recon import ReconPipeline

    t_start = time.monotonic()

    output_dir = Path(getattr(args, "output", "./output/recon"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target_faces = int(getattr(args, "target_faces", 1500))

    video_path_raw = getattr(args, "video", None)
    if not video_path_raw or not Path(video_path_raw).exists():
        raise AssetNotFoundError(
            f"Video frames file not found: {video_path_raw!r}",
            agent_fix_hint=(
                "Provide '--video <path>' pointing to a valid .npy, image directory, "
                "or .mp4/.mov/.mkv video file."
            ),
        )

    from synapse.recon.io import export_recon_fgpp_sequence, load_video_frames

    frames, video_meta = load_video_frames(video_path_raw)

    # FGPP re-ingestion: read EXR conditioning from --fgpp-dir
    fgpp_dir = getattr(args, "fgpp_dir", None)
    fgpp_conditioning: dict[str, Any] = {}
    if fgpp_dir and Path(fgpp_dir).is_dir():
        fgpp_conditioning = _load_fgpp_exr_dir(Path(fgpp_dir))

    # Load skeleton template for Redline #3 fail-closed skeleton gate
    skeleton_template = None
    skeleton_path_raw = getattr(args, "skeleton", None)
    if skeleton_path_raw:
        if not Path(skeleton_path_raw).is_file():
            raise AssetNotFoundError(
                f"Skeleton template file not found: {skeleton_path_raw!r}",
                agent_fix_hint=(
                    "Provide '--skeleton <path>' pointing to a valid "
                    "skeleton JSON file."
                ),
            )
        import json as _json_skel

        with open(skeleton_path_raw, encoding="utf-8") as _f_skel:
            skeleton_template = _json_skel.load(_f_skel)

    pipe = ReconPipeline(target_quad_faces=target_faces)
    result = pipe.run(frames, skeleton_template=skeleton_template, fps=video_meta.fps)

    output_info: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_frames": len(frames),
        "video_codec": video_meta.codec,
        "has_trajectory": result.camera_trajectory is not None,
        "has_geometry": result.mesh is not None,
    }

    # Save trajectory
    if result.camera_trajectory:
        traj_path = output_dir / "camera_trajectory.json"
        traj_path.write_text(
            json.dumps(result.camera_trajectory.to_dict(), indent=2),
            encoding="utf-8",
        )
        output_info["trajectory_path"] = str(traj_path)

    # Persist mesh as .obj + OpenPBR textures
    if result.mesh is not None:
        obj_path = output_dir / "mesh.obj"
        _write_obj(obj_path, result.mesh.vertices, result.mesh.faces, result.mesh.uvs)
        output_info["mesh_obj"] = str(obj_path)

        # Save albedo texture as PNG if delight available
        if result.delight is not None and result.delight.albedo is not None:
            try:
                from synapse.recon.loop.texture_extractor import extract_and_pack_pbr_textures

                extract_and_pack_pbr_textures(
                    result.delight.albedo,
                    output_dir=str(output_dir / "textures"),
                )
                output_info["pbr_textures_dir"] = str(output_dir / "textures")
            except Exception as exc:
                logger.warning(
                    "PBR texture extraction failed (%s); falling back to raw albedo.npy.",
                    exc,
                )
                # Fallback: save raw albedo npy
                np.save(str(output_dir / "albedo.npy"), result.delight.albedo)
                output_info["albedo_npy"] = str(output_dir / "albedo.npy")

        # Write RFC 0058 OpenUSD + OpenPBR + UsdSkel package and validate
        from synapse.recon.usd_packager import (
            build_package_manifest_summary,
            build_rfc0058_usda,
            validate_usd_package,
        )

        tex_paths: dict[str, Path] = {}
        tex_dir = output_dir / "textures"
        for role, fname in (
            ("albedo", "albedo.png"),
            ("normal", "normal.png"),
            ("roughness", "roughness.png"),
            ("metallic", "metallic.png"),
        ):
            cand = tex_dir / fname
            if cand.is_file():
                tex_paths[role] = cand

        usda_path = build_rfc0058_usda(
            result.mesh,
            output_dir,
            kinematics=result.kinematics,
            texture_paths=tex_paths or None,
            filename="mesh.usda",
        )
        usd_val = validate_usd_package(usda_path, strict=True)
        output_info["mesh_usda"] = str(usda_path)
        output_info["usd_validation"] = build_package_manifest_summary(usd_val)

        fgpp_out_raw = getattr(args, "fgpp_out", None)
        if fgpp_out_raw:
            exr_paths = export_recon_fgpp_sequence(
                frames=frames,
                camera_trajectory=result.camera_trajectory,
                mesh=result.mesh,
                output_dir=Path(fgpp_out_raw),
                fps=video_meta.fps,
            )
            output_info["fgpp_out"] = str(fgpp_out_raw)
            output_info["fgpp_exr_count"] = len(exr_paths)

    if fgpp_conditioning:
        output_info["fgpp_dir"] = fgpp_dir
        output_info["fgpp_channels_loaded"] = list(fgpp_conditioning.keys())

    # Record HITL timing via PersistentHitlLedger (Redline #3 circuit breaker)
    elapsed = time.monotonic() - t_start
    ledger = PersistentHitlLedger()
    ledger.record_action(
        CommandIRRecord(
            intent_group_id=f"recon-invert-{output_dir.name}",
            intent="video-to-3d-inverse-reconstruction",
            elapsed_seconds=elapsed,
        ),
        domain="asset_3d",
    )

    if not getattr(args, "json", False):
        print("[Recon] Video-to-3D inverse reconstruction complete")
        for k, v in output_info.items():
            print(f"  {k}: {v}")

    return output_info


def _load_fgpp_exr_dir(fgpp_dir: Path) -> dict[str, Any]:
    """Load FGPP OpenEXR channel files from a directory.

    Reads ``*.exr`` files from the given directory and returns them
    as FGPP frames keyed by stem name.  Falls back to an empty dict
    if ``OpenEXR`` is not available.
    """
    result: dict[str, Any] = {}
    try:
        from synapse.fgpp.io import import_fgpp
    except ImportError:
        # OpenEXR not available — return empty conditioning
        return result

    for exr_file in sorted(fgpp_dir.glob("*.exr")):
        try:
            frame = import_fgpp(exr_file)
            result[exr_file.stem] = frame
        except Exception as exc:
            logger.warning("Skipping unreadable FGPP EXR '%s': %s", exr_file, exc)
            continue
    return result


def _write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray | None = None,
) -> None:
    """Write a simple Wavefront OBJ file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Synapse reconstructed mesh\n")
        for v in np.asarray(vertices):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if uvs is not None:
            for uv in np.asarray(uvs):
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for face in np.asarray(faces):
            indices = [str(int(idx) + 1) for idx in face]
            if uvs is not None:
                face_str = " ".join(f"{idx}/{idx}" for idx in indices)
            else:
                face_str = " ".join(indices)
            f.write(f"f {face_str}\n")


def _cmd_recon_compress_3dgs(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compress a 3D Gaussian Splat cloud (.npz) into a .s3dgs container (PR-200)."""
    from synapse.recon.gs_quantizer import GaussianSplatCloud, quantize_3dgs, save_s3dgs

    inp_raw = getattr(args, "input", None)
    if not inp_raw:
        raise ContractValidationError(
            "Missing required '--input' argument for compress-3dgs",
            agent_fix_hint="Provide '--input <cloud.npz>' containing xyz, sh_dc, scales, rotations, opacities.",
        )
    inp_path = Path(inp_raw)
    if not inp_path.is_file():
        raise AssetNotFoundError(
            f"Input 3DGS cloud file not found: {inp_path}",
            agent_fix_hint="Check the path passed to '--input'.",
        )

    out_path = Path(getattr(args, "output", "./output/cloud.s3dgs"))
    require_draco = bool(getattr(args, "require_draco", False))

    if inp_path.suffix.lower() != ".npz":
        raise ContractValidationError(
            f"Unsupported 3DGS input extension '{inp_path.suffix}'; expected '.npz'.",
            agent_fix_hint="Export 3DGS tensors using numpy.savez(..., xyz=..., sh_dc=..., scales=..., rotations=..., opacities=...).",
        )

    with np.load(inp_path, allow_pickle=False) as data:
        for req_key in ("xyz", "sh_dc", "scales", "rotations", "opacities"):
            if req_key not in data.files:
                raise ContractValidationError(
                    f"Missing required 3DGS array '{req_key}' in {inp_path}",
                    agent_fix_hint="Ensure archive defines xyz, sh_dc, scales, rotations, opacities.",
                )
        sh_rest = data["sh_rest"] if "sh_rest" in data.files else None
        cloud = GaussianSplatCloud(
            xyz=np.asarray(data["xyz"], dtype=np.float32),
            sh_dc=np.asarray(data["sh_dc"], dtype=np.float32),
            sh_rest=np.asarray(sh_rest, dtype=np.float32) if sh_rest is not None else None,
            scales=np.asarray(data["scales"], dtype=np.float32),
            rotations=np.asarray(data["rotations"], dtype=np.float32),
            opacities=np.asarray(data["opacities"], dtype=np.float32),
        )
    quantized = quantize_3dgs(cloud, require_draco=require_draco)
    saved_path = save_s3dgs(quantized, out_path)
    return {
        "output_path": str(saved_path),
        "num_gaussians": cloud.num_gaussians,
        "sh_band_count": cloud.sh_band_count,
        "raw_bytes": quantized.metrics.raw_bytes,
        "compressed_bytes": quantized.metrics.compressed_bytes,
        "compression_ratio": quantized.metrics.compression_ratio,
        "xyz_rmse": quantized.metrics.xyz_rmse,
        "psnr_delta_db": quantized.metrics.psnr_delta_db,
        "recon_vs_orig_psnr_db": quantized.metrics.recon_vs_orig_psnr_db,
        "draco_used": quantized.metrics.draco_used,
    }


def _cmd_recon_closed_loop(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run 5-step inverse rendering closed loop (RFC 0062).

    Orchestrates forward-guide -> target-video -> inverse-optimisation
    -> aligned-render -> scalpel-patch.
    """
    import numpy as np

    from synapse.recon.loop.closed_loop_runner import ClosedLoopRunner

    output_dir = Path(getattr(args, "output", "./output/closed-loop"))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_iterations = int(getattr(args, "max_iterations", 5))
    n_preroll = int(getattr(args, "n_preroll", 0))
    backend_choice = str(getattr(args, "backend", "auto"))
    strict_backend = bool(getattr(args, "strict_backend", False))

    # Load frames — require real input (AUD-P0-01: no fake synthesis)
    video_path_raw = getattr(args, "video", None)
    if not video_path_raw:
        raise ContractValidationError(
            "Missing required '--video' argument for closed-loop",
            agent_fix_hint=(
                "Provide '--video <path>' pointing to a valid .npy "
                "file of shape (T, H, W, 3) float32."
            ),
        )
    if not Path(video_path_raw).is_file():
        raise AssetNotFoundError(
            f"Video frames file not found: {video_path_raw!r}",
            agent_fix_hint=(
                "Provide '--video <path>' pointing to an existing .npy "
                "file of shape (T, H, W, 3) float32."
            ),
        )
    frames = np.load(video_path_raw).astype(np.float32)

    t_count, h, w, _ = frames.shape

    # Real normals input required (H6-P1-05: no fake +Z synthesis)
    normals_path_raw = getattr(args, "normals", None)
    if not normals_path_raw:
        raise AssetNotFoundError(
            "Missing required '--normals' argument for closed-loop",
            agent_fix_hint=(
                "Provide '--normals <path>' pointing to an existing .npy "
                "file of shape (T, H, W, 3) float32."
            ),
        )
    if not Path(normals_path_raw).is_file():
        raise AssetNotFoundError(
            f"Normals file not found: {normals_path_raw!r}",
            agent_fix_hint=(
                "Provide '--normals <path>' pointing to an existing .npy "
                "file of shape (T, H, W, 3) float32."
            ),
        )
    normals = np.load(normals_path_raw).astype(np.float32)

    uv_maps_path_raw = getattr(args, "uv_maps", None)
    if uv_maps_path_raw:
        if not Path(uv_maps_path_raw).is_file():
            raise AssetNotFoundError(
                f"UV maps file not found: {uv_maps_path_raw!r}",
                agent_fix_hint=(
                    "Provide '--uv-maps <path>' pointing to an existing .npy "
                    "file of shape (T, H, W, 2) float32."
                ),
            )
        uv_maps = np.load(uv_maps_path_raw).astype(np.float32)
    else:
        uv_maps = np.zeros((t_count, h, w, 2), dtype=np.float32)
        u_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)
        v_lin = np.linspace(0.0, 1.0, h, dtype=np.float32)
        uv_maps[:, :, :, 0] = u_lin[np.newaxis, np.newaxis, :]
        uv_maps[:, :, :, 1] = v_lin[np.newaxis, :, np.newaxis]

    from synapse.recon.loop.mitsuba_optimizer import InverseOptimConfig
    from synapse.recon.loop.types import LoopConfig

    loop_config = LoopConfig(
        n_preroll=n_preroll,
        max_iterations=max_iterations,
    )
    optim_config = InverseOptimConfig(
        max_iterations=max_iterations,
        renderer_backend=cast(
            Literal["auto", "mitsuba", "fulgur_slang_ad", "numpy_oracle"],
            backend_choice,
        ),
        strict_backend=strict_backend,
    )

    runner = ClosedLoopRunner(
        loop_config=loop_config,
        optim_config=optim_config,
    )
    result = runner.run(
        source_frames=frames,
        normals=normals,
        uv_maps=uv_maps,
    )

    # Save output
    np.save(str(output_dir / "output_frames.npy"), result.output_frames)
    diag = result.to_dict()
    diag_path = output_dir / "closed_loop_diagnostics.json"
    diag_path.write_text(
        json.dumps(diag, indent=2, default=str),
        encoding="utf-8",
    )

    output_info: dict[str, Any] = {
        "output_dir": str(output_dir),
        **diag,
    }

    if not getattr(args, "json", False):
        print("[Recon] Closed-loop inverse rendering complete")
        for k, v in output_info.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")

    return output_info


def register_recon_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    json_parent: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Register the ``recon`` subcommand tree."""
    parents = [json_parent] if json_parent is not None else []
    recon_parser = subparsers.add_parser(
        "recon",
        help="3D reconstruction and inverse rendering commands",
        parents=parents,
    )
    recon_sub = recon_parser.add_subparsers(
        dest="recon_command",
    )

    recon_invert = recon_sub.add_parser(
        "invert-video",
        help="Inverse video-to-3D reconstruction (RFC 0060)",
        parents=parents,
    )
    recon_invert.add_argument(
        "--video",
        default=None,
        help="Path to video frames .npy file (T, H, W, 3)",
    )
    recon_invert.add_argument(
        "--fgpp-dir",
        default=None,
        help="Optional FGPP directory for re-ingestion",
    )
    recon_invert.add_argument(
        "--output",
        "-o",
        default="./output/recon",
        help="Output directory for reconstruction results",
    )
    recon_invert.add_argument(
        "--target-faces",
        type=int,
        default=1500,
        help="Target face count for quad mesh (default: 1500)",
    )
    recon_invert.add_argument(
        "--skeleton",
        default=None,
        help="Path to skeleton template JSON (Redline #3: fail-closed on missing)",
    )
    recon_invert.add_argument(
        "--fgpp-out",
        default=None,
        help="Optional output directory to re-export 15-channel FGPP EXR sequence",
    )

    recon_closed_loop = recon_sub.add_parser(
        "closed-loop",
        help="Run 5-step inverse rendering closed loop (RFC 0062)",
        parents=parents,
    )
    recon_closed_loop.add_argument(
        "--video",
        default=None,
        help="Path to video frames .npy file (T, H, W, 3) float32",
    )
    recon_closed_loop.add_argument(
        "--normals",
        default=None,
        help="Path to surface normals .npy file (T, H, W, 3) float32",
    )
    recon_closed_loop.add_argument(
        "--uv-maps",
        default=None,
        help="Path to UV maps .npy file (T, H, W, 2) float32",
    )
    recon_closed_loop.add_argument(
        "--output",
        "-o",
        default="./output/closed-loop",
        help="Output directory for closed-loop results",
    )
    recon_closed_loop.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max inverse optimisation iterations (default: 5)",
    )
    recon_closed_loop.add_argument(
        "--n-preroll",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="Number of preroll frames (default: 0)",
    )
    recon_closed_loop.add_argument(
        "--backend",
        choices=["auto", "fulgur_slang_ad", "mitsuba", "numpy_oracle"],
        default="auto",
        help="Inverse rendering backend selector (default: auto)",
    )
    recon_closed_loop.add_argument(
        "--strict-backend",
        action="store_true",
        default=False,
        help="Fail closed with exit 30 if selected GPU/Slang backend is unavailable",
    )

    recon_compress = recon_sub.add_parser(
        "compress-3dgs",
        help="Quantize and compress 3DGS SH/covariance attributes into .s3dgs (PR-200)",
        parents=parents,
    )
    recon_compress.add_argument(
        "--input",
        "-i",
        default=None,
        help="Input .npz containing 3DGS tensors (xyz, sh_dc, [sh_rest], scales, rotations, opacities)",
    )
    recon_compress.add_argument(
        "--output",
        "-o",
        default="./output/cloud.s3dgs",
        help="Output .s3dgs container path",
    )
    recon_compress.add_argument(
        "--require-draco",
        action="store_true",
        default=False,
        help="Require @aether//compress Draco encoder and exit 30 if unavailable",
    )
    return cast(argparse.ArgumentParser, recon_parser)


def dispatch_recon(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    recon_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse recon <sub-command>``."""
    recon_cmd = getattr(parsed, "recon_command", None)
    if recon_cmd == "invert-video":
        return _run_with_json_mode(
            _cmd_recon_invert_video,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif recon_cmd == "closed-loop":
        return _run_with_json_mode(
            _cmd_recon_closed_loop,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif recon_cmd == "compress-3dgs":
        return _run_with_json_mode(
            _cmd_recon_compress_3dgs,
            parsed,
            json_mode,
            trace_ctx,
        )
    else:
        if json_mode:
            err = SynapseError(
                "Missing recon sub-command (invert-video|closed-loop|compress-3dgs)",
                ExitCode.CONTRACT_ERROR,
                "Use 'synapse recon invert-video --help', 'closed-loop --help', or 'compress-3dgs --help'.",
            )
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
            return int(err.exit_code)
        recon_parser.print_help()
        return int(ExitCode.CONTRACT_ERROR)
