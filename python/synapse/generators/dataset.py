"""Multi-angle video & FGPP training/validation dataset generator."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from synapse.bench.camera import CameraTrajectoryType, generate_cameras
from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.generators.rasterizer import SoftwareRasterizer
from synapse.generators.visualizer import (
    colorize_depth,
    colorize_flow,
    colorize_mask,
    colorize_normal,
    make_contact_sheet,
)
from synapse.models.asset import ModelAsset
from synapse.models.loader import load_model


@dataclass
class DatasetConfig:
    """Configuration for multi-angle dataset generation."""

    model_name: str = "damaged_helmet"
    trajectory_type: str = "orbit"  # orbit, spiral, hemisphere, dolly, arc
    num_frames: int = 60
    width: int = 512
    height: int = 512
    fps: float = 24.0
    radius: float = 2.4
    fov_degrees: float = 50.0
    val_ratio: float = 0.2  # 20% holdout for novel view evaluation
    output_dir: str | Path = "./data/datasets/default"


def _generate_spiral_cameras(
    num_frames: int,
    radius: float,
    width: int,
    height: int,
    fov_degrees: float,
    min_elev: float = -10.0,
    max_elev: float = 65.0,
    rotations: float = 2.0,
) -> list[CameraParams]:
    """Generate a rising spiral camera trajectory around target."""
    from synapse.bench.camera import _make_camera

    cameras = []
    target = np.array([0.0, 0.0, 0.0])

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        azimuth = t * 360.0 * rotations
        elev = min_elev + t * (max_elev - min_elev)

        az_rad = math.radians(azimuth)
        el_rad = math.radians(elev)

        x = radius * math.cos(el_rad) * math.sin(az_rad)
        y = radius * math.sin(el_rad)
        z = radius * math.cos(el_rad) * math.cos(az_rad)
        eye = np.array([x, y, z])

        cam = _make_camera(eye, target, width, height, fov_degrees)
        cameras.append(cam)

    return cameras


class DatasetGenerator:
    """Generates ground-truth multi-angle video and FGPP training/validation datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.rasterizer = SoftwareRasterizer(config.width, config.height)

    def generate(self, model: Optional[ModelAsset] = None) -> Path:
        """Run the full generation pipeline and export all artifacts."""
        cfg = self.config
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = out_dir / "frames"
        fgpp_dir = out_dir / "fgpp"
        vis_dir = out_dir / "visualizations"
        for d in (frames_dir, fgpp_dir, vis_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1. Load model
        if model is None:
            model = load_model(cfg.model_name, normalize=True, target_radius=1.0)

        # 2. Generate camera trajectory
        if cfg.trajectory_type == "spiral":
            cameras = _generate_spiral_cameras(
                num_frames=cfg.num_frames,
                radius=cfg.radius,
                width=cfg.width,
                height=cfg.height,
                fov_degrees=cfg.fov_degrees,
            )
        else:
            traj_enum = CameraTrajectoryType(cfg.trajectory_type)
            cameras = generate_cameras(
                traj_enum,
                num_frames=cfg.num_frames,
                radius=cfg.radius,
                width=cfg.width,
                height=cfg.height,
                fov_degrees=cfg.fov_degrees,
            )

        # 3. Partition train / val split
        # Strategy: uniformly space validation frames across sequence
        num_frames = len(cameras)
        num_val = max(1, int(round(num_frames * cfg.val_ratio)))
        val_step = max(1, num_frames // num_val)
        val_indices = set(range(0, num_frames, val_step)[:num_val])
        train_indices = [i for i in range(num_frames) if i not in val_indices]
        val_indices_list = sorted(list(val_indices))

        # 4. Rendering loop
        rgb_frames: list[np.ndarray] = []
        depth_vis_frames: list[np.ndarray] = []
        fgpp_frames: list[FGPPFrame] = []
        frame_records: list[dict[str, Any]] = []

        import cv2  # lazy: heavy native dependency, not needed at import time

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_path = out_dir / "video_all.mp4"
        train_video_path = out_dir / "video_train.mp4"
        val_video_path = out_dir / "video_val.mp4"

        video_writer = cv2.VideoWriter(str(video_path), fourcc, cfg.fps, (cfg.width, cfg.height))
        train_writer = cv2.VideoWriter(str(train_video_path), fourcc, cfg.fps, (cfg.width, cfg.height))
        val_writer = cv2.VideoWriter(str(val_video_path), fourcc, cfg.fps, (cfg.width, cfg.height))

        max_flow_val = 0.0

        for i in range(num_frames):
            cam = cameras[i]
            prev_cam = cameras[i - 1] if i > 0 else cameras[0]

            render_res = self.rasterizer.render(model, cam, prev_cam)

            # Store stats
            if np.any(render_res.mask > 0):
                flow_mag = np.linalg.norm(render_res.motion_vector[render_res.mask > 0], axis=-1)
                if len(flow_mag) > 0:
                    max_flow_val = max(max_flow_val, float(np.max(flow_mag)))

            is_val = i in val_indices
            split_tag = "val" if is_val else "train"

            # Colorize channels for inspection
            depth_rgb = colorize_depth(render_res.depth, render_res.mask)
            normal_rgb = colorize_normal(render_res.world_normal, render_res.mask)
            flow_rgb = colorize_flow(render_res.motion_vector, render_res.mask, max_flow=max(1.0, max_flow_val))
            mask_rgb = colorize_mask(render_res.mask)

            # Save individual RGB frame
            rgb_filename = f"frame_{i:04d}.png"
            Image.fromarray(render_res.rgb).save(frames_dir / rgb_filename)

            # Save visualizations
            Image.fromarray(depth_rgb).save(vis_dir / f"depth_{i:04d}.png")
            Image.fromarray(normal_rgb).save(vis_dir / f"normal_{i:04d}.png")
            Image.fromarray(flow_rgb).save(vis_dir / f"flow_{i:04d}.png")

            # Write to video
            bgr_frame = cv2.cvtColor(render_res.rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
            if is_val:
                val_writer.write(bgr_frame)
            else:
                train_writer.write(bgr_frame)

            # Save NPZ for FGPP channel data
            fgpp_filename = f"frame_{i:04d}.npz"
            np.savez_compressed(
                fgpp_dir / fgpp_filename,
                depth=render_res.depth.astype(np.float32),
                world_normal=render_res.world_normal.astype(np.float16),
                motion_vector=render_res.motion_vector.astype(np.float16),
                mask=render_res.mask.astype(np.uint8),
            )

            # FGPP Frame dataclass
            fgpp_frame = FGPPFrame(
                camera=cam,
                depth=render_res.depth,
                world_normal=render_res.world_normal.astype(np.float16),
                motion_vector=render_res.motion_vector.astype(np.float16),
                frame_index=i,
            )
            fgpp_frames.append(fgpp_frame)

            # Manifest record
            pos = cam.position()
            frame_records.append({
                "frame_index": i,
                "split": split_tag,
                "rgb_path": f"frames/{rgb_filename}",
                "fgpp_path": f"fgpp/{fgpp_filename}",
                "camera_position": pos.tolist(),
                "camera": cam.to_dict(),
            })

            rgb_frames.append(render_res.rgb)
            depth_vis_frames.append(depth_rgb)

            # Generate contact sheet for middle frame or key frame
            if i == num_frames // 4:
                contact = make_contact_sheet(
                    render_res.rgb,
                    depth_rgb,
                    normal_rgb,
                    flow_rgb,
                    mask_rgb,
                    label=f"{cfg.model_name} [Frame {i:04d} - {split_tag.upper()}]",
                )
                Image.fromarray(contact).save(out_dir / "contact_sheet.png")

        video_writer.release()
        train_writer.release()
        val_writer.release()

        # 5. FGPP sequence metadata
        fgpp_seq = FGPPSequence(
            frames=fgpp_frames,
            fps=cfg.fps,
            scene_id=f"{cfg.model_name}_{cfg.trajectory_type}",
        )
        fgpp_seq.save_metadata(fgpp_dir)

        # 6. Overall Dataset Manifest
        manifest = {
            "version": "1.0",
            "dataset_id": f"synapse_{cfg.model_name}_{cfg.trajectory_type}_{cfg.num_frames}f",
            "model": model.get_summary(),
            "trajectory": {
                "type": cfg.trajectory_type,
                "radius": cfg.radius,
                "fov_degrees": cfg.fov_degrees,
                "width": cfg.width,
                "height": cfg.height,
                "fps": cfg.fps,
                "num_frames": num_frames,
            },
            "splits": {
                "train_count": len(train_indices),
                "val_count": len(val_indices_list),
                "train_frames": train_indices,
                "val_frames": val_indices_list,
            },
            "artifacts": {
                "video_all": "video_all.mp4",
                "video_train": "video_train.mp4",
                "video_val": "video_val.mp4",
                "contact_sheet": "contact_sheet.png",
                "frames_dir": "frames",
                "fgpp_dir": "fgpp",
                "visualizations_dir": "visualizations",
            },
            "frames": frame_records,
        }

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return out_dir
