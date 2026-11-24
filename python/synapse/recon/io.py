"""Video ingestion and FGPP EXR sequence projection helpers for ``synapse.recon``.

All heavy media codecs (``cv2``, ``av``, ``PIL``, ``OpenEXR``) are lazy-imported
so base SDK imports remain zero-overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    ContractValidationError,
)
from synapse.recon.types import CameraTrajectory, MeshGeometry

__all__ = [
    "VideoMetadata",
    "export_recon_fgpp_sequence",
    "load_video_frames",
]

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing loaded video frames."""

    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    source_path: str
    color_space: str = "srgb"


def _resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize ``(T, H, W, 3)`` array to ``(T, target_h, target_w, 3)``."""
    tw, th = int(target_size[0]), int(target_size[1])
    if tw <= 0 or th <= 0:
        raise ContractValidationError(
            f"Invalid target_size={target_size}; width and height must be positive.",
            agent_fix_hint="Pass target_size=(width, height) with positive integers.",
        )
    if frames.shape[1] == th and frames.shape[2] == tw:
        return frames
    try:
        import cv2

        out = np.empty((frames.shape[0], th, tw, frames.shape[3]), dtype=frames.dtype)
        for i in range(frames.shape[0]):
            out[i] = cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA)
        return out
    except ImportError:
        pass
    try:
        from PIL import Image

        out_list = []
        is_float = np.issubdtype(frames.dtype, np.floating)
        for i in range(frames.shape[0]):
            arr_u8 = (
                (np.clip(frames[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                if is_float
                else frames[i].astype(np.uint8)
            )
            resized = Image.fromarray(arr_u8).resize((tw, th))
            res_arr = np.asarray(resized)
            if is_float:
                res_arr = res_arr.astype(frames.dtype) / 255.0
            out_list.append(res_arr)
        return np.stack(out_list, axis=0)
    except ImportError:
        # Nearest-neighbor NumPy index sampling fallback
        ys = np.linspace(0, frames.shape[1] - 1, th).astype(np.int64)
        xs = np.linspace(0, frames.shape[2] - 1, tw).astype(np.int64)
        return np.asarray(frames[:, ys[:, None], xs[None, :], :])


def _load_image_directory(
    dir_path: Path,
    *,
    max_frames: int | None,
    stride: int,
) -> tuple[np.ndarray, VideoMetadata]:
    """Load sorted image sequence or .npy slices from a directory."""
    candidates = sorted(
        p
        for p in dir_path.iterdir()
        if p.is_file() and (p.suffix.lower() in _IMAGE_EXTENSIONS or p.suffix.lower() == ".npy")
    )
    if not candidates:
        raise AssetNotFoundError(
            f"No supported image or .npy frames found in directory: {dir_path}",
            agent_fix_hint="Populate the directory with .png/.jpg/.npy frame files.",
        )
    selected = candidates[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]

    frames_list: list[np.ndarray] = []
    for p in selected:
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
        else:
            try:
                from PIL import Image

                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            except ImportError as exc:
                try:
                    import cv2

                    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ContractValidationError(
                            f"Failed to decode image frame: {p}",
                            agent_fix_hint="Verify image file is not corrupted.",
                        ) from None
                    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except ImportError:
                    raise BackendUnavailableError(
                        f"Pillow or opencv-python-headless required to read {p.suffix}: {exc}",
                        agent_fix_hint="Install synapse[image].",
                    ) from exc
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        frames_list.append(arr)

    stacked = np.stack(frames_list, axis=0)
    _, h, w, _ = stacked.shape
    return stacked, VideoMetadata(
        width=int(w),
        height=int(h),
        fps=30.0,
        frame_count=int(len(stacked)),
        codec="image_sequence",
        source_path=str(dir_path),
    )


def _decode_video_container(
    path: Path,
    *,
    max_frames: int | None,
    stride: int,
) -> tuple[np.ndarray, VideoMetadata]:
    """Decode MP4/MOV/MKV/WebM via lazy cv2.VideoCapture or PyAV."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ContractValidationError(
                f"Cannot open video container: {path}",
                agent_fix_hint="Verify the video file is valid and non-corrupt.",
            )
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fps = fps_raw if fps_raw > 0.0 and np.isfinite(fps_raw) else 30.0
        frames: list[np.ndarray] = []
        idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                idx += 1
        finally:
            cap.release()
        if not frames:
            raise ContractValidationError(
                f"Video container decoded 0 frames: {path}",
                agent_fix_hint="Verify the video stream is non-empty and readable.",
            )
        stacked = np.stack(frames, axis=0)
        return stacked, VideoMetadata(
            width=int(stacked.shape[2]),
            height=int(stacked.shape[1]),
            fps=fps,
            frame_count=int(stacked.shape[0]),
            codec=path.suffix.lower().lstrip("."),
            source_path=str(path),
        )
    except ImportError:
        pass

    try:
        import av  # type: ignore[import-not-found]

        container: Any = av.open(str(path))
        frames = []
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % stride == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                if max_frames is not None and len(frames) >= max_frames:
                    break
        container.close()
        if not frames:
            raise ContractValidationError(
                f"Video container decoded 0 frames via PyAV: {path}",
                agent_fix_hint="Verify the video stream is non-empty.",
            )
        stacked = np.stack(frames, axis=0)
        return stacked, VideoMetadata(
            width=int(stacked.shape[2]),
            height=int(stacked.shape[1]),
            fps=fps,
            frame_count=int(stacked.shape[0]),
            codec=path.suffix.lower().lstrip("."),
            source_path=str(path),
        )
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Video container decoding for '{path.suffix}' requires opencv-python-headless (cv2) or PyAV (av): {exc}",
            agent_fix_hint="Install synapse[image] ('pip install opencv-python-headless') or pass .npy frames.",
        ) from exc


def load_video_frames(
    path: str | Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, VideoMetadata]:
    """Load video frames from ``.npy``, ``.npz``, an image directory, or video container.

    Supports ``.npy``/``.npz`` arrays of shape ``(T, H, W, 3)``, sorted directories of
    image frames, and standard containers (``.mp4``, ``.mov``, ``.mkv``, ``.webm``,
    ``.avi``) via lazy-loaded ``cv2`` or ``av``.
    """
    if stride <= 0:
        raise ContractValidationError(
            f"stride must be >= 1, got {stride}",
            agent_fix_hint="Pass a positive integer stride (default 1).",
        )
    if max_frames is not None and max_frames <= 0:
        raise ContractValidationError(
            f"max_frames must be >= 1 when specified, got {max_frames}",
            agent_fix_hint="Pass max_frames >= 1 or None.",
        )

    p = Path(path)
    if not p.exists():
        raise AssetNotFoundError(
            f"Video source not found: {path!r}",
            agent_fix_hint="Provide a valid path to an .npy/.npz array, image directory, or .mp4/.mov video.",
        )

    if p.is_dir():
        frames, meta = _load_image_directory(p, max_frames=max_frames, stride=stride)
    else:
        suffix = p.suffix.lower()
        if suffix == ".npy":
            try:
                arr = np.load(p, allow_pickle=False)
            except Exception as exc:
                raise ContractValidationError(
                    f"Failed to load .npy video frames from {p}: {exc}",
                    agent_fix_hint="Ensure the .npy file contains a valid numeric (T,H,W,3) array.",
                ) from exc
            if arr.ndim != 4 or arr.shape[-1] not in (3, 4) or arr.shape[0] == 0:
                raise ContractValidationError(
                    f"Invalid video .npy shape {arr.shape}; expected non-empty (T, H, W, 3).",
                    agent_fix_hint="Save frames with shape (T, H, W, 3).",
                )
            arr = arr[..., :3]
            arr = arr[::stride]
            if max_frames is not None:
                arr = arr[:max_frames]
            frames = arr
            meta = VideoMetadata(
                width=int(frames.shape[2]),
                height=int(frames.shape[1]),
                fps=30.0,
                frame_count=int(frames.shape[0]),
                codec="npy",
                source_path=str(p),
            )
        elif suffix == ".npz":
            try:
                with np.load(p, allow_pickle=False) as z:
                    key = "frames" if "frames" in z else (list(z.files)[0] if z.files else "")
                    if not key:
                        raise ContractValidationError(
                            f"Empty .npz archive: {p}",
                            agent_fix_hint="Save a 'frames' array inside the .npz archive.",
                        )
                    arr = np.asarray(z[key])
            except ContractValidationError:
                raise
            except Exception as exc:
                raise ContractValidationError(
                    f"Failed to load .npz video frames from {p}: {exc}",
                    agent_fix_hint="Verify .npz integrity.",
                ) from exc
            if arr.ndim != 4 or arr.shape[-1] not in (3, 4) or arr.shape[0] == 0:
                raise ContractValidationError(
                    f"Invalid video .npz array shape {arr.shape}; expected (T, H, W, 3).",
                    agent_fix_hint="Ensure the array shape is (T, H, W, 3).",
                )
            arr = arr[..., :3][::stride]
            if max_frames is not None:
                arr = arr[:max_frames]
            frames = arr
            meta = VideoMetadata(
                width=int(frames.shape[2]),
                height=int(frames.shape[1]),
                fps=30.0,
                frame_count=int(frames.shape[0]),
                codec="npz",
                source_path=str(p),
            )
        elif suffix in _VIDEO_EXTENSIONS:
            frames, meta = _decode_video_container(p, max_frames=max_frames, stride=stride)
        else:
            raise ContractValidationError(
                f"Unsupported video file format {suffix!r} for {p}",
                agent_fix_hint="Pass .mp4, .mov, .mkv, .webm, .npy, .npz, or a directory of images.",
            )

    if target_size is not None:
        frames = _resize_frames(frames, target_size)
        meta = VideoMetadata(
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            fps=meta.fps,
            frame_count=int(frames.shape[0]),
            codec=meta.codec,
            source_path=meta.source_path,
            color_space=meta.color_space,
        )

    return frames, meta


def export_recon_fgpp_sequence(
    frames: np.ndarray,
    camera_trajectory: CameraTrajectory | None,
    mesh: MeshGeometry,
    output_dir: str | Path,
    *,
    fps: float = 30.0,
    entity_name: str = "reconstructed_asset",
) -> list[Path]:
    """Project reconstructed geometry + camera trajectory into 15-channel FGPP EXRs.

    Writes ``frame_{i:06d}.exr`` and ``metadata.json`` under *output_dir* via
    :func:`synapse.fgpp.io.export_fgpp`.
    """
    from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
    from synapse.fgpp.io import export_fgpp
    from synapse.render_core.rasterizer import SoftwareRasterizer

    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[0] == 0:
        raise ContractValidationError(
            f"Expected (T, H, W, 3) frames for FGPP export, got {arr.shape}.",
            agent_fix_hint="Supply non-empty (T, H, W, 3) video frames.",
        )

    num_frames, height, width, _ = arr.shape
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f_raw = np.asarray(mesh.faces, dtype=np.int32)
    if len(v) == 0 or len(f_raw) == 0:
        raise ContractValidationError(
            "Cannot project empty reconstructed mesh to FGPP sequence.",
            agent_fix_hint="Ensure ReconPipeline produced non-empty mesh vertices and faces.",
        )

    if f_raw.ndim == 2 and f_raw.shape[1] == 4:
        tris = np.concatenate([f_raw[:, [0, 1, 2]], f_raw[:, [0, 2, 3]]], axis=0)
    else:
        tris = f_raw

    norms = (
        np.asarray(mesh.normals, dtype=np.float64)
        if mesh.normals is not None and len(mesh.normals) == len(v)
        else np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(v), 1))
    )

    intr = camera_trajectory.intrinsics if camera_trajectory else {}
    fx = float(intr.get("fx") or width)
    fy = float(intr.get("fy") or width)
    raw_cx = intr.get("cx")
    raw_cy = intr.get("cy")
    cx = float(raw_cx) if raw_cx is not None else width * 0.5
    cy = float(raw_cy) if raw_cy is not None else height * 0.5

    rasterizer = SoftwareRasterizer(width=width, height=height)
    fgpp_frames: list[FGPPFrame] = []
    # OpenCV w2c (+Z fwd, +Y down) -> OpenGL/CameraParams w2c (-Z fwd, +Y up)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])

    exts = camera_trajectory.extrinsics if camera_trajectory else []
    prev_cam: CameraParams | None = None
    for idx in range(num_frames):
        w2c_cv = (
            np.asarray(exts[idx].transform, dtype=np.float64)
            if idx < len(exts)
            else np.eye(4, dtype=np.float64)
        )
        # Ensure camera observes origin when trajectory identity sits at origin
        if np.allclose(w2c_cv, np.eye(4)):
            view_gl = np.eye(4, dtype=np.float64)
            view_gl[2, 3] = -2.5
        else:
            view_gl = cv_to_gl @ w2c_cv

        try:
            world_gl = np.linalg.inv(view_gl)
        except np.linalg.LinAlgError:
            world_gl = np.eye(4, dtype=np.float64)

        cam = CameraParams(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            view_matrix=view_gl,
            world_matrix=world_gl,
        )
        rendered = rasterizer.render(
            vertices=v,
            faces=tris,
            normals=norms,
            camera=cam,
            entity_name=entity_name,
            prev_camera=prev_cam,
        )
        rendered.frame_index = idx
        fgpp_frames.append(rendered)
        prev_cam = cam

    seq = FGPPSequence(frames=fgpp_frames, fps=float(fps), scene_id=entity_name)
    out_path = export_fgpp(seq, output_dir, strict=True)
    return sorted(out_path.glob("frame_*.exr"))
