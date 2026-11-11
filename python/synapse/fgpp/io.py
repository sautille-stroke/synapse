"""FGPP multi-channel OpenEXR I/O, sanitizer, and resize utilities.

Provides lossless round-trip export/import of 15-channel FGPP sequences
to OpenEXR 2.0 files, plus frame sanitisation and resize helpers.

All heavy dependencies (``OpenEXR``, ``Imath``, ``cv2``) are lazy-imported
inside the functions that need them.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload

import numpy as np

from synapse.fgpp.frame import CameraParams, FGPPFrame, FGPPSequence
from synapse.fgpp.schema import (
    FGPP_CHANNEL_SPECS,
    PixelType,
    get_logical_groups,
)

# ---------------------------------------------------------------------------
# Channel <-> FGPPFrame field mapping
# ---------------------------------------------------------------------------

# Map logical group to the EXR channel names (in FGPP_CHANNEL_SPECS order)
# and the FGPPFrame attribute name + component index.
_GROUP_TO_FIELD: dict[str, str] = {
    "base_color": "base_color",
    "depth": "depth",
    "cryptomatte_id": "cryptomatte_id",
    "direct_radiance": "direct_radiance",
    "motion_vector": "motion_vector",
    "roughness_metallic": "roughness_metallic",
    "world_normal": "world_normal",
}

# EXR channel name -> (field_name, component_index, pixel_type)
_CHANNEL_MAP: dict[str, tuple[str, int, PixelType]] = {}

_LOGICAL_GROUPS = get_logical_groups()
for _group, _channels in _LOGICAL_GROUPS.items():
    _field = _GROUP_TO_FIELD[_group]
    for _ci, _ch_name in enumerate(_channels):
        _pt = next(pt for n, pt, _ in FGPP_CHANNEL_SPECS if n == _ch_name)
        _CHANNEL_MAP[_ch_name] = (_field, _ci, _pt)

# Component ordering inside multi-component fields.
# EXR channels are ASCII-sorted, so BaseColor.B=0, G=1, R=2 in the spec,
# but in the numpy array we store [R, G, B] (index 0=R, 1=G, 2=B).
# We need an explicit mapping per group.
_COMPONENT_ORDER: dict[str, list[str]] = {
    "base_color": ["BaseColor.R", "BaseColor.G", "BaseColor.B"],
    "direct_radiance": [
        "DirectRadiance.R",
        "DirectRadiance.G",
        "DirectRadiance.B",
    ],
    "world_normal": ["WorldNormal.X", "WorldNormal.Y", "WorldNormal.Z"],
    "motion_vector": ["MotionVector.X", "MotionVector.Y"],
    "roughness_metallic": [
        "RoughnessMetallic.R",
        "RoughnessMetallic.G",
    ],
    "depth": ["CameraLinearDepth.Z"],
    "cryptomatte_id": ["Cryptomatte.ID"],
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_fgpp(
    seq: FGPPSequence,
    output_dir: str | Path,
    compression: str = "ZIP",
) -> Path:
    """Export an ``FGPPSequence`` to a directory of OpenEXR files.

    Each frame is written as ``frame_{idx:06d}.exr`` with all 15 scalar
    channels at their native precision.  ``metadata.json`` is written
    via :meth:`FGPPSequence.save_metadata`.

    Parameters
    ----------
    seq:
        Sequence to export.
    output_dir:
        Target directory (created if missing).
    compression:
        OpenEXR compression codec (``ZIP``, ``NONE``, ``ZIPS``, …).

    Returns
    -------
    Path
        The *output_dir* path.
    """
    import Imath  # type: ignore[import-untyped]
    import OpenEXR  # type: ignore[import-untyped]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve compression — Imath.Compression is a special type, not plain int
    _comp_attr = compression.upper() + "_COMPRESSION"
    compress_val = getattr(Imath.Compression, _comp_attr, None)
    if compress_val is None:
        compress_val = Imath.Compression.ZIP_COMPRESSION
    # Wrap in Imath.Compression to get the correct type
    compress_obj = Imath.Compression(compress_val)

    for frame in seq.frames:
        h, w = frame.height, frame.width
        header = OpenEXR.Header(w, h)
        header["compression"] = compress_obj

        chan_data: dict[str, bytes] = {}
        chan_types: dict[str, Imath.Channel] = {}

        for group, ch_names in _COMPONENT_ORDER.items():
            field_name = _GROUP_TO_FIELD[group]
            arr = getattr(frame, field_name, None)
            if arr is None:
                continue

            for ci, ch_name in enumerate(ch_names):
                _, _, pt = _CHANNEL_MAP[ch_name]

                if pt == PixelType.UINT32:
                    # Bit-reinterpret uint32 → float32 for EXR storage
                    raw = arr.astype(np.uint32).view(np.float32)
                    chan_types[ch_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    chan_data[ch_name] = raw.tobytes()
                elif pt == PixelType.FLOAT:
                    # Scalar float32 channel (depth)
                    raw = arr.astype(np.float32)
                    chan_types[ch_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    chan_data[ch_name] = raw.tobytes()
                elif pt == PixelType.HALF:
                    # Multi-component half channel
                    if arr.ndim == 2:
                        component = arr.astype(np.float16)
                    else:
                        component = arr[..., ci].astype(np.float16)
                    chan_types[ch_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
                    chan_data[ch_name] = component.tobytes()

        header["channels"] = chan_types
        exr_path = out / f"frame_{frame.frame_index:06d}.exr"
        exr_file = OpenEXR.OutputFile(str(exr_path), header)
        exr_file.writePixels(chan_data)
        exr_file.close()

    # Write metadata.json
    seq.save_metadata(out)
    return out


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _read_exr_frame(
    exr_path: Path,
    camera: CameraParams,
    frame_index: int,
    channels: set[str] | None = None,
) -> FGPPFrame:
    """Read a single EXR file into an ``FGPPFrame``."""
    import Imath  # type: ignore[import-untyped]
    import OpenEXR  # type: ignore[import-untyped]

    exr = OpenEXR.InputFile(str(exr_path))
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    available = set(exr.header()["channels"].keys())
    field_arrays: dict[str, np.ndarray] = {}

    for group, ch_names in _COMPONENT_ORDER.items():
        # Skip groups not requested
        if channels is not None and group not in channels:
            continue

        # Check all channels in this group are available
        if not all(cn in available for cn in ch_names):
            continue

        components: list[np.ndarray] = []
        for ch_name in ch_names:
            _, _, pt = _CHANNEL_MAP[ch_name]
            raw_bytes = exr.channel(ch_name)

            if pt == PixelType.UINT32:
                # Stored as float32 bit-pattern
                arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(h, w)
                arr = arr.view(np.uint32)
                components.append(arr)
            elif pt == PixelType.FLOAT:
                arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(h, w)
                components.append(arr)
            elif pt == PixelType.HALF:
                arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(h, w)
                components.append(arr)

        field_name = _GROUP_TO_FIELD[group]
        if len(components) == 1:
            field_arrays[field_name] = components[0].copy()
        else:
            field_arrays[field_name] = np.stack(components, axis=-1).copy()

    exr.close()

    return FGPPFrame(
        camera=camera,
        depth=field_arrays.get("depth", np.zeros((h, w), dtype=np.float32)),
        direct_radiance=field_arrays.get("direct_radiance"),
        world_normal=field_arrays.get("world_normal"),
        motion_vector=field_arrays.get("motion_vector"),
        base_color=field_arrays.get("base_color"),
        roughness_metallic=field_arrays.get("roughness_metallic"),
        cryptomatte_id=field_arrays.get("cryptomatte_id"),
        frame_index=frame_index,
    )


def _iter_fgpp_frames(
    input_dir: Path,
    meta_seq: FGPPSequence,
    channel_set: set[str] | None,
) -> Iterator[FGPPFrame]:
    """Yield frames one at a time (lazy loading)."""
    for meta_frame in meta_seq.frames:
        idx = meta_frame.frame_index
        exr_path = input_dir / f"frame_{idx:06d}.exr"
        if exr_path.exists():
            yield _read_exr_frame(
                exr_path, meta_frame.camera, idx, channel_set
            )


@overload
def import_fgpp(
    input_dir: str | Path,
    *,
    lazy: bool = True,
    channels: Sequence[str] | None = None,
) -> FGPPSequence | Iterator[FGPPFrame]: ...


def import_fgpp(
    input_dir: str | Path,
    *,
    lazy: bool = True,
    channels: Sequence[str] | None = None,
) -> FGPPSequence | Iterator[FGPPFrame]:
    """Import an FGPP sequence from a directory of OpenEXR files.

    Parameters
    ----------
    input_dir:
        Directory containing ``metadata.json`` and ``frame_*.exr`` files.
    lazy:
        If ``True`` (default), return a generator that yields one
        :class:`FGPPFrame` at a time to limit peak memory.
        If ``False``, load all frames into an :class:`FGPPSequence`.
    channels:
        Optional subset of logical groups to load (e.g.
        ``["depth", "world_normal"]``).  ``None`` loads all.

    Returns
    -------
    FGPPSequence | Iterator[FGPPFrame]
        Either a fully loaded sequence or a lazy frame iterator.
    """
    inp = Path(input_dir)
    meta_seq = FGPPSequence.load_metadata(inp)
    channel_set = set(channels) if channels is not None else None

    if lazy:
        return _iter_fgpp_frames(inp, meta_seq, channel_set)

    frames: list[FGPPFrame] = []
    for meta_frame in meta_seq.frames:
        idx = meta_frame.frame_index
        exr_path = inp / f"frame_{idx:06d}.exr"
        if exr_path.exists():
            frames.append(
                _read_exr_frame(exr_path, meta_frame.camera, idx, channel_set)
            )

    return FGPPSequence(
        frames=frames, fps=meta_seq.fps, scene_id=meta_seq.scene_id
    )


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


def sanitize_fgpp_frame(frame: FGPPFrame) -> FGPPFrame:
    """Sanitize an FGPP frame by clamping invalid values.

    - ``depth``: NaN / Inf / ≤ 0 → ``camera.z_far``
    - ``cryptomatte_id``: where depth was invalid → ``0``
    - ``world_normal``: re-normalise; zero-length → ``[0, 0, 1]``
    - ``motion_vector``: NaN / Inf → ``0``

    Returns a **new** ``FGPPFrame``; the input is not mutated.
    """
    z_far = frame.camera.z_far
    depth = frame.depth.copy()
    bad_depth = ~np.isfinite(depth) | (depth <= 0.0)
    depth[bad_depth] = z_far

    crypto = None
    if frame.cryptomatte_id is not None:
        crypto = frame.cryptomatte_id.copy()
        crypto[bad_depth] = np.uint32(0)

    normal = None
    if frame.world_normal is not None:
        normal = frame.world_normal.astype(np.float32).copy()
        norms = np.linalg.norm(normal, axis=-1, keepdims=True)
        zero_mask = norms.squeeze(-1) < 1e-8
        # Avoid division by zero
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        normal = normal / safe_norms
        normal[zero_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        normal = normal.astype(np.float16)

    mv = None
    if frame.motion_vector is not None:
        mv = frame.motion_vector.copy()
        mv_bad = ~np.isfinite(mv.astype(np.float32))
        mv = np.where(mv_bad, np.float16(0.0), mv)

    dr = None
    if frame.direct_radiance is not None:
        dr = frame.direct_radiance.copy()

    bc = None
    if frame.base_color is not None:
        bc = frame.base_color.copy()

    rm = None
    if frame.roughness_metallic is not None:
        rm = frame.roughness_metallic.copy()

    return FGPPFrame(
        camera=copy.deepcopy(frame.camera),
        depth=depth,
        direct_radiance=dr,
        world_normal=normal,
        motion_vector=mv,
        base_color=bc,
        roughness_metallic=rm,
        cryptomatte_id=crypto,
        frame_index=frame.frame_index,
    )


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


def _nearest_resize_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pure-NumPy nearest-neighbour resize for 2-D arrays."""
    h, w = arr.shape[:2]
    row_idx = (np.arange(target_h) * h / target_h).astype(int)
    col_idx = (np.arange(target_w) * w / target_w).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return arr[np.ix_(row_idx, col_idx)]


def resize_fgpp_frame(
    frame: FGPPFrame,
    target_hw: tuple[int, int],
) -> FGPPFrame:
    """Resize all channels of an FGPP frame.

    Float channels use ``cv2.INTER_AREA`` (or ``INTER_LINEAR`` for
    upscale).  ``cryptomatte_id`` and ``depth`` are always resized with
    **nearest-neighbour** to prevent interpolation artifacts.

    If OpenCV is not installed, a pure-NumPy nearest-neighbour fallback
    is used for all channels.

    Parameters
    ----------
    frame:
        Source frame.
    target_hw:
        ``(target_height, target_width)``.

    Returns
    -------
    FGPPFrame
        New frame with resized channels and updated camera.
    """
    th, tw = target_hw
    oh, ow = frame.height, frame.width

    # Try to import cv2 for quality resize of float channels
    try:
        import cv2  # type: ignore[import-untyped]
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    def _resize_float(arr: np.ndarray) -> np.ndarray:
        if not _has_cv2:
            return _nearest_resize_2d(arr, th, tw)
        # cv2 needs float32 at minimum
        was_f16 = arr.dtype == np.float16
        work = arr.astype(np.float32) if was_f16 else arr
        interp = cv2.INTER_AREA if (th < oh or tw < ow) else cv2.INTER_LINEAR
        out = cv2.resize(work, (tw, th), interpolation=interp)
        if was_f16:
            out = out.astype(np.float16)
        return out

    def _resize_nearest(arr: np.ndarray) -> np.ndarray:
        if not _has_cv2:
            return _nearest_resize_2d(arr, th, tw)
        if arr.dtype == np.uint32:
            # cv2 doesn't natively support uint32; use int32 view
            work = arr.view(np.int32)
            out = cv2.resize(work, (tw, th), interpolation=cv2.INTER_NEAREST)
            return out.view(np.uint32)
        return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)

    depth = _resize_nearest(frame.depth)
    dr = _resize_float(frame.direct_radiance) if frame.direct_radiance is not None else None
    wn = _resize_float(frame.world_normal) if frame.world_normal is not None else None
    mv = _resize_float(frame.motion_vector) if frame.motion_vector is not None else None
    bc = _resize_float(frame.base_color) if frame.base_color is not None else None
    rm = _resize_float(frame.roughness_metallic) if frame.roughness_metallic is not None else None
    crypto = _resize_nearest(frame.cryptomatte_id) if frame.cryptomatte_id is not None else None

    # Scale camera intrinsics proportionally
    sx = tw / ow
    sy = th / oh
    cam = copy.deepcopy(frame.camera)
    cam.cx *= sx
    cam.cy *= sy
    cam.fx *= sx
    cam.fy *= sy
    cam.width = tw
    cam.height = th

    return FGPPFrame(
        camera=cam,
        depth=depth,
        direct_radiance=dr,
        world_normal=wn,
        motion_vector=mv,
        base_color=bc,
        roughness_metallic=rm,
        cryptomatte_id=crypto,
        frame_index=frame.frame_index,
    )
