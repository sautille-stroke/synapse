"""FGPP multi-channel OpenEXR I/O, sanitizer, and resize utilities.

Provides lossless round-trip export/import of 15-channel FGPP sequences
to OpenEXR 2.0 files, plus frame sanitisation and resize helpers.

All heavy dependencies (``OpenEXR``, ``Imath``, ``cv2``) are lazy-imported
inside the functions that need them.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from synapse.errors import (
    AssetNotFoundError,
    BackendUnavailableError,
    ContractValidationError,
)
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
    *,
    strict: bool = True,
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
    try:
        import Imath
        import OpenEXR
    except ImportError as e:
        raise BackendUnavailableError(
            f"OpenEXR / Imath backend unavailable: {e}",
            agent_fix_hint="Install OpenEXR via pip install 'synapse[image]'.",
        ) from e

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Strict mode: require all 15 channels and sanitize boundary values
    if strict:
        from synapse.fgpp.schema import FGPP_NUM_CHANNELS, get_channel_names

        required_channels = set(get_channel_names())
        for frame in seq.frames:
            avail: set[str] = set()
            for group, ch_names in _COMPONENT_ORDER.items():
                field_name = _GROUP_TO_FIELD[group]
                if getattr(frame, field_name, None) is not None:
                    avail.update(ch_names)
            missing = required_channels - avail
            if missing:
                raise ContractValidationError(
                    f"Strict export: frame {frame.frame_index} missing channels: {sorted(missing)}.",
                    agent_fix_hint="Provide all 15 FGPP channels or set strict=False.",
                )
            # Boundary value sanitization
            if frame.depth is not None:
                if not np.all(np.isfinite(frame.depth)):
                    raise ContractValidationError(
                        f"Strict export: frame {frame.frame_index} depth contains NaN/Inf.",
                        agent_fix_hint="Sanitize depth with sanitize_fgpp_frame() before export.",
                    )

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
                    # Native uint32 for OpenEXR storage without float32 bit-reinterpretation
                    raw = np.ascontiguousarray(arr, dtype=np.uint32)
                    chan_types[ch_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.UINT))
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
        try:
            exr_file.writePixels(chan_data)
        finally:
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
    allow_missing_optional: bool = False,
) -> FGPPFrame:
    """Read a single EXR file into an ``FGPPFrame``."""
    if not exr_path.exists():
        raise AssetNotFoundError(
            f"EXR file does not exist: {exr_path}",
            agent_fix_hint="Ensure the EXR sequence files have been generated at the specified path.",
        )

    try:
        import Imath
        import OpenEXR
    except ImportError as e:
        raise BackendUnavailableError(
            f"OpenEXR / Imath backend unavailable: {e}",
            agent_fix_hint="Install OpenEXR via pip install 'synapse[image]'.",
        ) from e

    try:
        exr = OpenEXR.InputFile(str(exr_path))
    except Exception as e:
        raise ContractValidationError(
            f"Corrupted or invalid EXR file {exr_path}: {e}",
            agent_fix_hint="Regenerate or verify the EXR file integrity.",
        ) from e

    try:
        dw = exr.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        # Validate dataWindow matches camera resolution
        if camera.width > 0 and camera.height > 0:
            if w != camera.width or h != camera.height:
                raise ContractValidationError(
                    f"EXR {exr_path} dataWindow ({w}x{h}) does not match "
                    f"camera resolution ({camera.width}x{camera.height}).",
                    agent_fix_hint="Regenerate EXR files with matching resolution or fix camera metadata.",
                )

        available = set(exr.header()["channels"].keys())

        # Validate PixelType of each FGPP channel present in the EXR
        import Imath as _Imath

        _IMATH_PT_MAP = {
            _Imath.PixelType.UINT: PixelType.UINT32,
            _Imath.PixelType.HALF: PixelType.HALF,
            _Imath.PixelType.FLOAT: PixelType.FLOAT,
        }
        header_channels = exr.header().get("channels", {})
        for ch_name, (_, _, expected_pt) in _CHANNEL_MAP.items():
            if ch_name not in available:
                continue
            ch_info = header_channels.get(ch_name)
            if ch_info is None or not hasattr(ch_info, "type"):
                continue
            actual_imath_pt = ch_info.type
            pt_val = getattr(actual_imath_pt, "v", actual_imath_pt)
            actual_pt = _IMATH_PT_MAP.get(pt_val)
            # Allow legacy FLOAT storage for Cryptomatte.ID (backward compat)
            if ch_name == "Cryptomatte.ID" and pt_val == _Imath.PixelType.FLOAT:
                continue
            if actual_pt is not None and actual_pt != expected_pt:
                raise ContractValidationError(
                    f"EXR {exr_path} channel '{ch_name}' PixelType mismatch: "
                    f"expected {expected_pt.name}, got {actual_pt.name}.",
                    agent_fix_hint=f"Ensure channel '{ch_name}' uses {expected_pt.name} PixelType in EXR export.",
                )

        # Reject unknown channel group names
        if channels is not None:
            known_groups = set(_COMPONENT_ORDER.keys())
            unknown = set(channels) - known_groups
            if unknown:
                raise ContractValidationError(
                    f"Unknown FGPP channel groups: {sorted(unknown)}. Valid groups: {sorted(known_groups)}",
                    agent_fix_hint=f"Use only valid channel group names: {sorted(known_groups)}",
                )

        # If channels specified, validate requested groups are present
        if channels is not None and not allow_missing_optional:
            for group in channels:
                ch_names = _COMPONENT_ORDER.get(group)
                if ch_names is not None:
                    missing = [cn for cn in ch_names if cn not in available]
                    if missing:
                        raise ContractValidationError(
                            f"Missing required channel(s) in EXR {exr_path}: {missing}",
                            agent_fix_hint=f"Ensure the EXR file contains all requested channels: {channels}",
                        )

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

                arr: np.ndarray
                if pt == PixelType.UINT32:
                    header_channels = exr.header().get("channels", {})
                    ch_info = header_channels.get(ch_name)
                    is_legacy_float = (
                        ch_info is not None
                        and hasattr(ch_info, "type")
                        and ch_info.type == Imath.PixelType(Imath.PixelType.FLOAT)
                    )
                    if is_legacy_float:
                        warnings.warn(
                            f"EXR channel '{ch_name}' is stored as legacy FLOAT; decoding via bit reinterpret view. "
                            "Please migrate to native UINT32 OpenEXR format.",
                            UserWarning,
                            stacklevel=2,
                        )
                        arr = (
                            np.frombuffer(raw_bytes, dtype=np.float32).reshape(h, w).view(np.uint32)
                        )
                    else:
                        arr = np.frombuffer(raw_bytes, dtype=np.uint32).reshape(h, w)
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
    finally:
        exr.close()

    # Depth is mandatory when explicitly requested or when loading all channels
    depth_required = channels is None or "depth" in channels
    if depth_required and "depth" not in field_arrays:
        raise ContractValidationError(
            f"EXR {exr_path} missing required depth channel 'CameraLinearDepth.Z'.",
            agent_fix_hint="Ensure the EXR file contains the CameraLinearDepth.Z channel.",
        )

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
    allow_missing_optional: bool = False,
) -> Iterator[FGPPFrame]:
    """Yield frames one at a time (lazy loading)."""
    for meta_frame in meta_seq.frames:
        idx = meta_frame.frame_index
        exr_path = input_dir / f"frame_{idx:06d}.exr"
        if not exr_path.exists():
            raise AssetNotFoundError(
                f"EXR file for frame {idx} does not exist: {exr_path}",
                agent_fix_hint="Ensure all frame EXR files exist in the sequence directory.",
            )
        yield _read_exr_frame(
            exr_path,
            meta_frame.camera,
            idx,
            channel_set,
            allow_missing_optional=allow_missing_optional,
        )


def import_fgpp(
    input_dir: str | Path,
    *,
    lazy: bool = True,
    channels: Sequence[str] | None = None,
    allow_missing_optional: bool = False,
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
    allow_missing_optional:
        If ``True``, missing requested channels do not raise ContractValidationError.

    Returns
    -------
    FGPPSequence | Iterator[FGPPFrame]
        Either a fully loaded sequence or a lazy frame iterator.
    """
    inp = Path(input_dir)
    if not inp.exists():
        raise AssetNotFoundError(
            f"FGPP input directory does not exist: {inp}",
            agent_fix_hint="Ensure the specified sequence directory exists on disk.",
        )

    meta_seq = FGPPSequence.load_metadata(inp)
    channel_set = set(channels) if channels is not None else None

    if lazy:
        return _iter_fgpp_frames(
            inp, meta_seq, channel_set, allow_missing_optional=allow_missing_optional
        )

    frames: list[FGPPFrame] = []
    for meta_frame in meta_seq.frames:
        idx = meta_frame.frame_index
        exr_path = inp / f"frame_{idx:06d}.exr"
        if not exr_path.exists():
            raise AssetNotFoundError(
                f"EXR file for frame {idx} does not exist: {exr_path}",
                agent_fix_hint="Ensure all frame EXR files exist in the sequence directory.",
            )
        frames.append(
            _read_exr_frame(
                exr_path,
                meta_frame.camera,
                idx,
                channel_set,
                allow_missing_optional=allow_missing_optional,
            )
        )

    return FGPPSequence(frames=frames, fps=meta_seq.fps, scene_id=meta_seq.scene_id)


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
    # Physical range: clamp to (0, z_far]
    depth = np.clip(depth, np.finfo(np.float32).tiny, z_far)

    crypto = None
    if frame.cryptomatte_id is not None:
        crypto = frame.cryptomatte_id.copy()
        crypto[bad_depth] = np.uint32(0)

    normal = None
    if frame.world_normal is not None:
        normal = frame.world_normal.astype(np.float32).copy()
        # NaN / Inf -> 0.0 before norm calculation to avoid silent bypass
        normal = np.where(np.isfinite(normal), normal, 0.0)
        norms = np.linalg.norm(normal, axis=-1, keepdims=True)
        zero_mask = norms.squeeze(-1) < 1e-8
        # Avoid division by zero
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        normal = normal / safe_norms
        normal[zero_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        normal = normal.astype(frame.world_normal.dtype)

    mv = None
    if frame.motion_vector is not None:
        mv = frame.motion_vector.copy()
        mv_bad = ~np.isfinite(mv.astype(np.float32))
        mv = np.where(mv_bad, np.float16(0.0), mv)

    dr = None
    if frame.direct_radiance is not None:
        dr = frame.direct_radiance.copy()
        dr_bad = ~np.isfinite(dr.astype(np.float32))
        dr = np.where(dr_bad, np.float16(0.0), dr)
        # Physical range: non-negative, capped at HALF max to prevent Inf
        dr = np.clip(dr, np.float16(0.0), np.float16(65504.0))

    bc = None
    if frame.base_color is not None:
        bc = frame.base_color.copy()
        bc_bad = ~np.isfinite(bc.astype(np.float32))
        bc = np.where(bc_bad, np.float16(0.0), bc)
        # Physical range: albedo in [0, 1]
        bc = np.clip(bc, np.float16(0.0), np.float16(1.0))

    rm = None
    if frame.roughness_metallic is not None:
        rm = frame.roughness_metallic.copy()
        rm_bad = ~np.isfinite(rm.astype(np.float32))
        rm = np.where(rm_bad, np.float16(0.5), rm)
        # Physical range: roughness [0.04, 1.0] for channel 0, metallic [0, 1] for channel 1
        if rm.ndim == 3 and rm.shape[-1] >= 2:
            rm[..., 0] = np.clip(rm[..., 0], np.float16(0.04), np.float16(1.0))
            rm[..., 1] = np.clip(rm[..., 1], np.float16(0.0), np.float16(1.0))
        else:
            rm = np.clip(rm, np.float16(0.0), np.float16(1.0))

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
        import cv2

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

    depth = _resize_nearest(frame.depth)
    dr = _resize_float(frame.direct_radiance) if frame.direct_radiance is not None else None

    # Re-normalize world_normal to maintain unit length invariance after spatial interpolation
    wn = None
    if frame.world_normal is not None:
        wn_resized = _resize_float(frame.world_normal).astype(np.float32)
        norms = np.linalg.norm(wn_resized, axis=-1, keepdims=True)
        valid = norms.squeeze(-1) > 1e-6
        safe_norms = np.where(norms > 1e-6, norms, 1.0)
        wn_norm = wn_resized / safe_norms
        wn_norm[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        wn = wn_norm.astype(frame.world_normal.dtype)

    # Scale motion vector displacements by resolution change ratio
    mv = None
    if frame.motion_vector is not None:
        mv_resized = _resize_float(frame.motion_vector)
        mv = mv_resized.copy()
        mv[..., 0] *= sx
        mv[..., 1] *= sy

    bc = _resize_float(frame.base_color) if frame.base_color is not None else None
    rm = _resize_float(frame.roughness_metallic) if frame.roughness_metallic is not None else None
    crypto = _resize_nearest(frame.cryptomatte_id) if frame.cryptomatte_id is not None else None

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
