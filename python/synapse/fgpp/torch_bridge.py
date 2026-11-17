"""Zero-copy DLPack / PyTorch tensor bridge for FGPP frames.

``torch`` is lazy-imported — this module can be safely imported even
without PyTorch installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synapse.fgpp.frame import CameraParams, FGPPFrame

# Channel fields that carry image data
_IMAGE_FIELDS = (
    "depth",
    "direct_radiance",
    "world_normal",
    "motion_vector",
    "base_color",
    "roughness_metallic",
    "cryptomatte_id",
)


def frame_to_tensors(
    frame: FGPPFrame,
    device: str = "cpu",
) -> dict[str, Any]:
    """Convert an :class:`FGPPFrame` to a dict of PyTorch tensors.

    CPU-contiguous arrays use ``torch.from_numpy`` for **zero-copy**
    sharing.  ``cryptomatte_id`` (uint32) is reinterpreted as ``int32``
    because PyTorch has limited uint32 operator support.

    Parameters
    ----------
    frame:
        Source frame.
    device:
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).

    Returns
    -------
    dict[str, torch.Tensor]
        Keys match :class:`FGPPFrame` field names.
    """
    try:
        import torch
    except ImportError as e:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "PyTorch is required for tensor bridge",
            agent_fix_hint="Install PyTorch: pip install 'synapse[gpu]'",
        ) from e
    tensors: dict[str, Any] = {}

    for field in _IMAGE_FIELDS:
        arr = getattr(frame, field, None)
        if arr is None:
            continue

        if field == "cryptomatte_id":
            # uint32 → int32 bit-level reinterpret
            arr_u32 = np.ascontiguousarray(arr, dtype=np.uint32)
            arr_i32 = arr_u32.view(np.int32)
            t = torch.from_numpy(arr_i32)
        else:
            arr_c = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr_c)

        if device != "cpu":
            t = t.to(device)

        tensors[field] = t

    return tensors


def tensors_to_frame(
    tensors: dict[str, Any],
    camera: CameraParams,
    frame_index: int = 0,
) -> FGPPFrame:
    """Reconstruct an :class:`FGPPFrame` from a dict of PyTorch tensors.

    Reverses :func:`frame_to_tensors`, including the int32 → uint32
    reinterpretation for ``cryptomatte_id``.

    Parameters
    ----------
    tensors:
        Dict of tensors keyed by field name.
    camera:
        Camera parameters for this frame.
    frame_index:
        Frame index in the sequence.

    Returns
    -------
    FGPPFrame
    """

    kwargs: dict[str, Any] = {
        "camera": camera,
        "frame_index": frame_index,
    }

    for field in _IMAGE_FIELDS:
        if field not in tensors:
            continue

        t = tensors[field]
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        if hasattr(t, "contiguous"):
            t = t.contiguous()

        if field == "cryptomatte_id":
            # int32 tensor → uint32 numpy
            arr = t.numpy().view(np.uint32)
        else:
            arr = t.numpy()

        kwargs[field] = arr

    if "depth" not in kwargs:
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            "tensors dict must contain 'depth'",
            agent_fix_hint="Ensure the tensor dict includes a 'depth' key.",
        )

    return FGPPFrame(**kwargs)
