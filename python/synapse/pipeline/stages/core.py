"""Core built-in JobBundle stage handlers (FGPP validate, neural video, style, composite)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from synapse.pipeline.bundle_worker import BundleManifest, register_stage


@register_stage("fgpp_validate")
def handle_fgpp_validate(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Validate an FGPP sequence directory.

    Expects ``manifest.inputs["fgpp_dir"]`` to point to a directory
    containing ``metadata.json``.

    Returns:
        Dict with ``valid``, ``schema_version``, ``num_frames``, and
        ``output_paths`` (empty for a read-only validation stage).
    """
    from synapse.fgpp.schema import FGPP_SCHEMA_VERSION

    fgpp_dir = Path(manifest.inputs.get("fgpp_dir", ""))
    if not fgpp_dir.is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"FGPP directory does not exist: {fgpp_dir}",
            agent_fix_hint=(
                f"Ensure '{fgpp_dir}' exists and contains "
                "metadata.json produced by fulgur or synapse-bench."
            ),
        )

    meta_path = fgpp_dir / "metadata.json"
    if not meta_path.is_file():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"metadata.json not found in {fgpp_dir}",
            agent_fix_hint=(f"Run the FGPP exporter to generate metadata.json in '{fgpp_dir}'."),
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))

    schema_ver = raw.get("schema_version", "")
    if schema_ver != FGPP_SCHEMA_VERSION:
        from synapse.errors import ContractValidationError

        raise ContractValidationError(
            f"Schema version mismatch: expected '{FGPP_SCHEMA_VERSION}', got '{schema_ver}'",
            agent_fix_hint=(
                "Re-export with a compatible FGPP producer "
                f"(expected schema '{FGPP_SCHEMA_VERSION}')."
            ),
        )

    num_frames = raw.get(
        "num_frames",
        raw.get("total_frames", 0),
    )

    return {
        "valid": True,
        "schema_version": schema_ver,
        "num_frames": num_frames,
        "output_paths": {},
    }


@register_stage("neural_video_generate")
def handle_neural_video_generate(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Generate video frames using a neural adapter with OOM protection.

    OOM fallback logic:

    1. Estimate peak VRAM via ``MemoryForecaster.estimate_dit_peak_gb``.
    2. Compute ratio R = estimated / free_vram_gb.
    3. If 0.80 < R <= 1.05: inject ``spatial_tile="2x2"``.
    4. If R > 1.05: switch to ``window_frames=49``.
    5. If still over hard limit: reject with exit_code 30.

    Returns:
        Dict with ``output_paths``, ``fallback_level``, and
        ``reanchor_requests``.
    """
    from synapse.adapters import MockVideoAdapter, create_adapter
    from synapse.adapters.base import (
        VideoGenerationConfig,
        VideoResult,
    )
    from synapse.telemetry import MemoryForecaster

    params = dict(manifest.params)
    budget = manifest.hardware_budget

    num_frames = int(params.get("num_frames", 97))
    height = int(params.get("height", 512))
    width = int(params.get("width", 512))
    adapter_name = str(params.get("adapter", "mock"))
    default_variant = "wan2.1_1.3b" if adapter_name == "mock" else "wan2.1_14b"
    model_variant = str(params.get("model_variant") or default_variant)

    forecast = MemoryForecaster.estimate_dit_peak_gb(
        num_frames=num_frames,
        height=height,
        width=width,
        model_variant=model_variant,
    )

    fallback_level: str | None = None
    ratio = forecast.peak_gb / max(budget.free_vram_gb, 0.01)

    if ratio > 1.05:
        short_window = min(num_frames, 49)
        params["window_frames"] = short_window
        fallback_level = "LEVEL_2_CAUSAL_SHORT_WINDOW_49F"

        forecast2 = MemoryForecaster.estimate_dit_peak_gb(
            num_frames=short_window,
            height=height,
            width=width,
            model_variant=model_variant,
        )
        ratio2 = forecast2.peak_gb / max(budget.free_vram_gb, 0.01)
        if ratio2 > 1.05:
            from synapse.errors import BackendUnavailableError

            raise BackendUnavailableError(
                f"Estimated peak VRAM {forecast2.peak_gb:.1f} GiB "
                f"exceeds budget {budget.free_vram_gb:.1f} GiB "
                "even after fallback to 49-frame window",
                agent_fix_hint=(
                    "Increase hardware_budget.free_vram_gb or reduce resolution/frame count."
                ),
            )
    elif ratio > 0.80:
        params["spatial_tile"] = "2x2"
        fallback_level = "LEVEL_1_SPATIAL_TILE_2X2"

    adapter: Any
    if adapter_name == "mock":
        adapter = MockVideoAdapter()
    else:
        adapter = create_adapter(adapter_name)

    gen_cfg = VideoGenerationConfig(
        seed=int(params.get("seed", 42)),
        fps=float(params.get("fps", 24.0)),
        width=width,
        height=height,
        num_frames=num_frames,
        extra={
            k: v
            for k, v in params.items()
            if k
            not in {
                "num_frames",
                "height",
                "width",
                "model_variant",
                "adapter",
                "seed",
                "fps",
            }
        },
    )

    fgpp_dir_str = manifest.inputs.get("fgpp_dir", "")
    if not fgpp_dir_str or not Path(fgpp_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"FGPP directory not found: {fgpp_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['fgpp_dir'] pointing to a "
                "valid FGPP directory with metadata.json and EXR frames."
            ),
        )

    from synapse.fgpp import import_fgpp

    fgpp_result = import_fgpp(fgpp_dir_str, lazy=False)
    assert not isinstance(fgpp_result, Iterator)
    fgpp_seq = fgpp_result
    if not fgpp_seq.frames:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No EXR frames found in FGPP directory: {fgpp_dir_str!r}",
            agent_fix_hint=(
                "Ensure the FGPP directory contains frame_*.exr files alongside metadata.json."
            ),
        )
    conditions = adapter.prepare_conditions(fgpp_seq, gen_cfg)

    result: VideoResult = adapter.generate(
        conditions=conditions,
        config=gen_cfg,
    )

    ws_root = params.get("_workspace_root")
    default_out = (
        (Path(ws_root) / "output" / "video.npy") if ws_root else (Path("output") / "video.npy")
    )
    output_path = manifest.outputs.get(
        "video",
        manifest.outputs.get("result_video", str(default_out)),
    )
    out_p = Path(output_path)
    if out_p.suffix != ".npy":
        out_p = out_p.with_suffix(out_p.suffix + ".npy")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        np.save(f, result.frames_srgb_u8)

    output_paths: dict[str, str] = {"video": str(out_p)}
    reanchor: list[dict[str, Any]] | None = None

    return {
        "output_paths": output_paths,
        "fallback_level": fallback_level,
        "reanchor_requests": reanchor,
    }


@register_stage("style_extract_and_map")
def handle_style_extract_and_map(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run style extraction and mapping pipeline.

    Expects ``manifest.inputs["frames_dir"]`` and/or
    ``manifest.inputs["fgpp_dir"]`` to be set.

    Returns:
        Dict with ``output_paths`` to the style profile YAML.
    """
    from synapse.style import (
        extract_style_optical,
        save_style_profile_yaml,
    )

    params = dict(manifest.params)
    frames_dir = Path(
        manifest.inputs.get("frames_dir", ""),
    )
    num_frames = int(params.get("num_frames", 8))

    if not frames_dir.is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Frames directory does not exist: {frames_dir}",
            agent_fix_hint=(
                "Provide manifest.inputs['frames_dir'] pointing to a "
                "directory containing .png frame files."
            ),
        )

    frames: list[np.ndarray] = []
    for img_path in sorted(frames_dir.glob("*.png"))[:num_frames]:
        try:
            from PIL import Image

            img = np.array(
                Image.open(img_path).convert("RGB"),
                dtype=np.uint8,
            )
            frames.append(img)
        except ImportError:
            from synapse.errors import BackendUnavailableError

            raise BackendUnavailableError(
                "PIL (Pillow) is required for loading frame images but is not installed.",
                agent_fix_hint="Install Pillow: pip install 'Pillow>=9.0'",
            ) from None

    if not frames:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .png frame files found in {frames_dir}",
            agent_fix_hint="Ensure frames_dir contains .png frame files for style extraction.",
        )

    descriptor = extract_style_optical(np.asarray(frames, dtype=np.uint8))

    output_path = manifest.outputs.get(
        "style_profile",
        str(Path("output") / "style_profile.yaml"),
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    saved_path = save_style_profile_yaml(descriptor, out_p)

    return {"output_paths": {"style_profile": str(saved_path)}}


@register_stage("surgical_composite")
def handle_surgical_composite(
    manifest: BundleManifest,
) -> dict[str, Any]:
    """Run surgical compositing pipeline.

    Expects ``manifest.inputs["rendered_dir"]`` and
    ``manifest.inputs["neural_dir"]`` with frame sequences.

    Returns:
        Dict with ``output_paths`` to the composited output.
    """
    from synapse.composite import (
        composite_surgical_sequence,
        generate_surgical_mask,
    )

    params = dict(manifest.params)
    num_frames = int(params.get("num_frames", 4))

    rendered_dir_str = manifest.inputs.get("rendered_dir", "")
    neural_dir_str = manifest.inputs.get("neural_dir", "")

    if not rendered_dir_str or not Path(rendered_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Rendered frames directory not found: {rendered_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['rendered_dir'] pointing to "
                "a directory containing rendered .png frame files."
            ),
        )

    if not neural_dir_str or not Path(neural_dir_str).is_dir():
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"Neural frames directory not found: {neural_dir_str!r}",
            agent_fix_hint=(
                "Provide manifest.inputs['neural_dir'] pointing to "
                "a directory containing neural .png frame files."
            ),
        )

    try:
        from PIL import Image
    except ImportError:
        from synapse.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "PIL (Pillow) is required for loading frame images but is not installed.",
            agent_fix_hint="Install Pillow: pip install 'Pillow>=9.0'",
        ) from None

    rendered_dir = Path(rendered_dir_str)
    neural_dir = Path(neural_dir_str)

    rendered_list: list[np.ndarray] = []
    for img_path in sorted(rendered_dir.glob("*.png"))[:num_frames]:
        img = np.array(
            Image.open(img_path).convert("RGB"),
            dtype=np.uint8,
        )
        rendered_list.append(img)

    neural_list: list[np.ndarray] = []
    for img_path in sorted(neural_dir.glob("*.png"))[:num_frames]:
        img = np.array(
            Image.open(img_path).convert("RGB"),
            dtype=np.uint8,
        )
        neural_list.append(img)

    if not rendered_list:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .png frames found in rendered_dir: {rendered_dir}",
            agent_fix_hint="Ensure rendered_dir contains .png frame files.",
        )

    if not neural_list:
        from synapse.errors import AssetNotFoundError

        raise AssetNotFoundError(
            f"No .png frames found in neural_dir: {neural_dir}",
            agent_fix_hint="Ensure neural_dir contains .png frame files.",
        )

    rendered_frames = np.stack(rendered_list, axis=0)
    neural_frames = np.stack(neural_list, axis=0)

    height, width = rendered_frames.shape[1], rendered_frames.shape[2]

    mask_input_path = manifest.inputs.get("masks", "")
    if mask_input_path and Path(mask_input_path).is_file():
        masks = np.load(mask_input_path).astype(np.float32)
    else:
        masks = np.zeros(
            (rendered_frames.shape[0], height, width),
            dtype=np.float32,
        )
        for t in range(rendered_frames.shape[0]):
            diff = np.abs(
                rendered_frames[t].astype(np.float32) - neural_frames[t].astype(np.float32),
            )
            binary_mask = diff.mean(axis=-1) > 10.0
            masks[t] = generate_surgical_mask(binary_mask)

    composited = composite_surgical_sequence(
        neural_frames,
        rendered_frames,
        masks,
    )

    output_path = manifest.outputs.get(
        "composited",
        str(Path("output") / "composited.npy"),
    )
    out_p = Path(output_path)
    if out_p.suffix != ".npy":
        out_p = out_p.with_suffix(out_p.suffix + ".npy")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        np.save(f, composited)

    return {"output_paths": {"composited": str(out_p)}}
