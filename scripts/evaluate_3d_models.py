#!/usr/bin/env python3
"""Evaluate 3D models: Tier-0 topology checks, multi-view screenshots, metrics.

Produces a self-contained HTML report with per-model metrics tables,
pass/fail gate badges, multi-view screenshot galleries, and SVG radar
charts.  Also writes a machine-readable JSON results file.

Usage::

    python scripts/evaluate_3d_models.py \\
        --input-dir ./models/ \\
        --output-dir ./eval_output/ \\
        [--reference-dir ./reference_models/] \\
        [--profile TRACK_A_ASSET | --gate-profile track_a_asset] \\
        [--resolution 256] \\
        [--num-views 8 | --views 8] \\
        [--json]

Exit codes (per AGENTS.md §3)::

    0  — Success (all models passed quality gates, report generated)
    10 — Contract/argument validation error (e.g. invalid --profile)
    20 — Input directory or model files missing
    40 — Quality gate failures detected (report still generated)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from synapse.errors import SynapseError
from synapse.inspect.model_evaluator import (
    GATE_PROFILES,
    evaluate_3d_models_directory,
)

__all__ = ["GATE_PROFILES", "main"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for 3D model batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D models: topology, metrics, multi-view screenshots.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .obj/.glb/.ply model files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for HTML/JSON reports.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Optional directory with reference meshes for comparison.",
    )
    parser.add_argument(
        "--profile",
        "--gate-profile",
        dest="profile",
        default="TRACK_A_ASSET",
        help=(
            "Gate profile name (TRACK_A_ASSET, TRACK_B_VIDEO, "
            "RSI_PROMOTION, RSI_PROMOTION_TARGET; case-insensitive)."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Screenshot render resolution (default: 256).",
    )
    parser.add_argument(
        "--num-views",
        "--views",
        dest="num_views",
        type=int,
        default=8,
        help="Number of multi-view orbit screenshots (default: 8).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output pure JSON to stdout.",
    )
    parser.add_argument(
        "--backend",
        default="native",
        choices=["native", "python"],
        help="Geometry gate backend (default: native). "
             "'python' is for Oracle parity testing only.",
    )

    args = parser.parse_args(argv)

    def log_fn(msg: str) -> None:
        if not args.json_output:
            print(msg, file=sys.stdout)

    try:
        json_data = evaluate_3d_models_directory(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            reference_dir=Path(args.reference_dir) if args.reference_dir else None,
            profile_name=args.profile,
            resolution=args.resolution,
            num_views=args.num_views,
            log_fn=None if args.json_output else log_fn,
            geometry_backend=args.backend,
        )
    except SynapseError as exc:
        err = exc.to_json_dict()
        if args.json_output:
            print(json.dumps(err, indent=2, ensure_ascii=False))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return int(exc.exit_code)

    if args.json_output:
        print(json.dumps(json_data, indent=2, ensure_ascii=False))

    return int(json_data["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
