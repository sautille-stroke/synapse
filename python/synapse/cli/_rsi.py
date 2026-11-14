"""RSI subcommand handlers: sample-curriculum, verify, build-pairs, gate-promote."""

from __future__ import annotations

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


def _cmd_rsi_sample_curriculum(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Sample RSI curriculum scenarios."""
    from synapse.rsi.curriculum import (
        CurriculumConfig,
        CurriculumSampler,
    )

    seed = int(getattr(args, "seed", 42))
    num = int(getattr(args, "num_scenarios", 10))
    cfg = CurriculumConfig(seed=seed, num_scenarios=num)
    sampler = CurriculumSampler(cfg)

    # Use empty scores (all factors get default treatment)
    scores: dict[str, float] = {}
    scenarios = sampler.sample_scenarios(scores)
    return {
        "scenarios": [s.model_dump() for s in scenarios],
        "count": len(scenarios),
    }


def _cmd_rsi_verify(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run dual verification on a sample."""
    from synapse.rsi.dual_verifier import DualVerifierEngine

    sample_id = getattr(args, "sample_id", "unknown")
    metrics_raw = getattr(args, "physics_metrics", None)
    if not metrics_raw:
        raise ContractValidationError(
            "CLI 'synapse rsi verify' requires '--physics-metrics' "
            "(JSON string or path to JSON file). Hardcoded passing "
            "metrics are FORBIDDEN.",
            agent_fix_hint=(
                "Pass '--physics-metrics '{\"silhouette_iou\": 0.95, "
                "\"depth_rmse_m\": 0.01, "
                "\"normal_angular_error_deg\": 5.0}'' or a JSON file path."
            ),
        )

    metrics_path = Path(metrics_raw)
    try:
        if metrics_path.is_file():
            physics_metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        else:
            physics_metrics = json.loads(metrics_raw)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            f"Invalid JSON in --physics-metrics: {exc}",
            agent_fix_hint="Provide valid JSON for --physics-metrics.",
        ) from exc

    prm_score = float(getattr(args, "prm_score", 0.5))

    engine = DualVerifierEngine()
    verdict = engine.verify(
        sample_id=sample_id,
        physics_metrics=physics_metrics,
        prm_score=prm_score,
    )
    return {"verdict": verdict.model_dump()}


def _cmd_rsi_build_pairs(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build preference pairs for RSI training."""
    from synapse.rsi.pair_builder import (
        PreferenceDatasetBuilder,
        PreferencePair,
    )

    golden_file = getattr(args, "golden_file", None)
    if not golden_file or not Path(golden_file).is_file():
        raise AssetNotFoundError(
            f"Golden pairs file not found: {golden_file!r}. "
            "Hardcoded CLI golden pairs are FORBIDDEN.",
            agent_fix_hint=(
                "Pass '--golden-file <path>' pointing to a JSON/JSONL "
                "file of L0 golden PreferencePair records."
            ),
        )

    raw_text = Path(golden_file).read_text(encoding="utf-8").strip()
    if raw_text.startswith("["):
        items = json.loads(raw_text)
    else:
        items = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]

    builder = PreferenceDatasetBuilder()
    builder.add_golden_pairs([PreferencePair(**item) for item in items])

    self_file = getattr(args, "self_file", None)
    if self_file and Path(self_file).is_file():
        self_text = Path(self_file).read_text(encoding="utf-8").strip()
        self_items = (
            json.loads(self_text)
            if self_text.startswith("[")
            else [
                json.loads(line)
                for line in self_text.splitlines()
                if line.strip()
            ]
        )
        builder.add_self_pairs(
            [PreferencePair(**item) for item in self_items]
        )

    pairs = builder.build_pairs()
    return {
        "total": len(pairs),
        "golden_count": sum(
            1 for p in pairs if p.is_golden
        ),
    }


def _cmd_rsi_gate_promote(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate monotonic promotion gate."""
    from synapse.rsi.promotion_gate import (
        BenchReport,
        evaluate_monotonic_promotion,
    )

    baseline_overall_raw = getattr(args, "baseline_overall", None)
    candidate_overall_raw = getattr(args, "candidate_overall", None)

    if baseline_overall_raw is None or candidate_overall_raw is None:
        raise ContractValidationError(
            "Both --baseline-overall and --candidate-overall are "
            "required for 'synapse rsi gate-promote'. Hardcoded "
            "defaults are forbidden.",
            agent_fix_hint=(
                "Pass explicit '--baseline-overall <float>' and "
                "'--candidate-overall <float>' arguments."
            ),
        )

    baseline_overall = float(baseline_overall_raw)
    candidate_overall = float(candidate_overall_raw)

    baseline = BenchReport(
        overall_score=baseline_overall,
        geom_score=baseline_overall,
        temporal_score=baseline_overall,
    )
    candidate = BenchReport(
        overall_score=candidate_overall,
        geom_score=candidate_overall,
        temporal_score=candidate_overall,
    )

    result = evaluate_monotonic_promotion(
        baseline=baseline, candidate=candidate,
    )
    return {
        "promoted": result.promoted,
        "conditions": result.conditions,
    }


def register_rsi_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``rsi`` subcommand tree."""
    rsi_parser = subparsers.add_parser(
        "rsi",
        help="Recursive self-improvement commands",
    )
    rsi_sub = rsi_parser.add_subparsers(
        dest="rsi_command",
    )

    rsi_sample = rsi_sub.add_parser(
        "sample-curriculum",
        help="Sample curriculum scenarios for RSI",
    )
    rsi_sample.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    rsi_sample.add_argument(
        "--num-scenarios", type=int, default=10,
        help="Number of scenarios to sample (default: 10)",
    )

    rsi_verify = rsi_sub.add_parser(
        "verify",
        help="Run dual verification on a sample",
    )
    rsi_verify.add_argument(
        "--sample-id", default="unknown",
        help="Sample identifier (default: unknown)",
    )
    rsi_verify.add_argument(
        "--physics-metrics", default=None,
        help="JSON string or file path with measured physics metrics",
    )
    rsi_verify.add_argument(
        "--prm-score", type=float, default=0.8,
        help="PRM neural score in [0, 1] (default: 0.8)",
    )

    rsi_build = rsi_sub.add_parser(
        "build-pairs",
        help="Build preference pairs for RSI training",
    )
    rsi_build.add_argument(
        "--golden-file", default=None,
        help="Path to JSON/JSONL file with L0 golden preference pairs",
    )
    rsi_build.add_argument(
        "--self-file", default=None,
        help="Optional path to JSON/JSONL file with self-play pairs",
    )

    rsi_gate = rsi_sub.add_parser(
        "gate-promote",
        help="Evaluate monotonic promotion gate",
    )
    rsi_gate.add_argument(
        "--baseline-overall", type=float, default=None,
        help="Baseline overall score (required)",
    )
    rsi_gate.add_argument(
        "--candidate-overall", type=float, default=None,
        help="Candidate overall score (required)",
    )
    return rsi_parser


def dispatch_rsi(
    parsed: argparse.Namespace,
    json_mode: bool,
    trace_ctx: Any,
    rsi_parser: argparse.ArgumentParser,
) -> int:
    """Dispatch ``synapse rsi <sub-command>``."""
    rsi_cmd = getattr(parsed, "rsi_command", None)
    if rsi_cmd == "sample-curriculum":
        return _run_with_json_mode(
            _cmd_rsi_sample_curriculum, parsed,
            json_mode, trace_ctx,
        )
    elif rsi_cmd == "verify":
        return _run_with_json_mode(
            _cmd_rsi_verify, parsed,
            json_mode, trace_ctx,
        )
    elif rsi_cmd == "build-pairs":
        return _run_with_json_mode(
            _cmd_rsi_build_pairs, parsed,
            json_mode, trace_ctx,
        )
    elif rsi_cmd == "gate-promote":
        return _run_with_json_mode(
            _cmd_rsi_gate_promote, parsed,
            json_mode, trace_ctx,
        )
    else:
        if json_mode:
            err = SynapseError(
                "Missing rsi sub-command "
                "(sample-curriculum|verify|"
                "build-pairs|gate-promote)",
                ExitCode.CONTRACT_ERROR,
                "Use 'synapse rsi <sub-command>"
                " --help'.",
            )
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
            return int(err.exit_code)
        rsi_parser.print_help()
        return 0
