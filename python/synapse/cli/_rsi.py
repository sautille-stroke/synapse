"""RSI subcommand handlers: sample-curriculum, verify, build-pairs, gate-promote."""

from __future__ import annotations

from typing import cast

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

    # Use scores from CLI if provided, otherwise empty defaults
    scores_raw = getattr(args, "scores", None)
    scores: dict[str, float] = {}
    if scores_raw:
        scores_path = Path(scores_raw)
        if scores_path.is_file():
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        else:
            scores = json.loads(scores_raw)
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
                '"depth_rmse_m": 0.01, '
                "\"normal_angular_error_deg\": 5.0}'' or a JSON file path."
            ),
        )

    metrics_path = Path(metrics_raw)
    try:
        if metrics_path.is_file():
            physics_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
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
        items = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    builder = PreferenceDatasetBuilder()
    builder.add_golden_pairs([PreferencePair(**item) for item in items])

    self_file = getattr(args, "self_file", None)
    if self_file and Path(self_file).is_file():
        self_text = Path(self_file).read_text(encoding="utf-8").strip()
        self_items = (
            json.loads(self_text)
            if self_text.startswith("[")
            else [json.loads(line) for line in self_text.splitlines() if line.strip()]
        )
        builder.add_self_pairs([PreferencePair(**item) for item in self_items])

    pairs = builder.build_pairs()

    # Write output if --output specified
    output_path = getattr(args, "output", None)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps(
                [p.model_dump() for p in pairs],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    return {
        "total": len(pairs),
        "golden_count": sum(1 for p in pairs if p.is_golden),
        "output": str(output_path) if output_path else None,
    }


def _cmd_rsi_gate_promote(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate monotonic promotion gate."""
    from synapse.rsi.promotion_gate import (
        BenchReport,
        BenchScenarioScore,
        evaluate_monotonic_promotion,
    )

    baseline_report_raw = getattr(args, "baseline_report", None)
    candidate_report_raw = getattr(args, "candidate_report", None)

    if baseline_report_raw and candidate_report_raw:
        base_p = Path(baseline_report_raw)
        cand_p = Path(candidate_report_raw)
        if not base_p.is_file():
            raise AssetNotFoundError(
                f"Baseline report file not found: {baseline_report_raw!r}",
                agent_fix_hint=(
                    f"Ensure '{baseline_report_raw}' exists and is a valid BenchReport JSON file."
                ),
            )
        if not cand_p.is_file():
            raise AssetNotFoundError(
                f"Candidate report file not found: {candidate_report_raw!r}",
                agent_fix_hint=(
                    f"Ensure '{candidate_report_raw}' exists and is a valid BenchReport JSON file."
                ),
            )
        baseline = BenchReport.model_validate_json(base_p.read_text(encoding="utf-8"))
        candidate = BenchReport.model_validate_json(cand_p.read_text(encoding="utf-8"))
    else:
        baseline_overall_raw = getattr(args, "baseline_overall", None)
        candidate_overall_raw = getattr(args, "candidate_overall", None)
        baseline_geom_raw = getattr(args, "baseline_geom", None)
        candidate_geom_raw = getattr(args, "candidate_geom", None)
        baseline_temporal_raw = getattr(args, "baseline_temporal", None)
        candidate_temporal_raw = getattr(args, "candidate_temporal", None)

        missing_fields: list[str] = []
        if baseline_overall_raw is None:
            missing_fields.append("--baseline-overall")
        if baseline_geom_raw is None:
            missing_fields.append("--baseline-geom")
        if baseline_temporal_raw is None:
            missing_fields.append("--baseline-temporal")
        if candidate_overall_raw is None:
            missing_fields.append("--candidate-overall")
        if candidate_geom_raw is None:
            missing_fields.append("--candidate-geom")
        if candidate_temporal_raw is None:
            missing_fields.append("--candidate-temporal")

        if missing_fields:
            raise ContractValidationError(
                f"Missing required 3-axis score argument(s): {', '.join(missing_fields)}. "
                "Independent 3-axis scores (overall, geom, temporal) are required; "
                "single-score broadcasting across axes is forbidden.",
                agent_fix_hint=(
                    "Provide explicit values for --baseline-overall, --baseline-geom, --baseline-temporal, "
                    "--candidate-overall, --candidate-geom, and --candidate-temporal (or pass "
                    "--baseline-report and --candidate-report JSON files)."
                ),
            )

        baseline_overall = float(baseline_overall_raw)  # type: ignore[arg-type]
        candidate_overall = float(candidate_overall_raw)  # type: ignore[arg-type]
        baseline_geom = float(baseline_geom_raw)  # type: ignore[arg-type]
        candidate_geom = float(candidate_geom_raw)  # type: ignore[arg-type]
        baseline_temporal = float(baseline_temporal_raw)  # type: ignore[arg-type]
        candidate_temporal = float(candidate_temporal_raw)  # type: ignore[arg-type]

        # Parse baseline and candidate scenarios separately
        # NOTE: --scenarios is NOT used as a shared fallback; baseline and
        # candidate must supply their own --baseline-scenarios /
        # --candidate-scenarios to prevent data bleeding between the two.
        base_scenarios_raw = getattr(args, "baseline_scenarios", None)
        cand_scenarios_raw = getattr(args, "candidate_scenarios", None)

        def _parse_scenarios(raw: Any) -> list[BenchScenarioScore]:
            if not raw:
                return []
            sc_p = Path(raw)
            if sc_p.is_file():
                sc_data = json.loads(sc_p.read_text(encoding="utf-8"))
            else:
                sc_data = json.loads(raw)
            out: list[BenchScenarioScore] = []
            if isinstance(sc_data, list):
                for item in sc_data:
                    if isinstance(item, dict):
                        out.append(BenchScenarioScore.model_validate(item))
            return out

        baseline = BenchReport(
            overall_score=baseline_overall,
            geom_score=baseline_geom,
            temporal_score=baseline_temporal,
            scenarios=_parse_scenarios(base_scenarios_raw),
        )
        candidate = BenchReport(
            overall_score=candidate_overall,
            geom_score=candidate_geom,
            temporal_score=candidate_temporal,
            scenarios=_parse_scenarios(cand_scenarios_raw),
        )

    promote_kwargs: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "raise_on_reject": True,
    }
    if not base_scenarios_raw and not cand_scenarios_raw:
        promote_kwargs["min_scenarios"] = 0
    epsilon_min = getattr(args, "epsilon_min", None)
    if epsilon_min is not None:
        promote_kwargs["epsilon_min"] = epsilon_min
    noise_tolerance = getattr(args, "noise_tolerance", None)
    if noise_tolerance is not None:
        promote_kwargs["noise_tolerance"] = noise_tolerance

    result = evaluate_monotonic_promotion(**promote_kwargs)

    return {
        "promoted": result.promoted,
        "conditions": result.conditions,
        "failures": result.failures,
        "baseline": baseline.model_dump(),
        "candidate": candidate.model_dump(),
    }


def register_rsi_subparser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    json_parent: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Register the ``rsi`` subcommand tree."""
    parents = [json_parent] if json_parent is not None else []
    rsi_parser = subparsers.add_parser(
        "rsi",
        help="Recursive self-improvement commands",
        parents=parents,
    )
    rsi_sub = rsi_parser.add_subparsers(
        dest="rsi_command",
    )

    rsi_sample = rsi_sub.add_parser(
        "sample-curriculum",
        help="Sample curriculum scenarios for RSI",
        parents=parents,
    )
    rsi_sample.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    rsi_sample.add_argument(
        "--num-scenarios",
        type=int,
        default=10,
        help="Number of scenarios to sample (default: 10)",
    )
    rsi_sample.add_argument(
        "--scores",
        default=None,
        help="JSON string or path to JSON file with factor scores for curriculum weighting",
    )

    rsi_verify = rsi_sub.add_parser(
        "verify",
        help="Run dual verification on a sample",
        parents=parents,
    )
    rsi_verify.add_argument(
        "--sample-id",
        default="unknown",
        help="Sample identifier (default: unknown)",
    )
    rsi_verify.add_argument(
        "--physics-metrics",
        default=None,
        help="JSON string or file path with measured physics metrics",
    )
    rsi_verify.add_argument(
        "--prm-score",
        type=float,
        default=0.8,
        help="PRM neural score in [0, 1] (default: 0.8)",
    )

    rsi_build = rsi_sub.add_parser(
        "build-pairs",
        help="Build preference pairs for RSI training",
        parents=parents,
    )
    rsi_build.add_argument(
        "--golden-file",
        default=None,
        help="Path to JSON/JSONL file with L0 golden preference pairs",
    )
    rsi_build.add_argument(
        "--self-file",
        default=None,
        help="Optional path to JSON/JSONL file with self-play pairs",
    )
    rsi_build.add_argument(
        "--output",
        default=None,
        help="Path to write built preference pairs JSON output",
    )

    rsi_gate = rsi_sub.add_parser(
        "gate-promote",
        help="Evaluate monotonic promotion gate",
        parents=parents,
    )
    rsi_gate.add_argument(
        "--baseline-overall",
        type=float,
        default=None,
        help="Baseline overall score",
    )
    rsi_gate.add_argument(
        "--candidate-overall",
        type=float,
        default=None,
        help="Candidate overall score",
    )
    rsi_gate.add_argument(
        "--baseline-geom",
        type=float,
        default=None,
        help="Baseline geometry score",
    )
    rsi_gate.add_argument(
        "--candidate-geom",
        type=float,
        default=None,
        help="Candidate geometry score",
    )
    rsi_gate.add_argument(
        "--baseline-temporal",
        type=float,
        default=None,
        help="Baseline temporal score",
    )
    rsi_gate.add_argument(
        "--candidate-temporal",
        type=float,
        default=None,
        help="Candidate temporal score",
    )
    rsi_gate.add_argument(
        "--baseline-report",
        default=None,
        help="Path to baseline BenchReport JSON file",
    )
    rsi_gate.add_argument(
        "--candidate-report",
        default=None,
        help="Path to candidate BenchReport JSON file",
    )
    rsi_gate.add_argument(
        "--scenarios",
        default=None,
        help="Optional JSON string or path to JSON file with scenario scores list",
    )
    rsi_gate.add_argument(
        "--baseline-scenarios",
        default=None,
        help="Optional JSON string or path to JSON file with baseline scenario scores list",
    )
    rsi_gate.add_argument(
        "--candidate-scenarios",
        default=None,
        help="Optional JSON string or path to JSON file with candidate scenario scores list",
    )
    rsi_gate.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail closed with exit code 40 if candidate is not promoted",
    )
    rsi_gate.add_argument(
        "--epsilon-min",
        type=float,
        default=None,
        help="Minimum improvement epsilon threshold for promotion",
    )
    rsi_gate.add_argument(
        "--noise-tolerance",
        type=float,
        default=None,
        help="Noise tolerance for promotion gate comparison",
    )
    return cast(argparse.ArgumentParser, rsi_parser)


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
            _cmd_rsi_sample_curriculum,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif rsi_cmd == "verify":
        return _run_with_json_mode(
            _cmd_rsi_verify,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif rsi_cmd == "build-pairs":
        return _run_with_json_mode(
            _cmd_rsi_build_pairs,
            parsed,
            json_mode,
            trace_ctx,
        )
    elif rsi_cmd == "gate-promote":
        return _run_with_json_mode(
            _cmd_rsi_gate_promote,
            parsed,
            json_mode,
            trace_ctx,
        )
    else:
        if json_mode:
            err = SynapseError(
                "Missing rsi sub-command (sample-curriculum|verify|build-pairs|gate-promote)",
                ExitCode.CONTRACT_ERROR,
                "Use 'synapse rsi <sub-command> --help'.",
            )
            json.dump(
                err.to_json_dict(trace_ctx.trace_id),
                sys.stdout,
                default=str,
            )
            print()
            return int(err.exit_code)
        rsi_parser.print_help()
        return int(ExitCode.CONTRACT_ERROR)
