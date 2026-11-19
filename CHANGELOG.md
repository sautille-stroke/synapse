# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] — 2026-11-19

### Horizon 9 (M33–M36) — Holistic Remediation & Evolution

Major release consolidating Horizon 9 quality hardening across all subsystems.
14 hard quality gates verified; 1,488 tests (1,321 fast + 167 slow) passing.

#### Added

- **RFC-0071 `@aether//geo` integration** — All geometry, camera math, and
  metrics computations now delegate to the `aether_geo` C++20 native extension
  via `synapse.bridge.aether_bridge`, removing pure-Python fallback code.
  (PR-136, PR-137, PR-138, PR-139)
- **Wan 2.1 Diffusers backend** — `WanDiffusersTransport` for local in-process
  Wan 2.1 inference using HuggingFace Diffusers with ControlNeXt hooks and
  SSRF hardening. (PR-139)
- **LoRA training pipeline** — `StatelessLoRATrainer` with FeatureStore
  WebDataset streaming, tau-schedule, atomic checkpoint packager, and `lora`
  CLI sub-command. (PR-140)
- **Expanded `csrc/geometry` bindings** — Marching Cubes, Catmull-Clark
  subdivision, xatlas UV unwrapping, KD-Tree queries, robust ICP, and
  zero-copy float64 transfer via `@aether//geo`. (PR-141)
- **Inverse rendering & 3D reconstruction** — TSDF void carving, watertight
  quad remeshing with UV, world-space delight, Mitsuba GGX material fitting,
  and Redline #3 quality gate for `has_gen3d_highpoly`. (PR-142)
- **Local refinement pipeline** — Bypass 100 dB cap, EMA weight floor ≥ 0.85,
  cascade/uncertainty/Poisson refinement, outer border ring, HDR luminance
  ratio, ensemble blending, and HITL persistence. (PR-143)
- **Synapse-Bench** — 30 activated benchmark scenarios with scanned asset
  support, `applicable_keys` fix, and hardened metric thresholds. (PR-144)
- **3D model inspector** — Reference-free evaluator, world-scale estimation,
  `HttpJevTransport`, and RSI automated flywheel bridge. (PR-145)
- **Desktop inspector workspace** — Multi-entity mesh slots, 15-channel EXR
  AOV rendering, and 4-tab desktop UI. (PR-146)
- **Anti-tautology contract tests** — Cross-repo roundtrip verification with
  strict numerical assertions against `@aether//fgpp`. (PR-148)
- **Hypothesis property fuzz** — Property-based fuzz testing for FGPP schemas,
  malformed binary rejection, and ASan regression suite for native code.
  (PR-149)
- **Performance SLA** — Fibonacci vectorisation, geometry operator parallelism,
  and Bazel build cache reuse verification. (PR-150)
- **Three-tier documentation** — Consolidated archive structure with dynamic
  SSOT sync. (PR-151)
- **BCR & tooling distribution** — 3-tier BCR (Bazel Central Registry)
  configuration and cross-platform toolchain links. (PR-133-ext)

#### Changed

- **FGPP schema hardening** — Corrected `p02` projection sign convention,
  camera rotation matrix conventions, EXR channel validation, and wire guard
  `oneOf` schema support. (PR-134)
- **CLI/Worker routing** — RFC-0039 `JobBundle` routing, ancestor sandbox
  isolation, and `--json` output cleanup. (PR-135)
- **Composite & style** — EDT monotonic blend, ACEScg IRLS solver, McCamy CCT
  signed denominator fix, and symlink guard. (PR-137)
- **Render core refactor** — Mesh, camera, and procedural generator
  de-duplication, streaming FGPP reader, and low-memory style pipeline.
  (PR-138)
- **Dataset scripts** — Zero-crash provenance, `colab_runner` real GPU path,
  and offline asset fast-path. (PR-147)

#### Fixed

- **Inspector** — GLB overflow on large meshes, tri-platform `--shm-file`
  shared memory, and 3-slot tear-free ring buffer. (PR-136)
- **CI/Build** — Activated `.github` CI workflow, restored Bazel optimisation
  flags, and updated `pyproject.toml` dependencies. (PR-133)
- **Type safety** — Resolved all mypy strict errors, narrowed `Any` escapes
  in `wan.py` with `TYPE_CHECKING` guards, and eliminated duplicate function
  definitions.
- **Import DAG guard** — Added dynamic-import allowlist for `aether_bridge.py`.
- **Lint** — Resolved all ruff I001 (import-sort) and F401 (unused-import)
  violations.

## [3.6.0] — 2026-11-15

### Horizon 8 — Post-H7 Production Closure (M29–M32)

Baseline release prior to Horizon 9. Includes Horizons 1–8 cumulative work:
core FGPP pipeline, Wan 2.1 adapter with causal sliding window, ACEScg style
lock, Cryptomatte spatial masking, Synapse-Bench framework, 3D model
evaluation, and cross-repo contract verification.

[4.0.0]: https://github.com/sautille-stroke/synapse/compare/v3.6.0...v4.0.0
[3.6.0]: https://github.com/sautille-stroke/synapse/releases/tag/v3.6.0
