---
type: Guide
title: "Changelog"
tags: [synapse]
status: active
stale_after: 2027-11-19T00:00:00Z
---
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Traceboard Post-Review Hardening (`aether#569` `v0.1.1` & `synapse` polish)
- **Layer 0 `aether-traceboard v0.1.1` (`aether#569`)**: Made Python `_evaluate_comparison_pass` and C++ `ProbeContext::logComparison` fail closed on non-finite (`NaN`/`Inf`) metrics/thresholds, coerced NumPy scalar types (`np.bool_`, `np.integer`, `np.floating`) in JSON sanitization, held `Probe._lock` across `on_run_open` and `RUN_START` emission with `_active = False` until `RUN_START` completes, and fixed `FileCollector` UTF-8 multi-byte truncation boundaries.
- **Synapse hardening & docs reconciliation**: Enforced `workspace_root` sandboxing on manifest-supplied Traceboard paths in `bundle_worker.py`, emitted `LocalRefinementPipeline(strict=True)` Traceboard comparison/scalar telemetry before raising `QualityGateError`, forwarded `step=step` through `StatelessLoRATrainer._dpo_step()`, tightened `_accepts_kwarg()` against `**kwargs`, updated missing-dependency install hints to the pinned release URL, added `has_tb` CI smoke visibility, and reconciled Traceboard documentation/changelogs.

### Traceboard Remediation, Wire Parity & Rich-Media Hardening (`PR-TR01`–`PR-TR08`: `aether#557`–`#560`, `traceboard-python-v0.1.0`, `synapse#215`–`#219`, `caelum-core#557`, `all-overview#770`)
- **Layer 0 `aether-traceboard v0.1.0` Wire Parity & Release (`aether#557`, `#558`, `#560`, `traceboard-python-v0.1.0`)**: Fixed `.tbevt` `payloadLength = tagLen + len(payload)`, incremental CRC-32, strict RFC 8259 JSON, `STAGE_END` status & `ARTIFACT` metadata; established bi-directional C++↔Python golden interop tests (`bazel test //traceboard:test_golden_interop` + `pytest tests/golden/`); eliminated runtime `typing_extensions`, excluded `tests/` from wheel, cleared `mypy --strict` & `ruff`, and published `traceboard-python-v0.1.0`.
- **Wave TR-2 Runtime, CLI & SSOT Remediation (`synapse#215`–`#217`)**: Added `ProbeProtocol`/`StageProtocol` with `RLock` atomic rollback in `synapse.observability`; exposed shared `--traceboard [LOG_DIR]` across subcommands (`synapse bench --traceboard ...`, `synapse --traceboard ...`, `synapse worker run --traceboard ...`); fixed `BenchPipelineRunner` fail verdicts, monotonic `step` tracking, low-cardinality verifier/inspector tags + `compute_pillar_scores()` SSOT, Colab NVMe-first staging, `run_cpu_verification.py` `finally:` flushes, typed `MagicMock(spec=...)`, and CI `.tbevt` binary verification (`5/5` green).
- **Wave TR-3 Rich Media & Cross-Repo Closure (`synapse#218`, `PR-TR08`, `caelum-core#557`, `all-overview#770`)**: Added `bound_image_u8()`, `safe_log_image()`, and `safe_log_histogram()`; wired Traceboard into `bundle_worker` (`SYNAPSE_TRACEBOARD_DIR` / `JobBundle`), `ConditionScheduler`, `LocalRefinementPipeline` (`COMPARISON`), `BenchPipelineRunner` (`--tb-media`), and `render_comparison_html` (`ARTIFACT`); migrated `caelum-core` helpers to `aether-traceboard` v0.1 API and recorded `DUP-17` golden interop gate in `all-overview`.

### Traceboard Python SDK & Synapse Observability Integration (`#204`–`#212`, paired with `dervgen/aether#523`–`#525`)
- **Observability core & CLI (`#204`, `#205`, `#212`)**: Added PEP 562 lazy `synapse.observability` wrapper (`init_traceboard`, `get_probe`, `shutdown_traceboard`, `traceboard_stage`), registered `observability` in Tier 0 of `tools/import_dag_guard.py`, added `[traceboard]` optional extra (`aether-traceboard>=0.1.0`), and wired `--traceboard LOG_DIR` into `synapse` and `synapse-bench`.
- **Module instrumentation (`#206`–`#209`, `#212`)**: Added non-invasive Traceboard telemetry across 9 subsystems (`adapters`, `bench`, `composite`, `inspect`, `pipeline`, `recon`, `rsi`, `style`, `training`) with zero runtime overhead when uninitialized or absent.
- **Colab / CI / Dual-Track Export (`#210`, `#211`, `#212`)**: Added `--traceboard` to `scripts/colab_runner.py` and `scripts/run_cpu_verification.py`, added dual-track `synapse.bench.traceboard_export.export_reports_to_traceboard()`, documented integration architecture in `docs/DESIGN_TRACEBOARD_INTEGRATION.md`, and published complete plan & landing summary in `docs/plans/PLAN-TRACEBOARD-PYTHON-AND-SYNAPSE.md`.
- **Post-PR #213 Deep Review & Remediation Blueprint**: Completed deep cross-repo review (`docs/audits/REVIEW-PR213-TRACEBOARD-INTEGRATION-2026-11.md`, identifying 20 findings: 6 P0 / 9 P1 / 5 P2) and published an actionable 3-Wave, 8-PR remediation & hardening plan (`docs/plans/PLAN-TRACEBOARD-REMEDIATION-AND-HARDENING.md`).

## [5.0.0] — 2026-11-24

### Horizon 12–13B (M45–M52) — Video-to-3D Twin, 3D Evaluation, External Datasets & 4-Track Leaderboard

Release completing `PLAN-0100 Wave 4` (`M45–M48`), `PR #189` evaluation hardening & external datasets (`M49–M50`), and `Post-H10` strategic governance & 4-track industry comparator (`M51–M52`):

#### Horizon 13A · M49–M50 — 3D Evaluation Hardening, Perceptual Metrics & External Datasets (`PR-203 ~ PR-209`, GH `#191`, `#192`)
- Fixed `PR #189` topology/UV defects (`euler_char` vertex isolation, bowtie non-manifold detection, seam normal jumps, fail-closed UV gates, HTML rendering, `profiles` sync).
- Added perceptual & geometric metrics (`dino_cosine_sim`, `lpips_vgg`, `patch_lpips`, `discrete_mean_curvature_variance`, `surface_roughness_var`, `add_s_norm`, `reprojection_error_px`).
- Integrated external 3D asset sources (`GSO`, `ABO`, `OmniObject3D`, `Thingi10K`, license & Stage-0 filtering), expanding `Synapse-Bench` from 30 to **40 scenarios**.

#### Horizon 12 · M45–M48 — `PLAN-0100 Wave 4` Video-to-3D Twin & 3DGS Compression (`PR-193 ~ PR-202`, GH `#193`, `#194`)
- Unified streaming video decoder/FGPP re-exporter (`recon/io.py`), 12-service async `FoundaryClient` + affine metric scale anchoring, adaptive keyframes & Redline #3 character floor gating, and 16B-aligned `.fmesh.bin` `AetherGeoWorkerClient` (with fail-closed exit `50` on unimplemented `quadriflow`).
- Added dual-track `@aether//animation` IK bridge + fail-closed `backend="native"` (exit `30`) & `>0.5 cm` foot-sliding gate (exit `40`), `FulgurSlangADRendererAdapter`, DP Graph-Cut seam blending, RFC 0058 OpenUSD/OpenPBR/`UsdSkel` packager (`usd_packager.py`), 3DGS SH/Smallest-Three/Draco `.s3dgs` quantizer (`gs_quantizer.py` + `synapse recon compress-3dgs`), and Colab `E6` end-to-end twin benchmark.

#### Horizon 13B · M51–M52 — Tooling Governance, Module Consolidation & `v5.0.0` (`PR-210 ~ PR-215`, GH `#196`)
- Added weekly CUDA GPU smoke runner (`scripts/run_gpu_smoke.py` + `.github/workflows/gpu-smoke-weekly.yml`), `wire_guard.py --require-style`, and standalone-safe Ruff SSOT parity checker (`tools/sync_ruff_ssot.py`, superseding `#163`).
- Consolidated `synapse.generators` into `synapse.bench.{dataset_gen,dataset_visualizer}` with zero-warning shims and unified `bench.model_loader`; hardened RFC-0061 Foundary K=2 candidate hook and RFC-0062 `kappa` lookdev metrics.
- Added 4-track RFC-0063 industry leaderboard aggregator (`bench/compare.py`), HTML leaderboard (`render_comparison_html`), and `synapse bench compare` CLI.

## [4.2.0] — 2026-11-21

### Horizon 11 (M41–M44) — Full-Repository Hardening, Scientific Correctness & 16-Gate Closure

Release completing all 20 atomic PRs (`PR-173 ~ PR-192`, GH `#165 ~ #184`) across four milestones:

#### M41 — Infrastructure, Static Analysis & Security (PR-173 ~ PR-177)
- Hardened Colab workenv notebooks, Tailscale/VS Code TLS & secret handling (`PR-173`).
- Repaired GitHub Actions 5/5 jobs, dual `aether_geo`+`aether_metrics` auto-installation in `build_native.py`, and strict dependency guards (`PR-174`).
- Cleared all `mypy --strict`, `ruff`, and `DeprecationWarning` issues (`PR-175`).
- Pinned-socket `safe_urlopen` preserving POST/TLS/SNI/IPv6 + viewer/bundle symlink & XSS hardening (`PR-176`).
- Fail-closed atomic `PersistentHitlLedger` (`flock` + `fsync`) and deterministic CLI exit codes (`PR-177`).

#### M42 — Cross-Repo Contracts, Dual-Track Math & Worker Runtime (PR-178 ~ PR-182)
- 100% C++↔NumPy `camera_math` parity, CCW primitive winding, and `_generate_pan` pitch sign fix (`PR-178`).
- `MeshData` topology preservation, `load_obj` bounds, `model_evaluator` gates, and FGPP `FULGUR_FINITE_REVERSED_Z`/Vulkan/`bfloat16` hardening (`PR-179`).
- 1:1 Go `render-farm` RFC-0039 `JobBundle` alignment & tamper exit `20` (`PR-180`).
- Full 15ch `SoftwareRasterizer` dataset fallback & fail-closed SHA-256 verification (`PR-181`).
- Bidirectional `wire_guard.py`, sealed `import_dag_guard.py`, and synced `WIRE_CONTRACTS.md` (`PR-182`).

#### M43 — Video Adapters, Color & LoRA Training (PR-183 ~ PR-187)
- Pole-free $C^1$ quadratic soft-knee ACEScg bijection, calibrated `border_ring_lab_match`, and pyramid/mask fixes (`PR-183`).
- `WanAdapter` forward tonemap, cut reanchoring, `fuse_overlapping_latents` mean separation, `MultiLoRAManager` & `cloud.py` fixes (`PR-184`).
- `DiffusersWanBackend` strided `ControlNeXtPatchifyStem`, CFG broadcast, `AutoencoderKLWan` stats, and `colab_runner.py` Drive-only guard (`PR-185`).
- Full `StatelessLoRATrainer` `save_state_dict`/`load_state_dict` parity, auto-instantiated `LatentGeometryProbe`, and `.float()` loss precision (`PR-186`).
- Fail-closed streaming `FeatureStoreBuilder`, causal `[0:1],[1:5],...` binning, 4-block `_WhitelistDiT`, and real multi-bundle orthogonal `lora merge` (`PR-187`).

#### M44 — Recon, Refine, 30-Scenario Bench, Data Inspector & v4.2.0 Release (PR-188 ~ PR-192)
- Resolution-invariant de-rotated ray parallax + H/F GRIC 6-DoF, QEM Link Condition manifold guard, tight MC welding, aligned multiview delight, and strict Redline #3 (`PR-188`).
- Authoritative `anatomical_quality_score` routing, telescoping cascade identity, sharpness-weighted K=2 uncertainty fusion, exact 2D DST-I Poisson solver, closed-loop bbox cropping, and Mitsuba CPU fallback (`PR-189`).
- Activated all 30 Synapse-Bench scenario conditions (T1–T5), fail-closed NaN/missing/proxy-FVD gates, report-file `min_scenarios=30`, and `1e-2` consistency tolerance (`PR-190`).
- Monotonic SHM resize generation counter, standalone Bazel config + `@aether/fs/file_utils.h`, 4-tab DOM (`#timeline-controls`, `#comparison-metrics`, `#btn-export-screenshot`, `R/M` AOV), per-entity visibility, and GLB bounds (`PR-191`).
- 15x scalar-math `_qem_decimate` speedup, 13.5x single-pass `exr.channels()` batch OpenEXR reader, `FGPPSequence.save_metadata()` C++ `fulgur` parity, Test SLA closure, blind-spot unit tests, SSOT doc cleanup, and 16-gate `v4.2.0` release (`PR-192`).

## [4.1.0] — 2026-11-21

### Horizon 10 (M37–M40) — Post-Reconciliation True Closure & v4.1.0

Release consolidating Horizon 10 quality hardening: all 20 PRs (PR-153 ~ PR-172)
resolve the 56 defects identified in AUDIT-POST-HORIZON9-2026-11.md, pushing
health score from 61/100 to ≥95/100 with 14 anti-cheat hard gates verified
green on both `ubuntu24` and GitHub Actions.

#### M37 — 止血合流断裂、真绿 CI、静态门禁与 SSOT 纠偏

- **GitHub Actions CI 5/5 Jobs** — Corrected `sautille-stroke/aether` checkout,
  dual `aether_geo`/`aether_metrics` auto-build, and `native/.bazelrc` cleanup.
  (PR-153, GH #141)
- **mypy --strict zero errors** — Fixed package-base crash, cleared 20 errors
  across 216 files, truly enabled Ruff rules, sealed 4 `import_dag_guard`
  bypasses. (PR-154, GH #144)
- **Colab setup repair** — Fixed notebook watchdog `SyntaxError`, removed
  `curl -k`, hardened `tailscale.sh` secret handling. (PR-155, GH #143)
- **WIRE_CONTRACTS.md rewrite** — Removed phantom 16th channel, archived stale
  architecture docs, cleared 16 broken `csrc/geometry` links. (PR-156, GH #142)

#### M38 — 纯 Python 回退、@aether 加速激活、FGPP 相机与沙箱安全

- **Pure Python fallback** — Restored `requires-python >=3.10`, fixed sphere/cone
  watertight topology, activated full-resolution SIMD color and SSIM/CIEDE2000.
  (PR-157, GH #146)
- **FGPP camera fixes** — Corrected `cy=(1+p12)H/2`, Vulkan/Reversed-Z detection,
  NaN rotation/depth blocking, 8/8 StyleProfile schema lock. (PR-158, GH #145)
- **RFC 0039 sandbox hardening** — Pre-sandbox SHA256 validation, DNS IP binding
  anti-TOCTOU, `PersistentHitlLedger` atomic lock. (PR-159, GH #147)
- **Composite pipeline** — Linear EDT feathering, bounded `[0,1)` soft-knee,
  streaming detector (`<70 MB`), mesh container dedup. (PR-160, GH #150)

#### M39 — 真 Diffusers/PEFT、@aether//geo 焊接与逆向精修真值化

- **Real DiffusersWanBackend** — True `Wan2.1-Diffusers` signatures, latent-space
  fusion, 7ch ControlNeXt hooks, scene-cut reanchoring. (PR-161, GH #151)
- **LoRA trainer fixes** — Fixed ternary/Pickle RCE, streaming FeatureStore,
  whitelist-based `lora {train,merge}` CLI. (PR-162, GH #154/#155)
- **MC vertex welding** — Fixed `(nz,ny,nx)` axis order, CCW winding, xatlas UV
  rebuild with split-point, watertight quad remesh, 6-DoF scale. (PR-163, GH #149)
- **Redline #3 seal** — `has_gen3d_highpoly` defaults strict `False`, real Mitsuba
  AD connected graph with SH+GGX, world-space UV delighting. (PR-164, GH #153)
- **Refinement fusion** — Separated detection confidence from anatomical quality,
  non-degenerate Laplacian/Cascade/Poisson operators, anti-ghost stitcher.
  (PR-165, GH #152)

#### M40 — 真 30 场景基準、真 4-Tab Inspector、SLA 提速与 v4.1.0 封板

- **Synapse-Bench activation** — T1–T5 full parameters, real `light_dir_error_deg`,
  `applicable_keys` alias alignment, FVD `uint8` fix. (PR-166, GH #157)
- **3D evaluator world scale** — Removed default `normalize()`, sealed RSI 0.0
  wash-out, fixed viewer symlink and XSS. (PR-167, GH #156)
- **Data Inspector 4-Tab** — Wired 15ch EXR AOV, multi-entity slots, 4-tab
  rendering, fixed resize freeze and C++ UB. (PR-168, GH #159)
- **Script fixes** — Fixed `prepare_datasets.py` 15ch raster, `run_cpu_verification`
  E4 paths, `colab_runner` real E4 and SIGTERM protection. (PR-169, GH #158)
- **Anti-tautology tests** — Removed `MagicMock` false verification, added
  `WIRE_CONTRACTS` auto-check, real multi-view 3D roundtrip tests. (PR-170, GH #161)
- **Test SLA** — Single-process fast ≤20s, parallel slow ≤40s, zero
  `DeprecationWarning`. (PR-171, GH #160)
- **Release v4.1.0** — 14 anti-cheat hard gates verified green on `ubuntu24` and
  GitHub Actions, sealed Horizon 10. (PR-172)

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

[4.1.0]: https://github.com/sautille-stroke/synapse/compare/v4.0.0...v4.1.0
[4.0.0]: https://github.com/sautille-stroke/synapse/compare/v3.6.0...v4.0.0
[3.6.0]: https://github.com/sautille-stroke/synapse/releases/tag/v3.6.0
