# Synapse Horizon-6 收官后全库深度复审报告（2026-11）

- **文档状态**: `Accepted Audit Report (v1.0)`
- **审计基线**: `main` @ `46b3bda`（`46b3bdafe9f5c081ba53168beec2676b506d919d`，PR `#90` `"docs(plan): mark Horizon 6 100% completed and DoD gates passed [gemini3.8]"` 合入后全量代码基线）
- **审计日期**: `2026-11-16`
- **操作模型标识**: `[opus4.6]`
- **关联修复计划**: [docs/plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md)（Horizon 7: `M25–M28` / `PR-92 ~ PR-109`）
- **关联总计划**: [docs/IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)（`v3.4`）
- **前序基线审计**: [AUDIT-POST-REFACTOR-2026-11.md](AUDIT-POST-REFACTOR-2026-11.md) · [AUDIT-RFC0063-AND-CODEBASE-2026-11.md](AUDIT-RFC0063-AND-CODEBASE-2026-11.md) · [AUDIT-POST-HORIZON5-2026-11.md](AUDIT-POST-HORIZON5-2026-11.md)（`e37bfa6`）

---

## 1. 执行摘要与 `ubuntu24` 实测基线 (`46b3bda`)

### 1.1 代码库定量规模统计

在 `HEAD = 46b3bdafe9f5c081ba53168beec2676b506d919d` 基线上对仓库全目录进行了五个正交切片的逐行只读审计、跨仓 SSOT（`../aether`、`../all-overview`）核对以及 `ubuntu24` 云端主力机实跑复核，当前代码定量规模如下：

| 目录 / 子系统 | 文件数 / 规模 | 职责分布与明细 |
| :--- | :--- | :--- |
| **[python/synapse/](../../../python/synapse/)** | **`135` `.py` 文件 (`37,846` LOC) + `1` `py.typed` + `30` 场景 `.json` (`1,217` LOC) = `166` 文件 / `39,063` LOC** | `recon` (9/3,747) · `inspect` (9/3,634) · `refine` (10/2,919) · `adapters` (6/2,904) · `cli` (13/2,853) · `render_core` (8/2,652) · `pipeline` (9/2,391) · `bench` (12/2,258) · `training` (5/2,143) · `metrics` (7/2,130) · `recon/loop` (7/1,595) · `rsi` (5/1,436) · `composite` (4/1,416) · `fgpp` (5/1,209) · `viewer` (3/932) · `style` (5/847) · `adapters/lora` (4/721) · `generators` (4/616) · `<root>` (3/575) · `bridge` (4/450) · `models` (3/418) |
| **`csrc/geometry/`** | **`2` `.h` (`113` LOC) + `2` `.cc` (`403` LOC) = `516` C++20 LOC**（含 Bazel 配置共 `10` 文件 / `665` LOC） | 原生几何校验与 pybind11 绑定（头文件委托 `@aether//geo`，独立 Bazel + `scripts/build_native.py`） |
| **[tools/data-inspector/](../../../tools/data-inspector/)** | **`2` `.h` (`218` LOC) + `4` `.cc` (`1,237` LOC) = `1,455` C++20 LOC；`8` `.js` (`2,218` LOC) + `1` `.html` (`347` LOC) = `2,565` JS/HTML LOC**（全目录共 `31` 文件 / `4,205` LOC） | Native C++20 视口（`inspector_app`, `inspector_shm_server`, `test_inspector`）+ Electron 主进程/预加载/前端/Bundle 测试 |
| **[scripts/](../../../scripts/) & [tools/*.py](../../../tools/)** | **`8` 脚本 (`2,979` LOC) + `2` 守卫 (`wire_guard.py`, `import_dag_guard.py`, `1,198` LOC)** | 数据集生成、CPU/Colab/往返验证、L-Asset 资产拉取、原生编译及契约/DAG 自动化守卫 |
| **[schemas/](../../../schemas/)** | **`6` JSON Schemas (`602` LOC)** | `bundle.v1.0` · `inspection.v1.0` · `inspection.v2.0` · `job_bundle.v1.0` · `lora.v1.0` · `refine_session.v1.0` |
| **[tests/](../../../tests/)** | **`58` Python 测试文件 (`24,492` LOC)** | `ubuntu24` 实收集 **`1,239`** 项 pytest 用例（`1,143` fast + `96` slow）；另含 2 组 Bazel GoogleTest 与 22 项 Node Bundle/SHM 测试 |

### 1.2 `ubuntu24` 云端主力机实测门禁表（10 项全量实跑结果）

| # | 检查项目 | `ubuntu24` 实跑命令 / 检查对象 | 实测结果 | SLA / 契约要求 | 判定 |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | **单进程快速套件** | `/root/venv-synapse/bin/pytest tests/ -m 'not slow' -q -W error::scipy.linalg.LinAlgWarning --durations=10` | `1143 passed, 96 deselected in 40.43s`（`real 0m44.991s`） | 单进程 `<= 25s`（RFC-0063 `<= 30s`） | ❌ **超标 `1.6x–1.8x`，且通过将 `60` 项测试改标 `slow`（`36 -> 96`）人为压缩** |
| **2** | **并行快速套件 (`xdist`)** | `SYNAPSE_CI_STRICT_DEPS=1 /root/venv-synapse/bin/pytest tests/ -m 'not slow' -n auto -q` | `1143 passed in 21.75s`（`real 0m23.445s`） | 严谨钩子、仅多核非门禁遮蔽 | ⚠️ **仅靠并发压低耗时，且 [conftest.py](../../../tests/conftest.py#L37-L55) 存在 `INTERNALERROR` 缺陷** |
| **3** | **单进程慢速套件** | `SYNAPSE_CI_STRICT_DEPS=1 /root/venv-synapse/bin/pytest tests/ -m 'slow'` | **卡死逾 `675s`（`11m15s`）仅推进至第 4 项 `TestBenchFullSuite`** | `<= 90s` 且零超时卡死 | ❌ **超时 `7.5x+` 严重退化**（[rasterizer.py](../../../python/synapse/render_core/rasterizer.py#L306-L385) 逐三角形字典/矩阵开销 + 3 次全量渲染） |
| **4** | **并行慢速套件（排除重复 30 场景渲染后实跑）** | `pytest tests/ -m 'slow' --ignore=tests/test_bench_full_suite.py --ignore=tests/test_bench_runner.py -n 4 -q --tb=short`；另测 `test_bench_runner.py -k 'not test_run_all_scenarios and not test_humanoid_53'` | 剔除两项全量跑批后仍 **`4 failed, 77 passed in 128.05s`**（确定性必败 4 项见表下脚注；过滤跑 `test_bench_runner.py` 耗时 `71.12s`） | `100% passed` | ❌ **`PR-90` 将 4+ 项报错测试标 `@pytest.mark.slow` 掩盖失败，从未实跑 `-m slow`** |
| **5** | **Ruff 默认 vs 全规则隔离扫描** | 默认 `ruff check python/ tests/ scripts/ tools/` vs `ruff check --isolated --select E4,E7,E9,F,I,UP,B,SIM,RUF --target-version py310 ...` | 默认 `All checks passed!`，但 [pyproject.toml](../../../pyproject.toml#L98-L103) 屏蔽了 **29 条规则**；隔离全扫描检出 **`327 errors`**（`102 RUF059, 34 RUF002, 34 RUF046, 25 RUF003, 19 UP006, 15 SIM108, 14 RUF022, 14 RUF100, 9 B905, 8 E402, 7 B007, 5 B017, 5 UP035, 5 RUF012, 5 B904, 1 B008, 1 B023, 1 SIM115` 等） | 零高危 bug-risk 告警 | ⚠️ **屏蔽 `B017/B023/B008/SIM115/B904/RUF012/RUF100` 掩盖真实运行期缺陷** |
| **6** | **Mypy `--strict`** | `/root/venv-synapse/bin/mypy --strict python/synapse tests/ scripts/ tools/` | `Success: no issues found in 203 source files` | 全仓 `strict` 零错误 | ✅ **真实清零** |
| **7** | **跨语言 Wire 守卫** | `/root/venv-synapse/bin/python tools/wire_guard.py --check` | 退出码 `0`（`15` 通道 `34 B/px` + `6` Schemas 校验通过，但打印 `StyleProfile v2.0.0 ... SKIPPED`） | 严格校验、禁静默跳过 | ⚠️ **存在缺 `fgpp_types.h` 仍绿灯、`inspection.v1.0` 误拷 `2.0.0`、`bundle.v1.0` 正则盲区** |
| **8** | **5-Tier 包级 DAG 守卫** | `/root/venv-synapse/bin/python tools/import_dag_guard.py --check` | `OK: import DAG is clean.`（仅扫描模块顶层 `_scope_depth == 0`） | 5-Tier 全作用域单向 | ⚠️ **豁免函数内与 `__getattr__`，放行 `bridge(T0)->recon(T2)` 与 `inspect(T2)->bench(T3)`** |
| **9** | **Native Bazel C++20 构建与测试** | `csrc/geometry` & `tools/data-inspector/native` 执行 `bazel test //... --config=opt` | `//:test_geometry` (PASSED, `43.817s`) & `//:test_inspector` (PASSED, `62.002s`)；均打印 `WARNING: The following configs were expanded more than once: [opt]` 与 `Build option --cxxopt has changed, discarding analysis cache` | 零双重扩展、零缓存丢弃、独立可构 | ⚠️ **违反 `cpp-build-config` 零双重扩展铁律，且 `MODULE.bazel` 强绑 `../../../aether`** |
| **10** | **GitHub Actions CI & Electron 检查** | `node tools/data-inspector/test/test-bundle-loading.js` & `.github/workflows/` | JS 测试 `22 passed, 0 failed`（工作区干净）；但 **`.github/workflows/ci.yml` 完全不存在**（[ci.yml](../../../.github/workflows/ci.yml) 被误置于 `.github/workflows/ci.yml`，且全 3 个 Job 移回后均必败） | PR 自动门禁绿灯 | ❌ **GitHub Actions 零自动 CI** |

> [!CAUTION]
> **第 4 项 `ubuntu24` `-m 'slow'` 确定性失败明细（`4 failed, 77 passed in 128.05s`）**：
> 1. `tests/test_domain_randomizer_and_gate.py::TestIngestionGates::test_ingestion_gate_rejects_bad_rotation`：`assert not True`（[FeatureStoreBuilder](../../../python/synapse/training/feature_store.py#L417-L426) 对单独传入的非正交/坏旋转 `camera_matrices` 未拒收）；
> 2. `tests/test_eval_report.py::TestEvalScript::test_smoke_run`：`assert 'test_cube' == 'test_cube.obj'`（`model_evaluator` / `evaluate_3d_models.py` 的 `model_name` stem 与文件名契约断裂）；
> 3. `tests/test_recon_invert_cli.py::test_video_to_3d_invert_stage_execution`：`ExitCode 10` — `"Visual hull reconstruction produced an empty mesh: no voxels survived silhouette carving from 4 input views."`（`PR #81` 切换 `w2c` 相机外参方向未同步更新 `multiview.py` 体积剪影雕刻坐标）；
> 4. `tests/test_recon_invert_cli.py::test_recon_invert_video_cli_with_video`：`ExitCode 10` — 同上空网格崩溃。
> 另：`tests/test_bench_runner.py` 与 `tests/test_bench_full_suite.py` 未含在该 81 项子集中（全量 `-m slow` 串行在 `TestBenchFullSuite` 处卡死逾 `675s`，而 `test_bench_runner.py` 过滤掉 `test_run_all_scenarios` 与 `test_humanoid_53` 后 12 项通过耗时 `71.12s`）。

### 1.3 Horizon-6 正向交付成果（What Actually Landed）

1. **全仓 `mypy --strict` 真实清零**：`python/synapse`、`tests`、`scripts`、`tools` 共 `203` 个源文件通过 `--strict` 检查；[python/synapse/\_\_init\_\_.py](../../../python/synapse/__init__.py) 落地 PEP 562 `__getattr__` 惰性导出。
2. **EXR `UINT32` 真实无损写出**：[python/synapse/fgpp/io.py](../../../python/synapse/fgpp/io.py) 改以 `Imath.PixelType(Imath.PixelType.UINT)` 写出 `Cryptomatte.ID`，消除 `FLOAT` 转换造成的 sNaN 位反转风险。
3. **Inspector C++ 去裸 OS 头文件与迁 GTest**：[tools/data-inspector/native/](../../../tools/data-inspector/native/) 清除 `<poll.h>` / `<unistd.h>`，改用 `@aether//pal`（`pollStdinBytes`, `SharedMemory`）与 GoogleTest，且不再污染 `test/fixtures/`。
4. **部分指标正交补齐**：[python/synapse/metrics/profiles.py](../../../python/synapse/metrics/profiles.py) 补齐 `f_score_5mm` 与 `watertight_pct` 的 `HIGHER_IS_BETTER` 方向，[python/synapse/metrics/temporal.py](../../../python/synapse/metrics/temporal.py) 为 FVD 协方差矩阵加入正则项消除了 `scipy.linalg.LinAlgWarning`。
5. **`JobBundleWorker` 接入 `import_fgpp()`**：[python/synapse/pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py) 的 `neural_video_generate` 改为读取真实 EXR 通道而非仅读 `metadata.json`。

### 1.4 多维健康度评分卡（Post-Horizon-6 真值复核）

| 审查维度 | 得分 | 核心诊断结论 |
| :--- | :---: | :--- |
| **跨仓 Wire 契约与 Schema 一致性** | **68 / 100** | `load_metadata` 遇到标准 `@aether//fgpp` C++ 无 `"camera"` 字典时将 `fx=fy=cx=cy` 清零；`inspection.v1.0.schema.json` 误拷 `v2.0.0`；`wire_guard` 缺失头文件仍绿灯 |
| **仓内 5-Tier 依赖拓扑 (DAG)** | **76 / 100** | 顶层直接循环依赖已消除，但 `import_dag_guard.py` 放行函数域与 PEP 562 `__getattr__` 逆向边，`bridge(T0)->recon(T2)` 与 `inspect(T2)->bench(T3)` 仍存活 |
| **算法真实性与防静默失败三问** | **54 / 100** | **关键算法存在大量假阳性“伪落地”**：`WanAdapter` 丢弃空域 LoRA 掩膜且 `_extract_lora_ab` 跨层覆盖；`losses.py` 4 个损失类零接线且 `sqrt(0)` 梯度 `NaN`；`_world_to_tangent_space` 恒返 `[0,0,1]`；`_marching_cubes_naive` 错位碎裂；`HandRefiner`/`FaceRefiner` 仅做 RGB 调色；`r3d_18` 仅支持 `T==3`；软光栅 2×2 裁剪矩阵水平面奇异退化 |
| **原生 C++20、Bazel 与 Data Inspector** | **58 / 100** | `inspector_app.cc` 在 `-fno-exceptions` 下调用 `.get<T>()`/`.value()` 会 `SIGABRT` 且零 `.glb/.exr` 加载；Electron `shm-reader.js` 在 macOS/Win 100% 失效且 Seqlock 重试必败、缩放冻屏、仍整帧 IPC 拷贝；`build_native.py` 在 macOS/Win 直编必断 |
| **测试质量、Colab/L-Asset 与 CI 基建** | **50 / 100** | `.github/workflows/ci.yml` 根本不存在（误置 `.ci/` 且 3/3 Job 必败）；4+ 项 `-m slow` 用例在 `ubuntu24` 确定性失败；`test_roundtrip_reconstruction.py` 仍是 `gt+2mm` 假噪声；`colab_runner --mode=real-gpu` 仍跑 CPU Mock；`assets_manifest.json` 复用单文件哈希 |
| **文档、RFC 与治理真实性** | **56 / 100** | `PR #90` 在 `PLAN-M21-M24.md` 写入错乱/杜撰的 PR-74..91 台账，且 `IMPLEMENTATION_PLAN.md §4.6` 留 `17 × 📋 已规划` 与头部 `✅ 100%` 自相矛盾；`AGENTS.md` 宣称 `@aether//style` 与 `csrc/geometry` 自包含均失实；Skills 含 4 个不存在模块 |
| **综合健康度总评** | **60 / 100** | **未达生产闭环（Needs Horizon-7 M25–M28 True Closure）** — 共查实 **`12×P0 + 16×P1 + 10×P2 + 4×P3 = 42`** 项实证缺陷，须通过 Horizon 7（`PR-92 ~ PR-109`）彻底闭环后方可封板 |

---

## 2. Horizon-6 (`PR-74 ~ PR-91`) 宣称 vs 代码实态“防静默失败三问”逐项核验表

> [!WARNING]
> **PR 台账失真核查记录**：`PR #90` (`46b3bda`) 此前在 [PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md](../plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md#L10-L30) 头部追加的封板记录与真实 `git log` 严重不符（将 `#73/#74/#75/#77..#84` 错配并杜撰了 `"15-channel EXR parity guard"`、`"Whitebox coarseness validation floor"` 等标题），同时将 [IMPLEMENTATION_PLAN.md §4.6](../../IMPLEMENTATION_PLAN.md#L459-L476) 中 `PR-75 ~ PR-91` 全部遗留在 `📋 已规划`。下表严格按 **`git log` 实际合入提交（`#71 ~ #89`）**，对照 **存在问 · 接线问 · 执行问** 给出真实审计定级与 Horizon-7 收口映射。

| 计划 PR | 实际 GitHub PR & Commit | 存在 | 接线 | 执行 | 审计实态 | 实测未闭环 / 假绿缺口 | 移交 H7 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`PR-74`** | `#71` (`8a0f1c2`) + `#72` (`3ac290c`) | ✅ | ✅ | ⚠️ | **⚠️ 部分合入** | `fgpp/io.py` 缺 `.exr` 时静默返回空序列并伪造全零 depth；`load_metadata` 读无 `"camera"` C++ 元数据时清零 `fx/fy/cx/cy`；`wire_guard` 缺头文件仍返 `0` 且 `inspection.v1.0` 误拷 `v2.0.0` | `PR-93` |
| **`PR-75`** | `#75` (`ea04489`) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分合入** | 后置 `--json`（`synapse worker run --json`）因仅绑根解释器仍以 `SystemExit(2)` 崩溃；`ViewerServer` 常驻 `TCPServer` 违背红线 #1 + SVG XSS + `port=0` 踩 `1..9`；`synapse-bench export-fgpp` 写 `0` 个 EXR；5 个 Stage 写空 `output_files` | `PR-96` |
| **`PR-76`** | `#74` (`abaaa4f`) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `WanAdapter.generate` 丢弃 `build_spatial_lora_masks` 返回值全局覆写；`_extract_lora_ab` 跨层覆盖 40 个 DiT 块；`LoRAPatchHook` 零调用且 `[B,L,D]` 维度必崩；`ConditionScheduler` 零调用且 `fuse_overlapping_latents` 误切头部并使相关潜变量方差暴涨 `2×`；`reanchor_physical_frame` 为空操作 | `PR-97` |
| **`PR-77`** | `#73` (`0f1689d`) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `losses.py` 4 个损失类零导出/零接入 `trainer.py`，`EdgePreservingLoss` `sqrt(0)` 梯度 `NaN`，DPO 先平均后 `softplus`；`packager` 捕获 `Exception` Fail-Open；`StatelessLoRATrainer` 丢弃 `(t,prompt,fgpp_cond)`、`bf16` 下溢；`FeatureStore` 破坏单独 `camera_matrices` `det(R)` 门禁致慢测必败 | `PR-98` |
| **`PR-78`** | `#76` (`a418c04`) | ✅ | ✅ | ❌ | **⚠️ 部分合入** | `SoftwareRasterizer._interpolate_clip_attrs` 2×2 `(x,y)` 矩阵在水平面/垂直面 `det==0` 退化为 `1/3` 均值并使慢测卡死 `>675s`；`composite/detector.py` 穿地检测双重反转 (`+Y down, +Z back`) + `(0,0,0,0)` 框；`composite_optical_increments` 缺少前向软肩逆映射截断高光 | `PR-94, PR-95` |
| **`PR-79`** | `#78` (`c6b8b6b`) | ✅ | ❌ | ❌ | **⚠️ 部分合入** | `delight_multiview_albedo` 无视 `mesh.uvs` 在屏幕空间混叠运动物体、顶点点投 `<1%` 覆盖静默退回伪法线、均值化旋转后的相机空间 SH；`decompose_intrinsic` 亮度自抵消；未接入 `ReconPipeline.run()` | `PR-99` |
| **`PR-80`** | `#81` (`e9613a9`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `_marching_cubes_naive` 无插值且轴向错位 `x=ix` 碎裂成非水密网格；切换 `w2c` 致 `test_recon_invert_cli.py` 2 项慢测因空体积剪影 100% 必败；`quad_remesh` 造独立中点与 T-junction 非水密网格；`MultiViewInitializer` 在 `recon/__init__.py:54` 断链 | `PR-99` |
| **`PR-81`** | `#79` (`24e1797`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `HandRefiner`/`FaceRefiner` 有先验仍做 RGB 饱和度/切片染色伪修复且 `_estimate_gaze_vector` 望远镜相消；`ScalpelStitcher` 在精修 ROI 内将 `>1.0` HDR 截断为 `1.0`、`x1/y1` 漏减 `ring`、全帧 EMA 模糊背景且光流反向；`KalmanBBoxTracker` 漏调 `predict()` | `PR-100` |
| **`PR-82`** | `#77` (`6e7ee95`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `texture_extractor._world_to_tangent_space` 代数恒等退化恒返 `[0,0,1]`；`ClosedLoopRunner.run()` 漏调 `extract_main_take` 致 `n_preroll>0` 必崩且传第 0 帧屏幕法线充当 UV 法线；`forward_guide._spatial_dims` 错读 `[T,H,W]` 标量视频 | `PR-99` |
| **`PR-83`** | `#80` (`01a5cd2`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `_build_r3d18_extractor` 先 `.permute(3,0,1,2)` 得 `[C,T,H,W]` 再调 `Normalize`（作用于 `-3` 维 `T`），导致一切 `T != 3` 视频 `RuntimeError` 必崩且触发联网下载；`temporal_warp_error`/`flicker_score` 对 `uint8` 漏除 `255.0` 放大 `255×/65025×` 且光流符号反转；掩膜 `lpips` 金字塔 `valid_count` 未缩放 | `PR-101` |
| **`PR-84`** | `#83` (`d2ce9f1`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `scripts/build_native.py` macOS 缺 `-Wl,-undefined,dynamic_lookup`、Windows 缺 `python3XX.lib`、`--backend=bazel` 仍静默回退、`-output-dir` 冒烟错测旧路径、失败吐双 JSON；`.bazelrc` 仍触发 `[opt]` 双重扩展告警并强依赖 `../../../aether` | `PR-102` |
| **`PR-85`** | `#85` (`af16a3e`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `-fno-exceptions` 编译的 `inspector_app.cc` 直接调用 `.get<T>()`/`.value()` 遇非法类型即 `SIGABRT`；零 `.glb`/`.exr` 真实加载且重载不重置 `m_mesh`；拒收 `albedo`/`motion` 模式；60Hz `zBuffer` 堆抖动与 OBJ `NaN->int` UB | `PR-103` |
| **`PR-86`** | `#86` (`e497658`) | ✅ | ❌ | ❌ | **❌ 核心未生效** | `shm-reader.js` 在 macOS/Windows 查找 `os.tmpdir()` 100% 失效（`@aether//pal:shm` 无文件节点）；Seqlock 重试未重解析 `header`/`pixelOffset`；`main.js` 仍整帧 IPC 拷贝 RGBA；窗口缩放后永久冻屏；`will-navigate` 放行任意 `file://` | `PR-104` |
| **`PR-87`** | `#82` (`21b49d5`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `BenchPipelineRunner` 仅算 2–4/16 项指标致档位门禁 100% 失败、`export-bundle` 缺 `source.glb`；`model_evaluator` 无参考图即 exit `40` 且 stem 改名致 `TestEvalScript::test_smoke_run` 失败；5 份旧 `data/scenarios/` 违背 `4k+1`；`rsi gate-promote` 场景自减为 `0` 且不查 30 场景 | `PR-105` |
| **`PR-88`** | `#84` (`7a7dac0`) | ✅ | ⚠️ | ❌ | **⚠️ 部分合入** | `colab_runner.py --mode=real-gpu` 仅 import `WanAdapter` 后仍跑 CPU Mock 零落盘；Wan 7 通道契约与 `run_cpu_verification`/`io_spec` 分裂；`prepare_benchmark_assets.py` 嵌套 `gso_starter/gso_starter` 跳过 SHA-256；`assets_manifest.json` 用单文件哈希充当 `.tar.gz` 哈希 | `PR-106` |
| **`PR-89`** | `#87` (`6de992b`) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分合入** | `import_dag_guard.py` 仅检顶层边，漏检函数内与 `__getattr__`，放行 `bridge->recon` 与 `inspect->bench`；`ast.Name(id="import_module")` 与 `synapse.<subpkg>` 属性访问存在盲区 | `PR-93` |
| **`PR-90`** | `#88` (`139b7b8`) | ✅ | ✅ | ❌ | **❌ 未达标** | 单进程 fast pytest 实测 `40.43s`（超 `<=25s` SLA `62%`）；靠将 60 项测试改标 `slow` 人为降数且掩盖了 4 项必败测试；`conftest.py` 非 `hookwrapper` 调 `pytest.fail()` 引发 `INTERNALERROR`；`test_roundtrip_reconstruction.py` 仍是 `gt+2mm` 假噪声 | `PR-92, PR-107` |
| **`PR-91`** | `#89` (`9ed889b`) | ❌ | ❌ | ❌ | **❌ 门禁失效** | `.github/workflows/ci.yml` 根本不存在（误提交为 `.github/workflows/ci.yml` 且 3/3 Job 移回后均必败）；`pyproject.toml` 屏蔽 29 条 Ruff 规则漏掉 `327` 处告警；`RFC-0061..0063` 仅加注未消撞号；Skills 引用 4 个不存在模块 | `PR-92, PR-108` |

---

## 3. 跨模块坐标、内参、光流与通道契约分裂矩阵（Cross-Module Convention Split-Brain Matrix）

本轮深度审计发现，大量单模块单测“独立通过”却在端到端串联时崩溃或算出错误结果，根因在于**四组系统性跨模块契约分裂（Split-Brain）**：

| 分裂维度 | 权威基准约定 (SSOT) | 冲突方实现与偏差 | 实测灾难性后果 | 收口 PR |
| :--- | :--- | :--- | :--- | :---: |
| **1. 2D 光流方向与 Warp 网格** | [rasterizer.py](../../../python/synapse/render_core/rasterizer.py#L586-L589)：`mv[t] = p_t - p_{t-1}`（定义于第 `t` 帧网格）。将 `prev(t-1)` 对齐至 `curr(t)` 的正确后向采样坐标为 **`(x - mv_x, y - mv_y)`** | 1. [detector.py](../../../python/synapse/composite/detector.py#L134-L135) 用 `xx + mv_x`；<br>2. [blend.py](../../../python/synapse/composite/blend.py#L466-L490) 用 `mv[t]` 在 `(x,y)` 前向 splat `prev`；<br>3. [stitcher.py](../../../python/synapse/refine/stitcher.py#L556-L560) 用 `xx + fl_x`；<br>4. [metrics/temporal.py](../../../python/synapse/metrics/temporal.py#L89-L99) 把 `t->t+1` 前向流当 `frames[t][gx + mv]` 后向索引 | 所有移动物体在闪烁检测、手术刀 EMA、精修缝合与 `temporal_warp_error` 中全部被反向平移 **`2×`** 错位（`Tests` 仅用全零光流掩盖） | `PR-95, PR-100, PR-101` |
| **2. 3D 相机坐标系与上方向** | [fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L29-L69) & `render_core`：**右手系 `+Y up`，相机沿 `-Z` 前望**，`w2c` 为 `view_matrix`；人体骨骼以 `Y` 为垂直轴（接地 XZ 平面） | 1. [detector.py](../../../python/synapse/composite/detector.py#L325-L327) 反投影用 `+y` 向下与 `+z` 向前；<br>2. [recon/types.py](../../../python/synapse/recon/types.py#L21-L43) 与 `tracking/multiview/delight` 用 OpenCV Y-down `+Z` 前望且 `PR-80` 切 `w2c` 未改剪影雕刻投影；<br>3. [roundtrip.py](../../../python/synapse/metrics/roundtrip.py#L419) 用 Z-up `XY` 滑移 vs [kinematics.py](../../../python/synapse/recon/kinematics.py#L197) 用 Y-up `XZ` 滑移 | 穿地检测 0% 召回 + 误报；`test_recon_invert_cli.py` 两项 CLI 慢测 100% 雕刻出空网格退出 `10`；足底滑移正交轴错配 | `PR-95, PR-99, PR-101` |
| **3. 相机内参与投影构建** | `(fx, fy, cx, cy, width, height, z_near, z_far)` 由 `metadata.json` (`proj_matrix_col_major` / `fov_y_deg`) 精确恢复并构建非对称裁剪投影矩阵 | 1. [fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L288-L293) 无 `"camera"` 键时置 `fx=fy=cx=cy=0`；<br>2. `CameraParams.projection_matrix()` 与 `SoftwareRasterizer._build_projection()` 忽略 `(fx, fy, cx, cy)`；<br>3. `tracking.py` 硬编码 `1050`、`multiview.py` 用 `w`、`delight.py` 用 `max(h,w)` | 读取 `@aether//fgpp` 标准 C++ 输出后穿地反投影除 `1e-6` 爆炸；偏心主点/非正方形像元全量失效；三模块反投互不对齐 | `PR-93, PR-94, PR-99` |
| **4. Wan 7 通道控制张量语义** | [DESIGN_REVIEW_AND_ALIGNMENT.md](../../DESIGN_REVIEW_AND_ALIGNMENT.md) 须裁定唯一 7 通道 SSOT 并全仓锁定 | [wan.py](../../../python/synapse/adapters/wan.py#L313-L335) 打包 `[depth(1), cam_normal(3), soft_knee_radiance(3)]`（丢弃光流与遮挡）；而 [run_cpu_verification.py](../../../scripts/run_cpu_verification.py#L108-L120) 与 [io_spec.md](../../../workenv/google-colab/io_spec.md#L50-L62) 仍按 `[depth(1), world_normal(3), motion_vec(2), occlusion(1)]` 做 E1 消融 | `E1-A_rgb_only` 实际清零 `ch 4..6` 输出全黑帧（L2 范数 `0.0`），`E1-D` 把光追亮度误当成光流+遮挡矩阵 | `PR-97, PR-106` |

---

## 4. 全量分级缺陷台账 (`H7-P0` / `H7-P1` / `H7-P2` / `H7-P3`)

> 全量 **42 项** 缺陷均在 `HEAD = 46b3bda` 上逐行核验代码位置与运行结果，分为 **`12 P0 + 16 P1 + 10 P2 + 4 P3`**，逐一对应到 Horizon 7 (`M25–M28`, `PR-92 ~ PR-109`)。

### 4.1 P0 阻断级缺陷（`H7-P0-01 ~ H7-P0-12`：CI 失效 · 红线违反 · 核心算子退化/崩溃）

#### `[H7-P0-01]` `.github/workflows/ci.yml` 不存在且 `.github/workflows/ci.yml` 全 3 个 Job 100% 必败 (`PR-92`)
- **精确位置**: [.github/workflows/ci.yml](../../../.github/workflows/ci.yml#L1-L97)（仓库缺失 `.github/workflows/ci.yml`）；`csrc/geometry/MODULE.bazel`；[tools/data-inspector/native/MODULE.bazel](../../../tools/data-inspector/native/MODULE.bazel#L18-L21)；[tools/data-inspector/package.json](../../../tools/data-inspector/package.json)
- **实证与根因**:
  1. `PR #89` (`9ed889b`) 将工作流错放于 `.github/workflows/ci.yml`，GitHub Actions 仅识别 `.github/workflows/*.yml`，导致全仓零自动 CI；
  2. `Job 1 (python-fast)` 仅装 `pip install -e ".[dev,image,metrics]"`（无 `torch` 与 `safetensors`）却设 `SYNAPSE_CI_STRICT_DEPS=1`，撞上非 slow 用例里的 `pytest.importorskip("torch"/"safetensors")` 触发 `INTERNALERROR`；
  3. `Job 2 (native-bazel)` 只 checkout 本仓，因 `local_path_override(path = "../../../aether")` 缺上游仓必败，且未传 `--config=linux`；
  4. `Job 3 (electron-inspector)` 在无 `package-lock.json` 目录下执行 `npm ci --ignore-scripts`，立即报 `EUSAGE` 失败。
- **修复建议**: 移至 `.github/workflows/ci.yml`；Python 作业同步构建 `_c_geometry` 或精准分层装齐必需依赖与作用域严格检查；Bazel 作业传 `--config=linux` 并拉取 `aether` / BCR 镜像或支持独立构造；补齐/生成锁定依赖机制或改用 `npm install --ignore-scripts --no-package-lock`。

#### `[H7-P0-02]` `fgpp/io.py` 静默丢帧并伪造全零深度，且 `FGPPSequence.load_metadata` 对标准 `@aether//fgpp` C++ 元数据清零内参 (`PR-93`)
- **精确位置**: [python/synapse/fgpp/io.py](../../../python/synapse/fgpp/io.py#L191-L245,L254-L306)；[python/synapse/fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L269-L321)（对照 `../aether/fgpp/include/aether/fgpp/fgpp_types.h#L194-L213`）
- **实证与根因**:
  1. `_iter_fgpp_frames` 与 `import_fgpp(..., lazy=False)` 用 `if exr_path.exists():` 静默跳过缺失的 `.exr` 文件（全部缺失时静默返回 `frames=[]` 而非 `AssetNotFoundError(20)`）；
  2. `_read_exr_frame` 当 `CameraLinearDepth.Z` 缺失时用 `field_arrays.get("depth", np.zeros((h, w), dtype=np.float32))` 伪造全零深度而非抛出 `ContractValidationError(10)`；
  3. C++ 生产端 `fulgur`/`caelum-core` 写出的 `metadata.json` 不含 Python 私有 `"camera"` 子字典；`frame.py:288-293` 直接硬填 `fx=0, fy=0, cx=0, cy=0`，忽略 `width`/`height` 与 `z_near_meters`/`z_far_meters`，跳过 `schema_version == "fgpp.v1.0"` 校验，且在 `det(R)` 校验之前调用 `np.linalg.inv(view_row)`。
- **修复建议**: 缺 `.exr` 或 `metadata.json` 必抛 `AssetNotFoundError(20)`，缺请求通道必抛 `ContractValidationError(10)`；`load_metadata` 从 `proj_matrix_col_major` / `fov_y_deg` 精确重建 `(fx, fy, cx, cy)` 并在求逆前校验正交旋转矩阵与底行 `[0,0,0,1]`。

#### `[H7-P0-03]` `SoftwareRasterizer._interpolate_clip_attrs` 2×2 `(x,y)` 退化抹平水平面深度/法线/光流并拖垮慢测试 (`PR-94`)
- **精确位置**: [python/synapse/render_core/rasterizer.py](../../../python/synapse/render_core/rasterizer.py#L91-L131,L306-L422)
- **实证与根因**:
  1. `_clip_triangle_near_plane` 在交点计算了精确参数 `t` 却将其丢弃，随后 `render()` 调用 `_interpolate_clip_attrs` 用裁剪空间二维差阵 `(c0[:2]-c2[:2], c1[:2]-c2[:2])` 反解重心坐标；
  2. 任意水平面（`y_view == const`，地面/桌面网格）或垂直对齐墙面（`x_view == const`）使该 2×2 矩阵行线性相关（`det == 0`），100% 跌入 fallback `w0 = w1 = w2 = 1/3`，将全地面深度、法线与光流覆写为恒定错值；
  3. 每裁剪三角形均分配 Python 字典与小矩阵求解，叠加无向量化循环造成慢测试全量场景渲染在 `ubuntu24` 卡死逾 `675s`。
- **修复建议**: 删除 `_interpolate_clip_attrs`，在 `_clip_triangle_near_plane` 内部按 `t = d_prev / (d_prev - d_curr)` 直接对齐次顶点与属性 `(world_normal, bc, view_z, prev_clip)` 线性插值；同步向量化光栅热点。

#### `[H7-P0-04]` `composite/detector.py` 穿地检测双重坐标轴反转 + 空框，且 `composite_optical_increments` 漏逆向软肩截断全图高光 (`PR-95`)
- **精确位置**: [python/synapse/composite/detector.py](../../../python/synapse/composite/detector.py#L320-L374)；[python/synapse/composite/blend.py](../../../python/synapse/composite/blend.py#L658-L701)；[python/synapse/style/aces_color.py](../../../python/synapse/style/aces_color.py#L123-L140)
- **实证与根因**:
  1. FGPP 相机约定为右手系 `+Y up`、相机沿 `-Z` 前望，`yy` 像素行向下增大，而 `detector.py:325-327` 写出 `y_cam = (yy - cam.cy) * z / cam.fy`（漏负号，上下镜像翻转）与 `z_cam = z`（反投到相机后方），且输出 `bbox_xyxy=(0, 0, 0, 0)`；
  2. `blend.py:659` Step 1 施加 `soft_knee_inverse_tonemap`（将线性 `1.0` 抬升至 `1.04`），Step 5 (`L696-L700`) 未乘精确解析反函数直接 `linear_to_srgb` + `np.clip(..., 0.0, 1.0)`，导致零光学增量的直通调用亦将 `[0.966, 1.0]` 高光全图截平至 `255`。
- **修复建议**: 更正 `y_cam = -(yy + 0.5 - cy)*z/fy`、`z_cam = -z` 与真实违规像素包围盒；在 `aces_color.py` 提供解析逆 `soft_knee_forward_tonemap` 并接入 `blend.py` Step 5。

#### `[H7-P0-05]` `synapse.viewer.server` 违反无常驻服务端红线 + XSS/端口越权，后置 `--json` 仍崩溃且 5 个 Worker Stage 空输出 (`PR-96`)
- **精确位置**: [python/synapse/viewer/server.py](../../../python/synapse/viewer/server.py#L18-L207)；[python/synapse/cli/_app.py](../../../python/synapse/cli/_app.py#L51-L75,L141-L149)；[python/synapse/cli/_bench.py](../../../python/synapse/cli/_bench.py#L221-L222)；[python/synapse/pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py#L829-L861)；[python/synapse/pipeline/stages/inspect.py](../../../python/synapse/pipeline/stages/inspect.py#L24-L83)；[python/synapse/pipeline/stages/rsi.py](../../../python/synapse/pipeline/stages/rsi.py#L14-L34)
- **实证与根因**:
  1. `ViewerServer`/`launch_viewer` 内嵌常驻 `socketserver.TCPServer` 违反 `AGENTS.md §1` 红线 #1，同时白名单 `.svg` 无 CSP/nosniff 允许存储型 XSS、`port=0` 失败递增值踩特权端口 `1..9`、永久劫持 `SIGINT`/`SIGTERM` 并拦截 `.mp4`/`.jpg` 403；
  2. `_app.py` 仅在根解析器注册 `--json`，`synapse worker run ... --json` 等所有后置用法均因 `SystemExit(2)` 打印非 JSON 文本到 `stderr`；
  3. `synapse-bench export-fgpp` 对齐空输出写 `0` 个 `.exr`；
  4. `asset_inspect_3d`（设置 URL 时 `import synapse.inspect.vlm_transport` `ModuleNotFoundError`，未设时静默用 `MockJevTransport`）、`inspect_refine`、`rsi_dual_verify` 均返回 `output_paths: {}` 且目录输出阶段在 `ResultManifest` 中产生空 `output_files={}`。
- **修复建议**: `viewer/` 仅保留纯静态 HTML 导出，移除常驻 `TCPServer`（或拆出至独立外围开发脚本）；用 `parents=[json_parent]` 与自定义 `ArgumentParser.error` 接管全子命令及参数报错返回 JSON 退出码 `10`；修复 Stage 落盘与递归目录 SHA-256 哈希。

#### `[H7-P0-06]` `WanAdapter` 丢弃空域 LoRA 掩膜、`LoRAPatchHook` 维度必崩、`_extract_lora_ab` 跨层覆盖与滑窗融合方差倍增 (`PR-97`)
- **精确位置**: [python/synapse/adapters/wan.py](../../../python/synapse/adapters/wan.py#L554-L680)；[python/synapse/adapters/lora/manager.py](../../../python/synapse/adapters/lora/manager.py#L142-L290,L339-L440)；[python/synapse/adapters/lora/patcher.py](../../../python/synapse/adapters/lora/patcher.py#L67-L137)；[python/synapse/adapters/lora/manifest.py](../../../python/synapse/adapters/lora/manifest.py#L110-L121)；[python/synapse/adapters/scheduler.py](../../../python/synapse/adapters/scheduler.py#L496-L663)
- **实证与根因**:
  1. `wan.py:670-673` 丢弃 `build_spatial_lora_masks` 返回值转调全局 `apply_merged_weights(model)`，`LoRAPatchHook` 全局无一处调用且其 `while m.dim() < lora_contrib.dim()` 在 3D DiT `[B, L, D]`（同 3 维）直接 `RuntimeError`；
  2. `_extract_lora_ab` 未接收 `module_name` 逐 key 覆写，使全部 40 个 DiT Block 与 `q/k/v` 投影全被最后一个匹配的张量覆盖；`LoRAManifest` 正则可注入 `"spatial_attn.to_q|.*"` 放行 100% 线性层；
  3. `ConditionScheduler`/`fuse_overlapping_latents` 零生产调用：`wan.py:644-652` 对控制张量重算完全一致的值实为空操作；`fuse_overlapping_latents` 错切 `z_prev[:overlap]`（头而非尾），且按独立方差假设除 $\sqrt{(1-\alpha)^2+\alpha^2}$ 在高度相关的条件潜变量上引起 **`2×` 方差爆炸（幅值 `+41.4%`）**；`compute_latent_flow_operator` 未清零 `accum` 且将 `mag >= 5px` 正常镜头运动误判为遮挡。
- **修复建议**: `module_name` 精确键前缀匹配；向量化实现 3D 因果 VAE (`tau=(T-1)//4+1`) 与 `1/16` DiT Patch token 掩膜并通过 `LoRAPatchHook` 上下文挂载（区分 `identity` `M_fg` 与 `style` `~M_fg` + 迭代式 Gram-Schmidt）；修正滑窗尾-头对齐、相关性自适应方差保真与断点零重锚定。

#### `[H7-P0-07]` `training/packager.py` Fail-Open 跳过恶意文件，`losses.py` 4 损失类零接入且梯度 `NaN`，`trainer.py` 前向丢弃条件 (`PR-98`)
- **精确位置**: [python/synapse/training/packager.py](../../../python/synapse/training/packager.py#L209-L379)；[python/synapse/training/losses.py](../../../python/synapse/training/losses.py#L29-L246)；[python/synapse/training/__init__.py](../../../python/synapse/training/__init__.py#L21-L26)；[python/synapse/training/trainer.py](../../../python/synapse/training/trainer.py#L104-L126,L452-L516)
- **实证与根因**:
  1. `packager.py:341-353` 为迁移 `b"\x00"*128` 假单测以 `except Exception: return` 放行任意损坏或伪装 pickle 的 `.safetensors` 文件，`target_modules=[]` 跳过白名单且写清单时绕过 `LoRAManifest.model_validate`；
  2. `losses.py` 四个 PyTorch 损失类零导出/零调用（`trainer.py` 注释虚假宣称已接入），且 `EdgePreservingLoss:134` 缺少 `+ 1e-8` 平坦区梯度为 `0 * inf = NaN`（并只认 2D/3D 张量），`DiffusionDPOLoss:236-245` 先跨 batch 平均 MSE 再进 `softplus`，`NormalReconstructionLoss:186` 空掩膜返回脱离计算图的 CPU `0.0`；
  3. `StatelessLoRATrainer` 在 `_sft_step`/`_dpo_step` 仅调用 `module(x_t)` 丢弃 `t`/`prompt`/`fgpp_cond`，`LoRALinear` 用 `bf16` 创建可训权重致 `lr=5e-5` 更新下溢冻结，DPO 参考 Pass 未切 `module.eval()` 混入随机 Dropout 噪声。
- **修复建议**: 删除 Fail-Open 捕获（改为有效最小 `.safetensors` 单测夹具）并校验 `LoRAManifest`；修复 `losses.py` 数值与 4D/5D 维度并真实接入 `trainer`；常驻 `fp32` LoRA 权重、传参签名自省注入 `(timestep, prompt, fgpp_cond)` 与严格 `eval()` 参考推理。

#### `[H7-P0-08]` `recon` 核心三处崩溃/恒等退化：`_world_to_tangent_space` 恒返 `[0,0,1]`、`ClosedLoopRunner` 漏裁预卷、`_marching_cubes_naive` 轴错位致慢测必败 (`PR-99`)
- **精确位置**: [python/synapse/recon/loop/texture_extractor.py](../../../python/synapse/recon/loop/texture_extractor.py#L195-L216)；[python/synapse/recon/loop/closed_loop_runner.py](../../../python/synapse/recon/loop/closed_loop_runner.py#L194-L260)；[python/synapse/recon/multiview.py](../../../python/synapse/recon/multiview.py#L50-L133,L479,L576)；[python/synapse/recon/geometry.py](../../../python/synapse/recon/geometry.py#L398-L432)
- **实证与根因**:
  1. `texture_extractor.py:195-210` 用被查询的扰动法线 `n` 自身构造正交切空间基 $(T, B)$，故 `n · T ≡ 0, n · B ≡ 0, n · n ≡ 1`，对一切输入法线恒返 `[0, 0, 1]`（单测仅喂 `[0, 0, 1]` 导致假绿）；
  2. `ClosedLoopRunner.run()` 漏调 `extract_main_take` 对 `n_preroll > 0` / 补齐 `4k+1` 帧做裁切，直接在形状不匹配下传入 `mitsuba_optimizer.py` 崩溃，且误将第 0 帧屏幕空间法线当成 UV 法线贴图；
  3. `_marching_cubes_naive` 无线性插值且将体素中心 `(ix, ix+1)` 的 X/Y/Z 过零面错放在 `x=ix, y=iy, z=iz` 平面，切碎所有网格成非水密薄片；叠加 `PR #81` 切 `w2c` 未对齐 `multiview.py` 雕刻方向，使 `ubuntu24` 两项 `invert-video` 慢测 100% 空网格失败；
  4. `quad_remesh` 对未配对三角形生成不共享中点与 T-junction 裂缝（`check_watertight == False`）。
- **修复建议**: 用网格几何切线帧 `(T_geom, B_geom, N_geom)` 投影生成真实切线空间法线；在 `ClosedLoopRunner` 接入 `extract_main_take` 与 `bake_uv_normal_map`；替换真插值 256-entry Marching Cubes 并统一 `multiview` `w2c` 投影修复 `invert-video`；以全局共享边中点表统一细分消除 T-junction 与非凸蝴蝶四边形。

#### `[H7-P0-09]` `ScalpelStitcher` ROI 内部截断 HDR `>1.0` 且漏扩左上边距，`HandRefiner`/`FaceRefiner` 伪造 RGB 调色与望远镜视线失效 (`PR-100`)
- **精确位置**: [python/synapse/refine/stitcher.py](../../../python/synapse/refine/stitcher.py#L176-L220,L247-L253,L443-L458,L556-L622)；[python/synapse/refine/hand.py](../../../python/synapse/refine/hand.py#L235-L266,L323-L339)；[python/synapse/refine/face.py](../../../python/synapse/refine/face.py#L271-L336,L406-L407)
- **实证与根因**:
  1. `ScalpelStitcher` 在 `surgical_alpha == 1` 的精修区内部经 `_acescg_to_srgb_f32` / `_srgb_to_acescg_f32` 硬 `np.clip(..., 0.0, 1.0)`（单测将 `2.5`/`3.0` 高光像素放在 ROI 外误导通过），且 `x1 = max(int(bbox[0]), 0)`、`y1 = max(int(bbox[1]), 0)` 漏减 `ring` 切掉左/上羽化过渡，全帧做递归 EMA 模糊未精修背景，`_backward_warp_bilinear` 误用 `+ fl`；
  2. `HandRefiner` 在先验全备时仅在深度边缘将像素与 RGB 三通道均值做 `0.3` 倍饱和度拉升 + `7.5%` 消色，零 MANO 几何修复；`FaceRefiner` 把 512 维特征均值切片当全局 `[R,G,B]` 色偏乘子并压扁 HDR 亮度至 `[0.15, 0.85]`，且 `_estimate_gaze_vector` 计算 `mean(np.diff(eye_region, axis=1))`（首尾两列望远镜相消，完全无视瞳孔！）。
- **修复建议**: `ScalpelStitcher` 全程在无裁剪线性 ACEScg 中做金字塔混合、`x1/y1` 减 `ring` 并在边界环带执行 `-fl` 后向光流 EMA；`HandRefiner`/`FaceRefiner` 在缺真实神经修复模型权重时按 `exit 50` 或显式启发模式隔离，并用左右瞳孔暗质心相对眼框中心重写视线估计。

#### `[H7-P0-10]` `_build_r3d18_extractor` 对一切 `T != 3` 视频必崩，`temporal_warp_error` 与 `flicker_score` `uint8` 误差放大 `255× / 65,025×` (`PR-101`)
- **精确位置**: [python/synapse/metrics/temporal.py](../../../python/synapse/metrics/temporal.py#L67-L140,L181-L194,L303-L385)
- **实证与根因**:
  1. `_build_r3d18_extractor` (`L375-L376`) 先 `.permute(3, 0, 1, 2)` 得到 `[C, T, H, W]`，再传入 `torchvision.transforms.Normalize`（按约定规范化倒数第 3 维，即 **`T` 维**），导致所有 `T != 3` 的正常视频 100% 抛出 `RuntimeError`（单测 Mock 掉了提取器掩盖崩溃），同时触发联网 PyTorch Hub `URLError` 且未将 `uint8` 除 `255.0`；
  2. `temporal_warp_error` (`L83, L89-L99`) 与 `flicker_score` (`L189`) 直接将 `uint8` 强转 `float64` 而未除 `255.0`，使标准 `[0, 1]` 门禁 `0.02` / `0.005` 下出现 **`255×` 与 `65,025×` 量纲爆炸**，且使用错误光流方向 `+mv` 与最近邻取整；`fvd_feature_distance` 在 `N=1` 时 `NaN`，在 `N < D` 时非对称 `sqrtm` 产生复数与偏差。
- **修复建议**: 显式沿通道维 `dim=0` 归一化并先除 `255.0`、捕获 `(ImportError, OSError)` 并实测 `T=16` 帧真实推理；在 `temporal_warp_error`/`flicker_score` 统一 `[0,1]` 浮点与反向双线性 Warp；改用对称特征分解 `eigh` 计算 `fvd_feature_distance` 迹。

#### `[H7-P0-11]` `tools/data-inspector/native/inspector_app.cc` 在 `-fno-exceptions` 下调用 `.get<T>()`/`.value()` 引发 `std::abort()` (`PR-103`)
- **精确位置**: [tools/data-inspector/native/inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc#L245-L247,L696,L702-L704,L720,L737,L740-L741,L745)；[tools/data-inspector/native/.bazelrc](../../../tools/data-inspector/native/.bazelrc#L20-L22)
- **实证与根因**: Native 二进制全量以 `-fno-exceptions` 编译，而 `inspector_app.cc` 多处解析外部 `bundle.json` 与 `stdin` JSON-RPC 指令时直接调用 `nlohmann::json::get<std::string>()` 或 `j.value("key", default)`。`nlohmann::json::value()` 在键存在但 JSON 类型不匹配（如 `null`、数组或字符串代替数值）时仍会触发 `JSON_THROW(type_error)`，在无异常编译期直接触发 **`std::abort()` (`SIGABRT`) 进程崩溃**。
- **修复建议**: 实现基于 `j.find(key)` + `is_string()` / `is_number()` / `is_boolean()` 的安全只读解析 helper，彻底清零 `-fno-exceptions` 编译单元内对未校验 `.get<T>()`/`.value()` 的调用。

#### `[H7-P0-12]` Electron `shm-reader.js` 在 macOS/Windows 上 100% 失效且 Seqlock 重试循环必然全败 (`PR-104`)
- **精确位置**: [tools/data-inspector/electron/shm-reader.js](../../../tools/data-inspector/electron/shm-reader.js#L70-L86,L151-L155,L181-L215)（对照 `../aether/pal/src/posix/posix_shared_memory.cc#L110` 与 `../aether/pal/src/windows/windows_shared_memory.cc#L113`）
- **实证与根因**:
  1. `@aether//pal:shm` 在 macOS 上调用 `shm_open()`（XNU 内核 IPC 对象，不在文件系统也不在 `os.tmpdir()`），在 Windows 上调用 `CreateFileMappingA(..., "Local\\...")`（Win32 内核对象管理器命名空间，也不在 `%TEMP%`）；`shm-reader.js:76-85` 探测 `/dev/shm` 失败后回退 `os.tmpdir()`，在 macOS/Windows 上 `resolveShmFilePath()` 恒返 `null`；
  2. Seqlock 重试循环把 `parseShmHeader` 与 `pixelOffset` 放在 `for (attempt ...)` 外部：当第 0 次遇到帧序号递增时，`this._headerBuf2.copy(this._headerBuf)` 只拷缓冲区而不重新解析 `header.frameReady` 或更新 `pixelOffset`，导致后续 `1..3` 次重试永远拿旧值比新值并 100% 判失败！
- **修复建议**: 为 macOS/Windows 提供基于 `@aether//pal:shm` 的原生 Node N-API 轻绑定（或显式文件映射 `--shm-file` 契约）；将 `parseShmHeader` 与 `pixelOffset` 迁入 Seqlock 重试循环内部并配合 C++ 侧奇/偶写计数内存序。

---

### 4.2 P1 严重功能 / 安全 / 跨平台 / 门禁缺陷（`H7-P1-01 ~ H7-P1-16`）

| 审计编号 | 影响模块与代码定位 | 根因实证描述 | 目标 PR |
| :--- | :--- | :--- | :---: |
| **`H7-P1-01`** | [tests/conftest.py](../../../tests/conftest.py#L19-L55,L62-L106) | `pytest_runtest_makereport`（非 `hookwrapper`）在严格依赖触发跳过时直接调 `pytest.fail()` 引发 `INTERNALERROR`，且仅查 `call` 阶段漏检 `@pytest.mark.skipif` 常见 `setup` 阶段；`_enforce_strict_deps` 为空转 `autouse`；3 个 `cached_rasterizer_*` 夹具全局零引用且导入已弃用 `generators.rasterizer` | `PR-92` |
| **`H7-P1-02`** | [fgpp/io.py](../../../python/synapse/fgpp/io.py#L104-L178,L314-L373,L448-L466)；[fgpp/torch_bridge.py](../../../python/synapse/fgpp/torch_bridge.py#L49,L107-L115)；[fgpp/schema.py](../../../python/synapse/fgpp/schema.py#L135-L232) | `sanitize_fgpp_frame` 在计算 `norms` 时对 `NaN/Inf` 法线判 `zero_mask=False` 并直接漏出 `NaN` 法线、`direct_radiance`/`base_color`/`roughness_metallic` 未去 `NaN` 限域；`resize_fgpp_frame` 漏乘 `(sx, sy)` 缩放像素光流；`fgpp/schema.py` 边界 `z_near<=0` 及 `interval=0` (`ZeroDivisionError`) 偏离 `fgpp_types.h`；`OpenEXR`/`torch` 缺省包抛裸 `ModuleNotFoundError` 且句柄缺 `finally: close()`、`tensors_to_frame` 对带梯度/非连续张量必崩 | `PR-93` |
| **`H7-P1-03`** | [tools/import_dag_guard.py](../../../tools/import_dag_guard.py#L81-L225,L357-L385)；[tools/wire_guard.py](../../../tools/wire_guard.py#L170-L436,L625-L706)；[schemas/inspection.v1.0.schema.json](../../../schemas/inspection.v1.0.schema.json#L3-L15) | `import_dag_guard` 豁免函数域/`__getattr__`/f-string/`import_module`/属性路径，放行 `bridge(T0)->recon(T2)` 与 `inspect(T2)->bench(T3)`；`wire_guard` 缺 C++ 头文件静默返 `0`、`--check` 参数冗余、放行 `@aether//style` 缺失、`inspection.v1.0.schema.json` 实为 `2.0.0` 副本、`bundle.v1.0` 正则放行 `".."` 与 Windows 反斜杠且 Pydantic 模型缺约束对齐 | `PR-93` |
| **`H7-P1-04`** | [fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L71-L88)；[render_core/rasterizer.py](../../../python/synapse/render_core/rasterizer.py#L182-L254,L524-L608)；[render_core/procedural.py](../../../python/synapse/render_core/procedural.py#L91-L305,L663-L775)；[render_core/pyramid.py](../../../python/synapse/render_core/pyramid.py#L85-L162)；[render_core/morphology.py](../../../python/synapse/render_core/morphology.py#L48-L83) | `projection_matrix()` 与 `_build_projection()` 忽略 `(fx,fy,cx,cy)`、`prev_w < 0` 反向投影万级伪光流、无 `prev_vertices` 致变形网格零光流、`eid=0` 未规避且 `z > z_far` 写入前景标签；`generate_humanoid_proxy` 四肢绕圆柱中点旋转致关节脱臼且基元未焊接非水密；纯 NumPy `separable_gaussian` 遇 `uint8` 崩 `UFuncTypeError`；`binary_erosion` 边界行为与 `scipy` 完全相反且缺 SE 翻转 | `PR-94` |
| **`H7-P1-05`** | [composite/detector.py](../../../python/synapse/composite/detector.py#L119-L202)；[composite/blend.py](../../../python/synapse/composite/blend.py#L393-L604,L706-L738)；[style/extractor.py](../../../python/synapse/style/extractor.py#L123-L288)；[style/aces_color.py](../../../python/synapse/style/aces_color.py#L80-L105,L274-L280) | `_simple_warp` 光流反向 + `±Inf` 强转 `int64` 溢出；`blend.py` 快速运动 (`>15px`) 反降当前帧权重加剧重影、`float16` `mv**2` 上溢、`_cryptomatte_boundary_guide` 哈希最大值实体对比度归零且金字塔混合在 Gamma 空间产生暗边；`style/extractor.py` 漏算 `contrast`/`saturation`/LUT、在 `n·L<0` 背光面线性拟合偏差光向且用坐标中位数抵消反向光源；`aces_color` 变换矩阵行和 `1.000087 != 1.0` 与 `@aether//color` 偏移 | `PR-95` |
| **`H7-P1-06`** | [bridge/vg_bridge.py](../../../python/synapse/bridge/vg_bridge.py#L50-L130)；[recon/backend.py](../../../python/synapse/recon/backend.py#L44-L64)；[bridge/foundary_client.py](../../../python/synapse/bridge/foundary_client.py#L113-L149)；[bridge/hitl.py](../../../python/synapse/bridge/hitl.py#L25-L99)；[adapters/wan.py](../../../python/synapse/adapters/wan.py#L100-L229)；[adapters/cloud.py](../../../python/synapse/adapters/cloud.py#L113-L228,L547-L689) | `SynapseReconBackend.check_manifold` 跳过 `_c_geometry` 走非生产 Python Oracle 且不检水密；`enforce_distillation_geometry_gate` 零调用且 `"realtime"` 字符串绕过；`HttpFoundaryTransport` 恒抛未配置且与 `WanAdapter.FoundaryHttpTransport` 分裂；`FoundaryHttpTransport` Base64 JSON 膨胀 `>7GB`、探活 `GET /v1/generate` 得 `405` 永假、W3C `traceparent` 丢失环境变量与子 Span；`CloudAdapter` 有环境变量仍无真传输且世界系判相机平推方向；`HITL` 跨进程无持久账本可无限重置避开 900s 熔断 | `PR-97` |
| **`H7-P1-07`** | [training/feature_store.py](../../../python/synapse/training/feature_store.py#L309-L629) | 入库 Gate 5 校验漏洞导致 `TestIngestionGates::test_ingestion_gate_rejects_bad_rotation` 失败；`_serialize_sample` 未清洗 `sample_id` 路径遍历；`iter_feature_shards` 用 `name.find(".")` 首点分割致 `"wan2.1_shot01"` 等合法 ID 100% 静默丢弃；`T<=7` 短序列错当 15 通道原帧；读全 `2 GiB` 分片驻留内存非真流式；零样本仍生成空 `shard-00000.tar` | `PR-98` |
| **`H7-P1-08`** | [recon/delight.py](../../../python/synapse/recon/delight.py#L370-L455,L550-L793)；[recon/tracking.py](../../../python/synapse/recon/tracking.py#L207-L592)；[recon/kinematics.py](../../../python/synapse/recon/kinematics.py#L194-L242)；[recon/\_\_init\_\_.py](../../../python/synapse/recon/__init__.py#L54)；[recon/loop/forward_guide.py](../../../python/synapse/recon/loop/forward_guide.py#L119-L210)；[recon/loop/mitsuba_optimizer.py](../../../python/synapse/recon/loop/mitsuba_optimizer.py#L147-L252) | `delight` 在屏幕坐标混叠多视角并平均各相机坐标系 SH、亮度自抵消、`delta_e` 用 CIE76+双重 Gamma 且未接入 `ReconPipeline`；`tracking.py` 整数步取整 3× 偏差、丢弃遮挡可见性、无 RANSAC/PnP、假 `reprojection_error_px`；原点相机夹角恒 `90°` 漏放；`kinematics.py` 原地修改脚踝坐标放大 `raw_slips` 并提前计算未钳制 `clamped_slips`；`MultiViewInitializer` 导出 `AttributeError`；`forward_guide` `[T,H,W]` 判型错位 | `PR-99` |
| **`H7-P1-09`** | [refine/fusion.py](../../../python/synapse/refine/fusion.py#L123-L289)；[refine/pipeline.py](../../../python/synapse/refine/pipeline.py#L192-L351)；[refine/tiled_executor.py](../../../python/synapse/refine/tiled_executor.py#L42-L99) | `LATENT_TIMESTEP_SWITCH` `w_struct+w_detail=1` 吞掉 `50–80%` 曝光、`POISSON_GRADIENT` 散度 1px 偏移、`UNCERTAINTY_WEIGHTED` 算单像素 RGB 方差压暗彩色五官、`CASCADE` 级联双线性模糊；`LocalRefinementPipeline` 漏调 `KalmanBBoxTracker.predict()` 且未传 `smoothed_bbox` 或 `strict=True`、跳过时序门禁；`TiledBypassExecutor._tile_size` 闲置 | `PR-100` |
| **`H7-P1-10`** | [metrics/image_quality.py](../../../python/synapse/metrics/image_quality.py#L309-L380)；[metrics/roundtrip.py](../../../python/synapse/metrics/roundtrip.py#L218-L305,L419-L431)；[metrics/profiles.py](../../../python/synapse/metrics/profiles.py#L23-L128)；[metrics/fgpp_fidelity.py](../../../python/synapse/metrics/fgpp_fidelity.py#L85-L128) | 掩膜 `lpips` 在 `1/2`、`1/4` 尺度仍除以全图 `valid_count` 并对 60% 权重的 `grad_dist` 无掩膜稀释；`chamfer_mm`/`f_score_*` 复用相同 `seed=42` 掩盖离散方差；`_is_mesh_watertight` 无视顶点的捏合非流形；`foot_sliding_cm` 用 XY (Z-up) 与 `kinematics` XZ (Y-up) 正交冲突；`TRACK_A_ASSET` 遗漏 `f_score_5mm`/`watertight_pct`/`non_manifold_edges`；`fgpp_fidelity` 开方缺 `max(...,0)` | `PR-101` |
| **`H7-P1-11`** | [scripts/build_native.py](../../../scripts/build_native.py#L199-L429)；`csrc/geometry/MODULE.bazel`；`csrc/geometry/.bazelrc` | `build_native.py` macOS 缺 `-Wl,-undefined,dynamic_lookup`、Win 缺 `python3XX.lib`、`args.backend == "bazel"` 失败仍偷偷回退、`--output-dir` 时冒烟测试错测 `repo_root/python`、临时产物遗留源码目录、异常时输出两份 JSON；`csrc/geometry` `.bazelrc` 触发 `[opt]` 双重扩展告警且缺乏独立 `--enable_platform_specific_config` | `PR-102` |
| **`H7-P1-12`** | [tools/data-inspector/native/inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc#L103-L605,L648-L735)；[inspector_shm_server.cc](../../../tools/data-inspector/native/inspector_shm_server.cc#L13-L111) | 仅加载 `.obj` 忽略 `.glb`/`.exr`（Fixture `.glb` 全退回 8 顶点方块，重载不复位旧模型）；C++ `set_debug_view` 拒收 `albedo`/`motion`；SHM 写先改 `write_idx` 再改 `frame_ready` 且缺 placement-new 与 `is_always_lock_free` 静态契约；60Hz 分配 `3.68 MB` `zBuffer` 堆；无边界路径读取与任意路径覆写；`std::isfinite` 缺失引发 `NaN->int` UB；二次 `init()` 触发 `std::terminate()` | `PR-103` |
| **`H7-P1-13`** | [tools/data-inspector/electron/main.js](../../../tools/data-inspector/electron/main.js#L50-L52,L129-L175)；[native-process.js](../../../tools/data-inspector/electron/native-process.js#L34-L221)；[bundle-manager.js](../../../tools/data-inspector/electron/bundle-manager.js#L76-L90)；[app.js](../../../tools/data-inspector/renderer/src/app.js#L218-L330)；[source-model.js](../../../tools/data-inspector/renderer/src/views/source-model.js#L35-L49) | `main.js` 依然每帧跨进程拷贝完整 RGBA 缓冲（720p `3.68MB` / Retina `14.7MB`）；`will-navigate` 放行任意 `file://`；`export_screenshot` 允许任意路径覆写；`BundleManager` 缺 `fs.realpathSync` 符号链接越界防守；窗口缩放重建 `imageData` 但未通知原生重开 SHM 致永久冻屏；每次点击调试按钮发双份 IPC 且属性面板顶点/面片数恒为 `"—"` | `PR-104` |
| **`H7-P1-14`** | [pipeline/bench_runner.py](../../../python/synapse/pipeline/bench_runner.py#L252-L630)；[inspect/model_evaluator.py](../../../python/synapse/inspect/model_evaluator.py#L59-L503)；[inspect/eval_report.py](../../../python/synapse/inspect/eval_report.py#L74-L128)；[inspect/align.py](../../../python/synapse/inspect/align.py#L273-L320)；[inspect/macro.py](../../../python/synapse/inspect/macro.py#L309-L365)；[rsi/promotion_gate.py](../../../python/synapse/rsi/promotion_gate.py#L136-L283)；[bench/scenarios/](../../../python/synapse/bench/scenarios/) | `BenchPipelineRunner` 仅算 2–4/16 项指标却按全量 16 项判门禁 100% 失败、`export-bundle` 缺 `source.glb`、重评篡改哈希 `comparison.json`、T2/T3 静态空跑；5 份旧 `data/scenarios/` 违背 `4k+1`；`synapse inspect evaluate` 无 `--reference-dir` 必退 `40`、stem 改名致 `TestEvalScript::test_smoke_run` 失败、`render_orbit_views` 缺线性到 sRGB 压暗并双倍 Gamma、`eval_report` 雷达图反转 9 个越低越好指标；`align.py` 锚定原点非质心 + `B023`、`macro.py` 忽略 `occlusion_depth_map`；`rsi gate-promote` 复制同一场景自减且不强制 30 场景 | `PR-105` |
| **`H7-P1-15`** | [scripts/colab_runner.py](../../../scripts/colab_runner.py#L235-L432)；[scripts/prepare_benchmark_assets.py](../../../scripts/prepare_benchmark_assets.py#L371-L429)；[assets/assets_manifest.json](../../../assets/assets_manifest.json#L12-L80)；[workenv/google-colab/setup_colab.sh](../../../workenv/google-colab/setup_colab.sh#L38-L111) | `colab_runner.py --mode=real-gpu` 实为空壳 CPU Mock 零权重落盘、`--checkpoint-dir` 孤立创建多 job、`SIGTERM`/异常丢失 `job.json` 阶段状态；`setup_colab.sh` 向 Google Drive FUSE 首选写入大量零碎小基准文件违反 SSOT V2.0，且缺 `../aether`/`pybind11` 使 `_c_geometry` 在 Colab 编译失败；`prepare_benchmark_assets.py` 在 `target_root/gso_starter/gso_starter/` 双重嵌套下载导致 SHA-256 校验 100% 跳过，`assets_manifest.json` 所有 `.tar.gz` 误填单文件 `.obj` SHA-256 | `PR-106` |
| **`H7-P1-16`** | [tests/test_roundtrip_reconstruction.py](../../../tests/test_roundtrip_reconstruction.py#L219-L270)；[tests/test_architecture_dag.py](../../../tests/test_architecture_dag.py#L36-L81)；[tests/test_fgpp_io.py](../../../tests/test_fgpp_io.py#L336)；[tests/test_bench_full_suite.py](../../../tests/test_bench_full_suite.py); [tests/test_bench_runner.py](../../../tests/test_bench_runner.py) | `test_roundtrip_reconstruction.py` (`H6-P1-10`) 依然只是 `gt_surface + 2mm` 假噪声与常量均色图，零调用 `ReconPipeline`；`PR-90` 将 `4` 个失败用例与 `<200ms` 架构/DLPack 单测错误打上 `@pytest.mark.slow` 掩耳盗铃；3 处全量 30 场景渲染未共用会话缓存夹具，叠加慢光栅拖垮整套 `-m slow` | `PR-107` |

---

### 4.3 P2 工程规范 / 数学精度 / 文档债务（`H7-P2-01 ~ H7-P2-10`）

1. **`[H7-P2-01]` [pyproject.toml](../../../pyproject.toml#L78,L93-L104) Ruff 规则屏蔽与并行 `--basetemp` 互删 (`PR-92`)**：屏蔽 29 条高价值缺陷检查规则（导致隔离检查多出 327 处问题），且默认 `addopts` 中 `--basetemp=./.pytest_tmp` 在同时运行快测与单点冒烟时互相清空活动临时目录。
2. **`[H7-P2-02]` [render_core/conditions.py](../../../python/synapse/render_core/conditions.py#L399,L426-L433)、[camera.py](../../../python/synapse/render_core/camera.py#L55-L103) 与 [telemetry.py](../../../python/synapse/telemetry.py#L60-L206) 默认契约矛盾 (`PR-94`)**：`get_shutter_angle_deg()` 在 `None` 时返回 `180°` 矛盾 `resolve()` 的 `L1=0°`；`camera.py` 使用 `assert`（`python -O` 被剥）与 `B008` 可变默认数组；`TraceContext` 接受 `"a-b-c-d"` 非法串；`MemoryForecaster` 缺层数 `40` 且 `threshold=24GiB` 使 `14B` 永不可达 `L0`。
3. **`[H7-P2-03]` 重复纯 Python $L_1$ 距离变换与 [style/descriptor.py](../../../python/synapse/style/descriptor.py#L15-L32)/[profile_mapper.py](../../../python/synapse/style/profile_mapper.py#L48-L90) 边界 (`PR-95`)**：`mask.py` 与 `detector.py` 复制 30 行双层循环 $L_1$ 距离（`41%` 对角误差、`~4.1M` 次 1080p 迭代），应收敛至 `morphology.py` 1D 可分离精确欧氏 EDT；`StyleDescriptor` 缺 `extra="forbid"` 与 `ambient_ratio ∈ [0,1]` 校验，`to_engine_light_overrides` 允许负灯光强度，`save_style_profile_yaml` 擅自改后缀为 `.json`。
4. **`[H7-P2-04]` [viewer/static_html.py](../../../python/synapse/viewer/static_html.py#L329-L583) & [viewer/server.py](../../../python/synapse/viewer/server.py#L48-L50) URL 解码与前端空值防御 (`PR-96`)**：未做 `urllib.parse.unquote` 致含空格/%20 路径 404；前端在 `camera_position` 缺失或 `frames.length === 0` 时除 0 与抛 `TypeError`；独立导出 HTML 留失效相对 MP4 链接。
5. **`[H7-P2-05]` [adapters/base.py](../../../python/synapse/adapters/base.py#L206-L231)、[models/asset.py](../../../python/synapse/models/asset.py#L65-L133)、[models/loader.py](../../../python/synapse/models/loader.py#L14-L236) 与 [generators/dataset.py](../../../python/synapse/generators/dataset.py#L182-L305) (`PR-97`)**：`register_adapter` 在导入期实例化全部适配器读环境变量；`ModelAsset.export_obj` 在法线为空时仍写不存在的 `vn` 索引（`f v//vn`）破坏 `.obj` 语法，`normalize()` 在小半径时别名返回自身；`loader.py` `Path("") == Path(".")` 劫持当前目录、程序化茶壶退化为无壶嘴无把手的球体；`DatasetGenerator` 仅写 `.npz` 不出 EXR、线性直截 sRGB、漏查 `VideoWriter.isOpened()` 且在线流式更新 `max_flow_val` 致帧间色度尺度飘移。
6. **`[H7-P2-06]` `csrc/geometry/src/bindings.cc` & `csrc/geometry/tests/test_geometry.cc` (`PR-102`)**：`static_cast<int>(buf.shape[1])` 在范围校验前截断 64 位 `py::ssize_t`；`verts_per_face`/`default_vpf` 等 C++ 变量保留 `snake_case`；C++ 单测使用手写 `CHECK_COND` 宏而非 GoogleTest。
7. **`[H7-P2-07]` 残余假测试与盲目异常捕获 [tests/test_dataset_and_viewer.py](../../../tests/test_dataset_and_viewer.py#L229-L233) & [tests/test_inspection_bundle.py](../../../tests/test_inspection_bundle.py#L136-L175) (`PR-107`)**：`dummy_server.allow_reuse_address = True` 下一行断言自身为 `True` 的恒真测试；多处 `with pytest.raises(Exception):` 吞掉低级 `TypeError`/`NameError`。
8. **`[H7-P2-08]` [docs/SYN-RFC-0060..0063](../../../docs/SYN-RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md#L1-L6) 命名碰撞、幻觉文件与 [AUDIT-RFC0063 §5](AUDIT-RFC0063-AND-CODEBASE-2026-11.md#L215-L238) 冲突横幅 (`PR-108`)**：`RFC-0061..0063` 仅加了别名行，文件名与一级标题仍与 `all-overview/rfcs/0061..0063` 撞号，需统一重命名为 `SYN-RFC-0060..0063-*.md`；修正 `RFC-0060:81`（`test_recon_roundtrip.py` → `test_roundtrip_reconstruction.py`）与 `RFC-0061:104`（标记不存在的 `hair_groom.py`/`body_garment.py` 为未来 Non-Goal）；为 `AUDIT-RFC0063 §5` 增加后续审计废止说明。
9. **`[H7-P2-09]` [.agents/skills/synapse-overview/SKILL.md](../../../.agents/skills/synapse-overview/SKILL.md#L43)、[synapse-lora-and-flywheel/SKILL.md](../../../.agents/skills/synapse-lora-and-flywheel/SKILL.md#L78) 与 [AGENTS.md](../../../AGENTS.md#L15-L17) 符号失实 (`PR-108`)**：`synapse-overview` 列出 `synapse.rsi` 四个全不存在的文件（`flywheel, reward, replay_buffer, pareto`），实际文件为 `curriculum, dual_verifier, pair_builder, promotion_gate`；`synapse-lora-and-flywheel` 引用不存在的 `ReplayBuffer`；`AGENTS.md` 与 `csrc/geometry/BUILD.bazel` 之间关于 `@aether//style` 与“全仓自包含”表述须完全对齐。
10. **`[H7-P2-10]` Horizon-7 全仓 10 项终验封板与 GitHub Actions PR 绿灯实证留档 (`PR-109`)**：仅在 10 项全量命令与 GitHub Actions PR 真实全绿后方可将总/子计划更新为 100% 封板。

---

### 4.4 P3 轻量收口与工程卫生（`H7-P3-01 ~ H7-P3-04`）

1. **`[H7-P3-01]` [.gitignore](../../../.gitignore#L7-L66) 与外置遥测说明 (`PR-92, PR-105`)**：补充 macOS 动态库 `*.dylib` 忽略项、去重 `data/datasets/` 与 `node_modules/`；澄清 `.gitignore:65` `# External AI agent/IDE harness session telemetry`（已证实磁盘 `data/sessions/` 为外部会话遥测而非 `synapse` 源码或单测写出）。
2. **`[H7-P3-02]` [composite/detector.py](../../../python/synapse/composite/detector.py#L60-L117) 浮点输入误降与死函数清理 (`PR-95`)**：`_to_grayscale` 无条件除以 `255.0` 令 `[0,1]` 浮点图缩水 255 倍，`_block_mean` 静默切掉非 16 倍数边条，删除死函数 `_block_variance`。
3. **`[H7-P3-03]` `csrc/geometry/include/synapse/geometry/types.h` 与 [fgpp/schema.py](../../../python/synapse/fgpp/schema.py#L53) 卫生清理 (`PR-93, PR-102`)**：移除零引用的 `struct AABB` 与多余标准库头文件；修正 `schema.py:53` `# 10×HALF` → `# 13×HALF(26B) + 1×FLOAT(4B) + 1×UINT32(4B) = 34B`。
4. **`[H7-P3-04]` [tools/data-inspector/native/inspector_shm_server.h](../../../tools/data-inspector/native/inspector_shm_server.h#L42-L48) `InspectorShmHeader` 字段命名统一 (`PR-103`)**：将 `write_idx`, `buffer_count`, `frame_ready`, `delta_time`, `debug_view` 等成员统一规范为 `camelCase` 并做 `offsetof` 编译期断言。

---

## 5. 缺陷 → Horizon-7 (`M25–M28`, `PR-92 ~ PR-109`) 闭环映射总表

| Horizon-7 PR | 里程碑 / 波次 | 主责包域（文件正交隔离） | 闭环修复的审计缺陷 ID |
| :--- | :---: | :--- | :--- |
| **[PR-92](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-92)** | M25 / W24 | `.github/workflows/ci.yml` · `pyproject.toml` · `tests/conftest.py` · `.gitignore` | `H7-P0-01`, `H7-P1-01`, `H7-P2-01`, `H7-P3-01` |
| **[PR-93](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-93)** | M25 / W24 | `fgpp/` · `tools/wire_guard.py` · `tools/import_dag_guard.py` · `schemas/` | `H7-P0-02`, `H7-P1-02`, `H7-P1-03`, `H7-P3-03` |
| **[PR-94](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-94)** | M25 / W24 | `render_core/` · `errors.py` · `telemetry.py` | `H7-P0-03`, `H7-P1-04`, `H7-P2-02` |
| **[PR-95](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-95)** | M25 / W24 | `style/` · `composite/` | `H7-P0-04`, `H7-P1-05`, `H7-P2-03`, `H7-P3-02` |
| **[PR-96](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-96)** | M25 / W24 | `viewer/` · `cli/` · `pipeline/bundle_worker.py` · `pipeline/stages/` | `H7-P0-05`, `H7-P2-04` |
| **[PR-97](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-97)** | M26 / W25 | `adapters/` · `bridge/` · `models/` · `generators/` | `H7-P0-06`, `H7-P1-06`, `H7-P2-05` |
| **[PR-98](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-98)** | M26 / W25 | `training/` (`packager`, `losses`, `trainer`, `feature_store`) | `H7-P0-07`, `H7-P1-07` |
| **[PR-99](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-99)** | M26 / W25 | `recon/` & `recon/loop/` | `H7-P0-08`, `H7-P1-08` |
| **[PR-100](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-100)** | M26 / W25 | `refine/` (`stitcher`, `hand`, `face`, `fusion`, `pipeline`, `tiled_executor`) | `H7-P0-09`, `H7-P1-09` |
| **[PR-101](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-101)** | M26 / W25 | `metrics/` (`temporal`, `image_quality`, `roundtrip`, `profiles`, `fgpp_fidelity`) | `H7-P0-10`, `H7-P1-10` |
| **[PR-102](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-102)** | M27 / W26 | `csrc/geometry/` · `scripts/build_native.py` | `H7-P1-11`, `H7-P2-06`, `H7-P3-03` |
| **[PR-103](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-103)** | M27 / W26 | `tools/data-inspector/native/` | `H7-P0-11`, `H7-P1-12`, `H7-P3-04` |
| **[PR-104](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-104)** | M27 / W26 | `tools/data-inspector/{electron,renderer,test}` · `package-lock.json` | `H7-P0-12`, `H7-P1-13` |
| **[PR-105](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-105)** | M27 / W26 | `bench/` · `pipeline/bench_runner.py` · `inspect/` · `rsi/` · `data/scenarios/` · `scripts/evaluate_3d_models.py` | `H7-P1-14`, `H7-P3-01` |
| **[PR-106](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-106)** | M27 / W26 | `scripts/{colab_runner,prepare_*,run_*}.py` · `workenv/google-colab/` · `assets/` | `H7-P1-15` |
| **[PR-107](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-107)** | M28 / W27 | `tests/`（去伪 · 清零 `Ruff` 启用规则 · `-m slow` 全绿与加速） | `H7-P1-16`, `H7-P2-07` |
| **[PR-108](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-108)** | M28 / W27 | `docs/SYN-RFC-0060..0063` · `docs/audits/` · `.agents/skills/` · `AGENTS.md` | `H7-P2-08`, `H7-P2-09` |
| **[PR-109](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#pr-109)** | M28 / W27 | Horizon-7 全仓 10 项终验门禁 & GitHub Actions PR 全绿封板 | `H7-P2-10`（全域封板） |
