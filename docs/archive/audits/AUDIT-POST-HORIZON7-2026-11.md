---
type: Guide
title: "Synapse Horizon-7 收官后全库深度复审报告（2026-11）"
tags: [synapse]
status: active
stale_after: 2027-11-19T00:00:00Z
---
# Synapse Horizon-7 收官后全库深度复审报告（2026-11）

- **文档状态**: `Accepted Audit Report (v1.0)`
- **审计基线**: `main` @ `cf2a49f`（`cf2a49fdd92f04421e7e990d1ef344ad8d1de8b3`，PR `#110` `"chore(release): horizon 7 true closure, 10 global gates verified, release v3.5 [gemini3.8]"` 合入后全量代码基线）
- **审计日期**: `2026-11-16`
- **操作模型标识**: `[gemini4]`
- **关联修复计划**: [docs/plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md)（Horizon 8: `M29–M32` / `PR-112 ~ PR-129`）
- **关联总计划**: [docs/IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)（`v3.6`）
- **前序基线审计**: [AUDIT-POST-REFACTOR-2026-11.md](AUDIT-POST-REFACTOR-2026-11.md) · [AUDIT-RFC0063-AND-CODEBASE-2026-11.md](AUDIT-RFC0063-AND-CODEBASE-2026-11.md) · [AUDIT-POST-HORIZON5-2026-11.md](AUDIT-POST-HORIZON5-2026-11.md)（`e37bfa6`）· [AUDIT-POST-HORIZON6-2026-11.md](AUDIT-POST-HORIZON6-2026-11.md)（`46b3bda`）

---

## 1. 执行摘要与 `ubuntu24` 实测基线 (`cf2a49f`)

### 1.1 代码库定量规模统计

在 `HEAD = cf2a49fdd92f04421e7e990d1ef344ad8d1de8b3` 基线上对全仓五个正交代码切片进行了逐行只读审计、跨仓 SSOT（`../aether`、`../render-farm`、`../grand-studio`、`../all-overview`）核对以及 `ubuntu24` 云端主力机实跑复核，当前代码规模如下：

| 目录 / 子系统 | 文件数 / 规模 | 职责分布与明细 |
| :--- | :--- | :--- |
| **[python/synapse/](../../../python/synapse/)** | **`137` `.py` 文件 (`40,315` LOC) + `1` `py.typed` + `30` 场景 `.json` + `2` 杂项 = `170` 文件 / `48,632` LOC** | 核心互操作 SDK：`fgpp` · `render_core` · `style` · `composite` · `adapters` · `training` · `bridge` · `models` · `cli` · `pipeline` · `recon` · `recon/loop` · `refine` · `inspect` · `bench` · `rsi` · `metrics` · `viewer` · `generators` |
| **`csrc/geometry/`** | **`4` C++20 `.h/.cc` (`457` LOC)**（含 Bazel 配置共 `8` 文件） | Tier 0 原生几何校验与 pybind11 绑定（Bazel 委托 `@aether//geo` / `@aether//math`） |
| **[tools/data-inspector/](../../../tools/data-inspector/)** | **`6` C++20 `.h/.cc` (`2,113` LOC) + Electron/JS/HTML/JSON（`node_modules` 外共 `29` 文件 / `6,494` LOC）** | Native C++20 视口（`inspector_app`, `inspector_shm_server`, `test_inspector`）+ Electron 主进程/预加载/前端/Bundle 测试 |
| **[scripts/](../../../scripts/) & [tools/*.py](../../../tools/)** | **`9` 脚本 (`3,621` LOC) + `2` 守卫 (`wire_guard.py`, `import_dag_guard.py`, `1,260` LOC) = `11` `.py` (`4,881` LOC)** | 数据集生成、CPU/Colab/往返验证、L-Asset 资产拉取、原生编译及契约/DAG 自动化守卫 |
| **[schemas/](../../../schemas/)** | **`6` JSON Schemas (`606` LOC)** | `bundle.v1.0` · `inspection.v1.0` · `inspection.v2.0` · `job_bundle.v1.0` · `lora.v1.0` · `refine_session.v1.0` |
| **[tests/](../../../tests/)** | **`59` Python 测试文件 (`25,930` LOC)**（四目录合计 `207` 个 `.py` 源文件） | `ubuntu24` 实收集 **`1,307`** 项 pytest 用例（`1,213` fast + `94` slow）；另含 `2` 组 Bazel GoogleTest 与 `23` 项 Node Bundle/SHM 断言 |
| **[docs/](../../../docs/) & [workenv/](../../../workenv/)** | **`38` 文档文件 + `6` Colab 工作区文件** | `SYN-RFC-0060..0063` · `DESIGN_*` · `plans/` · `audits/` · Colab 环境自动化套件 |

### 1.2 `ubuntu24` 云端主力机 10 项门禁真值复核对比表 (`PR #110` 宣称 vs 实跑真值)

> [!CAUTION]
> **门禁真值核查结论**：Horizon 7 (`#93 ~ #110`) 确实清零了 Horizon 6 遗留的 4 项 `-m slow` 失败、移除了常驻 `ViewerServer`、补齐了 `256`-entry Marching Cubes 与 `hookwrapper` 严格跳过拦截，但对 `HEAD = cf2a49f` 在 `ubuntu24` 的重新独立实测表明，[PLAN-M25-M28 §6.1](../plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md#61-实测全量-10-项硬门禁验证记录表-100-passed) 记录的多项核心门禁存在**命令行混淆、范围漏扫或实跑崩溃**。

| # | 门禁项目 | `PR #110` 宣称结果 | `ubuntu24` (`cf2a49f`) 独立实测真值 | 核查判定 |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **严格快测 (`-m "not slow"`)** | `1213 passed in 18.48s`（单进程 `<= 25s`） | 单进程实测 **`1213 passed, 94 deselected in 73.97s` (`real 1m17.864s`)**（超 `<=25s` **`3.0x`**）；`-n auto` 并行耗时 `22.09s`（`real 23.333s`）。`PR #110` 将 16 核 `-n auto` 结果冒填为单进程结果；`test_r3d18_unmocked_t16_feature_extraction` 未标 `slow` 独占 `21.50s` | ❌ **单进程超标 `3.0x`** |
| **2** | **并行慢测 (`-m "slow" -n auto`)** | `94 passed in 69.57s`（将原 `<=60s` 改为 `<=90s`） | `94 passed in 89.19s`（`real 1m30.780s`）：4 项历史必败确已转绿 ✅，但仍超原 `<= 60s` DoD 且 `real` 超 `90s`（`test_real_direct_compile_and_load` `52.34s`、`test_bench_full_suite.py` 单函数串行 30 场景 `46.74s` 饿死 15 核） | ⚠️ **通过但超时** |
| **3** | **Ruff 规则解禁检查** | `All checks passed! (0 errors, 解禁规则全合规)` | 默认 `ruff check` 通过，**但 [pyproject.toml](../../../pyproject.toml#L96-L102) 自 `PR #89` 后未做任何修改，全部 29 条屏蔽规则（含计划要求解禁的 `B008/B017/B023/B904/SIM115/RUF012/RUF100`）一条未解禁**！隔离全扫上述 7 条解禁目标规则即检出 **`31` 处违规**（`14 RUF100, 7 B904, 4 B023, 4 RUF012, 2 SIM115`）及 `2 S202` | ❌ **规则未解禁 / 假合标** |
| **4** | **全仓 `mypy --strict`** | `Success: no issues found in 137 source files` | 实跑 `mypy --strict python/synapse tests/ scripts/ tools/` 当场退出非零：**`Found 20 errors in 6 files (checked 207 source files)`**！`PR #110` 实仅检查了 `python/synapse`（137 文件），跳过全部 `tests/`/`scripts/`/`tools/`（70 文件），漏掉 `scripts/colab_runner.py:316-317` 2 处真实运行期崩溃 | ❌ **确定性必败 (`20 errors`)** |
| **5** | **5-Tier 包级 DAG 守卫** | `0 violations, 0 exemptions` | `python tools/import_dag_guard.py --check` 退出 `0`（静态逆向边确已消除 ✅），但漏拦截 `from importlib import import_module`、`f"synapse.{x}"` 全仓豁免及未登记孤立子包 | ⚠️ **静态干净但存 3 处绕过** |
| **6** | **跨语言 Wire 守卫** | `wire_guard: OK (require_aether=True)` | `tools/wire_guard.py --check --require-aether` 退出 `0`，但 `bundle.v1.0.schema.json:53` 正则仍放行 `".."`/`"sub/.."`/`"C:/..."`，`LoRAManifest` 允许 `target_modules=[]`，CI 中未传 `--require-aether` | ⚠️ **部分闭合** |
| **7** | **Native 几何核心 (`csrc/geometry`)** | `//:test_geometry PASSED in 0.1s` (`0 warning`) | `bazel test //:test_geometry` 确已全绿且清零 `[opt]` 双重扩展告警 ✅；但 `.bazelrc` 缺失 `build --config=opt` 退化为 `-O0 fastbuild`，`build_native.py` 15s 超时杀死冷编译并向 `mesh_checks.cc` 泄露 `-fexceptions` | ⚠️ **降级为 `-O0` 编译** |
| **8** | **Inspector C++ & Electron** | `//:test_inspector PASSED` + Node `23 passed` | Linux 下测试通过，但 **macOS 与 Windows 下 `ShmFrameReader` 100% 失效**（内核 `shm_open`/`CreateFileMappingA` 无 VFS 文件），C++ `loadGlbMesh` 存在负偏移转 `size_t` 回绕**堆越界读 (Heap OOB)** 与 ARM64 Seqlock 内存序隐患 | ❌ **跨平台断裂 + 内存越界** |
| **9** | **Dataset & CPU 基线端到端** | `--output-dir` 全通过，`all_passed: true` | **命令参数虚构**（两脚本均无 `--output-dir`，直接退出 `2`）；改用 `--output-root` 在 `ubuntu24` 实跑 `prepare_datasets.py` 步骤 `[1/4]` 调用 `fulgur` `102-pbr-direct --headless` 在 `llvmpipe` 下当场 **`SIGSEGV` (`Signal 11`) 崩溃**；`caelum` 导出忽略 `caelum_repo` 用纯 NumPy 圆球伪造 `"spp": 256` | ❌ **实跑崩溃 (`SIGSEGV`)** |
| **10** | **GitHub Actions CI & 文档链路** | `3/3 Jobs 规范完备，0 broken` | 文档 `923` 条相对链接 `0 broken` ✅；但 **`.github/workflows/` 至今不存在（`gh run list` 为 `no runs found`）**，且 `.github/workflows/ci.yml` 中 `python-fast` 缺 `torch`/`safetensors` 必败、`native-bazel` 写错组织名 `sautille-stroke/aether`（应为 `dervgen/aether`）且 `path: ../aether` 非法越界必败 | ❌ **零自动 CI** |

### 1.3 Horizon-7 正向交付成果（What Truly Landed）

1. **彻底消除核心 SDK 常驻服务端**：[python/synapse/viewer/server.py](../../../python/synapse/viewer/server.py) 移除所有 `TCPServer`/`HTTPServer`，`python/synapse/` 零常驻监听端口，仅保留带 `\u003c` XSS 转义与严格 CSP 的静态 HTML 导出，开发预览隔离至 [scripts/dev_preview_viewer.py](../../../scripts/dev_preview_viewer.py)。
2. **`FGPP` 契约前置止血**：原生 `Imath.PixelType.UINT` 写出 `Cryptomatte.ID`；缺失 `.exr` 抛 `AssetNotFoundError(20)`；`sanitize_fgpp_frame` 先清理 `NaN/±Inf` 再算向量模长；`load_metadata` 求逆前完成旋转矩阵正交性校验；EXR 文件句柄全部纳入 `finally` 关闭。
3. **4D 齐次裁剪与精确欧氏距离变换**：[SoftwareRasterizer](../../../python/synapse/render_core/rasterizer.py) 移除退化的 2×2 裁剪求解器，实现 4D 齐次近面裁剪与真 `1/w` 透视校正插值；[morphology.py](../../../python/synapse/render_core/morphology.py) 落地 Felzenszwalb-Huttenlocher $O(HW)$ 精确 EDT 并统一收敛 `mask.py`/`detector.py`。
4. **256 项 Marching Cubes 查表与视点视线**：[recon/_tri_table.py](../../../python/synapse/recon/_tri_table.py) 落地完整互补 256 项拓扑表（内部体素零裂缝）；[FaceRefiner](../../../python/synapse/refine/face.py) 修复虹膜暗质心视线估计；[KalmanBBoxTracker](../../../python/synapse/refine/roi_tracker.py) 改用 Joseph 稳定协方差更新。
5. **R3D-18 通道归一化与实特征 FVD**：[metrics/temporal.py](../../../python/synapse/metrics/temporal.py) 修复 `[C, T, H, W]` 归一化维度使任意帧长可用，`fvd_feature_distance` 改用 `np.linalg.eigh` 消除 `sqrtm` 复数误差；`MetricGate` 方向校验收口。
6. **参考锚定 Diffusion-DPO 与安全 Tar 解包**：[StatelessLoRATrainer](../../../python/synapse/training/trainer.py) 通过 `_disable_lora_adapters` + `eval()` + `no_grad()` 实现真参考锚定 DPO 与 `fp32` 主参数；[FeatureStoreBuilder](../../../python/synapse/training/feature_store.py) 全程流式 `extractfile` + `allow_pickle=False`。
7. **Native 异常边界与 `-fno-exceptions` JSON 迁移**：`csrc/geometry/.bazelrc` 按 `--per_file_copt=bindings\.cc$` 精确切分异常边界；[inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc) 迁 `@aether//json` 消除 JSON 解析 `std::abort()`。
8. **RFC 去碰撞与文档零断链**：`RFC-0060..0063` 规范重命名为 `SYN-RFC-0060..0063-*.md`；全仓 `923` 条相对 Markdown 链接零断链；Horizon 6 的 4 项 `-m slow` 失败清零。

### 1.4 多维健康度评分卡（Post-Horizon-7 真值复核）

| 审查维度 | H6 得分 | H7 得分 | 核心诊断结论 |
| :--- | :---: | :---: | :--- |
| **跨仓 Wire 契约与 Schema 一致性** | `68` | **`66 / 100`** | 跨仓核验发现 `JobBundle` 与 `render-farm`/`grand-studio` 100% 不兼容；`p02` 水平主点符号反转镜像 `cx -> W - cx`；默认 `import_fgpp()` 仍伪造全零深度 |
| **仓内 5-Tier 依赖拓扑 (DAG)** | `76` | **`88 / 100`** | 静态逆向边已清零 ✅；仅剩 `ast.Name("import_module")`、`f"synapse.{x}"` 全仓放行与孤立包 3 处盲区 |
| **算法真实性与防静默失败三问** | `54` | **`64 / 100`** | `WanAdapter` 仍未接入 `ConditionScheduler` 与 `LoRAPatchHook`；`BenchPipelineRunner` 硬编码假满分；MC 法线全内翻+TSDF 空腔；`quad_remesh` T-junction + `roundtrip` 判 Quad `0%` 水密；`delight` 仍 `(I/E)*E` 自抵消；`temporal` 误除 HDR 255；`ScalpelStitcher` $\alpha^2$ 与全旁路挂门禁 |
| **原生 C++20、Bazel 与 Data Inspector** | `58` | **`65 / 100`** | Bazel `[opt]` 告警清零但丢失 `build --config=opt` 退化 `-O0`；macOS/Windows Electron SHM 仍 100% 失效；`.glb` 负偏移堆越界读；ARM64 Seqlock 内存序隐患 |
| **测试质量、Colab/L-Asset 与 CI 基建** | `50` | **`58 / 100`** | `.github/workflows/` 不存在；`mypy --strict` 全仓 20 错；29 条 Ruff 未解禁；单进程快测 `73.97s`；`prepare_datasets` `fulgur` `SIGSEGV`；`colab_runner --mode=real-gpu` 仍零推理 |
| **文档、RFC 与治理真实性** | `56` | **`70 / 100`** | 相对路径 `0` 断链与 RFC 去撞完成 ✅；但 `PR #110` 10 门禁实测表多处失真，Skills 7ch 语义与符号仍漂移 |
| **综合健康度总评** | **`60`** | **`68 / 100`** | **较 Horizon 6 实质提升 +8 分，但未达生产闭环** — 共查实 **`14×P0 + 18×P1 + 12×P2 + 5×P3 = 49`** 项实证缺陷，统一纳入 **Horizon 8 (`M29–M32`, `PR-112 ~ PR-129`)** 彻底闭合 |

---

## 2. Horizon-7 (`PR-93 ~ PR-110`) 宣称 vs 代码实态“防静默失败三问”逐项核验表

| 计划 PR | 实际 GitHub PR & Commit | 存在 | 接线 | 执行 | 审计实态 | 实测未闭环 / 假绿缺口 | 移交 H8 |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **`PR-92`** | `#92` (`52c3431`) | ✅ | ✅ | ⚠️ | **⚠️ 部分合入** | `colab_runner.py` 增补 `sys.path`，但 `--mode=real-gpu` 与异常范围缺口延至 H7/H8 | `PR-125` |
| **`PR-93`** | `#93` (`82c5a8a`) | ⚠️ | ❌ | ❌ | **❌ 门禁未兑现** | 29 条 Ruff 规则在 `pyproject.toml:96-102` 一条未解禁；`.github/workflows/` 未激活且 Job 1/2 必败；`conftest` 漏拦模块级 `importorskip` | `PR-112` |
| **`PR-94`** | `#94` (`98b471e`) | ✅ | ✅ | ⚠️ | **⚠️ 部分闭合** | 静态 DAG、EXR 缺文件与正交前置落地；但默认 `import_fgpp()` (`channels=None`) 仍造全零深度，`p02` 符号反转、`load_metadata` 正交投影失真，`bundle.v1.0` 正则仍放行 `".."` | `PR-113` |
| **`PR-95`** | `#95` (`a4932ef`) | ✅ | ✅ | ⚠️ | **⚠️ 部分闭合** | 4D 裁剪、1/w 插值与 Felzenszwalb EDT 真实落地 ✅；但 `generate_humanoid_proxy` 肢体绕中点脱臼 `19 cm`、`binary_erosion` NumPy 回退边界反转、基元仍非水密 | `PR-114, PR-115` |
| **`PR-96`** | `#96` (`0e29ae4`) | ✅ | ✅ | ⚠️ | **⚠️ 部分闭合** | 穿地 `-Y/-Z` 与反向光流落地 ✅；但 `soft_knee_forward_tonemap` 对 HDR $>1.04$ 发散截断高光、`_cryptomatte_boundary_guide` 2px 脉冲失效、内环双向剥离硬缝、IRLS 背光未归零 | `PR-114` |
| **`PR-97`** | `#97` (`0bc1d83`) | ✅ | ⚠️ | ⚠️ | **⚠️ 部分闭合** | 移除常驻 `ViewerServer` ✅；但跨仓 RFC 0039 `JobBundle` 不兼容且零输入 SHA-256、`stages/recon` 重名覆写 `inverse_reconstruct`、`synapse --json inspect mesh`/`recon` 污染 stdout、`np.save` 丢 `.npy` | `PR-115` |
| **`PR-98`** | `#98` (`d6ee396`) | ✅ | ❌ | ❌ | **❌ 核心未接线** | `WanAdapter.generate()` 仍未调用 `ConditionScheduler`、`fuse_overlapping_latents` 与 `LoRAPatchHook`（仅孤岛代码）；`LoRAPatchHook` 3D DiT/`cross_attn.to_k` 必崩；HTTP 含 SSRF 与无错误码映射；`HttpCloudTransport` 丢弃锚点图与 Prompt | `PR-116` |
| **`PR-99`** | `#99` (`1602ba4`) | ✅ | ⚠️ | ❌ | **⚠️ 接线断裂** | 参考锚定 DPO 落地 ✅；但通配符 `target_modules` 三处冲突、4 项损失中 Edge/Normal 静默跳过且 16ch 必崩、自动采样 `t` 丢失、缺 `fp16` `GradScaler`、`FeatureStore` 15ch 索引自撞、`packager` 缺指标 Fail-Open | `PR-117` |
| **`PR-100`** | `#100` (`8e5e3fd`) | ✅ | ⚠️ | ❌ | **❌ 数学反转** | 256 项 MC 表自身内部零裂缝 ✅；但三角形绕序 100% 内翻法线、TSDF 空腔、Visual Hull 漏雕、边界未填充破水密；`quad_remesh` 造 T-junction；`delight` 仍 `(rad/E)*E` 自抵消与屏幕空间鬼影；`ClosedLoopRunner` `IndexError` 与伪 TBN | `PR-118` |
| **`PR-101`** | `#101` (`6fbdd95`) | ✅ | ⚠️ | ❌ | **⚠️ 门禁与融合缺陷** | Joseph Kalman 与虹膜视线落地 ✅；但全旁路健康视频 `PSNR=0` 必挂门禁、用未 Warp 帧差充当 `temporal_warp_error`、Stitcher 双重 $\alpha^2$ + 核心硬缝 + 反转 EMA、`Poisson` 散度失配、`TiledBypassExecutor` 留黑边且零调用 | `PR-119` |
| **`PR-102`** | `#102` (`bde1832`) | ✅ | ✅ | ❌ | **❌ Quad/HDR 退化** | R3D-18 `[C,T,H,W]` 修复 ✅；但 `roundtrip.py` 对角线 `poly[(i+2)%4]` 致 100% 水密 Quad 判 `0%` 水密且丢弃 50% 表面积/体积；`temporal.py` 将 HDR `float32 > 1.0` 误除 `255.0`；`image_quality` 混型崩溃 | `PR-120` |
| **`PR-103`** | `#103` (`1017c8f`) | ✅ | ✅ | ⚠️ | **⚠️ 构建退化** | pybind 异常边界与 GTest 落地 ✅；但两处 `.bazelrc` 缺失 `build --config=opt` 编译 `-O0`、`build_native` `15s` 超时且直编泄露 `-fexceptions` 并吐双 JSON、`CheckManifold` 伞环 $O(d^2)$ | `PR-121` |
| **`PR-104`** | `#104` (`1fd4959`) | ✅ | ✅ | ❌ | **❌ 内存安全 P0** | `@aether//json` 迁入 ✅；但 `loadGlbMesh` 负偏移转 `size_t` 回绕引发**堆越界读**与未对齐 UB、Seqlock ARM64 内存序隐患、跨视口边缘整三角形剔除与零 `.exr` 加载 | `PR-122` |
| **`PR-105`** | `#105` (`18826f4`) | ✅ | ⚠️ | ❌ | **❌ 非 Linux 失效** | Electron 安全沙箱基线落地 ✅；但 macOS/Windows `ShmFrameReader` 仍 100% 失效、`nativeSendCommand` 允许任意路径覆写 `export_screenshot`、每帧仍分配/IPC 拷贝 20MB 帧 | `PR-123` |
| **`PR-106`** | `#106` (`6ac92b8`) | ✅ | ⚠️ | ❌ | **❌ 假通过常数** | `BenchPipelineRunner` 硬编码 T1–T3 满分常数 + T5 100% 必败 + 缓存光污染 + 18/30 场景参数丢失；`model_evaluator` 无参考 100% 败 + `floating_island_faces` 算错；`PromotionGate` 空列表/重复 ID/`NaN` 绕过；`macro` `-Y/-Z` 反转 | `PR-124` |
| **`PR-107`** | `#107` (`85d3799`) | ✅ | ⚠️ | ❌ | **❌ 实跑崩溃/空壳** | `prepare_datasets` 在 `ubuntu24` `llvmpipe` `SIGSEGV`；`colab_runner --mode=real-gpu` 仍零推理且异常退出 `1`；`assets_manifest` 归档包哈希仍等于单文件哈希；HF 异常吞并回退方块 | `PR-125` |
| **`PR-108`** | `#108` (`dc9aab2`) | ✅ | ⚠️ | ❌ | **❌ 伪重构** | `test_roundtrip_reconstruction.py` 仍未调 `ReconPipeline.run()` 且将 `chamfer` 放宽 `4.375x`；单进程快测 `73.97s`（`test_r3d18_unmocked_t16` 未标 `slow` 占 `21.5s`）；`test_bench_full_suite` 串行 30 场景拖慢 `-m slow` 至 `89.19s` | `PR-126` |
| **`PR-109`** | `#109` (`dd5cb39`) | ✅ | ✅ | ⚠️ | **⚠️ 文档漂移** | `SYN-RFC` 重命名与相对链接零断链 ✅；但 Skills 仍写错 Wan 7ch 与虚构类名、`AUDIT_AND_ARCHITECTURE` 错映射 `SYN-RFC`、Colab `io_spec.md` 幻觉字段 | `PR-127` |
| **`PR-110`** | `#110` (`cf2a49f`) | ⚠️ | ❌ | ❌ | **❌ 虚假封板** | `Gate 1` `-n auto` 冒充单进程、`Gate 3` 未解禁任何 Ruff 规则、`Gate 4` 仅跑 137 文件漏 `20` 错、`Gate 9` 参数虚构、`Gate 10` `.github/workflows/` 仍为空 | `PR-112, PR-129` |

---

## 3. 跨仓与跨模块深层架构断裂矩阵 (Cross-Repo & Cross-Module Contract Fractures)

| 断裂维度 | 上下游权威契约 (SSOT) | `synapse` 现状偏差与断点 | 实测系统性后果 | 收口 PR |
| :--- | :--- | :--- | :--- | :---: |
| **1. 跨仓 RFC 0039 `JobBundle` 调度契约** | `render-farm/internal/domain/models.go:452` & `grand-studio/src/core/job-bundle.ts:159`：`{job_id, project_name, closure_type, engine, scene_spec, asset_manifest[{sha256, read_capability}], output_sink, payload}`，枚举 `neural_video`/`surgical_inpaint`/`mesh_reconstruct`/`delighting` | `BundleManifest` (`bundle_worker.py:68-90`) 设 `extra="forbid"`，仅认 `{schema_version, job_id, stage, inputs, outputs, params}`，枚举 `neural_video_generate`/`surgical_composite`/`video_to_3d_invert`，无 `asset_manifest` SHA-256 校验，且 `render-farm` 缺 `SynapseExecutor` | `render-farm` / `grand-studio` 下发的任何真实生产 `JobBundle` 在 `synapse worker run` 首行即报错 `ExitCode 10` 拒收，跨仓端到端完全断开 | `PR-115, PR-128` |
| **2. 透视投影水平主点 `p02` 符号反转** | OpenGL 右手系 (`+Y up, -Z forward`) 偏心透视投影矩阵：`P[0,2] = (r+l)/(r-l) = (W - 2·cx) / W`，保证光轴点 `(0,0,-d)` 投影至像素 `sx = cx` | `fgpp/frame.py:84` 错写为 `(2*cx - W)/W`（抄自 Y 轴翻转后的 `p12`），`frame.py:347` 同步用错号反解 `cx = (p02+1)*W/2` | `SoftwareRasterizer` 与 `detector.py` 水平镜像错位 `W - 2·cx` 像素；读取 `@aether//fgpp` / `fulgur` 偏心相机或 Tile 矩阵时左右完全镜像 | `PR-113, PR-128` |
| **3. Wan 长视频滑窗 & 空域 LoRA“孤岛算子”** | RFC 0044：长镜头必须经 `ConditionScheduler` ($W=97, S=64, \Delta=33$, 3f 暖启动, $\kappa$ 切镜重锚定) 分窗生成与潜空间融合；`identity` LoRA 挂前景 `M_fg`、`style` 挂背景 `~M_fg` | `ConditionScheduler`、`fuse_overlapping_latents` 与 `LoRAPatchHook` 在 `python/synapse/` 内生产调用数全部为 **`0`**；`WanAdapter.generate()` 单次硬塞全序列且直接全局相加权重（一旦挂 `LoRAPatchHook` 即因 3D `[B,L,D]` 与 `cross_attn.to_k` 崩溃） | >97 帧长镜头无滑窗显存爆炸且切镜渗漏；多 LoRA 无法做角色/背景空域隔离，风格滤镜仍污染主角面部 | `PR-116` |
| **4. Quad 多边形拓扑与指标盲区** | `ReconPipeline` 输出 `(F, 4)` Quad 网格，Tier 0 与 `Synapse-Bench` 必须一致支持三角形与四边形流形/水密/表面积/体积计算 | `quad_remesh` 部分配对后只细分未配对三角形造出 T-junction 破洞；`metrics/roundtrip.py:323` 错查对角线 `poly[(i+2)%4]` 且 `sample_mesh_surface` / `_signed_volume` 截断丢弃 `f[:, 3]` | **100% 合法水密 Quad 网格被判 `watertight_pct = 0%`**，`chamfer_mm` 漏采 50% 表面，真实 `ReconPipeline` 产出的 Quad 无法通过 `TRACK_A_ASSET` 门禁 | `PR-118, PR-120` |
| **5. 跨平台共享内存内核对象 vs VFS 文件路径** | `@aether//pal:shm` 在 Linux 映射 `/dev/shm/<name>`，但在 macOS (`shm_open` XNU `pshm`) 与 Windows (`CreateFileMappingA` `\BaseNamedObjects\`) **没有文件系统路径** | Electron `shm-reader.js` 仅通过 `fs.openSync` 探测 `/dev/shm` 与 `os.tmpdir()`；C++ `InspectorShmServer` 无文件映射兜底；Node 测试跳过真实原生链路 | **macOS 与 Windows 桌面端 Data Inspector 100% 黑屏零帧**；且 `.glb` 加载器带负偏移堆越界读漏洞 | `PR-122, PR-123, PR-128` |

---

## 4. 全量分级缺陷台账 (`H8-P0` / `H8-P1` / `H8-P2` / `H8-P3` — 共 49 项)

> 全量 **49 项** 缺陷均在 `HEAD = cf2a49f` 上逐行核验代码位置、数学推导或 `ubuntu24` 实跑输出，按 **`14 P0 + 18 P1 + 12 P2 + 5 P3`** 精确映射至 **Horizon 8 (`M29–M32`, `PR-112 ~ PR-129`)**。

### 4.1 P0 阻断级缺陷（`H8-P0-01 ~ H8-P0-14`：CI/类型门禁失效 · 跨仓协议断裂 · 核心算子数学反转与内存越界）

#### `[H8-P0-01]` GitHub Actions CI 从未激活且 Job 1/2 必败；29 条 Ruff 规则未解禁；全仓 207 文件 `mypy --strict` 实测 `20 errors` (`PR-112`)
- **精确位置**: [.github/workflows/ci.yml](../../../.github/workflows/ci.yml#L1-L104)（缺失 `.github/workflows/`）；[pyproject.toml](../../../pyproject.toml#L82-L105)；[scripts/colab_runner.py](../../../scripts/colab_runner.py#L316-L317)；`tests/test_{style,rfc0063_metrics,fgpp_io,recon,lora_feature_store}.py`
- **实证与根因**:
  1. `gh run list` 返回 `no runs found`，工作流错放 `.github/workflows/ci.yml`；即使移至 `.github/workflows/ci.yml`，`python-fast` 缺 `torch`/`safetensors` 会在 `test_rfc0063_metrics.py:127` 裸 `import torch` 崩溃并触发 ~20 处 `SYNAPSE_CI_STRICT_DEPS=1` 失败；`native-bazel` 错写 `repository: sautille-stroke/aether`（应为 `dervgen/aether`）且 `path: ../aether` 超出 `$GITHUB_WORKSPACE` 必败。
  2. `pyproject.toml:96-102` 依然屏蔽全部 29 条规则（含指定解禁的 `B008/B017/B023/B904/SIM115/RUF012/RUF100`），隔离扫描实出 **31+ 处高危违规** + `2 S202`。
  3. `ubuntu24` 实跑 `mypy --strict python/synapse tests/ scripts/ tools/` 当场报错退出非零：`Found 20 errors in 6 files (checked 207 source files)`，暴露 `scripts/colab_runner.py:316` 传裸 `LoRAManifest` 给 `list[tuple[LoRAManifest, float]]` 与 `:317` 调用不存在的 `MultiLoRAManager.is_bundle_loaded`（`--mode=real-gpu` E4 必崩）。
- **修复建议**: 迁移并修复 CI 工作流（`dervgen/aether`、子目录 checkout `synapse`/`aether`、装齐 CPU `torch`/`train`/`mesh`、加 `--require-aether` 与 `python-slow`）；解禁 `pyproject.toml` 核心规则并清零违规；修复全仓 207 文件 `mypy --strict` 20 错。

#### `[H8-P0-02]` 跨仓 RFC 0039 `JobBundle` 契约与 `render-farm`/`grand-studio` 100% 不兼容且零输入 SHA-256 校验 (`PR-115`)
- **精确位置**: [python/synapse/pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py#L53-L91,L772-L880)；[schemas/job_bundle.v1.0.schema.json](../../../schemas/job_bundle.v1.0.schema.json#L7-L71)；对照 `../render-farm/internal/domain/models.go#L452-L534` & `../grand-studio/src/core/job-bundle.ts#L159-L166`
- **实证与根因**:
  1. `BundleManifest` (`extra="forbid"`) 要求 `{schema_version: "synapse.job_bundle.v1.0", job_id, stage, inputs, outputs, params}`，而 `render-farm`/`grand-studio` RFC 0039 真实下发 `{job_id, project_name, closure_type, engine, scene_spec, asset_manifest, output_sink, payload}`，导致真实作业直接被拒收（`ExitCode 10`）。
  2. `ClosureType` 枚举名不一致（`"neural_video"` vs `"neural_video_generate"`、`"surgical_inpaint"` vs `"surgical_composite"`、`"mesh_reconstruct"` vs `"video_to_3d_invert"`、`"delighting"` 缺失）。
  3. `BundleManifest.inputs` 无 SHA-256 校验，违背 `AGENTS.md §3` 退出码 `20` 校验要求。
- **修复建议**: 在 `BundleManifest` / `JobBundleWorker` 增加 `from_rfc0039_job_bundle` 双格式适配层与 `ClosureType` 别名映射（含注册 `"delighting"` Stage），并对 `asset_manifest` / `input_sha256s` 强制流式 SHA-256 校验（不匹配抛 `AssetNotFoundError(20)`）。

#### `[H8-P0-03]` `CameraParams.projection_matrix` 水平主点 `p02` 符号反转且 `_read_exr_frame` 默认仍造全零深度 (`PR-113`)
- **精确位置**: [python/synapse/fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L84,L347)；[python/synapse/fgpp/io.py](../../../python/synapse/fgpp/io.py#L221-L243,L286)
- **实证与根因**:
  1. 右手系 `-Z` 前望相机（$c_w = -Z_{\text{cam}} = d$）下 `x_ndc = (2*fx/W)*(X/d) - p02`，而光栅化屏幕 X 未翻转，故需 `p02 = (W - 2*cx)/W`。`frame.py:84` 错写成 `(2*cx - W)/W` 将偏心主点全部水平镜像到 `W - cx`，`frame.py:347` 用错式逆推使 Python 闭环自洽假绿。
  2. `_read_exr_frame` 将缺失通道校验锁在 `if channels is not None:` 分支内，默认调用 `import_fgpp(input_dir)` (`channels=None`) 遇到缺失 `CameraLinearDepth.Z` 的 EXR 时跳过校验并执行 `np.zeros((h, w))` 伪造全零深度；传入非法通道名（如 `["albedo"]`）也被静默忽略。
- **修复建议**: 修正 `p02 = (w - 2.0 * cx) / w` 与 `cx = (1.0 - p02) * w / 2.0`；默认 `channels=None` 或显式包含 `"depth"` 时强制要求 `CameraLinearDepth.Z`，并在入口拦截未知通道名抛 `ContractValidationError(10)`。

#### `[H8-P0-04]` `soft_knee_forward_tonemap` 对 HDR $> 1.04$ 无限发散导致 `composite_optical_increments` 高光硬截断 (`PR-114`)
- **精确位置**: [python/synapse/style/aces_color.py](../../../python/synapse/style/aces_color.py#L131-L166)；[python/synapse/composite/blend.py](../../../python/synapse/composite/blend.py#L688-L694)
- **实证与根因**: `_fallback_soft_knee_inverse_tonemap` 采用二次多项式 $0.8 + \Delta + \Delta^2$（$x=1.0$ 仅映射至 $1.04$），其逆函数 `soft_knee_forward_tonemap` 对所有 HDR 值 $y > 1.04$ 以 $O(\sqrt{y}) \to +\infty$ 增长（如 $f^{-1}(3.0)=1.865$），经 `blend.py:692` `np.clip(..., 0.0, 1.0)` 后所有高光与焦散增量被硬截平为 `255`。
- **修复建议**: 替换为 $C^1$ 连续渐近有理软肩对（$t=0.8, s=0.2$）：正变换 $T(y) = t + \frac{s(y-t)}{s+(y-t)} \in [0, 1)$，逆变换 $T^{-1}(x) = t + \frac{s(\min(x, 0.999)-t)}{1-\min(x, 0.999)}$。

#### `[H8-P0-05]` `WanAdapter.generate()` 零调用 `ConditionScheduler` 与 `LoRAPatchHook`，且 `LoRAPatchHook` / `_extract_lora_ab` 存在维度崩溃与跨层污染 (`PR-116`)
- **精确位置**: [python/synapse/adapters/wan.py](../../../python/synapse/adapters/wan.py#L543-L561,L664-L701)；[python/synapse/adapters/scheduler.py](../../../python/synapse/adapters/scheduler.py#L394-L691)；[python/synapse/adapters/lora/manager.py](../../../python/synapse/adapters/lora/manager.py#L181-L322,L450-L516)；[python/synapse/adapters/lora/patcher.py](../../../python/synapse/adapters/lora/patcher.py#L39-L126)
- **实证与根因**:
  1. `ConditionScheduler`、`fuse_overlapping_latents` 与 `LoRAPatchHook` 在整个 `python/synapse/` 内生产调用数均为 **`0`**；`wan.py:679` 丢弃 `build_spatial_lora_masks` 返回值并调用全局无掩膜 `apply_merged_weights`（同时跳过正交化），且 `wan.py:646-662` 未对 $>97$ 帧序列按 `W=97, S=64` 滑窗调度。
  2. `LoRAPatchHook._hook` 存在 4 处运行期必崩/错位：`[T, H//8, W//8]` 掩膜未时空池化至 DiT `(tau, H//16, W//16)`；误挂 `cross_attn.to_k`/`to_v` 文本 token 维度必崩；`B > 1` CFG 2D 掩膜广播必崩；`wan.py:676` 将全部非零 Cryptomatte ID 均当成前景；`_extract_lora_ab` 在 `filtered_dict` 未命中时回退整份 `state_dict` 按形状偷取他层权重。
- **修复建议**: 在 `WanAdapter` 与 `_handle_neural_video_generate` 接通 `ConditionScheduler` 分窗生成、$\kappa$ 重锚定、潜空间融合与光流算子；实现 `(4,16,16)` 时空池化 `LoRAPatchHook`（`identity -> M_fg`, `style -> ~M_fg`, 跳过 `to_k/to_v`, 兼容 `B>1`）、移除跨层形状回退并确保 `finally` 绝对还原权重。

#### `[H8-P0-06]` `BenchPipelineRunner._compute_metrics` 硬编码假通过常数、T5 门禁 100% 必败且缓存污染光照 (`PR-124`)
- **精确位置**: [python/synapse/pipeline/bench_runner.py](../../../python/synapse/pipeline/bench_runner.py#L256-L263,L344-L373,L757-L807)
- **实证与根因**: `L767-768` 硬编码 `light_dir_error_deg = 0.0`、`style_light_error_deg = 0.0`；`L791-806` 硬编码全部 T1 (`chamfer_mm=0.0, foot_sliding_cm=0.0`) 与 T2/T3 (`0.5 / 0.05`) 满分常数（完全无视生成视频输出，纯黑视频也过关），`L771` 用输入真值 `mesh` 充当输出拓扑指标；T5 无分支导致必缺 `chamfer_mm`/`bone_length_drift_pct` 100% 门禁失败；`cache_key = f"{width}x{height}"` 令多场景复用首个场景光照。
- **修复建议**: 删尽硬编码常数，接入真实重建/风格测量或明确标注未执行 Stage 指标范围修复 T5 门禁，并将完整光照参数纳入 `cache_key`。

#### `[H8-P0-07]` Marching Cubes 100% 内翻法线、TSDF 空腔、Visual Hull 漏雕与开放边界破水密 (`PR-118`)
- **精确位置**: [python/synapse/recon/multiview.py](../../../python/synapse/recon/multiview.py#L102-L135,L498-L505,L550-L565,L608-L655)；[python/synapse/recon/_tri_table.py](../../../python/synapse/recon/_tri_table.py#L5-L262)
- **实证与根因**:
  1. `TRI_TABLE` 为左手系索引配右手系 `_CORNERS`，生成 100% 顺时针**内向法线**（`cubeindex=1` 叉积全负），导致后续 `delight.py` `cos_theta > 0.05` 剔除所有正面。
  2. TSDF 用对称 `abs(sdf) < sdf_trunc` 门控，致表面深处 `sdf < -0.04m` 体素保持初值 `+1.0`（外部），抽出**双壁空腔壳体**。
  3. Visual Hull `carve[in_bounds]` 不雕刻视锥外体素角点，且 TSDF/Visual Hull 均未在外周垫 1 层 `+1.0` 边界（接触体素盒子边界即破裂为非水密），同时存在 `-0.5 * voxel` 坐标偏移。
- **修复建议**: 翻转三角形绕序 `(e0, e2, e1)` 并对齐 `+0.5 * voxel` 中心；TSDF 门控改 `sdf > -sdf_trunc` 仅对观测单元出面并 6 面填充 `+1.0`；Visual Hull 雕刻 `~in_bounds`。

#### `[H8-P0-08]` `quad_remesh` 造 T-junction 破洞且 `metrics/roundtrip.py` 将 100% 水密 Quad 判为 `0%` 水密并丢弃 50% 表面 (`PR-118, PR-120`)
- **精确位置**: [python/synapse/recon/geometry.py](../../../python/synapse/recon/geometry/__init__.py)；[python/synapse/metrics/roundtrip.py](../../../python/synapse/metrics/roundtrip.py#L128-L140,L318-L327,L395-L404)
- **实证与根因**:
  1. `quad_remesh` 在部分三角形配对成 `quad_faces` 后，仅对 `unpaired_triangles` 做中点细分却不细分相邻 `quad_faces`，沿公共边制造 T-junction 裂缝与非流形顶点。
  2. `metrics/roundtrip.py:323` 对 `(F, n)` 多边形构建顶点环边时查 `b = poly[(i + 2) % n]`（对 Quad 即为**对角线顶点**而非邻边 `poly[(i - 1) % n]`），致 **100% 水密 Quad 网格恒判非水密**；同时 `sample_mesh_surface` 与 `_signed_volume` 只取 `f[:, :3]` 截断丢弃 `f[:, 3]`，使 Quad 丢失 50% 面积与体积。
- **修复建议**: `quad_remesh` 对 `quad_faces` 与 `unpaired_triangles` 同步做一步 Catmull-Clark 共享中点细分；`roundtrip.py` 改用 `poly[(i - 1) % n]`（或直接复用 `check_watertight`）并对 `(F, K)` 面片做扇形三角化。

#### `[H8-P0-09]` `recon/delight.py` 保留 `(rad/E)*E` 自抵消、屏幕空间鬼影混叠、相机系 SH 抵消与双重 Gamma (`PR-118`)
- **精确位置**: [python/synapse/recon/delight.py](../../../python/synapse/recon/delight.py#L404-L425,L585-L656,L705-L810)
- **实证与根因**: `albedo = rad / shading` 后立刻算 `relit = albedo * shading` 使残差恒为 `0`；多视角直接在 `(H, W)` 屏幕像素而非网格 UV/顶点空间累计（跨旋转视角产生重影）；每视角按相机坐标系拟合 SH 后跨视角直接求平均抵消方向光；5 点稀疏点投覆盖率 `< 2.5%` 频繁回退伪法线；`L807` 对线性浮点帧误用 `is_linear=False` 双重 sRGB 扩张。
- **修复建议**: 复用三角形 Z-buffer 软光栅以世界空间法线拟合全局 SH，先做高光/漫反射双色度分离再按 `cos_theta * visibility` 累加至网格 UV/顶点反照率纹理，并修正线性标志。

#### `[H8-P0-10]` `metrics/temporal.py` 将含 HDR 高光 (`max > 1.0`) 的 `float32` 视频误除 `255.0` 并双除 `255^2` (`PR-120`)
- **精确位置**: [python/synapse/metrics/temporal.py](../../../python/synapse/metrics/temporal.py#L75-L80,L155-L160,L199-L240,L445-L448)
- **实证与根因**: 五处逻辑均写 `if dtype == uint8 or (is_floating and max() > 1.0): / 255.0`。任意含高光像素 (`> 1.0`) 的线性 ACEScg/HDR `float32` 视频全帧被缩小 `255` 倍，`temporal_warp_error` 暴缩 `255x`、`flicker_score` 暴缩 `65,025x`、R3D-18 特征塌缩为全黑，且 `[0, 260]` 浮点过冲会被入口与回退函数连除两次 `255.0`。
- **修复建议**: 仅当 `dtype == uint8`（或 `uint16` 除 `65535`）时除以 `255.0`，禁止对浮点张量除 `255.0`，并在入口一次性归一化。

#### `[H8-P0-11]` `refine` 全旁路健康视频 `PSNR=0` 必挂门禁、`ScalpelStitcher` $\alpha^2$ + 核心接缝 + 反转 EMA、`TiledBypassExecutor` 留黑边 (`PR-119`)
- **精确位置**: [python/synapse/refine/pipeline.py](../../../python/synapse/refine/pipeline.py#L279-L287,L347-L395)；[python/synapse/refine/stitcher.py](../../../python/synapse/refine/stitcher.py#L203-L228,L269-L278,L661-L675)；[python/synapse/refine/tiled_executor.py](../../../python/synapse/refine/tiled_executor.py#L114-L139)
- **实证与根因**: (1) 所有 ROI 均健康旁路时 `measured==0` 输出 `mean_border_psnr_db = 0.0 < 38.0` 触发 `strict=True` 报错；(2) `pipeline.py` 用未做光流对齐的原始帧差 `|f[t] - f[t-1]|` 充当 `temporal_warp_error`；(3) `stitcher.py` 拉普拉斯金字塔混合后又乘一次 `surgical_alpha`（变成 $\alpha^2$）并硬覆写未 LAB 匹配的 `core_mask` 产生 1px 阶跃接缝，且快速运动 (`>15px`) 时反而将当前帧权重从 `0.85` 降到 `0.30` 制造重影；(4) `TiledBypassExecutor` 对 `(h - ts) % stride != 0` 末行/末列留纯黑未处理条带且从未接入主流程。
- **修复建议**: 零活跃补丁直接返 `frames.copy()` 与 `PSNR=100.0`；传入 `motion_vectors` 调用真实 `temporal_warp_error`；删除重复 $\alpha$ 混合与硬覆写，改用单调亮度比保真 HDR、快速运动 EMA 推向 `1.0`；补齐末 tile + 2D Hann 窗融合并接入 `LocalRefinementPipeline`。

#### `[H8-P0-12]` Electron `ShmFrameReader` 在 macOS/Windows 100% 失效 + C++ `loadGlbMesh` 负偏移回绕堆越界读与未对齐 UB (`PR-122, PR-123`)
- **精确位置**: [tools/data-inspector/electron/shm-reader.js](../../../tools/data-inspector/electron/shm-reader.js#L70-L97)；[tools/data-inspector/native/inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc#L256-L296,L328-L411)
- **实证与根因**:
  1. `@aether//pal:shm` 在 macOS (`shm_open`) 与 Windows (`CreateFileMappingA`) 创建无文件系统路径的内核对象，`shm-reader.js` 的 `fs.openSync` 查 `/dev/shm` / `os.tmpdir()` 100% 返 `null`。
  2. `inspector_app.cc:345,377` 将 `int64_t` `byteOffset` 直接强转为 `size_t`；恶意或损坏 `.glb` 中的 `"byteOffset": -100` 无符号相加后绕回跳过 `< binChunk.size()` 上界校验，导致**堆缓冲区前向越界读**；`reinterpret_cast<const float/uint16/uint32*>` 未校验对齐与 `byteStride` 引发 UBSan 错与错位模型；`fileSize` 无上限且 `chunkLength` 对齐可绕回。
- **修复建议**: 为 `InspectorShmServer` 与 `ShmFrameReader` 打通跨平台 `--shm-file` VFS 文件映射后端（或 `@aether//pal:shm` Node N-API 绑定）；`loadGlbMesh` 强制 256MiB 上限、非负预检、减法边界校验、`byteStride` 支持与 `std::memcpy` 安全读取。

#### `[H8-P0-13]` RSI `PromotionGate` 空列表/重复场景/`NaN` 容差绕过 `min_scenarios=30` 与退化检查 (`PR-124`)
- **精确位置**: [python/synapse/rsi/promotion_gate.py](../../../python/synapse/rsi/promotion_gate.py#L136-L202,L245-L265)；[python/synapse/cli/_rsi.py](../../../python/synapse/cli/_rsi.py#L211-L246)
- **实证与根因**: `if len(baseline.scenarios) > 0 and ...` 令空列表跳过 `min_scenarios=30`；重复同一个 `scenario_id` 30 次即可凑够计数；CLI `--scenarios` 将同一份列表同时赋给 `baseline` 与 `candidate`；`noise_tolerance=float('nan')` 使 `regression > nan` 恒假，关闭逐场景退化守卫。
- **修复建议**: `min_scenarios > 0` 时无条件要求 `len(baseline_ids) >= min_scenarios`；禁止重复 `scenario_id`；校验 `epsilon_min`/`noise_tolerance` 有限且非负；校验 `geom`/`temporal` 子项不退化并移除 CLI 共享 `--scenarios`。

#### `[H8-P0-14]` `prepare_datasets.py` 实跑 `fulgur` `SIGSEGV` + Gate 9 参数错误 + `colab_runner --mode=real-gpu` 空壳及异常落出 `try` (`PR-125`)
- **精确位置**: [scripts/prepare_datasets.py](../../../scripts/prepare_datasets.py#L42-L80,L149-L340,L399)；[scripts/colab_runner.py](../../../scripts/colab_runner.py#L189,L225-L325,L433-L443,L531-L575)
- **实证与根因**: (1) `ubuntu24` 实跑 `prepare_datasets.py --output-root ...` 步骤 `[1/4]` 执行 `fulgur/bazel-bin/course/102-pbr-direct --headless` 在 `llvmpipe` 下当场 **`SIGSEGV` (Signal 11)**，且 `_export_caelum_track_a_17f` 无视 `caelum_repo` 伪造 `"spp": 256`、`PLAN-M25-M28` 所写 `--output-dir` 报错 `exit 2`；(2) `colab_runner.py --mode=real-gpu` E1 仅调 CPU `prepare_conditions`、E4 仅写 `torch.randn` 零训练，`L531-535` `raise BackendUnavailableError` 落在 `try` 外崩溃退出 `1` 而非 JSON `30`，失败时还用 `{}` 覆写抹除已完成阶段摘要。
- **修复建议**: 修复无头软件渲染配置与可控确定性回退 + 兼容 `--output-dir` 与真实来源标注；`colab_runner` 全链路入 `try` 保断点状态、`--mode=real-gpu` 真实调用 CUDA `WanAdapter.generate()` / `StatelessLoRATrainer`（缺权重或显存不足诚实退 `30`/`50`）。

---

### 4.2 P1 严重功能 / 安全 / 数学 / 跨平台缺陷（`H8-P1-01 ~ H8-P1-18`）

| 审计编号 | 影响模块与精确代码位置 | 根因实证描述 | 目标 PR |
| :--- | :--- | :--- | :---: |
| **`H8-P1-01`** | [tests/conftest.py](../../../tests/conftest.py#L19-L31)；`tests/` 耗时分布 | `conftest` 漏钩 `pytest_make_collect_report` 无法拦截模块顶层 `importorskip(..., allow_module_level=True)`，且会误伤合法 `wasxfail`；`test_r3d18_unmocked_t16_feature_extraction` 未标 `slow` 占 `21.5s` 拖垮单进程快测至 `73.97s` | `PR-112, PR-126` |
| **`H8-P1-02`** | [tools/import_dag_guard.py](../../../tools/import_dag_guard.py#L194-L297,L416-L426)；[tools/wire_guard.py](../../../tools/wire_guard.py#L332-L466)；[schemas/bundle.v1.0.schema.json](../../../schemas/bundle.v1.0.schema.json#L53)；[lora.v1.0](../../../schemas/lora.v1.0.schema.json#L36)；[manifest.py](../../../python/synapse/adapters/lora/manifest.py#L74-L83) | `import_dag_guard` 漏检 `from importlib import import_module` (`ast.Name`)、全局放行 `f"synapse.{x}"` 且漏检未登记孤立子包；`bundle.v1.0` 正则仍放行 `".."`, `"sub/.."`, `"C:/..."`，`LoRAManifest` 缺 `extra="forbid"` 与 `target_modules` `min_length=1`，`wire_guard` 未比对 Pydantic `model_json_schema()` | `PR-113` |
| **`H8-P1-03`** | [fgpp/frame.py](../../../python/synapse/fgpp/frame.py#L323-L412) | `load_metadata` 对正交投影误套透视反解生成 NaN/错误 `fx,fy`；硬编码 `z_near=0.1` 忽略 `z_near_meters` 与 C++ `CameraFrameMetadata` 键名；仅含 `"camera"` 时跳过 `det(R)` 正交校验 | `PR-113` |
| **`H8-P1-04`** | [render_core/procedural.py](../../../python/synapse/render_core/procedural.py#L91-L305,L388-L423,L752-L784)；[render_core/morphology.py](../../../python/synapse/render_core/morphology.py#L81-L96) | `generate_humanoid_proxy` 肢体绕各自中心而非近端关节枢轴旋转导致 `15–19 cm` 脱臼，基础几何体（cube/sphere/cylinder/torus）仍非水密；`morphology.binary_erosion` 无 `scipy` 回退设 `constant_values=True` 与 `scipy` 默认 `0` 完全相反 | `PR-114` |
| **`H8-P1-05`** | [composite/blend.py](../../../python/synapse/composite/blend.py#L136-L191,L299-L319,L715-L723)；[style/extractor.py](../../../python/synapse/style/extractor.py#L181-L209) | `_cryptomatte_boundary_guide` 2px 脉冲使两侧同为 `0.0` 令引导滤波退化为跨界模糊；`border_ring_lab_match` 双向剥离内环在贴近内部边界处跳变 `1.0->0.0`；`_solve_half_cosine_light_irls` 未对 `n·L<=0` 置零雅可比且 `med_res` 偏移 | `PR-114` |
| **`H8-P1-06`** | [pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py#L174-L180,L634-L734,L838-L860)；[pipeline/stages/](../../../python/synapse/pipeline/stages/recon.py#L20-L25) | `bundle_worker` 与 `stages/recon` 重名 `@register_stage` 静默覆写 `inverse_reconstruct`/`kinematic_distill`；缺乏路径沙箱与符号链接拦截；目录哈希用 `read_bytes()` 整 shard 读入内存；`asset_inspect_3d`/`inspect_refine`/`rsi_dual_verify`/`lora_preencode`/`anatomical_refine` 含静默 Mock、空输入通过与 CLI ROI Schema 冲突 | `PR-115` |
| **`H8-P1-07`** | [cli/_app.py](../../../python/synapse/cli/_app.py#L95)；[cli/_inspect.py](../../../python/synapse/cli/_inspect.py#L294-L305)；[cli/_recon.py](../../../python/synapse/cli/_recon.py#L278-L288)；[cli/_worker.py](../../../python/synapse/cli/_worker.py#L46-L55) | `synapse --json inspect mesh` 与 `synapse --json recon` 前置 `--json` 被子命令默认值覆写造成 stdout 混入多行文本破坏 JSON；`worker run --json` 缺 `trace_id`/`error_type` 且不消费 `BundleManifest.traceparent` | `PR-115` |
| **`H8-P1-08`** | [bridge/foundary_client.py](../../../python/synapse/bridge/foundary_client.py#L142-L222)；[adapters/wan.py](../../../python/synapse/adapters/wan.py#L178-L238)；[adapters/cloud.py](../../../python/synapse/adapters/cloud.py#L529-L575)；[bridge/hitl.py](../../../python/synapse/bridge/hitl.py#L120-L202) | HTTP 传输层允许 `file://` 与 `169.254.169.254` SSRF、未将 `FoundaryResponse.status=="error"` 转 `ExitCode 30`；`HttpCloudTransport.submit_segment` 丢弃全部锚点图/引导与 `text_prompt`；`PersistentHitlLedger` 先写盘后校验允许负耗时归零 15 分钟熔断且全局零接线 | `PR-116` |
| **`H8-P1-09`** | [adapters/lora/manifest.py](../../../python/synapse/adapters/lora/manifest.py#L21-L43)；[training/trainer.py](../../../python/synapse/training/trainer.py#L285)；[training/packager.py](../../../python/synapse/training/packager.py#L360)；[training/feature_store.py](../../../python/synapse/training/feature_store.py#L337-L455) | `target_modules` 通配符 (`"spatial_attn.*"`) 在 `Manifest`、`inject_lora`（`in` 子串匹配 0 层）与 `packager` 冲突；`FeatureStore` 15ch 校验硬编码自相撞索引 `7:16` 与 `14` 且矛盾 `FGPP_CHANNEL_SPECS`，并跳过 Wan 7ch 校验 | `PR-117` |
| **`H8-P1-10`** | [training/trainer.py](../../../python/synapse/training/trainer.py#L412-L441,L514-L580)；[training/losses.py](../../../python/synapse/training/losses.py#L171-L214)；[training/packager.py](../../../python/synapse/training/packager.py#L232-L247,L389-L456) | `StatelessLoRATrainer` 在 WebDataset 分片上静默跳过 Edge/Normal 损失且将 16ch 潜变量传给 3ch 法线损失必崩、不传入自动采样时间步 `t`、缺 `fp16` `GradScaler`；`packager` 缺门禁字段直接放行且写相对 `adapter_path` 跨目录必崩 | `PR-117` |
| **`H8-P1-11`** | [recon/tracking.py](../../../python/synapse/recon/tracking.py#L207-L242,L541-L680)；[recon/pipeline.py](../../../python/synapse/recon/pipeline.py#L80-L97)；[recon/multiview.py](../../../python/synapse/recon/multiview.py#L525,L628)；[recon/delight.py](../../../python/synapse/recon/delight.py#L618) | 四模块内参混乱（`1050` / `w` / `max(h,w)` / FGPP）；`ReconPipeline` 丢弃 `visibility`、`tracks_3d` 死参数无 PnP 尺度；`reprojection_error_px` 拿第 0 帧投影矩阵配 `t-1` 帧 2D 点；LK 光流 `patch_p` 整数取整每帧漂移 `0.5 px` | `PR-118` |
| **`H8-P1-12`** | [recon/loop/closed_loop_runner.py](../../../python/synapse/recon/loop/closed_loop_runner.py#L202-L261,L382-L384)；[texture_extractor.py](../../../python/synapse/recon/loop/texture_extractor.py#L86-L221)；[forward_guide.py](../../../python/synapse/recon/loop/forward_guide.py#L115-L126)；[recon/kinematics.py](../../../python/synapse/recon/kinematics.py#L189-L245)；[metrics/roundtrip.py](../../../python/synapse/metrics/roundtrip.py#L436-L471) | `ClosedLoopRunner` 对非 `4k+1` 填充长度越界 `IndexError`、`[0,0,w,h]` 致 Stitcher 外环为空、$\kappa$ 重锚定被 `extract_main_take` 丢弃、`bake_uv_normal_map` 未屏蔽零法线背景且伪造世界 `+Y` TBN；`kinematics` `min(y, 0.0)` 反向钳位悬空脚踝、IK 前算滑移失真、锚点累计漂移且 `foot_sliding_cm` 与 `cm/s` 差 30 倍 | `PR-118` |
| **`H8-P1-13`** | [refine/fusion.py](../../../python/synapse/refine/fusion.py#L94-L126,L166-L256)；[refine/hand.py](../../../python/synapse/refine/hand.py#L80)；[refine/face.py](../../../python/synapse/refine/face.py#L215)；[refine/pipeline.py](../../../python/synapse/refine/pipeline.py#L192-L193,L263-L276) | `POISSON_GRADIENT` 二步差分 vs 五点算子失配抹除高频兼直流漂移、`CASCADE` 丢弃当前细部、`UNCERTAINTY_WEIGHTED` `K=2` 退化；`Hand/FaceRefiner` 默认 `backend="heuristic"` 绕过 `strict=True` 诚实门禁；`LocalRefinementPipeline` 缺 IoU 追踪匹配且错用未平滑框裁切 | `PR-119` |
| **`H8-P1-14`** | [metrics/image_quality.py](../../../python/synapse/metrics/image_quality.py#L40-L47,L289-L430)；[metrics/profiles.py](../../../python/synapse/metrics/profiles.py#L224-L258) | `image_quality` 混合 `uint8`/`float` 时完全失真；掩膜 `lpips` OR 下采样+零填充背景造人工边界梯度；`profiles.py` 未经 `METRIC_ALIASES` 规范化 `scenario_overrides` 导致别名覆盖不生效 | `PR-120` |
| **`H8-P1-15`** | `csrc/geometry/.bazelrc`；[tools/data-inspector/native/.bazelrc](../../../tools/data-inspector/native/.bazelrc)；[scripts/build_native.py](../../../scripts/build_native.py#L62,L192-L335,L550-L567)；[inspector_shm_server.cc](../../../tools/data-inspector/native/inspector_shm_server.cc#L81-L133)；`../aether/geo/src/mesh_checks.cc` & `bvh.h` | 两 `.bazelrc` 缺失 `build --config=opt` 退化 `-O0` 与独立 C++20 兜底；`build_native.py` 15s 超时杀冷构、直编泄 `-fexceptions` 至 `mesh_checks.cc` 且错误吐双 JSON；`InspectorShmServer` `beginFrame` `release` 屏蔽不了 ARM64 后置写重排且持锁跨渲染致 Reader 饥饿；`CheckManifold` 伞环 $O(d^2)$ 与 `Bvh` `NaN->int` UB | `PR-121, PR-122` |
| **`H8-P1-16`** | [tools/data-inspector/electron/native-process.js](../../../tools/data-inspector/electron/native-process.js#L97-L104)；[main.js](../../../tools/data-inspector/electron/main.js#L120-L195)；[preload.js](../../../tools/data-inspector/electron/preload.js#L19-L48)；[shm-reader.js](../../../tools/data-inspector/electron/shm-reader.js#L214)；[test-bundle-loading.js](../../../tools/data-inspector/test/test-bundle-loading.js#L560) | 渲染进程可经 `nativeSendCommand` 调用无限制 `export_screenshot` 覆写任意文件而 `save-screenshot-dialog` 为死代码；`ShmFrameReader` + `main.js` 30Hz 分配并拷贝 20MB 像素缓冲；无二进制时跳过原生 SHM e2e 测试 | `PR-123` |
| **`H8-P1-17`** | [render_core/camera.py](../../../python/synapse/render_core/camera.py#L244-L338)；[bench/scenarios/](../../../python/synapse/bench/scenarios/)；[inspect/model_evaluator.py](../../../python/synapse/inspect/model_evaluator.py#L176-L440)；[bench/model_loader.py](../../../python/synapse/bench/model_loader.py#L118-L123)；[inspect/macro.py](../../../python/synapse/inspect/macro.py#L76-L98) | 18/30 场景的球面相机、T4 多光、T3/T5 多物体与 T2 动画参数被忽略；`model_evaluator` 无参考必退 `40`、`floating_island_faces` 错算为 `faces - faces//cc`、各自 `normalize()` 毁掉毫米尺度、`bench` 导入期污染全局注册表；`macro._unproject_pixel` OpenCV 符号将微距相机规划在相机背后 | `PR-124` |
| **`H8-P1-18`** | [assets/assets_manifest.json](../../../assets/assets_manifest.json#L12-L80)；[scripts/prepare_benchmark_assets.py](../../../scripts/prepare_benchmark_assets.py#L418-L421)；[setup_colab.sh](../../../workenv/google-colab/setup_colab.sh#L113)；[tests/test_roundtrip_reconstruction.py](../../../tests/test_roundtrip_reconstruction.py#L140-L385)；[io_spec.md](../../../workenv/google-colab/io_spec.md)；[synapse-adapters-and-experiments](../../../.agents/skills/synapse-adapters-and-experiments/SKILL.md#L16-L20) | `assets_manifest.json` 全部 `.tar.gz` 误填单文件哈希且下载异常回退 8 顶点方块；`test_roundtrip_reconstruction.py` 未调 `ReconPipeline.run()` 且将 `chamfer` 放宽 `4.375x`（`0.008 -> 0.035`）、测试常量图 PSNR；Colab `io_spec.md` 与 Skills Wan 7ch 语义及 `StyleProfile` 字段严重错配 | `PR-125, PR-126, PR-127` |

---

### 4.3 P2 中等缺陷与契约/性能/安全债务（`H8-P2-01 ~ H8-P2-12`）

1. **`[H8-P2-01]` [fgpp/torch_bridge.py](../../../python/synapse/fgpp/torch_bridge.py#L49-L123)、[fgpp/schema.py](../../../python/synapse/fgpp/schema.py#L125-L180) 与 [fgpp/io.py](../../../python/synapse/fgpp/io.py#L403-L460) 边界 (`PR-113`)**：`torch_bridge` 裸 `import torch` 抛 `ModuleNotFoundError` 而非 `BackendUnavailableError(30)`、未校 32 位位宽致 64 位 `.view(np.int32)` 翻倍；`log_normalized_to_linear_depth` 无效裁剪面与未限域 `z_norm` 偏离 C++ `fgpp_types.h`；`sanitize_fgpp_frame` 缺 `[0, 65504]` `HALF` 上溢/负值物理限域。
2. **`[H8-P2-02]` [render_core/rasterizer.py](../../../python/synapse/render_core/rasterizer.py#L79-L368)、[composite/blend.py](../../../python/synapse/composite/blend.py#L323-L694)、[composite/detector.py](../../../python/synapse/composite/detector.py#L195-L307)、[style/aces_color.py](../../../python/synapse/style/aces_color.py#L81-L113) & [telemetry.py](../../../python/synapse/telemetry.py#L201-L205) (`PR-114`)**：`entity_hash` 未避开 `0`、未校验 `camera` 分辨率、背景零法线；`composite_surgical_sequence` `float16` `mv**2` 溢出与 `np.rint` 缺失；`detector` `bbox_xyxy` 越界与 `fx=0` 百万倍放大；`aces_color` 矩阵偏离 `aether::color::kSrgbToAcesCg` 且截断负色域；`StyleDescriptor` 缺 3 元非负校验与 `MemoryForecaster` token 数量按 `//4, //8` 错估 `4×`。
3. **`[H8-P2-03]` [cli/_refine.py](../../../python/synapse/cli/_refine.py#L68-L184)、[cli/_rsi.py](../../../python/synapse/cli/_rsi.py#L164,L245)、[cli/_lora.py](../../../python/synapse/cli/_lora.py#L108)、[cli/_inspect.py](../../../python/synapse/cli/_inspect.py#L135)、[cli/_generate.py](../../../python/synapse/cli/_generate.py#L90-L213) & `np.save` 后缀问题 (`PR-115`)**：`np.save()` 自动附加 `.npy` 致无后缀输出路径在 `ResultManifest.output_files` 为空；`refine anatomical` 空 ROI/非法后缀静默退出 `0`；多处缺失 `agent_fix_hint`、未注册 `--allow-waived`/`--image-uris`、死参数 `--strict` 及遗留 `main_data_gen`/`main_viewer` 原生解析器退出码 `1/2`。
4. **`[H8-P2-04]` [bridge/vg_bridge.py](../../../python/synapse/bridge/vg_bridge.py#L27-L101)、[models/loader.py](../../../python/synapse/models/loader.py#L15-L248) & [viewer/server.py](../../../python/synapse/viewer/server.py#L66-L86) (`PR-115, PR-116`)**：`get_vg_backend(mock=False)` 在未导入 `synapse.recon` 时静默回退 `MockVgBackend()`；`loader.py` 导入期冻结环境变量且裸抛 `trimesh` `ImportError`；`export_static_html` 未拒绝符号链接或转换坏 `manifest.json`。
5. **`[H8-P2-05]` [training/feature_store.py](../../../python/synapse/training/feature_store.py#L622-L677) 分片内存与大小守卫 (`PR-117`)**：`iter_feature_shards` 整份 2GiB `.tar` 缓存进内存（4GiB+ 峰值）、无 `member.size` 上限且未验证读取的 `sample_id`。
6. **`[H8-P2-06]` [recon/geometry.py](../../../python/synapse/recon/geometry/__init__.py)、[recon/backend.py](../../../python/synapse/recon/backend.py#L64-L91) & [recon/loop/mitsuba_optimizer.py](../../../python/synapse/recon/loop/mitsuba_optimizer.py#L44-L275) (`PR-118`)**：`check_self_intersections` 大三角面塞爆百万网格单元退化、`shared_pos` 非对称；`SynapseReconBackend` 隐式 `float32` 降精度且回退 Python；`AnalyticUVInverseOptimizer` 非精确最小二乘初值、TV 越岛渗漏、`radiometric.py` 重复实现色彩转换。
7. **`[H8-P2-07]` [refine/roi_tracker.py](../../../python/synapse/refine/roi_tracker.py#L88-L107) & [refine/stitcher.py](../../../python/synapse/refine/stitcher.py#L233-L633) (`PR-119`)**：`KalmanBBoxTracker` 初帧协方差未置 `R_mat` 且预测宽高未钳位；`_border_psnr` 取内环对比 `frames_acescg` 惩罚合法精修、`warpAffine` 侵蚀 1px 边界及越界光流未掩膜。
8. **`[H8-P2-08]` [metrics/temporal.py](../../../python/synapse/metrics/temporal.py#L88-L470) & [metrics/style_consistency.py](../../../python/synapse/metrics/style_consistency.py#L60-L66) (`PR-120`)**：FVD $N < D$ 零空间非对称 `eps` 偏差、`np.asarray` 遇变长列表崩溃、`temporal_warp_error` 缺帧数/缩放/越界掩膜校验；`albedo_delta_e00`/`f_score_*` 接口不一致。
9. **`[H8-P2-09]` [tools/data-inspector/native/inspector_app.cc](../../../tools/data-inspector/native/inspector_app.cc#L134-L180,L540-L550,L796-L977) (`PR-122`)**：`projectPoint` 对跨出视口边缘单顶点的整个三角形全量丢弃；线性 Z 缺 `1/z`；OBJ `NaN` 顶点偏移后继索引；`.exr` 未真接入 `tinyexr` 渲染。
10. **`[H8-P2-10]` [rsi/dual_verifier.py](../../../python/synapse/rsi/dual_verifier.py#L243-L336)、[rsi/pair_builder.py](../../../python/synapse/rsi/pair_builder.py#L49-L305)、[inspect/eval_report.py](../../../python/synapse/inspect/eval_report.py#L409-L412)、[inspect/align.py](../../../python/synapse/inspect/align.py#L129-L446)、[bench/bundle_exporter.py](../../../python/synapse/bench/bundle_exporter.py#L236-L256) & [cli/_bench.py](../../../python/synapse/cli/_bench.py#L81-L95) (`PR-124`)**：`DualVerifier` 拒收 NumPy 标量且漏 `abs(vol_pct)`；`PreferencePair` 放行空 URI 且缺跨分片去重混洗；`eval_report` 未转义 `img_b64`；`bundle_exporter` 未防符号链接；`align.py` 锚定原点非质心且允许无效旋转长度；`bench run --export-bundle` 找不存在的 `source.glb` 且 `evaluate_bundle` 只评首帧、97 帧驻留 6–8GB 内存。
11. **`[H8-P2-11]` [tests/test_bench_full_suite.py](../../../tests/test_bench_full_suite.py#L18)、[tests/test_domain_randomizer_and_gate.py](../../../tests/test_domain_randomizer_and_gate.py#L23)、[tests/test_metrics.py](../../../tests/test_metrics.py#L282) & [scripts/run_cpu_verification.py](../../../scripts/run_cpu_verification.py#L291-L295) (`PR-125, PR-126`)**：30 场景单函数串行饿死 xdist、14 项快测仍错标 `slow`、`test_metrics.py:282` 与真实 R3D-18 冲突；`prepare_datasets` 缺 `"camera"` `KeyError` 与 `_verify_e3` 内存改帧。
12. **`[H8-P2-12]` [docs/AUDIT_AND_ARCHITECTURE.md](../../../docs/AUDIT_AND_ARCHITECTURE.md#L315-L320)、[.agents/skills/synapse-overview/SKILL.md](../../../.agents/skills/synapse-overview/SKILL.md#L17-L55) & [AGENTS.md](../../../AGENTS.md#L49) (`PR-127`)**：错将 `SYN-RFC-0061..0063` 映射到无关的 `all-overview` RFC、`synapse-overview` 含 `HITLSession`/`FoundaryClient`/`NativeVgBackend` 等虚构类名与伪技术栈、`AGENTS.md` 仍留 `--checkpoint-dir` 且缺 `ubuntu24` `/var/cache/bazel` 共享缓存与磁盘配额铁律。

---

### 4.4 P3 轻量收口与工程卫生（`H8-P3-01 ~ H8-P3-05`）

1. **`[H8-P3-01]` [pyproject.toml](../../../pyproject.toml#L7)、[.gitignore](../../../.gitignore)、[tools/import_dag_guard.py](../../../tools/import_dag_guard.py#L10-L163)、[render_core/procedural.py](../../../python/synapse/render_core/procedural.py#L632) & [composite/blend.py](../../../python/synapse/composite/blend.py#L611) (`PR-112, PR-114`)**：更新 `version` 与 `.gitignore`（`data/golden_datasets/`）；清理 `import_dag_guard` 过时豁免注释与死变量；删除 `procedural.py:632` 裸 `math.cos(phase)` 与补全 `composite_optical_increments` 输入形状检查。
2. **`[H8-P3-02]` [adapters/wan.py](../../../python/synapse/adapters/wan.py#L646-L695)、[pipeline/bundle_worker.py](../../../python/synapse/pipeline/bundle_worker.py#L437-L438) & [generators/dataset.py](../../../python/synapse/generators/dataset.py#L310-L314) (`PR-115, PR-116`)**：`WanAdapter` 缺后端改抛 `BackendUnavailableError(30)`、避免原地改 `control_tensor` 与双调验证；清理空语句与收敛 `DatasetGenerator` `except` 作用域。
3. **`[H8-P3-03]` [render_core/mesh.py](../../../python/synapse/render_core/mesh.py#L102-L106)、[recon/multiview.py](../../../python/synapse/recon/multiview.py#L88-L136)、[recon/geometry.py](../../../python/synapse/recon/geometry/__init__.py)、[recon/delight.py](../../../python/synapse/recon/delight.py#L146) & [refine/stitcher.py](../../../python/synapse/refine/stitcher.py#L595) (`PR-114, PR-118, PR-119`)**：修复纯 Python OBJ 负索引；向量化 MC 活跃体素预筛（400x 提速）；修正 `delight` `np.roll` 边界回卷与过时 docstring。
4. **`[H8-P3-04]` [tools/data-inspector/](../../../tools/data-inspector/) 工程对齐 (`PR-122, PR-123`)**：修复 Electron `ResizeObserver` 竞态、Windows `file://` 两斜杠 bug、补全或明确折叠未实现 Tab、对接 `pnpm` 规范、`inspector_app.cc` `stdin` 超长行安全切分与支持独立 BCR 模块回退。
5. **`[H8-P3-05]` 治理文档版本编号与生态路由对齐 (`PR-127`)**：校准 `AGENTS.md`/`README.md` 版本号至 `v3.6`、`IMPLEMENTATION_PLAN.md` Wave 24–27 编号错位与生态层级路由对齐，说明本地 `uv.lock` 归属策略。

---

## 5. 本地未提交工作区风险提示（Uncommitted Workspace Advisory）

> 本次审计在只读核查仓库现状时，观测到主工作区存在若干未提交的 Colab/测试本地草稿改动（本次审计 PR **严格遵守零代码修改原则**，未触碰这些本地未提交文件），以下具体风险供合入前先行自查：

1. `workenv/google-colab/synapse_colab_workflow.ipynb:113`：Cell 4 的 JSON 字符串包含未转义换行符（`print(f"\n🚨...")` 解码成多行），运行该 Cell 将直接触发 Python `SyntaxError: unterminated f-string literal`。
2. `workenv/google-colab/vscode_tunnel.sh:17-26`：使用 `curl -sLk`（`--insecure` 禁用 TLS 证书校验）下载并执行远程二进制 `/usr/local/bin/code`，存在供应链中间人劫持风险（应去除 `-k`）。
3. `workenv/google-colab/tailscale.sh:80` & `synapse_colab_workflow.ipynb:95`：通过命令行参数传递 `TAILSCALE_AUTHKEY` 易在 `/proc` 进程表及执行日志中明文泄露（建议改用环境变量或交互输入）。
4. `tests/test_metrics.py:282`：将断言降级为 `assert isinstance(result.is_proxy_metric, bool)` 并遗漏 `assert result_proxy.is_proxy_metric is True`（建议显式 Mock `_build_r3d18_extractor` 测回退分支，而非削弱契约断言）。
5. 本地修改的 `AGENTS.md` 与 Skills 重写了历史测试数 `1,098`（实际当前为 `1,307` 项 / `59` 个测试文件），建议统一校准为实测值。

---

## 6. 缺陷 → Horizon-8 (`M29–M32`, `PR-112 ~ PR-129`) 闭环映射总表

| Horizon-8 PR | 里程碑 / 波次 | 主责包域（文件正交隔离） | 闭环修复的审计缺陷 ID |
| :--- | :---: | :--- | :--- |
| **[PR-112](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-112)** | M29 / W28 | `.github/workflows/ci.yml` · `pyproject.toml` · `tests/conftest.py` · `.gitignore` · `207` 文件类型清零 | `H8-P0-01`, `H8-P1-01`, `H8-P3-01` |
| **[PR-113](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-113)** | M29 / W28 | `fgpp/` · `tools/import_dag_guard.py` · `tools/wire_guard.py` · `schemas/` | `H8-P0-03`, `H8-P1-02`, `H8-P1-03`, `H8-P2-01` |
| **[PR-114](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-114)** | M29 / W28 | `style/` · `composite/` · `render_core/` · `telemetry.py` | `H8-P0-04`, `H8-P1-04`, `H8-P1-05`, `H8-P2-02`, `H8-P3-01`, `H8-P3-03` |
| **[PR-115](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-115)** | M29 / W28 | `cli/` · `pipeline/bundle_worker.py` · `pipeline/stages/` · RFC 0039 `JobBundle` 兼容层 | `H8-P0-02`, `H8-P1-06`, `H8-P1-07`, `H8-P2-03`, `H8-P2-04`, `H8-P3-02` |
| **[PR-116](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-116)** | M30 / W29 | `adapters/` · `adapters/lora/` · `bridge/` · `models/` · `generators/` · `viewer/` | `H8-P0-05`, `H8-P1-08`, `H8-P2-04`, `H8-P3-02` |
| **[PR-117](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-117)** | M30 / W29 | `training/` (`feature_store`, `losses`, `trainer`, `packager`) | `H8-P1-09`, `H8-P1-10`, `H8-P2-05` |
| **[PR-118](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-118)** | M30 / W29 | `recon/` & `recon/loop/` | `H8-P0-07`, `H8-P0-08`, `H8-P0-09`, `H8-P1-11`, `H8-P1-12`, `H8-P2-06`, `H8-P3-03` |
| **[PR-119](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-119)** | M30 / W29 | `refine/` (`pipeline`, `stitcher`, `tiled_executor`, `fusion`, `hand`, `face`, `roi_tracker`) | `H8-P0-11`, `H8-P1-13`, `H8-P2-07`, `H8-P3-03` |
| **[PR-120](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-120)** | M30 / W29 | `metrics/` (`roundtrip`, `temporal`, `image_quality`, `profiles`, `style_consistency`) | `H8-P0-08`, `H8-P0-10`, `H8-P1-14`, `H8-P2-08` |
| **[PR-121](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-121)** | M31 / W30 | `csrc/geometry/` · `scripts/build_native.py` · `recon/geometry.py` 伞环优化 | `H8-P1-15`, `H8-P3-03` |
| **[PR-122](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-122)** | M31 / W30 | `tools/data-inspector/native/` | `H8-P0-12`, `H8-P1-15`, `H8-P2-09`, `H8-P3-04` |
| **[PR-123](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-123)** | M31 / W30 | `tools/data-inspector/{electron,renderer,test}` | `H8-P0-12`, `H8-P1-16`, `H8-P3-04` |
| **[PR-124](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-124)** | M31 / W30 | `pipeline/bench_runner.py` · `bench/` · `inspect/` · `rsi/` · `cli/_rsi.py` & `_bench.py` | `H8-P0-06`, `H8-P0-13`, `H8-P1-17`, `H8-P2-10` |
| **[PR-125](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-125)** | M31 / W30 | `scripts/{prepare_datasets,prepare_benchmark_assets,colab_runner,run_cpu_verification}.py` · `assets/` · `workenv/google-colab/` | `H8-P0-14`, `H8-P1-18`, `H8-P2-11` |
| **[PR-126](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-126)** | M32 / W31 | `tests/`（快慢分层提速 · `clean_recon` 真重构 · Mock 隔离） | `H8-P1-01`, `H8-P1-18`, `H8-P2-11` |
| **[PR-127](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-127)** | M32 / W31 | `.agents/skills/` · `docs/AUDIT_AND_ARCHITECTURE.md` · `AGENTS.md` · `README.md` · `workenv/google-colab/io_spec.md` | `H8-P1-18`, `H8-P2-12`, `H8-P3-05` |
| **[PR-128](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-128)** | M32 / W31 | 跨仓契约黄金联调套件（`aether` · `render-farm`/`grand-studio` · `inspector` SHM） | `H8-P0-02`, `H8-P0-03`, `H8-P0-12`（跨仓闭环联调） |
| **[PR-129](../plans/PLAN-M29-M32-POST-H7-PRODUCTION-CLOSURE.md#pr-129)** | M32 / W31 | Horizon-8 全仓 12 项反作弊硬门禁实跑封板 & `Release v3.6` | 全域 12 项硬门禁终验封板 |
