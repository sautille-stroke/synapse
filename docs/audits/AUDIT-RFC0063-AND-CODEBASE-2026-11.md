# Synapse RFC-0063 与全库架构及工程质量深度审计报告（2026-11）

- **审计基线**: `b98c6e5` (`PR #51` / `main` @ `2026-11-14`)
- **前序审计基线**: [AUDIT-POST-REFACTOR-2026-11.md](AUDIT-POST-REFACTOR-2026-11.md) (`87e88b1`，Horizon 4 发起源头)
- **被审核心对象**: [RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md](../RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md)（`PR #48` / `717d40a`）、`PR #49`（`b7e5b96`）、`PR #50`（`9ce5037`）、`PR #51`（`b98c6e5`）及 Horizon 4（`M13–M16`）合入后的全量代码库
- **整改落地方案**: [PLAN-M17-M20-QUALITY-GOVERNANCE-AND-BENCHMARK-LANDING.md](../plans/PLAN-M17-M20-QUALITY-GOVERNANCE-AND-BENCHMARK-LANDING.md)（Horizon 5: `PR-58 ~ PR-73`）
- **审计模型标识**: `[gemini4]`

---

## 1. 执行摘要与 `ubuntu24` 实测门禁基线

### 1.1 总体结论

尽管 `PR #49` (`b7e5b96`) 在 [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) 中将 Horizon 4 全部 14 个 PR（`PR-44 ~ PR-57`）统一标记为 `✅ 已合入`，但对照当前 `main`（`b98c6e5`）源码、自动化守卫与 `ubuntu24` 实测结果进行的逐项核验表明：

1. **Horizon 4 尚未完整闭环**：仅 `PR-51`（`synapse.viewer` 安全硬化）达到完整合入标准；`PR-55`（`data-inspector` 真实 3D/EXR 视口与低拷贝 IPC，含 `inspector_app.cc` 标准 JSON 解析与线程安全 `join`）**完全未实施（`❌`）**，且 `tools/data-inspector/native` 的 Bazel 构建在 `ubuntu24` 上直接失败；其余 12 个 PR（`PR-44 ~ PR-50`、`PR-52 ~ PR-54`、`PR-56 ~ PR-57`）处于**部分合入·待收口（`⚠️`）**状态。
2. **RFC-0063 与基准数据集基础设施存在三重碰撞与立即崩溃缺陷**：
   - **生态编号三重冲突**：本地 `RFC-0061` / `RFC-0062` / `RFC-0063` 与生态总仓 `all-overview/rfcs/` 同号提案内容完全不同，且本地 `RFC-0063` 未满足 `GOVERNANCE.md` 与 L-Asset `RFC 0062` 规范；
   - **阈值与量纲偏移**：`RFC-0063 §5` 将 `TRACK_A_ASSET` 阈值误写为 Track-B，`temporal_warp_error` 纯 NumPy 实现仍计算 RMSE 而非 RFC 定义的 L1 $E_{\text{warp}}$，`roundtrip.py` Chamfer 与行业惯例差 $2\times$；
   - **Colab / 数据准备必崩**：`workenv/google-colab/setup_colab.sh` 在 `set -euo pipefail` 下必然因 `prepare_benchmark_assets.py` 跨目录调用 `relative_to(REPO_ROOT)` 触发 `ValueError` 崩溃，且 `#47`（`d0aa898`）重构 `@aether//geo` 后破坏了 `scripts/build_native.py` 无 Bazel 环境编译链路。
3. **静默回退与绕过尚未清零**：`cli/_recon.py` 与 `cli/_refine.py` 在缺失输入视频时仍合成 `np.random.default_rng(42)` 伪帧（且后者写入 `confidence=0.9` 直接触发 `TiledBypassExecutor` 跳过精修）；`vg_bridge.py` 与 `closed_loop_runner.py` 通过 `importlib.import_module` 绕过 `tools/import_dag_guard.py`；`model_evaluator.py` 缺失参考网格时 Fail-Open 跳过 Chamfer 门禁。

### 1.2 `ubuntu24` 实测检查基线（Commit `717d40a` / 代码与 `b98c6e5` 一致）

> 说明：以下实测均在云端主力机 `ubuntu24`（`~/work/dervgen/synapse`，`/root/venv-synapse`）执行；`#49/#50/#51`（至 `b98c6e5`）在 `python/`、`tests/`、`csrc/` 与 `tools/` 源码上仅修改 `scripts/prepare_benchmark_assets.py` 一行 f-string，`docs/IMPLEMENTATION_PLAN.md` 与 `workenv/google-colab/benchmarks.json` 为非测试运行时文件。

| 检查项 | 执行命令（`ubuntu24`） | 实测结果 | SLA / 预期 | 判定 |
| :--- | :--- | :--- | :--- | :---: |
| 快速单元套件 | `/root/venv-synapse/bin/pytest tests/ -m 'not slow' -q` | `1005 passed, 46 deselected in 151.59s`（二次预热 `143.57s`） | `<= 30s` | ❌ 超标 ~5x |
| Ruff 静态检查 | `/root/venv-synapse/bin/ruff check python/ tests/ scripts/ tools/` | `All checks passed!` | 0 error | ✅ 通过 |
| Mypy 主包默认配置 | `/root/venv-synapse/bin/mypy python/synapse` | `Success: no issues found in 133 source files` | 0 error | ⚠️ 放水（`warn_return_any = false`） |
| Mypy 严格返回检查 | `/root/venv-synapse/bin/mypy --warn-return-any python/synapse` | `Found 88 errors in 37 files (checked 133 source files)` | 0 error | ❌ 88 处 Any 泄漏 |
| Mypy 测试/脚本/工具 | `/root/venv-synapse/bin/mypy tests/ scripts/ tools/` | 失败：`scripts/run_cpu_verification.py` 双模块路径冲突 | 0 error | ❌ 模块映射冲突 |
| 跨语言线格式守卫 | `/root/venv-synapse/bin/python tools/wire_guard.py --check` | `RESULT: All checks passed`（`15` 通道 / `34 B/px` / 4 Schema） | 全覆盖校验 | ⚠️ 缺 PixelType/Style/bundle 覆盖 |
| 包级单向依赖守卫 | `/root/venv-synapse/bin/python tools/import_dag_guard.py --check` | `OK: import DAG is clean.` | 零动态绕过 | ⚠️ 被 `importlib` 与白名单绕过 |
| C++ 几何核 Bazel | `cd csrc/geometry && bazel test //... --config=linux` | `//:test_geometry (cached) PASSED in 0.0s` | Colab 亦可编 | ⚠️ 无 Bazel 兜底已坏 |
| C++ Inspector Bazel | `cd tools/data-inspector/native && bazel test //... --config=linux` | **失败**：`bcr.bazel.build` 获取 `spdlog/1.17.0` TLS 握手终止 | `--config=linux` 全绿 | ❌ 缺私服/离线构建配置 |

本轮全库 + RFC-0063 共确认 **18 项未闭环缺陷（5×P0 + 8×P1 + 5×P2）**，全部纳入 Horizon 5（`M17–M20`，`PR-58 ~ PR-73`）。

---

## 2. `RFC-0063` 与 `PR #48 / #50 / #51` 深度评审

### 2.1 生态三重 RFC 编号碰撞与治理规范违例

1. **跨仓 RFC 编号三重碰撞（最高治理风险）**：
   本地 `synapse/docs/` 与生态权威索引 `all-overview/rfcs/` 存在 3 组同号不同题冲突：

   | RFC 编号 | `all-overview/rfcs/`（生态权威 SSOT） | `synapse/docs/`（仓内本地文档） | 冲突影响 |
   | :--- | :--- | :--- | :--- |
   | **`RFC 0061`** | `0061-3d-and-runtime-compression-architecture.md`（3D 与运行时压缩） | `RFC-0061-LOCAL-REFINEMENT-AND-MULTI-MODEL-FUSION.md`（局部解剖精修） | 跨仓引用与 Agent 检索歧义 |
   | **`RFC 0062`** | `0062-lightweight-asset-distribution-and-dual-track-fallback.md`（L-Asset 资产分发） | `RFC-0062-INVERSE-RENDERING-AND-3D-NEURAL-CLOSED-LOOP.md`（可微逆向闭环） | 仓库规范与设计文档语义碰撞 |
   | **`RFC 0063`** | `0063-geometry-physics-and-runtime-foundational-toolkits-architecture.md`（基础几何物理工具链 CoACD/xatlas/Manifold/ozz） | `RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md`（测试、正确性与行业基准） | `#48` 新增文档继续加剧同号碰撞 |

2. **`GOVERNANCE.md` 规范违例（[RFC-0063](../RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md)）**：
   - 元数据状态为非法值 `Proposed (v1.0)`，缺失创建日期、作者/审核人、`Alternatives Considered`、`4-Repo Readiness Matrix` 与 `Landing Matrix`（仅在第 6 节给出了过时/错位的 Horizon 4 Mermaid 图）。
   - 第 6 节将 **10 个 GSO 黄金资产与 TRELLIS/Hunyuan3D 雷达图**归入已收口的 `PR-54 / PR-57`（`M15–M16`），将 `pytest <= 20s` 归于 `PR-48`、Quad 与 `.bazelignore` 归于 `PR-50`——不仅与 [PLAN-M13-M16](../plans/PLAN-M13-M16-POST-REFACTOR-REMEDIATION.md) 职责编号错位，且该 10 个 GSO 资产、真实 HF 下载器与 `<= 30s` 测试分层迄今均未完整落地。

3. **违背生态 `RFC 0062`（轻量级资产分发 L-Asset）**：
   `#48` 引入 [data/manifests/benchmark_assets_manifest.json](../../data/manifests/benchmark_assets_manifest.json)（`112` 个条目、`197.73 MB`）与独立脚本 `scripts/prepare_benchmark_assets.py`，未遵循 `all-overview/rfcs/0062` 要求的 `assets/assets_manifest.json` 统一清单结构、多源镜像级联（`.assets.lock` / OSS+GH+HF Fallback）与运行时双轨契约。

### 2.2 `RFC-0063 §3–§4` 核心机制三问审计（存在 · 接线 · 执行）

| 条款 | RFC-0063 声明 | 存在问 & 接线问 | 执行问与实测差距 |
| :--- | :--- | :--- | :--- |
| `§3.1` Tier-1 快速测试 | “默认测试套件 `<= 20s`” | `pyproject.toml` 已注册 `slow` 标记 | `ubuntu24` 实测 `pytest -m 'not slow'` 仍耗时 **151.59s / 143.57s**，主要瓶颈为 `tests/test_inspect_align.py` 等重型光栅化用例未标 `slow` |
| `§3.1` Tier-2 输出非退化检查 | “检验输出非零矩阵、方差 `Var(X) > 1e-4`” | 全局检索代码库，[telemetry.py](../../python/synapse/telemetry.py) 仅有 `TraceContext` 与 `MemoryForecaster` | **不存在**通用运行时 `Var(X) > 1e-4` 输出退化门禁函数，Adapter/Pipeline 未接线 |
| `§3.2` 异常类名与路径 | 点名 `QualityGateCircuitBreakerError` 与 `adapters/wan_adapter.py` | 真实类为 [errors.py:132,153](../../python/synapse/errors.py#L132-L171) `CircuitBreakerError` / `QualityGateError`（且构造不接受位置参数 `40`）；文件为 [wan.py](../../python/synapse/adapters/wan.py) | 规范文档自身符号名与文件路径失真 |
| `§4.1` 48 帧物理锚点与对数深度可逆性 | “强制每 48 帧插入真实物理锚点；Log-Depth 编解码误差 `< 1e-12`” | [schema.py:60](../../python/synapse/fgpp/schema.py#L60) 定义了 `DEFAULT_PHYSICAL_ANCHOR_INTERVAL_FRAMES = 48`；`float64` 单测可过 | 该常量**未接入 `WanAdapter` / `ConditionScheduler` 推理调度**；且 [wan.py:417,422](../../python/synapse/adapters/wan.py#L417-L422) 在入出均强制 `.astype(np.float32)`，导致端到端精度被截断在 ~`1e-7`，无法达到 `< 1e-12` |
| `§4.4` 2D-3D-4D 多视与闭合约束 | “2D VBench + 3D 闭合度 + 4D 物理动态一致性” | `roundtrip.py` / `model_evaluator.py` 存在 | 多项关键行业指标未实现或 fail-open（见 2.3） |

### 2.3 `RFC-0063 §5` 四轨道评测基准：阈值冲突与指标缺失

1. **Track 1（3D 模型生成，对标 TRELLIS / Hunyuan3D 2.0 / TripoSG）**：
   - `Chamfer Distance` 量纲与定义分歧：[roundtrip.py:69](../../python/synapse/metrics/roundtrip.py#L69) 实现为双侧均值 `mean(d_a2b + d_b2a) / 2.0 * 1000`（mm）；行业部分基准（PointNet/TRELLIS 标准点云协议）采用无 `/2` 之和或单位包围盒归一化比例，报告对比时缺少统一协议声明，且仅实现了 `f_score_1mm`，缺失 `RFC-0063` 要求的 `f_score_5mm` 与批量 `watertight_pct`。
   - [model_evaluator.py:152-178](../../python/synapse/inspect/model_evaluator.py#L152-L178) 的 `evaluate_profile_gates` **违背其 docstring `(fail-closed)`**：当未传 `reference_mesh` 导致 `chamfer_mm` 等指标缺席时，`if value is None: continue` 直接跳过检查；且 [model_evaluator.py:403](../../python/synapse/inspect/model_evaluator.py#L403) 使用非递归 `input_dir.iterdir()`，无法扫描 GSO/Objaverse 常见的 `<asset_id>/meshes/model.obj` 嵌套目录。
   - [bench_runner.py:191](../../python/synapse/pipeline/bench_runner.py#L191) 将 `scenario.asset.id` 直接传入 `load_mesh()`，导致全部 6 个 `humanoid_53` 相关场景加载失败（而测试用例通过仅因吞错或单测隔离）；更严重的是，[bench_runner.py:486-501](../../python/synapse/pipeline/bench_runner.py#L486-L501) 在计算几何/法线指标时用**生成帧亮度代理深度**与 **`gen_f * 2 - 1` 代理世界法线**直接同真实 GT 比较，不具备真实几何测量意义。

2. **Track 2（物理条件可控视频，对标 Wan 2.1 / CogVideoX / HunyuanVideo）**：
   - **文档阈值错位（SSOT 漂移）**：[RFC-0063:123](../../docs/RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md#L123) 声称 Track-B 门禁为 `FVD <= 150.0, E_warp <= 0.015, Foot-sliding <= 0.3 cm`——该组数字实际是 [profiles.py:89,91,100](../../python/synapse/metrics/profiles.py#L89-L100) 中的 **`TRACK_A_ASSET`**；真实 `TRACK_B_VIDEO`（[profiles.py:70,72,81](../../python/synapse/metrics/profiles.py#L70-L81)）为 `temporal_warp_error = 0.020, fvd = 300.0, foot_sliding_cm = 0.5`。
   - **$E_{\text{warp}}$ 数学定义冲突**：RFC-0063 将 $E_{\text{warp}}$ 定义为光流遮挡掩码 **L1** 误差，而 [temporal.py:53-99](../../python/synapse/metrics/temporal.py#L53-L99) `_fallback_temporal_warp_error` 计算的是像素 **RMSE**（L2 开方），且 `_warn_fallback`（43–45 行）为空操作。
   - **VBench 与 FVD 覆盖度严重不足**：`fvd_feature_distance`（[temporal.py:165](../../python/synapse/metrics/temporal.py#L165)）仅计算预先传入矩阵的高斯 Fréchet 距离，缺乏真实 I3D/VideoMAE 特征提取器；[prepare_benchmark_assets.py:46-70](../../scripts/prepare_benchmark_assets.py#L46-L70) 中的 `CANONICAL_VBENCH_PROMPTS` 仅覆盖 5 个维度共 10 条 Prompt，远低于 RFC-0063 宣称的 VBench 全 16 维评测协议。
   - **`#50` Human3.6M 注册与下载器断裂**：[benchmarks.json:133-146](../../workenv/google-colab/benchmarks.json#L133-L146) 新增 `ryushinn/Human3.6M`（`source_type: huggingface`），但 `prepare_benchmark_assets.py` 完全无 HuggingFace 拉取代码，也未声明许可门禁与离线校验契约。

3. **Track 3 & Track 4（逆向 3D / 闭环重渲染）**：
   - Track 3：`RFC-0063` 提及 `Depth AbsRel`，却遗漏 `synapse.metrics.profiles` 强制校验的核心字段 `depth_rmse_m`（Track-B `0.05` / Track-A `0.04`）。
   - Track 4：名称写作 `Border-ring ΔE00`，应规范为 SSOT 字段名 `border_ring_ciede2000`；更关键的是逆向管线核心存在算法退化与占位实现（见 §3.3）。

### 2.4 `#48` / `#50` / `#51` 数据准备脚本与 Colab 安装链路缺陷

直接审查 [workenv/google-colab/setup_colab.sh](../../workenv/google-colab/setup_colab.sh) 与 [scripts/prepare_benchmark_assets.py](../../scripts/prepare_benchmark_assets.py) 发现以下 P0 级阻断：

1. **Colab 安装脚本必崩链路（`#47` + `#48` 叠加）**：
   - `setup_colab.sh:5` 启用 `set -euo pipefail`；
   - `setup_colab.sh:44` 未安装 `[mesh]`（缺失 `trimesh`）与 `huggingface_hub`；
   - `setup_colab.sh:60` 调用 `scripts/build_native.py` 构建 C++ 几何扩展，而 `#47`（`d0aa898`）删除了 `csrc/geometry/src/mesh_checks.cc` 并令 `mesh_checks.h:11` 直接 `#include <aether/geo/mesh_checks.h>`，`build_native.py:141-148` 无 Bazel 兜底编译路径仍硬编码该已删除 `.cc` 文件且无法在无同级 `../aether` 的 Colab 容器中解析 `@aether//geo`；
   - `setup_colab.sh:108-109` 向 `prepare_benchmark_assets.py --target-dir` 传入 `/content/datasets/...` 与 `/content/drive/...`（位于 `REPO_ROOT` 之外），而 [prepare_benchmark_assets.py:97,120,158,182,218,240](../../scripts/prepare_benchmark_assets.py#L97) 均执行无异常捕获的 `dst.relative_to(REPO_ROOT)`——**在 Colab 环境下 100% 抛出 `ValueError` 中止安装**。
2. **`prepare_benchmark_assets.py` 数据来源与清单篡改缺陷**：
   - `F:/dervgen/open_data` Windows 私有盘符硬编码（第 43 行）；
   - 当本地无资产时（第 227–244、284–298 行），既不从 HuggingFace / Poly Haven / Stanford 网络下载，也不报错退出，而是合成 1 个 `unit_cube.obj` 并直接**覆盖仓库受控文件 `data/manifests/benchmark_assets_manifest.json`**（原有 `112` 个文件 / `197.73 MB` 清单记录会被静默破坏）；
   - `--tier`（4 级仅在 argparse 中解析但主逻辑未使用）与 `--force`（参数未使用）为空参数，且 `benchmarks.json`（7 个条目）的 ID/层级与脚本完全脱节。

---

## 3. Post-`M13–M16` 全库四维架构与工程质量审计 (`HEAD: b98c6e5`)

### 3.1 Horizon 4（`PR-44 ~ PR-57`）真实落地核查表

> 对照 `#49`（`b7e5b96`）所标记的 “14 个 PR 全部 `✅ 已合入`”，经逐文件源码级审计重新定级如下：

| PR | 原声明状态 | 审计实态 | 已验证的未闭环项 |
| :---: | :---: | :---: | :--- |
| `PR-44` | ✅ 已合入 | **⚠️ 部分合入** | `stages/` 清理较好，但 `cli/_recon.py:92-109` 与 `cli/_refine.py:63-92` 仍保留 `np.random` 合成帧回退；`cli/_bench.py` 门禁失败仍返回退出码 `0` |
| `PR-45` | ✅ 已合入 | **⚠️ 部分合入** | `cli/_refine.py:89` 默认伪造 `confidence=0.9` 直接命中 `TiledBypassExecutor`（`>0.8` 旁路）；`refine/pipeline.py:317-362` 未把像素块传入 `evaluate_refinement_gates` 且非 strict 模式复制原图 |
| `PR-46` | ✅ 已合入 | **⚠️ 部分合入** | `refine/fusion.py:184` 留存死表达式语句（`weights[-1] / ...`）；`temporal.py` 未切 L1 $E_{\text{warp}}$；`composite/blend.py` 未与 `refine/fusion.py` 彻底去重 |
| `PR-47` | ✅ 已合入 | **⚠️ 部分合入** | `recon/delight.py:198-200` 先转 `float32` 致 `uint8` 判断为死代码，`235` 行硬编码 `delta_e_residual=1.85`；`stages/recon.py:81` 传 `cameras=[]`；`mitsuba_optimizer.py:244-255` TV 梯度单边缺失 |
| `PR-48` | ✅ 已合入 | **⚠️ 部分合入** | `tools/import_dag_guard.py` 存在 3 处结构性盲区（不检 `importlib`、`len(parts)<2` 跳过根模块、`inspect->bench` 硬编码白名单），`vg_bridge.py` 与 `closed_loop_runner.py` 实际仍在动态环路 |
| `PR-49` | ✅ 已合入 | **⚠️ 部分合入** | `vg_bridge.py` 用 `importlib` 动态加载 `synapse.recon.backend`；`pyproject.toml` `[full]` 等 extra 同步与 Colab `[mesh]` 缺失仍需收口 |
| `PR-50` | ✅ 已合入 | **⚠️ 部分合入** | `#47`（`d0aa898`）迁移 `@aether//geo` 后破坏了 `scripts/build_native.py` 无 Bazel 兜底构建；`setup_colab.sh:60` 在 Colab 必败 |
| `PR-51` | ✅ 已合入 | **✅ 已合入** | `synapse.viewer` 静态 HTML 导出、回环强制与 `--serve-local` + `--json` 互斥校验均已到位 |
| `PR-52` | ✅ 已合入 | **⚠️ 部分合入** | `csrc/geometry/src/bindings.cc` 已实现 N-gon 扇形剖分与 GIL 释放；但 Python Oracle/Evaluator 与 C++ 之间的门禁指标（`GATE_PROFILES` 重复、`fail-open`）尚未收口 |
| `PR-53` | ✅ 已合入 | **⚠️ 部分合入** | `cli/__init__.py` 已拆分为领域子模块，但 `cli/_generate.py`（`main_data_gen` 无 `--json`）、`cli/_bench.py`（exit 0）、`cli/_rsi.py`（单分广播三轴）、`cli/_fgpp.py`（不读 EXR）仍有明显缺口 |
| `PR-54` | ✅ 已合入 | **⚠️ 部分合入** | `wire_guard.py` 已对齐 15 通道，但 `WorldNormal` 错允 `FLOAT`、`_RE_PIXEL_TYPE` 死正则、未实际解析 `@aether//style`、缺 `bundle.json` Schema；局部 RFC `0061–0063` 仍与总仓冲突 |
| **`PR-55`** | ✅ 已合入 | **❌ 未实施** | **`tools/data-inspector` 完全停留在原型脚手架**：C++ 侧 `std::cin.rdbuf(nullptr)` UB、FNV 伪波形/UV 渐变渲染、手写 string-find JSON 解析、Bazel 缺私服失败；Electron 侧 `SourceEntry` 字典崩溃、IPC 传整帧数组、`currentView` 死状态 |
| `PR-56` | ✅ 已合入 | **⚠️ 部分合入** | `composite/blend.py` 仍存在 ~150+ 行未彻底收敛的金字塔/形态学代码；`generators/dataset.py` 缺少 `VideoWriter` 异常释放保护 |
| `PR-57` | ✅ 已合入 | **⚠️ 部分合入** | 快测试仍需 **143.57s–151.59s**（非 `< 30s`）；`mypy` 因 `warn_return_any = false` 掩盖 **88 处错误**；`.agents/skills/synapse-overview/SKILL.md:68` 仍写着不存在的 `QualityGateCircuitBreakerError` 与过时的 “C++ 自包含” 描述 |

### 3.2 架构与跨仓集成问题 (`ARCH`)

1. **`AUD-P0-02` / `build_native.py` 与 `#47` 构建撕裂（P0）**：
   - `d0aa898`（`#47`）正确地将 `csrc/geometry` 的几何校验逻辑去重委托给 `@aether//geo`（`csrc/geometry/include/synapse/geometry/mesh_checks.h:11` 直接 `#include <aether/geo/mesh_checks.h>`）并删除了 `csrc/geometry/src/mesh_checks.cc`。
   - 但 [scripts/build_native.py:141-148](../../scripts/build_native.py#L141-L148) 的非 Bazel 直编 fallback 依然把 `csrc/geometry/src/mesh_checks.cc` 列在源文件列表，且 `_find_aether_math_include`（126–135 行）未发现/挂载 `aether/geo/include`，同时孤立 Google Colab 容器并无同级 `../../../aether` 源码树，导致 Colab 及无 Bazel 开发环境无法构建 `_c_geometry`。
2. **`AUD-P1-02` / `import_dag_guard.py` 绕过与静默环路（P1）**：
   - [tools/import_dag_guard.py:58-66,97,201](../../tools/import_dag_guard.py#L58-L97) 仅分析静态 `ast.Import` / `ast.ImportFrom`，跳过 `len(parts) < 2` 的包顶层（如 `python/synapse/__init__.py`），且内置白名单 `{"cli", "pipeline", "viewer", "inspect"} -> "bench"`。
   - [bridge/vg_bridge.py:181-215](../../python/synapse/bridge/vg_bridge.py#L181-L215) 与 [recon/loop/closed_loop_runner.py:226-229,352-357](../../python/synapse/recon/loop/closed_loop_runner.py#L226-L229) 通过 `importlib.import_module("synapse.recon.backend")` 与 `importlib.import_module("synapse.refine.stitcher")` / `"synapse.refine.types"` 运行时动态引入逆向或跨域模块，在静态守卫全绿的表象下维持隐藏依赖耦合。
3. **`AUD-P1-03` / `wire_guard.py` 覆盖盲区与 Schema 错位（P1）**：
   - [tools/wire_guard.py:118-120](../../tools/wire_guard.py#L118-L120) `_EXPECTED_PIXEL_TYPES` 允许 `WorldNormal.{X,Y,Z}` 使用 `FLOAT`（与 `FGPP_CHANNEL_SPECS` 锁定的 `HALF` / `34 B/px` 矛盾）；第 373 行 `_RE_PIXEL_TYPE` 已定义却未使用（并未校验每个通道的实际 C++ `PixelType`）；第 185–220 行 `_check_style_profile` 未解析 `@aether//style` C++ 头文件；第 238–241 行以 `inspection.v1.0.schema.json` 之名校验 `const "2.0.0"`；另缺 `data-inspector` 导出的 `bundle.json` 跨端 Schema。
4. **`AUD-P1-06` / `GATE_PROFILES` 双处真值源冲突（P1）**：
   - [inspect/model_evaluator.py:36-62](../../python/synapse/inspect/model_evaluator.py#L36-L62) 硬编码独立的 `GATE_PROFILES`，未从 `synapse.metrics.profiles`（受 `test_profiles_doc_sync.py` 保护的唯一真值源）派生，且缺失 `f_score_1mm`、`lpips`、`ciede2000` 等维度。

### 3.3 防静默失败与核心算法正确性缺陷 (`CORRECTNESS`)

1. **`AUD-P0-01` / CLI 逆向与精修伪输入 + 置信度旁路（P0）**：
   - [cli/_recon.py:92-109](../../python/synapse/cli/_recon.py#L92-L109) 与 [cli/_refine.py:63-92](../../python/synapse/cli/_refine.py#L63-L92)：当未提供 `--video` 或路径不存在时，未按约定抛出 `AssetNotFoundError (20)`，反而通过 `np.random.default_rng(42)` 合成随机噪声视频帧执行并返回伪结果；更隐蔽的是 `cli/_refine.py:89` 将合成 ROI 设为 `confidence=0.9`，直接命中 [tiled_executor.py:88-97](../../python/synapse/refine/tiled_executor.py#L88-L97) 的 `roi.confidence > 0.8` 旁路分支，彻底跳过精修。
2. **`AUD-P0-04` / 精修门禁与基准评测假阳（P0）**：
   - [refine/pipeline.py:317-362](../../python/synapse/refine/pipeline.py#L317-L362)：从未将 `original_patch` / `refined_patch` 传给 `evaluate_refinement_gates`，且在 `self.strict == False` 时直接复制原始输入帧作输出。
   - [refine/fusion.py:184](../../python/synapse/refine/fusion.py#L184)：存在无赋值的裸除法 dead statement（`weights[-1] / max(float(np.sum(weights)), 1e-8)`）。
   - [inspect/model_evaluator.py:152-178,323-325,351-352,427-428](../../python/synapse/inspect/model_evaluator.py#L152-L178)：缺失参考模型时跳过 gate 评估（fail-open），捕获 `Exception` 后静默丢弃渲染截图与指标。
   - [pipeline/bench_runner.py:191,486-501](../../python/synapse/pipeline/bench_runner.py#L191)：`humanoid_53` 加载失败及亮度/颜色通道充当深度/法线代理。
3. **`AUD-P1-01` / 逆向去光照、微距优化器与闭环核心缺陷（P1）**：
   - [recon/delight.py:198-200,235](../../python/synapse/recon/delight.py#L198-L200)：在 `if rad.dtype == np.uint8:` 之前已执行 `.astype(np.float32)`，导致 `uint8` 分支成为死代码（`[0..255]` 输入直接饱和裁剪为 `1.0`），且第 235 行将 `delta_e_residual=1.85` 写死为常量；
   - [pipeline/stages/recon.py:81](../../python/synapse/pipeline/stages/recon.py#L81)：调用 `delight_multiview_albedo(frames=frames, cameras=[], mesh=result.mesh)` 时硬编码空相机列表 `cameras=[]`；
   - [recon/loop/mitsuba_optimizer.py:244-255,346-350](../../python/synapse/recon/loop/mitsuba_optimizer.py#L244-L255)：Total Variation 正则化仅在 `[:-1, :]` 和 `[:, :-1]` 累加前向差分梯度，**未在 `[1:, :]` / `[:, 1:]` 减去对应对偶项**，破坏离散散度对称性与像素均值守恒；同时 `MitsubaInverseOptimizer.optimize` 仍仅为校验 import 的空占位；
   - [recon/loop/closed_loop_runner.py:189,268-296](../../python/synapse/recon/loop/closed_loop_runner.py#L189)：`_build_synthetic_conditioning` 以 `(1 - luminance)` 和恒定 `(0, 0, 1)` 法线构造伪 FGPP 条件；`step4` 仅按固定光源 `(0,0,1)` 对最近邻 UV 做简化 Lambert 采样；
   - [recon/loop/forward_guide.py:112-117,170-180](../../python/synapse/recon/loop/forward_guide.py#L112-L117)：CPU mock 生成 `0.5` 灰帧 + 5500K 暖偏色调；另当真实生产实现仅返回部分帧而 `n_preroll > total_locked` 时存在切片假设风险（需显式形状断言）。
4. **`AUD-P1-05` / CLI 确定性退出码与协议一致性缺陷（P1）**：
   - [cli/_bench.py:46-112](../../python/synapse/cli/_bench.py#L46-L112)：`_cmd_bench_run` 在 `gate_verdict.passed == False` 时依然返回退出码 `0`（应触发 `QualityGateError` / `40` 或提供显式 `--strict` 门禁模式）；`_cmd_bench_export_fgpp:174` 直接传 `scenario.asset.id` 给 `load_mesh`；
   - [cli/_rsi.py:171-180](../../python/synapse/cli/_rsi.py#L171-L180)：`_cmd_rsi_gate_promote` 将单一标量 `baseline_overall` / `candidate_overall` 同时复制给 `overall_score`、`geom_score`、`temporal_score`，令多轴单调晋升退化为重言式；
   - [cli/_fgpp.py:19-73](../../python/synapse/cli/_fgpp.py#L19-L73)：`synapse fgpp validate` 仅解析 `metadata.json`（`schema_version` 与 `num_frames`），不打开任何 `.exr` 验证 15 通道名、像素类型或有限值；
   - [cli/_generate.py:20-100](../../python/synapse/cli/_generate.py#L20-L100)：独立入口 `main_data_gen` 缺失 `--json` 机器可读输出与统一信封；
   - [scripts/colab_runner.py:205](../../scripts/colab_runner.py#L205)：未将 `REPO_ROOT` 注入 `sys.path` 导致独立运行报 `ModuleNotFoundError: No module named 'scripts'`，`EXPERIMENTS` 注册表遗漏 `e3`，且默认执行 CPU mock 流程；[scripts/prepare_datasets.py:321,327,333,345](../../scripts/prepare_datasets.py#L321-L345) 默认路径泄露 `/root/work/...`。

### 3.4 可视化检查器 `tools/data-inspector` (`PR-55` 遗留欠账)

1. **`AUD-P0-05` / C++ Native SHM Server 内存安全、构建与假渲染（P0）**：
   - **C++20 标准未定义行为 / 死锁**：[inspector_app.cc:109-114](../../tools/data-inspector/native/inspector_app.cc#L109-L114) 在工作线程仍在 `std::getline(std::cin, line)` 阻塞读取期间，主线程直接调用 `std::cin.rdbuf(nullptr);` 企图打断阻塞读，再执行 `stdin_thread.join()`，触发并发流竞争与退出挂死风险；
   - **手写字符串解析 JSON**：`inspector_app.cc:46-93` 通过 `line.find(...)` 手写抽取 SEP 字段；
   - **假渲染画布**：`inspector_app.cc:268-348`（`renderViewport`）当前仅按 `mesh_name` 哈希输出 `fnv1aHash` 噪点或 UV 渐变图，未加载真实 `InspectionBundle` 的网格或 `.exr` AOV；
   - **Bazel 构建断裂与配置重复**：`tools/data-inspector/native/.bazelrc` 缺少 `bcr.dervgen.com` 私服/本地 vendor 回退，导致 `ubuntu24` 访问外部 `bcr.bazel.build` 获取 `spdlog/1.17.0` TLS 失败；且 `native/.bazelrc` 与 `native/BUILD.bazel` `COMMON_COPTS` 重复声明 `-fno-exceptions -fno-rtti`。
2. **`AUD-P1-04` / Electron 壳契约错配与高拷贝 IPC（P1）**：
   - **Bundle 数据契约崩溃**：[bundle-manager.js:61,83](../../tools/data-inspector/electron/bundle-manager.js#L61-L83) 直接对 `manifest.sources`（Python `export_inspection_bundle` 写出的是对象列表 `SourceEntry`，而非纯字符串路径）拼接 `path.join(bundleDir, sourceRel)`，导致运行时 `TypeError`；
   - **非共享内存整帧 IPC**：[main.js:128-139](../../tools/data-inspector/electron/main.js#L128-L139) 与 [renderer/src/app.js:186-193](../../tools/data-inspector/renderer/src/app.js#L186-L193) 通过 Electron JSON IPC 回传整帧数组而非真实 POSIX SHM 零拷贝消费，`currentView` 切换状态未驱动 C++ 通道切换。

### 3.5 类型门禁、测试性能与技术债务 (`ENGINEERING`)

1. **`AUD-P1-07` / Mypy 与 Test 分层门禁未收紧（P1）**：
   - [pyproject.toml:83](../../pyproject.toml#L83) `warn_return_any = false` 掩盖了 **`37` 个文件中 `88` 处**返回 `Any` 的类型泄漏（已由 `ubuntu24` 实测确认，含 `fgpp/schema.py`、`adapters/wan.py`、`inspect/` 等核心模块）；
   - `mypy tests/ scripts/ tools/` 在 `scripts/run_cpu_verification.py` 上因模块名解析重名而报错；
   - `pytest -m 'not slow'` 在 `ubuntu24` 耗时 `143.57s–151.59s`（主要因未标 `slow` 的高分辨率光栅化、多视角齐次优化与跨 Schema 循环测试仍在 fast tier 运行）。
2. **`AUD-P2-01..05` / 代码去重、资源安全与文档同步（P2）**：
   - [composite/blend.py](../../python/synapse/composite/blend.py) 仍保留 ~150+ 行局部形态学与金字塔融合实现，与 `synapse/refine/fusion.py` 未彻底收敛；
   - [generators/dataset.py:190-296](../../python/synapse/generators/dataset.py#L190-L296) 创建 3 个 `cv2.VideoWriter` 句柄未包裹 `try ... finally: release()`；
   - [pipeline/bench_runner.py](../../python/synapse/pipeline/bench_runner.py) 等报告/导出路径及 `scripts/run_roundtrip_verification.py:278` 留有 `0.5` 灰图渲染占位；
   - [.agents/skills/synapse-overview/SKILL.md:68](../../.agents/skills/synapse-overview/SKILL.md#L68) 仍残留已删除的错误异常名 `QualityGateCircuitBreakerError` 与过时的 "`csrc/geometry` 自包含"表述。

---

## 4. 全量缺陷台账总表与 M17–M20（PR-58 ~ PR-73）映射

| 缺陷 ID | 严重度 | 类别 | 精确位置（文件 / 符号） | 核心问题摘要 | 目标 PR |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **`AUD-P0-01`** | P0 | CORRECTNESS | `cli/_recon.py:92-109`, `cli/_refine.py:63-92`, `tiled_executor.py:88-97` | 缺输入视频时伪造 `np.random` 帧，且 `confidence=0.9` 直接旁路精修 | `PR-60` |
| **`AUD-P0-02`** | P0 | ARCH | `scripts/build_native.py:141-148`, `workenv/google-colab/setup_colab.sh:60`, `#47` | `#47` 删 `mesh_checks.cc` 依赖 `@aether//geo` 后破坏无 Bazel/Colab 原生扩展编译 | `PR-62` |
| **`AUD-P0-03`** | P0 | BENCH / COLAB | `scripts/prepare_benchmark_assets.py`, `setup_colab.sh:44,108-109`, `benchmarks.json` | Colab `relative_to(REPO_ROOT)` 必崩、缺 HF/HTTP 下载器、覆盖受控清单、违背生态 `RFC 0062` | `PR-66` |
| **`AUD-P0-04`** | P0 | CORRECTNESS | `refine/pipeline.py:317-362`, `refine/fusion.py:184`, `model_evaluator.py:152-178`, `bench_runner.py:191,486-501` | `evaluate_refinement_gates` 未传真实像素块；`model_evaluator` 无参考网格 Fail-Open；`bench_runner` 6 场景资产失败 + 伪通道比较 | `PR-60`, `PR-68` |
| **`AUD-P0-05`** | P0 | TOOLING | `tools/data-inspector/native/inspector_app.cc:46-114,268-348`, `native/.bazelrc` | `std::cin.rdbuf(nullptr)` UB、伪渲染画布、手写 JSON、`bcr.bazel.build` 获取 `spdlog` 失败（`PR-55` 遗留） | `PR-64` |
| **`AUD-P1-01`** | P1 | CORRECTNESS | `recon/delight.py:198-200,235`, `stages/recon.py:81`, `mitsuba_optimizer.py:244-255`, `forward_guide.py` | `delight` 死 `uint8` 分支与硬编码 $\Delta E=1.85$；`cameras=[]`；单边 TV 梯度；`Mitsuba` 占位与伪条件通道 | `PR-61` |
| **`AUD-P1-02`** | P1 | ARCH | `tools/import_dag_guard.py:58-97,201`, `vg_bridge.py:181-215`, `closed_loop_runner.py:226-229` | 通过 `importlib.import_module` 绕过 AST DAG 守卫；跳过根模块且白名单过宽 | `PR-63` |
| **`AUD-P1-03`** | P1 | WIRE / GOV | `tools/wire_guard.py:118-120,185-241,373`, `docs/RFC-0061..0063` | `wire_guard` 放行 `WorldNormal=FLOAT`、未解析 `@aether//style`、Schema 版本名错位；生态三重 RFC 碰撞 | `PR-58`, `PR-59` |
| **`AUD-P1-04`** | P1 | TOOLING | `tools/data-inspector/electron/{bundle-manager.js,main.js}`, `renderer/src/app.js` | `SourceEntry` 对象拼接崩溃、IPC 回传整帧数组非 SHM、`currentView` 无效果 | `PR-65` |
| **`AUD-P1-05`** | P1 | CLI | `cli/{_bench.py,_rsi.py,_fgpp.py,_generate.py}`, `scripts/colab_runner.py:205` | 门禁失败 exit 0；RSI 三轴同值；`fgpp validate` 不开 EXR；`main_data_gen` 缺 `--json`；`colab_runner` `sys.path` 崩溃与缺 `e3` | `PR-69`, `PR-71` |
| **`AUD-P1-06`** | P1 | METRICS | `RFC-0063:123`, `metrics/{temporal.py,roundtrip.py}`, `model_evaluator.py:36-62,403` | RFC 错引 Track-A 作 Track-B；$E_{\text{warp}}$ 用 RMSE 非 L1；Chamfer `/2` 歧义与缺 `f_score_5mm`；独立 `GATE_PROFILES`；非递归 `iterdir` | `PR-67`, `PR-68` |
| **`AUD-P1-07`** | P1 | QUALITY | `pyproject.toml:83`, `tests/`, `scripts/run_cpu_verification.py` | 88 处 `warn_return_any` 报错；`mypy tests/ scripts/ tools/` 模块名冲突；fast pytest 143.6s–151.6s（~5x 超标） | `PR-72` |
| **`AUD-P1-08`** | P1 | ADAPTER / GATE | `adapters/wan.py:417,422`, `fgpp/schema.py:60`, `telemetry.py` | 48 帧物理锚点未接入调度；`float32` 提前截断破坏 `1e-12` 对数深度可逆性；缺失 `Var(X) > 1e-4` 运行时退化门禁 | `PR-70` |
| **`AUD-P2-01`** | P2 | DEDUP | `composite/blend.py`, `refine/fusion.py` | 未迁移的金字塔与形态学实现重复，缺少跨模块共享原语或统一委派 | `PR-61`, `PR-71` |
| **`AUD-P2-02`** | P2 | RELIABILITY | `generators/dataset.py:190-296` | 3 路 `cv2.VideoWriter` 无 `try/finally` 释放保护，异常时破坏 MP4 容器 | `PR-71` |
| **`AUD-P2-03`** | P2 | BENCH / SCRIPT | `scripts/{run_roundtrip_verification.py:278,prepare_datasets.py:321}`, `bench/` | `0.5` 灰图占位评估、`prepare_datasets.py` 泄露 `/root/work/...` 硬编码路径 | `PR-68`, `PR-71` |
| **`AUD-P2-04`** | P2 | BENCH | `prepare_benchmark_assets.py:46-70`, `temporal.py:165` | `VBench` 仅有 5 维/10 条，缺 16 维完整套件与可接入真视频的 I3D/FVD 特征抽取桥接 | `PR-66`, `PR-67` |
| **`AUD-P2-05`** | P2 | DOCS / SKILLS | `.agents/skills/synapse-overview/SKILL.md:68`, `docs/IMPLEMENTATION_PLAN.md` | 错写 `QualityGateCircuitBreakerError`、过时“自包含 C++”描述；`#49` 对 `PR-44~57` 误标全绿 | 本文 + `PR-58` + `PR-73` |
