# Synapse 整体架构设计深度评审与跨文档对齐决议 (Design Review & Cross-Spec Alignment)

- **文档状态**: `Approved Review & Alignment Baseline (v2.0)`
- **评审日期**: 2026-11-11
- **操作模型标识**: `[gemini4]`
- **评审范围**:
  - 仓内全部 7 篇架构与专项设计文档（`AUDIT_AND_ARCHITECTURE.md`、`DESIGN_SYSTEM_ENHANCEMENT.md`、`DESIGN_INDUSTRY_GAP_AND_COMPENSATION.md`、`DESIGN_LORA_PIPELINE.md`、`DESIGN_3D_INSPECTION_AND_REFINEMENT.md`、`DESIGN_RSI_FLYWHEEL.md`、`DESIGN_DATA_INSPECTOR.md`）及原 `IMPLEMENTATION_PLAN.md (v1.2)`；
  - 仓内现存 `v0.1.0` 代码基线（`python/synapse/**`、`pyproject.toml`、`tests/**`）；
  - 同级跨仓契约真值源：`/Volumes/work/dervgen/aether` (`@aether//fgpp`)、`/Volumes/work/dervgen/vg-core`、`/Volumes/work/dervgen/fulgur` 及 `/Volumes/work/dervgen/all-overview`（RFC 0039 / 0043 / 0044 / 0048 / 0050 / 0052 / 0053 / 0058）。
- **关联主实施计划**: [docs/IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (`v2.0`, M1–M8 共 30 个原子 PR)

---

## 1. 整体设计评审结论 (Executive Summary)

### 1.1 架构优势肯定
`synapse` 的顶层定位与技术路线在工业界具有极强的差异化竞争力：
1. **定位清晰、严守生态红线**：将重型 PyTorch / Diffusers / 视频生成适配与跨模态评测收敛在 Layer 2 `synapse`（无状态 CLI Worker + Python SDK），既保护了 `vg-core` 纯 C++20/CUDA 算子的零依赖纯洁性，又遵守了 **Rule 11（服务端零 Python，DAG 调度归属 Go `render-farm` / `narrative-rail`）** 与 **NG-5（桌面端零重型 Python 依赖）**。
2. **用确定性物理底座代偿通用底模短板**：通过 `fulgur`/`caelum-core` 15 通道真值 FGPP、48 帧物理强锚点、双曲正切软肩 ACEScg 逆色调映射、Visual Jev 微距校准置信度以及物理验证器驱动的递归自我提升（RSI），形成了逻辑严密的正逆向闭环。

### 1.2 核心问题诊断：多篇专项设计并行合入引发“跨文档分裂与落地断层”
由于 PR #3 至 PR #8 的 6 篇专项设计规范在短时间内独立起草并合入，对照底层 C++ 真值头文件与现有仓库代码逐行交叉核验后，发现 **10 项必须立即纠偏的设计冲突与工程隐患（`3×P0` 阻塞性契约冲突 + `5×P1` 算法/边界断层 + `2×P2` 规范与卫生问题）**：
- 若不先对齐文档即启动编码，将直接导致 **FGPP 跨语言字节校验 100% 报错、多 Agent 并行开发文件路径撞车、4 大新增核心模块无落地 PR 承接、以及隔离虚拟环境构建因隐式依赖崩溃**。

本报告在 §2 逐项给出评审发现与修正决议，并在 §3 锁死全仓唯一的 **目录树、CLI 语法树、`JobBundle` 阶段契约与三级质量门禁阈值表**，直接指导 `IMPLEMENTATION_PLAN.md (v2.0)` 及其 5 份子计划拆解。

---

## 2. 十大评审发现与架构对齐决议 (`RV-01` ~ `RV-10`)

### 🔴 P0 级：阻塞性契约错误、模块路径分裂与落地计划断层

#### [RV-01 | P0] `DESIGN_SYSTEM_ENHANCEMENT.md` §4.3 出现幻觉第 16 通道与通道后缀错误
- **问题定位**：`DESIGN_SYSTEM_ENHANCEMENT.md` 原 §4.3 声称“严格为 15 个标量通道、34 Bytes/pixel”，但列出的通道清单实际有 **16 个**，凭空多出了 `@aether//fgpp` 中不存在的 **`ResidualOcclusion.R`**，且将粗糙度/金属度误写为 **`RoughnessMetallic.B`, `RoughnessMetallic.G`**！
- **真值核对**：对照 Layer 0 唯一事实源 `aether/fgpp/include/aether/fgpp/fgpp_types.h`（第 78–105 行 `getAlphabeticalEXRChannelSpecs()`），权威的 15 个 ASCII 字典序标量通道为：
  `BaseColor.B`, `BaseColor.G`, `BaseColor.R`, `CameraLinearDepth.Z` (`FLOAT32`), `Cryptomatte.ID` (`UINT32`, `allow_spatial_filtering=false`), `DirectRadiance.B`, `DirectRadiance.G`, `DirectRadiance.R`, `MotionVector.X`, `MotionVector.Y`, `RoughnessMetallic.G`, `RoughnessMetallic.R`, `WorldNormal.X`, `WorldNormal.Y`, `WorldNormal.Z`（共 $1\times 4 + 1\times 4 + 13\times 2 = 34\text{ B/px}$；环境接触闭塞已按 `DESIGN-0044-01` §4.1 烘焙进 `DirectRadiance.RGB`）。
- **对齐决议**：立即修正 `DESIGN_SYSTEM_ENHANCEMENT.md` §4.3 通道清单；在 `PR-02` 中以解析 `fgpp_types.h` 的自动化断言作为唯一准绳。

#### [RV-02 | P0] 多篇设计文档的 Python 模块路径与 CLI 入口点“各自命名、相互冲突”
- **问题定位**：不同设计文档对同一功能定义了完全不同的文件路径与 CLI 命令：
  1. **FGPP 契约文件**：`IMPLEMENTATION_PLAN` 定为 `fgpp/schema.py`，而 `DESIGN_SYSTEM_ENHANCEMENT` §9 定为 `fgpp/wire_schema.py` 与 `fgpp/reader.py`；
  2. **流式滑窗与退火**：`IMPLEMENTATION_PLAN` 定为 `adapters/scheduler.py`，而 `DESIGN_SYSTEM_ENHANCEMENT` §9 拆至 `pipeline/streaming.py` + `composite/annealing.py`，甚至将运行时的 `DiscontinuityDetector` 误放入评测包 `bench/conditions.py`；
  3. **色彩与融合**：`aces_color.py` vs `color_space.py`；`composite/blend.py` vs `composite/blending.py`；`pipeline/bundle_worker.py` vs `pipeline/worker.py`；
  4. **CLI 入口分裂**：`PR-04` 规定收敛为 `synapse` 与 `synapse-bench` 双入口，但 `DESIGN_LORA_PIPELINE` 新增了第 3 个顶层命令 `synapse-lora`，而 `DESIGN_DATA_INSPECTOR` §7.1 调用了未在 `PR-04`/`PR-14a` 定义的 `synapse-bench evaluate --bundle <path>`。
- **对齐决议**：
  - 全仓强制统一至 **§3.1 权威模块目录树** 与 **§3.2 统一双入口 CLI 语法树**（仅保留 `synapse` 与 `synapse-bench` 两个顶层可执行入口；LoRA、3D 质检、RSI 统一挂载为 `synapse lora ...`、`synapse inspect ...`、`synapse rsi ...` 子命令；补齐 `synapse-bench evaluate --bundle` 子命令对接 Data Inspector）。

#### [RV-03 | P0] 四大专项设计悬空（无落地 PR）且 `Data Inspector` 存在 Bazel 构建死结
- **问题定位**：
  1. 原 `IMPLEMENTATION_PLAN.md (v1.2)` 仅规划了 M1–M4（`PR-01` ~ `PR-15`），导致后合入的 **LoRA 微调流水线**、**3D 智能质检与精修飞轮**、**物理驱动 RSI 自演进引擎**、**Electron Data Inspector** 四大模块完全没有对应的原子 PR、依赖拓扑与验收门禁；
  2. `DESIGN_DATA_INSPECTOR.md` §5.1 编写了 `synapse/tools/data-inspector/native/BUILD.bazel` 并直接依赖 `"//engine/rhi"` 与 `"//course/common:headless"`。但 `synapse` 是纯 Python 仓库（根级无 `MODULE.bazel`），`//engine/rhi` 在跨仓视角下必须写为 `@fulgur//engine/rhi`，且 `@fulgur//course/common:headless` 属于课程内部目标而非稳定公开契约。
- **对齐决议**：
  - 将实施总计划升级为 **`v2.0` 双地平线、8 阶段（M1–M8）、30 个原子 PR（`PR-01` ~ `PR-29`，含 `14a/14b`）**，并按领域拆分为 `docs/plans/` 下的 **5 份精细化子计划**；
  - `tools/data-inspector/native/` 采用独立子工作区（自带 `MODULE.bazel`，通过 `local_path_override` 引用同级 `../../../../fulgur` 与 `../../../../aether`），仅依赖 `@fulgur` 公开模块（`@fulgur//engine/rhi`, `@fulgur//engine/renderer`, `@fulgur//engine/asset`, `@fulgur//engine/animation`, `@fulgur//engine/physics`, `@aether//pal`），内建轻量 `InspectorShmServer` 而不耦合 `fulgur/course/` 私有目录。

---

### 🟠 P1 级：跨仓职责重叠、因果潜空间相位陷阱与现存代码隐式依赖

#### [RV-04 | P1] `synapse.inspect` 与 `vg-core`（NG-3）及 `RFC 0053`（3D 修复评测）的职责去重定界
- **问题定位**：
  - `RFC 0053`（已在 `grand-studio`、`forgekit`、`asset-forge`、`scene-forge` 落地）已实现 `RepairSandbox` 无头修复沙箱、缺陷注入与 8 视角摄影棚评测；
  - `vg-core` 已承载 C++20/CUDA 流形拓扑检查、BVH 距离、四边面重拓扑、物理去光照与 Humanoid-53 足底零滑移 IK。
  - 若 `synapse.inspect` 在 Python 中重新实现网格流形检测、BVH 点面距离或重拓扑循环，将严重违反 **NG-3（严禁在 Python 中重写密集几何/IK 循环）**。
- **对齐决议**：精准收敛 `synapse.inspect` 边界——**只做神经感知判别、微距相机规划与强化学习样本提纯**：
  1. 姿态粗检索（DINOv2 球面特征匹配）+ 调用 `vg.recon` 精配准；
  2. 多通道残差热力图计算 + 虚拟微距相机（Macro Camera）3D 视锥规划；
  3. **Visual Jev 共享视觉上下文推理客户端**（从 vLLM / 本地端点直读校准 Logits 置信度）；
  4. 通过 `synapse.bridge.vg_bridge` / `foundary_client` 编排 `Trellis-Refine` 与 `vg-core` C++ 几何校验，禁止在 Python 手写半边网格或 IK 算子；
  5. 解析 `forgekit` 导出的 RFC 0020/0021 操作日志，转化为 DPO 偏好对与 PRM 步骤级 JSONL 训练数据。

#### [RV-05 | P1] `Wan 2.1` 3D Causal VAE 首帧非对称压缩在 `W=97, S=64, Δ=33` 滑窗中的相位偏移陷阱
- **问题定位**：`DESIGN_SYSTEM_ENHANCEMENT.md` §2.1 设计了窗口 $W=97$（$\tau=25$）、步长 $S=64$、重叠 $\Delta=33$（帧 `64..96`，$\tau_{\text{overlap}}=9$）。但在 `Wan 2.1` 的 $4\times 8\times 8$ 3D 因果 VAE 中：
  - 窗口 0（帧 `0..96`）：$\tau=0$ 单独编码第 `0` 帧（1 帧），$\tau=1..24$ 每步编码 4 帧（`[1..4], ..., [61..64], [65..68], ..., [93..96]`）。其中 $\tau=16$ 对应 `[61..64]`，$\tau=17..24$ 对应 `[65..96]`；
  - 窗口 1（帧 `64..160`）：若直接从第 `64` 帧独立编码，$\tau'=0$ 仅含第 `64` 帧（缺失 `61..63` 因果卷积上下文），而 $\tau'=1..8$ 恰好精确对应 `[65..68] .. [93..96]`（与窗口 0 的 $\tau=17..24$ **100% 四帧同相对齐**）！
- **对齐决议**：在 `PR-06` 的 `ConditionScheduler` 中明确 **“3 帧因果前导暖启动（3-Frame Causal Context Warm-Start）”** 规约：
  - 编码非首窗口 $k \ge 1$（名义帧区间 $[64k, 64k+96]$）时，向 VAE 额外喂入前序 3 帧 `[64k-3, 64k-2, 64k-1]` 激活 3D 因果卷积缓存（并对首潜步 $\tau'=0$ 施加边界掩膜权重 $\alpha(0) = 0$ 完全保留 $\mathbf{z}_{k-1}[16]$），使 $\tau'=1..8$ 与 $\mathbf{z}_{k-1}[17..24]$ 在完全同相的 4 帧时间网格上执行保方差 Sigmoid 融合，彻底消除接缝鬼影。

#### [RV-06 | P1] 拓扑突变重锚定（$\kappa > 0.65$）的无状态容器边界澄清
- **问题定位**：`DESIGN_SYSTEM_ENHANCEMENT.md` §3.2 提出当检测到 $\kappa(t) > 0.65$ 时“由 `fulgur` 补算 $\pm 2$ 帧局部插值 FGPP”。但 `synapse` 在 `foundary` 容器内无状态运行，不持有原始 3D 工程文件与 `fulgur` 渲染管线。
- **对齐决议**：拆分为**两级协同协议**：
  1. **容器内即时自愈（Level-A，单次运行内完成）**：立即将 $t_{\text{event}}$ 提升为动态强锚点（Dynamic Anchor），清空潜空间光流积分缓冲并衰减局部时间注意力偏置；
  2. **跨阶段重烘焙请求（Level-B，上报 Go 调度器）**：若突变超出单帧插值容限，在 `result_manifest.json` 中写入结构化 `reanchor_requests: [{frame_index, window: [-2, +2], kappa}]`（必要时返回 `EXIT_CIRCUIT_BREAKER = 40`），由上游 `render-farm`（Go）决策是否回调 `fulgur` 补算子帧 FGPP。

#### [RV-07 | P1] 上游 `@aether//style` C++ 库尚未落盘的解耦防阻塞策略
- **问题定位**：核查 `/Volumes/work/dervgen/aether` 发现：`aether/fgpp/` 已完整落盘，但 `RFC 0043`（`DESIGN-0043-01`）规划的 `aether/style/` C++ 目录目前仍处于架构设计阶段，尚未在 `aether` 仓内落地头文件。
- **对齐决议**：`PR-09`（`synapse.style`）**不阻塞等待 C++ 侧施工**，直接依据 `all-overview/designs/0043-multi-style-rendering/01-nexus-style-profile-spec.md` 的权威 `StyleProfile v2.0.0` 七段式 Schema（`Metadata`, `ShadingConfig`, `MaterialRemapConfig`, `PostProcessConfig`, `AnimationStepConfig`, `EnvironmentModifier`, `AudioModifier`, `TransitionConfig`）在 `synapse/style/profile_mapper.py` 中内建 Pydantic v2 强类型模型与 JSON/YAML 序列化器，并预留 `wire_guard.py --check-style` 钩子以便后续 `aether/style/` 落盘后一键对齐。

#### [RV-08 | P1] 现存 `v0.1.0` 代码存在未声明重型顶层依赖（`numba`/`trimesh`/`cv2`/`PIL`）与 Windows 硬编码路径
- **问题定位**：审计现存源码发现：
  1. `python/synapse/generators/rasterizer.py:18` 顶层直接 `from numba import njit, prange`，`python/synapse/models/loader.py:10` 顶层直接 `import trimesh`，而 `pyproject.toml` 完全未声明 `numba` 与 `trimesh`！在干净环境中仅安装 `synapse[dev]` 运行测试会立即 `ModuleNotFoundError`；
  2. `python/synapse/generators/dataset.py:11-13` 顶层直接 `import cv2` 与 `PIL`，违反 `AGENTS.md` §3 第 3 条“重型依赖延迟按需加载”铁律；
  3. `python/synapse/models/loader.py:19` 硬编码了 Windows 宿主路径 `Path("H:/dervgen/fulgur/assets/models")`。
- **对齐决议**：合入 **`PR-01` 必改清单**：
  - 将软光栅重构为**纯 NumPy 向量化实现作为零依赖默认主路径**（`numba` 仅作可选 JIT 加速装饰器：若未安装 `numba` 自动退回纯 NumPy 路径，确保 `synapse[dev]` 零额外编译依赖即可跑通 `export-fgpp`！）；
  - 在 `pyproject.toml` 新增可选依赖组 `mesh = ["trimesh>=4.0.0", "scipy>=1.10.0"]` 与 `jit = ["numba>=0.58.0"]`，所有 `cv2`/`PIL`/`trimesh`/`openexr`/`torch` 强制改为函数内按需延迟导入（缺失时抛出携带 `agent_fix_hint` 的 `SynapseError`）；
  - 彻底清除 `H:/dervgen/...` 硬编码路径，统一通过环境变量 `SYNAPSE_ASSET_CACHE` / 相对同级探测 / 包内资源解析。

---

### 🟡 P2 级：指标门禁阈值口径归一与仓库安全卫生

#### [RV-09 | P2] 四篇文档间同名评测指标及格阈值不一致
- **问题定位**：`normal_angular_error_deg` 在 `IMPLEMENTATION_PLAN`/`3D_INSPECTION` 中为 `15°`，在 `LORA`/`RSI` 中为 `12°`；`depth_rmse` 出现 `0.05m` 与 `0.03m`；`volume_breathing_pct` 出现 `5%` 与 `3%`；`albedo_delta_e00` 出现 `5.0` 与 `2.5`。
- **对齐决议**：不再混用单一魔数，在 `synapse.metrics` 中正式确立 **三级质量门禁配置档（Quality Gate Profiles）**（详见 §3.4）：`track_b_video`（快速预演基线）、`track_a_asset`（生产资产精修准出）、`rsi_promotion`（模型自演进严苛准出）。

#### [RV-10 | P2] 本地预览器全网卡监听隐患与工作区未追踪文件卫生
- **问题定位**：
  1. `python/synapse/viewer/server.py:70` 使用 `TCPServer(("", cur_port), ...)` 绑定了 `0.0.0.0` 全网卡接口，在云端共享机器或开发机上存在未授权局域网访问隐患，且削弱了 NG-1 豁免边界；
  2. 主工作区存在未追踪的 `data/sessions/` 目录，`.gitignore` 未覆盖运行时会话与评测临时产物。
- **对齐决议**：在 `PR-01` 中强制将 `ViewerServer` 默认绑定地址锁死为回环地址 `127.0.0.1`，并完善 `.gitignore`（忽略 `data/sessions/`, `data/datasets/`, `.pytest_tmp/`, `*.egg-info/`，仅保留 `data/scenarios/*.json` 版本化资产）。

---

## 3. 全仓唯一事实源（Canonical SSOT Contracts）

> 以下四张契约表为 `synapse` 全仓设计与编码的**最高优先级命名与接口准绳**。任何子文档或历史段落若与下表存在字面出入，一律以下表为准。

### 3.1 权威模块目录树（Canonical Module Tree）

```text
synapse/
├── AGENTS.md / CLAUDE.md                # Agent 行为铁律与云端测试规范
├── README.md                            # 仓库总览与导航
├── pyproject.toml                       # 依赖分层: core / image / mesh / metrics / gpu / train / dev
├── tools/
│   ├── wire_guard.py                    # 从 @aether//fgpp (fgpp_types.h) 生成并校验 schema.py
│   └── data-inspector/                  # [Phase M8] Electron + fulgur Native SHM 可视化检查器
│       ├── package.json                 # Electron 33 + @dervgen/electronkit + @dervgen/forgekit
│       ├── electron/                    # 主进程 (SHM 读取、Bundle 管理、synapse-bench CLI 桥接)
│       ├── renderer/                    # 5 大视图 (Source / Anim / Video+FGPP / Recon / Compare)
│       └── native/                      # 独立 MODULE.bazel C++20 渲染进程 (依赖 @fulgur 公开目标)
├── python/synapse/
│   ├── py.typed                         # PEP 561 强类型标记
│   ├── errors.py                        # SynapseError 异常族、退出码 (0/10/20/30/40)、agent_fix_hint
│   ├── telemetry.py                     # W3C traceparent/tracestate 解析与 MemoryForecaster 显存预检
│   ├── fgpp/                            # [M1] @aether//fgpp Python 官方镜像 SDK
│   │   ├── schema.py                    # 15 标量通道规格、34B/px、Log-Depth、Coarseness Floor、Anchor
│   │   ├── frame.py                     # CameraParams (行列主序无损互转)、FGPPFrame、FGPPSequence
│   │   ├── io.py                        # OpenEXR 2.0 异构通道读写、消毒器、Cryptomatte Nearest 缩放
│   │   └── torch_bridge.py              # NumPy <-> PyTorch DLPack 零拷贝桥接
│   ├── adapters/                        # [M2 & M5] 双轨视频生成适配与运行时 Multi-LoRA
│   │   ├── base.py                      # VideoModelAdapter 抽象基类、VideoResult、注册表
│   │   ├── mock.py                      # 零 GPU 确定性合成器 + 可控缺陷注入器 (DefectInjectionConfig)
│   │   ├── scheduler.py                 # W=97/S=64/Δ=33 因果滑窗(含3帧暖启动)、Sigmoid 退火、突变重锚定、光流池化
│   │   ├── wan.py                       # Tier-1 白盒: Wan 2.1 + ControlNeXt + 软光流偏置 + MultiLoRA 挂载
│   │   ├── cloud.py                     # Tier-2 黑盒: Seedance 2.5 / Kling 1.5 / Veo 锚定帧降级适配
│   │   └── lora/                        # [M5] 运行时 LoRA 加载与隔离
│   │       ├── manifest.py              # lora_manifest.json (synapse.lora.v1.0) Pydantic 模型
│   │       ├── manager.py               # MultiLoRAManager: 快速权重加和、Cryptomatte 掩膜空间隔离、ZipLoRA 正交化
│   │       └── patcher.py               # DiT Attention Processor 挂钩与零残留卸载
│   ├── style/                           # [M3] 风格提取、软肩 ACEScg 与 RFC 0043 映射
│   │   ├── aces_color.py                # sRGB <-> ACEScg 变换、双曲正切软肩逆色调映射 (L_th=4, max=12)、CIEDE2000
│   │   ├── descriptor.py                # StyleDescriptor 数据结构
│   │   ├── extractor.py                 # histogram_optical & fgpp_intrinsic 球谐主光提取
│   │   └── profile_mapper.py            # 导出 @aether//style StyleProfile v2.0.0 (YAML/JSON) & USD 灯光覆盖
│   ├── composite/                       # [M3] 3D+SAM2 局部手术刀合成
│   │   ├── detector.py                  # 闪烁、轮廓漂移、浮空穿地自动异常区域检测
│   │   ├── mask.py                      # M_3D ∪ M_SAM2 联合掩膜 + 光流自适应膨胀算子
│   │   └── blend.py                     # 16px 环带 CIE LAB 直方图匹配、时域泊松融合、Cryptomatte 引导光学合成
│   ├── bridge/                          # [M3] 跨库/跨容器薄桥接层 (严禁重写 C++ 算法)
│   │   ├── vg_bridge.py                 # 零拷贝调用 vg-core (去光照/Graph-Cut/重拓扑/IK) + Track-B 白模反投门禁
│   │   ├── foundary_client.py           # 调用 foundary (TRELLIS/Hunyuan3D/SAM2/GVHMR/Visual-Jev) 标准客户端
│   │   └── hitl.py                      # RFC 0020 CommandIRRecord 五元组与 15 分钟 (900s) 熔断状态机
│   ├── metrics/                         # [M4] 四维 16+ 项评测指标与三级门禁档位
│   │   ├── profiles.py                  # 三级门禁阈值定义 (track_b_video / track_a_asset / rsi_promotion)
│   │   ├── temporal.py                  # temporal_warp_error, flicker_score, fvd
│   │   ├── fgpp_fidelity.py             # depth_rmse, normal_angular_error_deg, flow_endpoint_error_px, silhouette_iou
│   │   ├── roundtrip.py                 # chamfer_mm, f_score_1mm, volume_breathing_pct, foot_sliding_cm, bone_drift
│   │   └── style_consistency.py         # ciede2000, light_dir_error_deg, tone_curve_rmse, histogram_intersection
│   ├── pipeline/                        # [M4] 无状态单步 Worker 与离线评测串联器
│   │   ├── bench_runner.py              # BenchPipelineRunner (本地/CI 场景串联与 Bundle 导出)
│   │   └── bundle_worker.py             # JobBundleWorker (消费 RFC 0039 bundle_manifest.json 单次执行即退出)
│   ├── bench/                           # [M1 & M4] Synapse-Bench 30 场景、软光栅与报告
│   │   ├── scenarios/*.json             # T1–T5 共 30 个标准场景 JSON (打包进 wheel package-data)
│   │   ├── scenario.py                  # Scenario 模型 (含 temporal_mode 与多级资源寻址)
│   │   ├── conditions.py                # 11×4 正交渲染条件因子
│   │   ├── camera.py                    # 6 类轨迹生成 (修复 360° endpoint、±90° 奇异点、因子接线)
│   │   ├── procedural.py                # 基础几何体 + Humanoid-53 (>=1500 顶点) 铰接水密代理白模
│   │   ├── model_loader.py              # 整合自原 models/ (纯 NumPy OBJ + 可选 trimesh glTF 加载)
│   │   ├── rasterizer.py                # 整合自原 generators/ (纯 NumPy 向量化主路径 + 可选 Numba JIT)
│   │   ├── registry.py                  # 外部测试资产 SHA256 缓存拉取器
│   │   ├── bundle_exporter.py           # [M8] 导出标准化 InspectionBundle (inspection.v1.0)
│   │   └── report.py                    # 零外部依赖自包含离线 HTML 报告生成器
│   ├── training/                        # [M5] 无状态 LoRA 离线特征预编码与 PEFT 训练核
│   │   ├── feature_store.py             # 3D Causal VAE + T5 + FGPP Min-Depth 离线预编码与 WebDataset 打包
│   │   ├── losses.py                    # Flow-Matching + Sobel 深度边缘对齐 + 法线一致性 + DPO 损失
│   │   ├── trainer.py                   # 5 类正交 LoRA (style/identity/control/whitebox/motion) 训练器
│   │   └── packager.py                  # Safetensors 导出、lora_manifest.json 封装与准出门禁评估
│   ├── inspect/                         # [M6] 3D 智能质检、Visual Jev 判别与 RL 飞轮样本导出
│   │   ├── schemas.py                   # CameraPose, DefectAnnotation3D, InspectionReport (v2.0.0)
│   │   ├── align.py                     # Fibonacci 球面 DINOv2 粗检索 + vg.recon 6DoF 精配准
│   │   ├── macro.py                     # 多通道残差热力图定位 + 虚拟微距相机 3D 反投解算
│   │   ├── jev_critic.py                # Visual Jev 共享视觉上下文批量质检与校准 Logits 置信度提取
│   │   ├── flywheel_exporter.py         # 对接 RFC 0020/0021 导出 IntentGroup / PRM StepAnnotation / Undo-DPO JSONL
│   │   └── eval_report.py               # [已实施] 3D 模型拓扑/渲染评估数据模型与自包含 HTML 报告生成器 (SVG 雷达图)
│   ├── rsi/                             # [M7] 物理驱动递归自我提升 (RSI) 无状态阶段算子
│   │   ├── curriculum.py                # 自博弈课程域随机化参数采样器 Ω_task(θ)
│   │   ├── dual_verifier.py             # 融合 vg_bridge 物理硬判决 V_phys 与 Visual Jev V_prm 的双重验算器
│   │   ├── pair_builder.py              # D_self / D_human 偏好对提纯与 >=20% L0 黄金锚点经验回放混洗器
│   │   └── promotion_gate.py            # Synapse-Bench T1-T5 全量单调递增防退化准出守门器
│   ├── viewer/                          # [保留] 本地 127.0.0.1 轻量调试预览器 (NG-1 豁免)
│   └── cli/                             # 统一 CLI 路由层
│       ├── __init__.py                  # `synapse` 与 `synapse-bench` 双入口分发
│       └── ...                          # 按子域拆分的命令处理器
├── tests/                               # 单元、跨语言黄金契约与端到端 Mock 回归套件
└── scripts/
    ├── evaluate_3d_models.py             # [已实施] 3D 模型评估 CLI (OBJ 扫描→拓扑检查→多视角截图→6 维指标→三级门禁→HTML 报告)
    ├── prepare_datasets.py               # FGPP 黄金数据集准备
    ├── run_cpu_verification.py            # E1–E5 无卡 CPU 基线验证
    └── colab_runner.py                    # Google Colab 断点续跑实验执行器
```

### 3.2 统一 CLI 双入口语法树（Unified CLI Grammar）

所有子命令强制支持 `--json`（纯 JSON `stdout`，零杂散日志）与统一退出码（`0 / 10 / 20 / 30 / 40`）：

| 顶层入口 | 子命令族 | 具体命令签名 | 归属里程碑 |
| :--- | :--- | :--- | :---: |
| **`synapse`** | `fgpp` | `synapse fgpp validate <dir> [--json]`<br/>`synapse fgpp inspect <dir> [--export-vis <out>] [--json]` | M1 (`PR-04`) |
| **`synapse`** | `worker` | `synapse worker run --bundle <manifest_or_dir> [--json]` | M4 (`PR-14b`) |
| **`synapse`** | `lora` | `synapse lora prepare-data --input <dir> --out-wds <path> [--json]`<br/>`synapse lora train --config <yaml> --dataset <wds> --out <dir> [--json]`<br/>`synapse lora eval --lora-bundle <dir> --scenario <id> [--json]`<br/>`synapse lora merge --base <ckpt> --lora <dir:scale>... --out <ckpt> [--json]` | M5 (`PR-16~19`) |
| **`synapse`** | `inspect` | `synapse inspect align --ref-image <img> --mesh <glb> --out <pose.json> [--json]`<br/>`synapse inspect eval --ref-image <img> --mesh <glb> --camera <pose.json> [--json]`<br/>`synapse inspect refine --mesh <glb> --report <rep.json> --out <glb> [--json]`<br/>`synapse inspect export-rl-data --journal-dir <dir> --format dpo\|prm --out <jsonl> [--json]`<br/>`synapse inspect evaluate --input-dir <dir> --output-dir <dir> --gate-profile <p> [--views N] [--resolution N] [--json]` ✅ | M6 (`PR-20~23`) |
| **`synapse`** | `rsi` | `synapse rsi sample-curriculum --round <t> --count <n> --out <dir> [--json]`<br/>`synapse rsi verify --candidate-dir <dir> --fgpp-dir <dir> [--json]`<br/>`synapse rsi build-pairs --verified-dir <dir> --l0-golden <dir> --out <dir> [--json]`<br/>`synapse rsi gate-promote --baseline <json> --candidate <json> [--json]` | M7 (`PR-24~25`) |
| **`synapse`** | `generate` / `view` | `synapse generate ...`（开发合成数据集） / `synapse view <dir>`（本地 `127.0.0.1` 预览） | M1 (`PR-04`) |
| **`synapse-bench`** | 评测基准套件 | `synapse-bench list [--tier T1..T5] [--json]`<br/>`synapse-bench export-fgpp --scenario <id> --out <dir> [--json]`<br/>`synapse-bench run [--scenario <id> \| --tier <T> \| --all] --adapter <id> [--profile <p>] [--report <html_path>] [--export-bundle <dir>] [--json]`<br/>`synapse-bench evaluate --bundle <inspection_bundle_dir> [--profile <p>] [--json]` | M1 / M4 / M8 |

### 3.3 `JobBundleWorker` 八大无状态容器阶段契约（RFC 0039）

`render-farm`（Go）通过 `synapse worker run --bundle <dir> --json` 统一调度全部单次容器计算阶段：

| `stage` 枚举值 | 核心输入载荷 (`inputs`) | 核心输出产物 (`outputs`) | 落地 PR |
| :--- | :--- | :--- | :---: |
| `fgpp_validate` | `fgpp_dir` | 校验摘要与通道统计 (`result_manifest.json`) | `PR-14b` |
| `neural_video_generate` | `fgpp_dir`, `prompt`, `adapter_config`, `lora_bundles[]` | `generated_video.mp4`, 逐帧 PNG, `reanchor_requests[]` | `PR-14b` / `PR-16` |
| `style_extract_and_map` | `video_or_frames`, `fgpp_dir` (可选) | `style_profile.yaml` (RFC 0043 v2.0.0), `usd_light_overrides.json` | `PR-14b` |
| `surgical_composite` | `base_video`, `patch_video`, `fgpp_new`, `sam2_mask`, `caelum_aov` | `composited_video.mp4`, 环带色差诊断 | `PR-14b` |
| `lora_preencode` | `fgpp_dir`, `target_video`, `caption` | `WebDataset` 潜空间分片 (`shard-*.tar`) | `PR-17` |
| `lora_train` | `wds_shards[]`, `lora_train_config.yaml`, `base_checkpoint` | `lora_bundle/` (`adapter_model.safetensors` + `lora_manifest.json`) | `PR-18` |
| `asset_inspect_3d` | `mesh_glb`, `ref_image`, `rendered_gbuffer_dir` | `inspection_report.json` (`DefectAnnotation3D[]`) | `PR-22` |
| `rsi_dual_verify` | `candidate_dir`, `gt_fgpp_dir`, `l0_anchor_manifest` | `verification_verdict.json` + 提纯偏好对记录 | `PR-24` |

### 3.4 三级质量门禁统一阈值矩阵（Reconciled Quality Gate Profiles）

| 指标名称 | 物理意义 | `track_b_video`<br/>(快速白模视频预演) | `track_a_asset`<br/>(高保真 3D 资产准出) | `rsi_promotion`<br/>(LoRA/RSI 演进硬门禁) |
| :--- | :--- | :---: | :---: | :---: |
| `temporal_warp_error` | 光流对齐帧间扭曲残差 | $\le 0.020$ | $\le 0.015$ | $\le 0.015$ (且不劣于上代) |
| `flicker_score` | 高频时域拉普拉斯闪烁 | $\le 0.025$ | $\le 0.020$ | $\le 0.018$ |
| `depth_rmse_m` | 米制线性深度均方根误差 | $\le 0.05\text{ m}$ | $\le 0.04\text{ m}$ | $\le 0.03\text{ m}$ |
| `normal_angular_error_deg` | 表面法线平均余弦夹角 | $\le 15.0^\circ$ | $\le 12.0^\circ$ | $\le 12.0^\circ$ |
| `silhouette_iou` | 前景剪影交并比 | $\ge 0.85$ | $\ge 0.88$ | $\ge 0.90$ |
| `chamfer_mm` | 双边采样点云倒角距离 | $\le 8.0\text{ mm}$ | $\le 5.0\text{ mm}$ | $\le 4.0\text{ mm}$ |
| `volume_breathing_pct` | 跨帧/精修前后体积变化率 | $\le 5.0\%$ | $\le 3.0\%$ | $\le 2.5\%$ |
| `foot_sliding_cm` | 触地相足底水平位移 | $\le 0.5\text{ cm}$ | $\le 0.3\text{ cm}$ | $\le 0.2\text{ cm}$ |
| `albedo_delta_e00` | 去光照固有色 CIEDE2000 | $\le 5.0$ | $\le 2.5$ | $\le 2.5$ |
| `border_ring_ciede2000` | 16px 合成环带边界色差 | $\le 3.0$ | $\le 2.5$ | $\le 2.0$ |
| `non_manifold_edges` | 非流形边数量（3D 网格） | N/A | $\equiv 0$ | $\equiv 0$ |
