# Synapse (神经突触)

> **3D CG 渲染与神经视频生成互操作算法 SDK、无状态推理容器 Worker 与跨模态评测基准 (`Synapse-Bench`)**  
> Interoperability SDK, Stateless GPU Worker Runtime & Benchmark Suite between 3D CG Rendering and Generative Video Models.

---

## 1. 核心定位与生态边界

当前 3D 物理渲染（确定性时空几何与辐射度量）与生成式视频扩散模型（潜空间时序分布）存在根本性的表示鸿沟。  
**`synapse`** 位于 DervGen 生态 **Layer 2**，扮演连接两者的“神经突触”算法与评测层（落实 **RFC 0044** 神经-物理混合工业流水线）：

- **向上承接编排**：作为无状态单步 GPU Worker（消费 **RFC 0039 `JobBundle`**），由 Go 分布式农场 `render-farm` 与影视管线 `narrative-rail` 在 `foundary` 推理容器中按需拉起执行，**绝不自建常驻 Python 服务端**；
- **向下严守契约**：Python 端 `synapse.fgpp` 严格镜像 Layer 0 **`@aether//fgpp`** C++20 线格式 SSOT，与 `fulgur`（Track B 毫秒级光栅）和 `caelum-core`（Track A 物理光追）实现多通道 OpenEXR 2.0 零歧义互通；
- **横向原生闭环**：风格描述符对齐 **`@aether//style` (`RFC 0043`)** 与 **ACEScg (`RFC 0058`)**；4D 点追踪、6-DoF 运镜解算、流形闭合校验与 Quad 重拓扑、Humanoid-53 足底防滑锁定 IK（$<0.3\text{ cm/s}$）以及球谐/Retinex 混合去光照全面由原生算法模块 **`synapse.recon`** 提供，实现零外部 C++ 强依赖开箱即用，并向下兼容 `synapse.bridge.vg_bridge` 抽象。

---

## 2. 混合渲染管线协同全景（5 阶段）

```mermaid
flowchart LR
    A["① 分镜布景\n(scene-forge + film-studio + wgp)"] --> B["② 白模引导视频生成\n(fulgur/caelum FGPP → synapse.adapters)"]
    B --> C["③ 精细 3D 资产反哺\n(foundary Gen3D + vg-core + asset-forge)"]
    C --> D["④ 局部重渲染与风格对齐\n(synapse.style + synapse.composite)"]
    D --> E["⑤ 终审光学合成与成片\n(caelum-core 增量 + narrative-rail)"]

    style A fill:#e3f2fd,stroke:#1565c0
    style B fill:#fff3e0,stroke:#e65100
    style C fill:#e8f5e9,stroke:#2e7d32
    style D fill:#fce4ec,stroke:#c2185b
    style E fill:#f3e5f5,stroke:#6a1b9a
```

1. **分镜布景**：在 `grand-studio` 宿主内通过 `scene-forge`、`film-studio` 与 `wgp` 完成 3D 代理白模（角色满足 $\ge 1500$ 顶点粗糙度下限）与运镜轨迹规划。
2. **白模引导生成**：`fulgur` / `caelum-core` 导出 15 标量通道 **FGPP** OpenEXR 序列，`synapse.adapters` 驱动白盒主力（`Wan 2.1 14B` + ControlNeXt + 3D Causal VAE 软光流偏置）或黑盒商业 API（`Seedance 2.5` / `Kling 1.5` / `Veo`）生成时空受控视频。
3. **精细 3D 资产反哺**：经 `foundary` 正交关键帧 3D 重建（`TRELLIS`/`Hunyuan3D`）破解白模反投悖论，由 `vg-core` 完成混合去光照与 Graph-Cut 缝合，并由 L2 美术在 `asset-forge` 限时精修（15 分钟硬熔断）。
4. **局部手术刀重渲染与风格对齐**：`synapse.style` 提取视频色温、曝光与球谐主光映射为 `StyleProfile` 驱动 3D 局部重渲；`synapse.composite` 构建 `3D ∪ SAM2` 运动膨胀掩膜，在 CIE LAB 与线性 ACEScg 空间完成无缝融合。
5. **终审合成与评测闭环**：叠加高阶物理接触阴影/焦散，全流程由 `Synapse-Bench` 四维 16 项指标量化守门。

---

## 3. 模块架构与当前就绪状态

> 整体设计与跨文档对齐决议见 [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](docs/DESIGN_REVIEW_AND_ALIGNMENT.md)，架构审查见 [docs/AUDIT_AND_ARCHITECTURE.md](docs/AUDIT_AND_ARCHITECTURE.md)，历次深度审计见 [AUDIT-POST-REFACTOR-2026-11](docs/audits/AUDIT-POST-REFACTOR-2026-11.md)、[AUDIT-RFC0063-AND-CODEBASE-2026-11](docs/audits/AUDIT-RFC0063-AND-CODEBASE-2026-11.md)、[AUDIT-POST-HORIZON5-2026-11](docs/audits/AUDIT-POST-HORIZON5-2026-11.md)（`e37bfa6`）与 [AUDIT-POST-HORIZON6-2026-11](docs/audits/AUDIT-POST-HORIZON6-2026-11.md)（`46b3bda`），M1–M28 全量 110-PR 实施总计划 (`v3.4`) 见 [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)。当前 `ubuntu24` 全仓实测收集 **1,239** 项 pytest 用例（`1,143` fast + `96` slow），经复审确认 `4` 项 `-m slow` 失败、串行慢测 `>675s` 超时、缺失 `.github/workflows/ci.yml` 及 `42` 项残余假绿/契约债务（`12×P0 + 16×P1 + 10×P2 + 4×P3`），已统一排入 Horizon 7（`M25–M28`, `PR-92 ~ PR-109`，详见 [PLAN-M25-M28](docs/plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md)）。

| 模块路径 | 核心职责 | 当前状态 | 对应实施里程碑 / 交付 |
| :--- | :--- | :---: | :---: |
| `csrc/geometry/` | Tier 0 现代 C++20 几何绑定与校验门禁（Bazel 9.3.1，委托 `@aether//geo` + `@aether//math` `DVec3`），流形/水密/简并/自交校验 | 🟡 Bazel 可测 (`[opt]` 双重扩展待 `PR-102`) | Tier 0 核心算子 (`PR-47`, `PR-84`, `PR-102`) |
| `python/synapse/render_core/` | 无状态底层软件光栅化、相机模型与基础几何生成原语（从 `bench/` 剥离解耦） | 🟡 已抽离 (齐次近面裁剪与向量化待 `PR-94`) | `PR-48`, `PR-78`, `PR-94` |
| `python/synapse/recon/` | Video-to-3D 几何逆向重建与资产蒸馏（RFC 0060/0062）：4D 点追踪、6-DoF 运镜解算、流形闭合 Python Oracle、Quad 重拓扑、足底防滑零位移 IK (<0.2 cm)、球谐与 Retinex 本征去光照、双轨可微逆渲染 | 🟡 骨架就绪 (`test_recon_invert_cli` 失败与烘焙待 `PR-99`) | RFC 0060/0062 (`PR-79~82`, `PR-99`) |
| `python/synapse/refine/` | 局部解剖智能修正与多模型融合（RFC 0061）：MANO 手部先验、Canonical 面部身份保持、五维多模型融合核、512×512 瓦片与光流 Warp 手术刀缝合 | 🟡 已合入 (HDR 缝合与 `strict` 门禁待 `PR-100`) | M10 (`PR-33~37`, `PR-81`, `PR-100`) |
| `python/synapse/bench/` | `Synapse-Bench` 30 场景、11×4 条件因子、纯 NumPy 软光栅与 InspectionBundle 导出器 | 🟢 已就绪并测试覆盖 (去陈旧 `data/scenarios/` 待 `PR-105`) | `PR-01`, `PR-15`, `PR-26`, `PR-105` |
| `python/synapse/fgpp/` | 对齐 `@aether//fgpp` 的 15 标量通道 (`34B/px`) OpenEXR 读写、消毒器与 DLPack 桥接 | 🟡 `UINT32` 已合入 (Fail-Closed & 内参重建待 `PR-93`) | `PR-02`, `PR-03`, `PR-74`, `PR-93` |
| `python/synapse/cli/` | 双入口统一 CLI（`synapse` / `synapse-bench`、确定性退出码 `0/10/20/30/40/50`、显存预检） | 🟡 已就绪 (后置 `--json` 与空子命令待 `PR-96`) | `PR-04`, PR #22, `PR-96` |
| `python/synapse/adapters/` | 双轨视频适配器（`Mock` / 白盒 `Wan 2.1` / 黑盒 API）、97f 因果滑窗与 Multi-LoRA 空间隔离 | 🟡 已合入 (空域掩膜与滑窗收敛待 `PR-97`) | `PR-05~08`, `PR-16`, `PR-76`, `PR-97` |
| `python/synapse/style/` | 双曲正切软肩 ACEScg 逆色调映射、球谐主光提取与 `StyleProfile v2.0` 映射 | 🟡 已就绪 (`IRLS`/`S^2` 求解与前向软肩待 `PR-95`) | `PR-09`, `PR-78`, `PR-95` |
| `python/synapse/composite/` | 穿模/闪烁检测、`3D ∪ SAM2` 膨胀掩膜、16px 环带 LAB 与光学增量拉普拉斯金字塔融合 | 🟡 已就绪 (穿地坐标/光流方向与 EDT 收口待 `PR-95`) | `PR-10`, `PR-11`, `PR-95` |
| `python/synapse/bridge/` | `foundary` 客户端、HITL 15m 熔断状态机与 `vg_bridge`（内建 `MockVgBackend` 回退） | 🟡 已就绪 (单一 Foundary 传输与跨进程 HITL 待 `PR-97`) | `PR-12`, `PR-97` |
| `python/synapse/metrics/` | 四维 16 项指标与三级质量门禁档位（`track_b` / `track_a` / `rsi_promotion`，SSOT 对齐） | 🟡 方向与正则已修 (R3D18 通道维/`uint8` 除 255 待 `PR-101`) | `PR-13`, PR #23, `PR-83`, `PR-101` |
| `python/synapse/pipeline/` | 无状态 `JobBundleWorker` 容器执行核、离线 `BenchPipelineRunner` 与解耦的 Stage 适配器 | 🟡 `import_fgpp` 已接入 (5 Stage 落盘与全指标待 `PR-96,105`) | `PR-14a`, `PR-14b`, `PR-75`, `PR-96`, `PR-105` |
| `python/synapse/training/` | 3D VAE / FGPP 离线潜空间 `WebDataset` 缓存与正交 LoRA 训练器（参考锚定 Diffusion-DPO） | 🟡 已合入 (Fail-Closed/Losses/旋转门禁待 `PR-98`) | `PR-17~19`, PR #27, `PR-77`, `PR-98` |
| `python/synapse/inspect/` | Tier 0 几何硬门禁、Fibonacci 粗到精配准、微距 3D 反投、Visual Jev 判别与 3D 模型评估报告（SVG 雷达图） | 🟡 已就绪 (`TestEvalScript` stem/XSS 待 `PR-105`) | `PR-20~23`, PR #31, `PR-87`, `PR-105` |
| `python/synapse/rsi/` | 课程域随机化、物理+PRM 双重验算、20% L0 黄金锚点经验回放与单调递增发布门禁 | 🟡 模块已落地 (`promotion_gate` 30 场景闭环待 `PR-105`) | `PR-24`, `PR-25`, `PR-89`, `PR-105` |
| `python/synapse/viewer/` | 静态 HTML 比较报告导出器（`PR-96` 移除常驻 `ViewerServer` `TCPServer` 遵守 Zero-Python 红线） | 🔴 常驻 Daemon 违背红线 #1 & XSS (退场至 `PR-96`) | PR-01, PR-51, `PR-96` |
| `tools/data-inspector/` | Electron 33 + `@aether//pal:shm` Native C++ 检查器 | 🟡 Native C++ 去裸 POSIX 已合入 (`.value()` `SIGABRT`/Seqlock/macOS SHM 待 `PR-103,104`) | `PR-27~29`, `PR-64~65`, `PR-85~86`, `PR-103~104` |

---

## 4. 文档导航

- **[AGENTS.md](AGENTS.md)**：AI Agent 与开发者行为铁律、测试机器分工与编码规范
- **[docs/DESIGN_REVIEW_AND_ALIGNMENT.md](docs/DESIGN_REVIEW_AND_ALIGNMENT.md)**：**整体设计深度评审与跨文档对齐决议**（`RV-01~RV-17` 纠偏、权威目录树、统一 CLI 语法树与三级门禁阈值表）
- **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)**：**M1–M28 二十八阶段 110 个原子 PR 可落地实施总计划 (`v3.4`)** 及 `docs/plans/` 十份精细化执行子计划：
  - [PLAN-M1-M2](docs/plans/PLAN-M1-M2-FOUNDATION-AND-ADAPTERS.md) · [PLAN-M3-M4](docs/plans/PLAN-M3-M4-STYLE-COMPOSITE-AND-BENCH.md) · [PLAN-M5](docs/plans/PLAN-M5-LORA-TRAINING-PIPELINE.md) · [PLAN-M6-M7](docs/plans/PLAN-M6-M7-INSPECTION-AND-RSI-FLYWHEEL.md) · [PLAN-M8](docs/plans/PLAN-M8-DATA-INSPECTOR-DESKTOP.md) · [PLAN-M9-M12](docs/plans/PLAN-M9-M12-HORIZON3-LANDING.md) · [PLAN-M13-M16](docs/plans/PLAN-M13-M16-POST-REFACTOR-REMEDIATION.md) · [PLAN-M17-M20](docs/plans/PLAN-M17-M20-QUALITY-GOVERNANCE-AND-BENCHMARK-LANDING.md) · [PLAN-M21-M24](docs/plans/PLAN-M21-M24-POST-H5-PRODUCTION-HARDENING.md) · [PLAN-M25-M28](docs/plans/PLAN-M25-M28-POST-H6-TRUE-CLOSURE.md)
- **[docs/AUDIT_AND_ARCHITECTURE.md](docs/AUDIT_AND_ARCHITECTURE.md)**：整体目标审查、跨仓职责定界与 10 大现状问题诊断报告
- **[docs/audits/AUDIT-POST-REFACTOR-2026-11.md](docs/audits/AUDIT-POST-REFACTOR-2026-11.md)**：**重构后全库架构、工程实现与技术债务深度审计报告（2026-11）**
- **[docs/audits/AUDIT-RFC0063-AND-CODEBASE-2026-11.md](docs/audits/AUDIT-RFC0063-AND-CODEBASE-2026-11.md)**：**`RFC-0063` 与全库质量基线深度评审报告（`b98c6e5`）**
- **[docs/audits/AUDIT-POST-HORIZON5-2026-11.md](docs/audits/AUDIT-POST-HORIZON5-2026-11.md)**：**Horizon-5 收官后全库深度复审报告（`e37bfa6`，`10×P0 + 10×P1 + 5×P2 + 2×P3` 台账与 `M21–M24` 映射）**
- **[docs/audits/AUDIT-POST-HORIZON6-2026-11.md](docs/audits/AUDIT-POST-HORIZON6-2026-11.md)**：**Horizon-6 合入后全库五切片深度复审报告（`46b3bda`，`ubuntu24` 十项实测、`12×P0 + 16×P1 + 10×P2 + 4×P3` 台账与 `M25–M28` 映射）**
- **[RFC 0060](docs/RFC-0060-REAL-VIDEO-TO-3D-DIGITAL-TWIN.md)** · **[详细设计](docs/DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md)**：**真实视频逆向 3D 动态数字孪生与联合反演规范**
- **[RFC 0061](docs/RFC-0061-LOCAL-REFINEMENT-AND-MULTI-MODEL-FUSION.md)** · **[详细设计](docs/DESIGN_LOCAL_REFINEMENT_AND_MULTI_MODEL_FUSION.md)** · **[ForgeKit 工具链](docs/DESIGN_FORGEKIT_ANATOMICAL_REFINE_TOOLING.md)**：**局部解剖智能修正、多模型融合与手术刀缝合**
- **[RFC 0062](docs/RFC-0062-INVERSE-RENDERING-AND-3D-NEURAL-CLOSED-LOOP.md)** · **[详细设计](docs/DESIGN_INVERSE_RENDERING_CLOSED_LOOP.md)**：**3D 引导生成、可微逆向渲染与局部修补双向闭环**
- **[RFC 0063](docs/RFC-0063-TESTING-CORRECTNESS-AND-BENCHMARKS.md)** · **[行业对比详细设计](docs/DESIGN_INDUSTRY_GAP_AND_COMPENSATION.md)** · **[PLAN-M17-M20](docs/plans/PLAN-M17-M20-QUALITY-GOVERNANCE-AND-BENCHMARK-LANDING.md)**：**质量保障、5 层测试金字塔与 4 轨道横向评测基准**
- **[docs/PLAN-ALGO-TRAINING-DATA-PIPELINE.md](docs/PLAN-ALGO-TRAINING-DATA-PIPELINE.md)**：算法、训练与数据流水线攻坚执行计划 (`v2.0`)
- **[docs/DESIGN_3D_MODEL_ENHANCEMENT_AND_DATA_SPEC.md](docs/DESIGN_3D_MODEL_ENHANCEMENT_AND_DATA_SPEC.md)**：3D 生成模型质量提升方法论、评估体系全景矩阵与训练数据工程规范
- **[docs/DESIGN_INDUSTRY_GAP_AND_COMPENSATION.md](docs/DESIGN_INDUSTRY_GAP_AND_COMPENSATION.md)**：行业成熟流水线对比差距与代偿性架构设计规范
- **[docs/DESIGN_SYSTEM_ENHANCEMENT.md](docs/DESIGN_SYSTEM_ENHANCEMENT.md)**：Synapse 2.0 系统设计增强规范（流式潜空间切片、拓扑突变保护、Wire Guard、软肩 ACEScg）
- **[docs/DESIGN_LORA_PIPELINE.md](docs/DESIGN_LORA_PIPELINE.md)**：LoRA & PEFT 微调流水线设计规范
- **[docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md](docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md)**：3D 建模智能质检、多尺度对齐与精修数据飞轮设计规范
- **[docs/DESIGN_RSI_FLYWHEEL.md](docs/DESIGN_RSI_FLYWHEEL.md)**：递归自我提升 (RSI) 与物理驱动自演进飞轮设计规范
- **[docs/DESIGN_DATA_INSPECTOR.md](docs/DESIGN_DATA_INSPECTOR.md)**：训练数据可视化检查器方案（Data Inspector）
- **[docs/plans/PLAN-3D-ROUNDTRIP-RECONSTRUCTION-VERIFICATION.md](docs/plans/PLAN-3D-ROUNDTRIP-RECONSTRUCTION-VERIFICATION.md)**：3D 多视角生成与逆向重建闭环验证计划
- **[docs/archive/](docs/archive/README.md)**：**历史评审与交付归档**（含 [PR #17/#18 联合整改报告](docs/archive/PLAN-REVIEW-PR17-PR18.md)、[3D 模型评估交付 Review](docs/archive/REVIEW_3D_EVAL_DELIVERY.md)）

---

## 5. 快速上手

```bash
# 安装开发环境
pip install -e ".[dev]"

# 运行单元测试（日常全量回归优先在云端主力机 ubuntu24 执行）
ssh ubuntu24 "cd ~/work/dervgen/synapse && pytest tests/ -v"
```
