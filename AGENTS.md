# Synapse Agent 指引与开发者规范

> 本文件为 AI Agent 及开发者在 `synapse` 仓库中进行规划、开发、测试与代码审查时的最高优先级行为准则。

---

## 1. 仓库定位与核心边界

**`synapse`** 是 sautille-stroke / DervGen 生态 Layer 2 的 **3D CG 渲染与神经视频生成互操作算法 SDK、无状态推理容器执行核（Worker Runtime）与跨模态往返评测基准（`Synapse-Bench`）**。

### 上下游生态协同关系

| 协同方向 | 目标仓库 | 职责边界与交互契约 |
| :--- | :--- | :--- |
| **底层跨语言契约 SSOT** | `aether` (`@aether//fgpp`, `@aether//style`) | Layer 0 C++20 定义 `FGPP` 15 标量 EXR 通道布局、`metadata.json` (`fgpp.v1.0`) 与 `StyleProfile` Schema；`synapse` 在 Python 端严格镜像并通过跨语言黄金测试锁死，**严禁私自篡改线格式** |
| **正向物理 FGPP 生产端** | `fulgur` (Track B) / `caelum-core` (Track A) | 导出多通道 OpenEXR 序列 + `metadata.json`；`synapse.fgpp` 负责无损读取、校验与张量转换 |
| **全仓自包含原生算法核** | `synapse` (`csrc/geometry/` + Python Oracle) | 已彻底脱离外部 `vg-core`，自包含原生 C++20 几何算子；Python 模块 (`synapse.recon.geometry`) 作为算法正确性验证的 Ground Truth Oracle，双轨互验 |
| **上游分布式调度与编排** | `render-farm` (Go) / `narrative-rail` (Go) | 生产级 DAG 编排与 API 服务统一由 Go 承载；通过 RFC 0039 自洽 `JobBundle` 调度 `synapse` 在 `foundary` 容器内完成单步 GPU 计算后即刻退出 |
| **上层创作工坊与总控** | `grand-studio` / `scene-forge` / `asset-forge` / `film-studio` | 负责白模布景、运镜编辑、2D 瑕疵框选、网格/材质 L2 限时人工精修与 `CommandIR` 采集；**桌面端严禁直接内嵌 `synapse` 重型 PyTorch 环境** |

### 绝对红线（Always-On Redlines）
1. **非服务端铁律（Server-side Zero-Python）**：严禁在 `synapse` 中编写常驻 HTTP/WebSocket 微服务、数据库连接池或任务队列守护进程。`synapse` 只能是纯 SDK 库 + 单次执行即退出的无状态 CLI Worker。
2. **契约零漂移铁律**：`FGPP` 的 OpenEXR 通道名（15 个 ASCII 排序标量通道，`34 B/px`）、`CameraLinearDepth.Z`（米制线性深度）、`WorldNormal.XYZ`、`Cryptomatte.ID`（`UINT32` 严禁插值滤波）以及相机矩阵主序标记必须与 `aether/fgpp/include/aether/fgpp/fgpp_types.h` 100% 对齐。
3. **白模反投防呆铁律（RFC 0044 §4.1 & §4.4）**：
   - 角色白模代理必须满足顶点数 $\ge 1500$ 且具备铰接四肢（`validate_whitebox_coarseness_floor`）；
   - Track B 粗白模流严禁直接将 AI 视频反投回光滑白模 UV，必须先经生成式 3D 重建（`TRELLIS`/`Hunyuan3D`）+ 重拓扑门禁；
   - 单镜头或单资产人工精修超过 **15 分钟** 必须触发工程熔断降级。
4. **防静默失败三问**：任何计划条目或测试验收在标记 `✅` 前必须同时通过 **存在问**（模块存在）、**接线问**（真实挂入主流程而非孤立死代码）、**执行问**（产出非空有效张量/图像并通过数值阈值断言）。

---

## 2. 环境角色定位与测试执行铁律（保护本地开发机）

| 环境节点 | 连接方式 | 角色定位与使用规范 |
| :--- | :--- | :--- |
| **Ubuntu 云端主力机 (`ubuntu24`)** | `ssh ubuntu24` | **P0 首选全量测试与 CI 验证环境**。全套 `pytest`、`mypy`、`ruff`、OpenEXR 读写压测及 `MockVideoAdapter` 端到端回归**默认全部在 `ubuntu24` 上执行** |
| **Windows GPU 节点 (`win-dev`)** / **`foundary` 容器** | `ssh win-dev` / 算力集群 | **GPU 专有验证环境**。仅在验证真实 `Wan 2.1` / `SAM2` CUDA 推理或跨平台引擎（`fulgur`/`caelum-core`）联调时调用 |
| **macOS 本地开发机** | 本机 (`/Volumes/work/dervgen/synapse`) | **主控代码与文档编辑环境**。仅限单文件极速轻量语法/单测冒烟，**严禁在本地 Mac 拉起数十 GB 视频大模型或跑全量重型基准套件** |

```bash
# 推荐首选：在云端主力机 ubuntu24 运行全套校验、黄金数据集生成与 E1-E5 无卡基线验证
ssh ubuntu24 "cd ~/work/dervgen/synapse && /root/venv-synapse/bin/pytest tests/ -q && /root/venv-synapse/bin/ruff check python/ tests/ scripts/ && /root/venv-synapse/bin/mypy python/synapse"
ssh ubuntu24 "cd ~/work/dervgen/synapse && /root/venv-synapse/bin/python scripts/prepare_datasets.py && /root/venv-synapse/bin/python scripts/run_cpu_verification.py"

# Google Colab GPU 节点：必须遵循 Google Colab 与 Google Drive SSOT V2.0 铁律（零污染 MyDrive 根目录，计算存储分离）
# 完整环境规范见 workenv/google-colab/README.md 及 architecture/google-colab-and-drive-standards.md
bash workenv/google-colab/setup_colab.sh
python scripts/colab_runner.py --bundle /content/synapse_colab_bundle_v1.tar.gz --checkpoint-dir /content/drive/MyDrive/dervgen/work/yanglei/synapse/$(date +%F)/e1-e4/checkpoints --exp all

# 本地极轻量单点冒烟（仅限纯 CPU 毫秒级单测）
pytest tests/test_bench.py -k "test_scenario_create" -v
```

---

## 3. Python 编码与 Agent-Native 接口规范

1. **Python 版本与类型安全**：兼容 Python `>=3.10`（主力生产容器 Python `3.12`），所有公开函数、方法与 `dataclass` 必须具备完整 Type Hints，确保 `mypy` 零告警并携带 `python/synapse/py.typed`。
2. **零隐藏全局状态**：适配器、提取器与评测器通过显式配置构造注入，严禁在 `import synapse` 阶段隐式分配 GPU 显存、加载大模型权重或发起网络请求。
3. **重型依赖延迟按需加载（Lazy Optional Imports）**：基础包仅依赖 `numpy` 与 `pydantic`；`openexr`、`opencv-python-headless`、`torch` 等重型依赖放在可选 extra（`[image]`, `[metrics]`, `[gpu]`）中，并在缺失时抛出带明确安装指引的结构化错误。
4. **CLI-First 与机器可读契约**：
   - 所有 CLI 子命令支持 `--json` 输出纯 JSON 结构，禁止在 `--json` 模式下向 `stdout` 混入非结构化日志；
   - 统一确定性退出状态码：
     - `0`：执行成功；
     - `10`：参数或 FGPP / StyleProfile 契约校验失败；
     - `20`：输入场景、资产或 EXR 文件缺失 / SHA256 校验不匹配；
     - `30`：GPU 显存不足或下游推理后端不可达；
     - `40`：触发 HITL 15 分钟超时工程熔断或质量门禁未达标；
     - `50`：命令或 Stage 尚未实现（严禁未实现桩代码以 `0` 退出）；
   - 报错 JSON 必须包含 `"agent_fix_hint"` 字段，给出可直接执行的修复动作建议。

---

## 4. Git 与 PR 提交铁律（Worktree 隔离模式）

1. **严禁直接 push `main`**：所有代码与文档变更必须遵循 `github-pr-workflow` Skill，使用 `/tmp/synapse-wt-<branch>` 隔离工作区，走「分支 → PR → Squash Merge → 清理 Worktree」流程。
2. **强制携带 AI 模型标识**：Commit、PR 标题及正文首行必须显式标注当前操作的 AI 模型标识（如 `[gemini4]`, `[gemini3.8]`, `[opus4.6]`）：
   - Commit / PR 标题格式：`<type>(<scope>): <description> [<model-tag>]`
   - PR 正文首行：`> **Operated by**: \`<model-tag>\``

---

## 5. 核心文档导航

- **整体设计深度评审与跨文档契约对齐（必读 SSOT）**：[docs/DESIGN_REVIEW_AND_ALIGNMENT.md](docs/DESIGN_REVIEW_AND_ALIGNMENT.md)
- **全景可落地实施总计划（M1–M8, 30 PRs）**：[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
  - 细分执行蓝图：[PLAN-M1-M2](docs/plans/PLAN-M1-M2-FOUNDATION-AND-ADAPTERS.md) · [PLAN-M3-M4](docs/plans/PLAN-M3-M4-STYLE-COMPOSITE-AND-BENCH.md) · [PLAN-M5](docs/plans/PLAN-M5-LORA-TRAINING-PIPELINE.md) · [PLAN-M6-M7](docs/plans/PLAN-M6-M7-INSPECTION-AND-RSI-FLYWHEEL.md) · [PLAN-M8](docs/plans/PLAN-M8-DATA-INSPECTOR-DESKTOP.md)
- **架构审查与跨仓边界定界**：[docs/AUDIT_AND_ARCHITECTURE.md](docs/AUDIT_AND_ARCHITECTURE.md)
- **行业成熟流水线对比差距与代偿设计规范**：[docs/DESIGN_INDUSTRY_GAP_AND_COMPENSATION.md](docs/DESIGN_INDUSTRY_GAP_AND_COMPENSATION.md)
- **行业前沿技术追踪、演进趋势与架构启示录**：[docs/INDUSTRY_TRACKING_AND_TRENDS.md](docs/INDUSTRY_TRACKING_AND_TRENDS.md)
- **全系统工程设计增强规范**：[docs/DESIGN_SYSTEM_ENHANCEMENT.md](docs/DESIGN_SYSTEM_ENHANCEMENT.md)
- **LoRA & PEFT 微调流水线设计规范**：[docs/DESIGN_LORA_PIPELINE.md](docs/DESIGN_LORA_PIPELINE.md)
- **训练数据可视化检查器方案**：[docs/DESIGN_DATA_INSPECTOR.md](docs/DESIGN_DATA_INSPECTOR.md)
- **3D 建模智能质检与精修数据飞轮设计规范**：[docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md](docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md)
- **3D 生成模型质量提升方法论、评估体系全景矩阵与训练数据工程规范**：[docs/DESIGN_3D_MODEL_ENHANCEMENT_AND_DATA_SPEC.md](docs/DESIGN_3D_MODEL_ENHANCEMENT_AND_DATA_SPEC.md)
- **3D 模型评估体系完整交付清单与 Review 指南**：[docs/REVIEW_3D_EVAL_DELIVERY.md](docs/REVIEW_3D_EVAL_DELIVERY.md)
- **真实视频逆向 3D 动态数字孪生与联合反演设计规范 (RFC 0060)**：[docs/DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md](docs/DESIGN_REAL_VIDEO_TO_3D_INVERSE_RENDERING.md)
- **算法、训练与数据流水线攻坚执行计划**：[docs/PLAN-ALGO-TRAINING-DATA-PIPELINE.md](docs/PLAN-ALGO-TRAINING-DATA-PIPELINE.md)
- **PR #17 / PR #18 联合评审与整改落地报告**：[docs/plans/PLAN-REVIEW-PR17-PR18.md](docs/plans/PLAN-REVIEW-PR17-PR18.md)
- **递归自我提升 (RSI) 与物理驱动自演进飞轮设计规范**：[docs/DESIGN_RSI_FLYWHEEL.md](docs/DESIGN_RSI_FLYWHEEL.md)
- **局部解剖智能修正与多模型融合设计规范**：[docs/DESIGN_LOCAL_REFINEMENT_AND_MULTI_MODEL_FUSION.md](docs/DESIGN_LOCAL_REFINEMENT_AND_MULTI_MODEL_FUSION.md)
- **3D 多视角生成与逆向重建闭环验证计划**：[docs/plans/PLAN-3D-ROUNDTRIP-RECONSTRUCTION-VERIFICATION.md](docs/plans/PLAN-3D-ROUNDTRIP-RECONSTRUCTION-VERIFICATION.md)
- **Google Colab 实验与微调运行环境指南**：[workenv/google-colab/README.md](workenv/google-colab/README.md)

---

## 6. 内置 Agent Skills 导航（`.agents/skills/`）

| Skill 名称 | 路径 | 适用场景 |
| :--- | :--- | :--- |
| **`synapse-overview`** | [.agents/skills/synapse-overview/SKILL.md](.agents/skills/synapse-overview/SKILL.md) | 架构全景、全模块索引（含 3D 模型评估、csrc/geometry 与 synapse.recon 逆向反演）、跨仓协作边界、CLI 确定性退出码契约（含 50）与多机测试规范 |
| **`synapse-fgpp-and-datasets`** | [.agents/skills/synapse-fgpp-and-datasets/SKILL.md](.agents/skills/synapse-fgpp-and-datasets/SKILL.md) | 15 通道 OpenEXR 线格式（`34 B/px`）、`fulgur` Track-B 与 `caelum-core` Track-A 导出、`prepare_datasets.py` 与 Colab 数据包构建 |
| **`synapse-adapters-and-experiments`** | [.agents/skills/synapse-adapters-and-experiments/SKILL.md](.agents/skills/synapse-adapters-and-experiments/SKILL.md) | `WanAdapter` 7 通道条件构建、$W=97$ 因果滑窗与切镜重锚定、ACEScg 风格锁、16px 手术刀合成及 `ubuntu24` + Colab 双阶段实验执行 |
| **`synapse-lora-and-flywheel`** | [.agents/skills/synapse-lora-and-flywheel/SKILL.md](.agents/skills/synapse-lora-and-flywheel/SKILL.md) | WebDataset 潜变量分片、`LoRAManifest` 安全校验、Cryptomatte 空域注意力掩码、3D 智能质检与 RSI 飞轮 |
