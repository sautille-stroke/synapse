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
| **原生视觉与角色算法核** | `vg-core` (`csrc/` + `pybind11`) | 承载去光照、Graph-Cut 贴图缝合、四边面重拓扑、Humanoid-53 刚体骨长与足底零速度 IK；`synapse.bridge` 通过零拷贝 FFI 调用，**严禁在 Python 中重写密集几何/IK 循环** |
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
# 推荐首选：在云端主力机 ubuntu24 运行全套校验与单元测试
ssh ubuntu24 "cd ~/work/dervgen/synapse && pytest tests/ -v && ruff check python/ tests/ && mypy python/synapse"

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
   - 报错 JSON 必须包含 `"agent_fix_hint"` 字段，给出可直接执行的修复动作建议。

---

## 4. Git 与 PR 提交铁律（Worktree 隔离模式）

1. **严禁直接 push `main`**：所有代码与文档变更必须遵循 `github-pr-workflow` Skill，使用 `/tmp/synapse-wt-<branch>` 隔离工作区，走「分支 → PR → Squash Merge → 清理 Worktree」流程。
2. **强制携带 AI 模型标识**：Commit、PR 标题及正文首行必须显式标注当前操作的 AI 模型标识（如 `[gemini4]`, `[gemini3.8]`, `[opus4.6]`）：
   - Commit / PR 标题格式：`<type>(<scope>): <description> [<model-tag>]`
   - PR 正文首行：`> **Operated by**: \`<model-tag>\``

---

## 5. 核心文档导航

- **架构审查与跨仓边界定界**：[docs/AUDIT_AND_ARCHITECTURE.md](docs/AUDIT_AND_ARCHITECTURE.md)
- **分阶段可落地实施计划（M1–M4, 15 PRs）**：[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- **LoRA & PEFT 微调流水线设计规范**：[docs/DESIGN_LORA_PIPELINE.md](docs/DESIGN_LORA_PIPELINE.md)
- **训练数据可视化检查器方案**：[docs/DESIGN_DATA_INSPECTOR.md](docs/DESIGN_DATA_INSPECTOR.md)
- **3D 建模智能质检与精修数据飞轮设计规范**：[docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md](docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md)
