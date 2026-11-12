---
name: synapse-overview
description: >-
  Synapse（Layer 2）3D CG 渲染与神经视频生成互操作算法 SDK、无状态推理容器执行核与跨模态往返评测基准（Synapse-Bench）全景指南。涵盖架构边界、全模块索引（含 3D 模型评估脚本与 HTML 报告）、跨仓协作契约（@aether//fgpp、fulgur、caelum-core、vg-core）、CLI 确定性状态码以及 ubuntu24 + Google Colab 双阶段实验工作流。在进入 synapse 开发、定位模块或执行测试前必读。
---

# Synapse 架构全景与开发指南

> `synapse` 是 sautille-stroke / DervGen 生态 **Layer 2** 的 **3D CG 渲染与神经视频生成互操作算法 SDK、无状态推理容器执行核（Worker Runtime）与跨模态往返评测基准（`Synapse-Bench`）**。

---

## 一、上下游生态协同边界（SSOT）

| 协同方向 | 目标仓库 | 职责边界与交互契约 |
| :--- | :--- | :--- |
| **底层跨语言契约 SSOT** | `aether` (`@aether//fgpp`, `@aether//style`) | Layer 0 C++20 定义 `FGPP` 15 标量 EXR 通道布局（`34 B/px`）、`metadata.json` (`fgpp.v1.0`) 与 `StyleProfile` (`2.0.0`)；`synapse` 在 Python 端严格镜像并通过跨语言黄金测试锁死。 |
| **实时光栅 FGPP 生产端 (Track B)** | `fulgur` (`//engine/fgpp_export`) | 导出 15 通道 OpenEXR 序列 + `metadata.json`；当开启 `--export-fgpp` 时，`SampleApp` 自动挂载 `PickingPass` (`Editor_PickingEntityId`, `R32_UINT`) 生成非零 `Cryptomatte.ID`。 |
| **离线光追 FGPP + 真值生产端 (Track A)** | `caelum-core` (`caelum.camera.fgpp_exporter`) | 将路径追踪 `AOVBuffer` 与 `Camera` 导出为 15 通道 `frame_XXXXXX.exr` (`render_track = "offline_pathtraced"`) + 配对路径追踪 `beauty_XXXXXX.exr` Ground Truth。 |
| **原生几何与角色算法核** | `vg-core` (`csrc/` + `pybind11`) | 承载去光照、Graph-Cut 贴图缝合、四边面重拓扑、Humanoid-53 刚体骨长与足底零速度 IK；`synapse.bridge` 通过零拷贝 FFI 调用，严禁在 Python 中重写密集几何/IK 循环。 |
| **分布式调度与上层创作工坊** | `render-farm` / `foundary` / `grand-studio` | 服务端由 Go 承载；`synapse` 仅作为纯算法 SDK 与单次执行即退出的无状态 CLI Worker（遵循 Server-side Zero-Python 铁律）。 |

---

## 二、核心模块地图（`python/synapse/`）

| 模块路径 | 核心类 / 函数 | 职责说明 |
| :--- | :--- | :--- |
| `synapse.fgpp` | `FGPPFrame`, `FGPPSequence`, `CameraParams`, `export_fgpp`, `import_fgpp`, `sanitize_fgpp_frame`, `resize_fgpp_frame`, `validate_whitebox_coarseness_floor` | 15 通道 OpenEXR 读写、通道净化、分辨率缩放、PyTorch 张量桥接与白模底线门禁（$\ge 1500$ 顶点 + 铰接四肢）。 |
| `synapse.adapters` | `WanAdapter`, `MockVideoAdapter`, `KlingAdapter`, `SeedanceAdapter`, `VeoAdapter`, `ConditionScheduler`, `CausalWindowPlan`, `DiscontinuityDetector`, `fuse_overlapping_latents` | 视频生成模型适配器、7 通道控制张量构建（`[7, T, H, W]`）、$W=97, S=64, \Delta=33$ 因果滑窗调度、潜空间保方差 Sigmoid 融合与切镜 $\kappa$ 重锚定。 |
| `synapse.adapters.lora` | `LoRAManifest`, `MultiLoRAManager`, `LoRAPatchHook`, `patch_wan_adapter_with_lora` | `synapse.lora.v1.0` 安全清单（强制 `.safetensors` + SHA256 + 模块白名单）、Gram-Schmidt 正交化与 `Cryptomatte.ID` 空域注意力掩码门控。 |
| `synapse.style` | `StyleDescriptor`, `extract_style_optical`, `extract_style_fgpp_intrinsic`, `to_style_profile_dict`, `to_engine_light_overrides`, `aces_color` | ACEScg 对数色度内蕴风格提取、高光软肩压缩（Soft-Shoulder Rolloff）与 `StyleProfile v2.0.0` 引擎灯光回写。 |
| `synapse.composite` | `generate_surgical_mask`, `border_ring_lab_match`, `laplacian_pyramid_blend`, `composite_surgical_sequence`, `composite_optical_increments` | 运动自适应手术刀掩码 $M_{\text{surgical}}(t) = \text{Dilate}_{\mathbf{v}(t)}(M_{\text{3D}} \cup M_{\text{SAM2}})$、16px 边界环 CIE LAB 仿射色彩对齐与多层拉普拉斯金字塔无缝融合。 |
| `synapse.training` | `FeatureStoreBuilder`, `iter_feature_shards`, `format_structured_caption`, `losses`, `trainer` | 离线 VAE/文本潜变量预编码、WebDataset `.tar` 分片构建与流式读取、5 段式结构化 Caption 以及 LoRA 混合损失训练器。 |
| `synapse.inspect` | `schemas`, `align`, `macro`, `jev_critic`, `flywheel_exporter`, `eval_report` | 3D 网格拓扑/法线/UV 质检（`InspectionReport` v2.0）、Fibonacci 球面配准、多通道残差热力图、Visual Jev 判别、RL 飞轮样本导出、**3D 模型评估脚本与 HTML 报告**（`TopologyResult`/`ModelEvalResult`/`EvalReportData`）及 L0/L1/L2 三级精修路由（15 分钟 HITL 熔断）。 |
| `synapse.rsi` | `flywheel`, `reward`, `replay_buffer`, `curriculum`, `pareto` | 递归自我提升（RSI）物理驱动自演进飞轮、多目标帕累托前沿筛选与反遗忘经验回放。 |
| `synapse.bench` | `list_scenarios`, `load_scenario`, `SoftwareRasterizer`, `generate_humanoid_proxy`, `export_inspection_bundle` | `Synapse-Bench` 30 个标准评测场景（T1–T5 五大 Tier）、纯 NumPy 软光栅真值生成器与 HTML 诊断报告。 |
| `synapse.metrics` | `psnr`, `ssim`, `ciede2000`, `temporal_warp_error`, `flicker_score`, `chamfer_distance` | 图像/色度/时序/3D 几何全维度量化评测指标。 |
| `tools/data-inspector` | Electron + React + Three.js 桌面端 | 训练数据分片、15 通道 EXR 与 3D 质检包可视化检查器。 |

---

## 三、多机分工与测试执行铁律

| 环境节点 | 角色定位 | 标准执行命令 |
| :--- | :--- | :--- |
| **`ubuntu24` (P0 云端主力机)** | **全量单测 (`pytest`)、`ruff`/`mypy` 静态门禁、C++ 引擎无头导出、黄金数据集打包与 E1–E5 无卡基线验证** | `ssh ubuntu24 "cd ~/work/dervgen/synapse && /root/venv-synapse/bin/pytest tests/ -q && /root/venv-synapse/bin/ruff check python/ tests/ scripts/ && /root/venv-synapse/bin/mypy python/synapse"` |
| **Google Colab / win-dev (P1 GPU)** | **重型视频大模型 (Wan 2.1) 推理与 LoRA 微调实验**<br>详见 workenv/google-colab/ 一站式环境套件（含 setup_colab.sh、models.json、io_spec.md 及 Notebook），使用 ubuntu24 导出的 synapse_colab_bundle_v1.tar.gz 与 Google Drive 断点续跑 | ash workenv/google-colab/setup_colab.sh && python scripts/colab_runner.py --bundle /content/synapse_colab_bundle_v1.tar.gz --checkpoint-dir /content/drive/MyDrive/dervgen/work/$(whoami)/synapse/$(date +%F)/e1-e4/checkpoints --exp all |
| **macOS 本地开发机** | **代码主控与 Git Worktree 管理**<br>严禁在本地拉起视频大模型或跑全量重型套件 | `pytest tests/test_bench.py -k "test_scenario_create" -v` |

---

## 四、CLI 确定性退出状态码契约

所有 CLI 子命令支持 `--json` 纯 JSON 输出，报错时必须携带 `"agent_fix_hint"` 字段，并遵循统一退出码：
- `0`：执行成功；
- `10`：参数或 `FGPP` / `StyleProfile` 契约校验失败（`ContractValidationError`）；
- `20`：输入场景、资产或 EXR 文件缺失 / SHA256 校验不匹配（`AssetNotFoundError`）；
- `30`：GPU 显存不足或下游推理后端不可达（`BackendUnavailableError`）；
- `40`：触发 HITL 15 分钟人工精修超时熔断或质量门禁未达标（`QualityGateCircuitBreakerError`）。
