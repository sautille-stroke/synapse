---
name: synapse-overview
description: >-
  Synapse（Layer 2）3D CG 渲染与神经视频生成互操作算法 SDK、无状态推理容器执行核与跨模态往返评测基准（Synapse-Bench）全景指南。涵盖架构边界、全模块索引（含 3D 模型评估、csrc/geometry 自包含双轨几何核、synapse.recon 逆向反演）、跨仓协作契约（@aether//fgpp、fulgur、caelum-core、foundary）、CLI 确定性状态码（含 50）以及 ubuntu24 + Google Colab 双阶段实验工作流。在进入 synapse 开发、定位模块或执行测试前必读。
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
| **全仓自包含原生算法与逆向重建核** | `synapse` (`csrc/geometry/` + `synapse.recon`) | 自包含现代 C++20 几何算子（Bazel + `@aether//math`，`synapse._c_geometry`）与 Python Ground Truth Oracle（`synapse.recon.geometry`）100% 双轨互验；实现 2-流形/水密闭合校验、简并面剔除、共面 2D SAT + Möller–Trumbore 自穿插检测、4D 点追踪、6-DoF 运镜解算、足底防滑零位移 IK 与本征去光照（RFC 0060）。 |
| **跨仓 FFI 与 DCC 精修委托** | `synapse.bridge` (`vg_bridge`, `foundary_client`, `hitl`) | 通过薄桥接层按需调用 `foundary`（TRELLIS/SAM2）或 `vg-core` 生产级重拓扑；内建 `MockVgBackend` 保证脱离外部 C++ 库时全套测试与门禁 100% 跑通；实现 15 分钟 HITL 熔断状态机。 |
| **分布式调度与上层创作工坊** | `render-farm` / `foundary` / `grand-studio` | 服务端由 Go 承载；`synapse` 仅作为纯算法 SDK 与单次执行即退出的无状态 CLI Worker（遵循 Server-side Zero-Python 铁律）。 |

---

## 二、核心模块地图（`python/synapse/` 与 `csrc/`）

| 模块路径 | 核心类 / 函数 | 职责说明 |
| :--- | :--- | :--- |
| `csrc/geometry/` | `checkManifold`, `checkWatertight`, `checkDegenerateFaces`, `checkSelfIntersections` | 现代 C++20 原生几何拓扑算子核（Bazel 构建，使用 `@aether//math` DVec3），单线程极速毫秒级流形/闭合/自交校验。 |
| `synapse.recon` | `ReconPipeline`, `check_manifold`, `check_watertight`, `solve_camera_trajectory`, `solve_zero_slip_ik`, `delight` | Video-to-3D 几何逆向重建与资产蒸馏（RFC 0060）：4D 点追踪、6-DoF 运镜解算、流形闭合 Python Oracle、Quad 重拓扑、足底防滑零位移 IK 与本征去光照。 |
| `synapse.refine` | `RefinementROI`, `roi_tracker`, `tiled_executor`, `hand`, `face`, `MultiModelRefineFusion`, `ScalpelStitcher`, `LocalRefinementPipeline` | 局部解剖智能修正与多模型手术刀融合（RFC 0061）：MANO 手部先验、Canonical 肖像面部 ID 保持、5 大模型融合算子、512x512 瓦片与光流 Warp 手术刀逆向缝合。 |
| `synapse.bridge` | `VgBackend`, `MockVgBackend`, `NativeVgBackend`, `FoundaryClient`, `HITLSession` | 跨库/跨容器薄桥接层，含 15 分钟（900s）HITL 熔断状态机与无外部依赖 Mock 回退。 |
| `synapse.fgpp` | `FGPPFrame`, `FGPPSequence`, `CameraParams`, `export_fgpp`, `import_fgpp`, `sanitize_fgpp_frame`, `resize_fgpp_frame`, `validate_whitebox_coarseness_floor` | 15 通道 OpenEXR 读写、通道净化、分辨率缩放、PyTorch 张量桥接与白模底线门禁（$\ge 1500$ 顶点 + 铰接四肢）。 |
| `synapse.adapters` | `WanAdapter`, `MockVideoAdapter`, `KlingAdapter`, `SeedanceAdapter`, `VeoAdapter`, `ConditionScheduler`, `CausalWindowPlan`, `DiscontinuityDetector`, `fuse_overlapping_latents` | 视频生成模型适配器、7 通道控制张量构建（`[7, T, H, W]`）、$W=97, S=64, \Delta=33$ 因果滑窗调度、潜空间保方差 Sigmoid 融合与切镜 $\kappa$ 重锚定。 |
| `synapse.adapters.lora` | `LoRAManifest`, `MultiLoRAManager`, `LoRAPatchHook`, `patch_wan_adapter_with_lora` | `synapse.lora.v1.0` 安全清单（强制 `.safetensors` + SHA256 + 模块白名单）、Gram-Schmidt 正交化与 `Cryptomatte.ID` 空域注意力掩码门控。 |
| `synapse.style` | `StyleDescriptor`, `extract_style_optical`, `extract_style_fgpp_intrinsic`, `to_style_profile_dict`, `to_engine_light_overrides`, `aces_color` | ACEScg 对数色度内蕴风格提取、高光软肩压缩（Soft-Shoulder Rolloff）与 `StyleProfile v2.0.0` 引擎灯光回写。 |
| `synapse.composite` | `generate_surgical_mask`, `border_ring_lab_match`, `laplacian_pyramid_blend`, `composite_surgical_sequence`, `composite_optical_increments` | 运动自适应手术刀掩码 $M_{\text{surgical}}(t) = \text{Dilate}_{\mathbf{v}(t)}(M_{\text{3D}} \cup M_{\text{SAM2}})$、16px 边界环 CIE LAB 仿射色彩对齐与多层拉普拉斯金字塔无缝融合。 |
| `synapse.training` | `FeatureStoreBuilder`, `iter_feature_shards`, `format_structured_caption`, `losses`, `trainer` | 离线 VAE/文本潜变量预编码、WebDataset `.tar` 分片构建与流式读取、5 段式结构化 Caption、参考锚定 Diffusion-DPO 以及 LoRA 混合损失训练器。 |
| `synapse.inspect` | `schemas`, `geometry_gate`, `model_evaluator`, `eval_report`, `align`, `macro`, `jev_critic`, `flywheel_exporter` | Tier 0 双轨几何硬门禁（`validate_mesh_geometry`）、3D 模型批量多维评估与 HTML 雷达图报告（`synapse inspect evaluate` / `scripts/evaluate_3d_models.py`）、`InspectionReport` v2.0、Fibonacci 球面配准、多通道残差热力图、Visual Jev 判别、RL 飞轮样本导出及 L0/L1/L2 三级精修路由（15 分钟 HITL 熔断）。 |
| `synapse.rsi` | `flywheel`, `reward`, `replay_buffer`, `curriculum`, `pareto`, `dual_verifier` | 递归自我提升（RSI）物理驱动自演进飞轮、多目标帕累托前沿筛选与反遗忘经验回放（$\ge 20\%$ 黄金锚点）。 |
| `synapse.pipeline` | `BenchPipelineRunner`, `JobBundleWorker`, `stages` (`inspect`, `rsi`, `training`) | 无状态单步容器执行核、离线评测串联器与解耦的 Stage 适配器。 |
| `synapse.bench` | `list_scenarios`, `load_scenario`, `SoftwareRasterizer`, `generate_humanoid_proxy`, `export_inspection_bundle` | `Synapse-Bench` 30 个标准评测场景（T1–T5 五大 Tier）、纯 NumPy 软光栅真值生成器与 HTML 诊断报告。 |
| `synapse.metrics` | `psnr_db`, `ssim`, `ciede2000`, `lpips`, `temporal_warp_error`, `flicker_score`, `chamfer_mm`, `f_score_1mm`, `evaluate_metrics_gate` | 图像画质（`image_quality.py`）、色度风格、时序一致性与 3D 几何往返全维度量化评测指标及三级门禁档位（含 `METRIC_ALIASES`）。 |
| `tools/data-inspector` | Electron + React + Three.js + Native C++ SHM | 训练数据分片、15 通道 EXR 与 3D 质检包可视化检查器（基于 `@aether//pal:shm`）。 |

---

## 三、多机分工与测试执行铁律

| 环境节点 | 角色定位 | 标准执行命令 |
| :--- | :--- | :--- |
| **`ubuntu24` (P0 云端主力机)** | **全量单测 (`pytest`)、`ruff`/`mypy` 静态门禁、`csrc/geometry` Bazel 构建与 C++/Python 双轨一致性校验、黄金数据集打包与 E1–E5 无卡基线验证** | `ssh ubuntu24 "cd ~/work/dervgen/synapse && /root/venv-synapse/bin/pytest tests/ -q && /root/venv-synapse/bin/ruff check python/ tests/ scripts/ && /root/venv-synapse/bin/mypy python/synapse"` |
| **Google Colab / win-dev (P1 GPU)** | **重型视频大模型 (Wan 2.1) 推理与 LoRA 微调实验**<br>详见 `workenv/google-colab/` 一站式环境套件（含 `setup_colab.sh`、`models.json`、`io_spec.md` 及 Notebook），使用 `ubuntu24` 导出的 `synapse_colab_bundle_v1.tar.gz` 与 Google Drive 断点续跑 | `bash workenv/google-colab/setup_colab.sh && python scripts/colab_runner.py --bundle /content/synapse_colab_bundle_v1.tar.gz --checkpoint-dir /content/drive/MyDrive/dervgen/work/$(whoami)/synapse/$(date +%F)/exp-all/checkpoints --exp all` |
| **macOS 本地开发机** | **代码主控与 Git Worktree 管理**<br>严禁在本地拉起视频大模型或跑全量重型套件 | `pytest tests/test_bench.py -k "test_scenario_create" -v` |

---

## 四、CLI 确定性退出状态码契约

所有 CLI 子命令支持 `--json` 纯 JSON 输出，报错时必须携带 `"agent_fix_hint"` 字段，并遵循统一退出码：
- `0`：执行成功；
- `10`：参数或 `FGPP` / `StyleProfile` 契约校验失败（`ContractValidationError`）；
- `20`：输入场景、资产或 EXR 文件缺失 / SHA256 校验不匹配（`AssetNotFoundError`）；
- `30`：GPU 显存不足或下游推理后端不可达（`BackendUnavailableError`）；
- `40`：触发 HITL 15 分钟人工精修超时熔断或质量门禁未达标（`QualityGateCircuitBreakerError`）；
- `50`：命令或 Stage 尚未实现（严禁未实现桩代码以 `0` 退出）。
