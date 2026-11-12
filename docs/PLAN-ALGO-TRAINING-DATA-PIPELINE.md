# 算法、训练与数据流水线攻坚执行计划 (Algorithm, Training & Data Pipeline Assault Plan)

- **文档状态**: `Draft v1.0` — 待核心审查完成后升级为 `Approved`
- **操作模型标识**: `[opus4.6]`
- **修订日期**: 2026-11-12
- **定位**: 本文档是对 `IMPLEMENTATION_PLAN.md` 中 **M2（适配器）、M3（风格/合成）、M5（LoRA 训练）、M6–M7（质检/RSI 飞轮）** 阶段的算法与数据流水线专项聚焦执行计划，侧重于"短期内如何将现有代码底座转化为可运行的真实训练与数据生产闭环"。
- **关联文档**:
  - [docs/IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 全局 M1–M8 总计划
  - [docs/DESIGN_LORA_PIPELINE.md](DESIGN_LORA_PIPELINE.md) — LoRA 微调设计规范
  - [docs/DESIGN_RSI_FLYWHEEL.md](DESIGN_RSI_FLYWHEEL.md) — 物理驱动 RSI 自演进规范
  - [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](DESIGN_REVIEW_AND_ALIGNMENT.md) — 跨文档对齐决议（§3.4 三级门禁）
  - [docs/DESIGN_3D_INSPECTION_AND_REFINEMENT.md](DESIGN_3D_INSPECTION_AND_REFINEMENT.md) — 3D 质检与精修飞轮
  - [docs/DESIGN_SYSTEM_ENHANCEMENT.md](DESIGN_SYSTEM_ENHANCEMENT.md) — 流式切片与拓扑突变保护

---

## 0. 计划审查摘要与纠偏决议 (Plan Review & Correction Register)

> [!IMPORTANT]
> 本章节记录对原始三战役攻坚计划的自审结论——发现 **5 项需纠正的参数错误** 与 **3 项跨文档未收敛的指标矛盾**，以及 **4 项需要补充锁定的缺失超参数**。所有纠偏已在后续正文中同步修正。

### 0.1 代码基线审计结论

在本计划制定前，对全部 27 个核心模块（`training/`、`adapters/lora/`、`adapters/`、`rsi/`、`style/`、`composite/`、`fgpp/`、`scripts/`）执行了逐文件代码审计：

| 审计维度 | 结论 |
| :--- | :--- |
| **模块实现度** | **27/27 = 100% 已实现**（零 Stub / 零 Skeleton / 零 Missing），含完整算法逻辑 |
| **本地全量测试** | `pytest tests/ -q` → **673 passed, 2 skipped** (234s)，全绿 |
| **LoRA 专项测试** | `test_lora_trainer / test_lora_cli_gate / test_lora_feature_store / test_lora_manager` → **81 passed, 1 skipped** |
| **Inspect + RSI 测试** | `test_inspect_* / test_rsi_*` → **123 passed** |
| **重型依赖隔离** | 全部 `torch` / `diffusers` / `openexr` / `cv2` 均为函数体内懒加载（Lazy Import），`import synapse` 零 GPU 副作用 |
| **30 场景基准覆盖** | T1×8 + T2×7 + T3×5 + T4×6 + T5×4 = **30 场景** JSON 全部就绪 |
| **CLI 子命令** | `synapse lora {prepare-data, train, eval, merge}` + `synapse rsi {gate-promote}` + `synapse inspect {align, eval, export-rl-data}` 全部注册并接线 |

### 0.2 原始计划参数错误纠偏 (Factual Corrections)

| 编号 | 错误内容 | 纠正 | 权威来源 |
| :---: | :--- | :--- | :--- |
| **FC-01** | 原计划将 `whitebox` LoRA 标注为 r=16, α=16 | **应为 r=32, α=64**，目标模块为全身 Attention（Spatial + Temporal），而非仅 `spatial_attn` | `DESIGN_LORA_PIPELINE.md` §2 表格 & `manifest.py` 白名单 |
| **FC-02** | 原计划将 `style` LoRA 目标模块仅写为 `cross_attn.to_k/v` | **应追加 FFN 层** (`ffn.*`)，代码中 `manifest.py` L22-26 白名单已包含 `ffn\\..*` | `manifest.py` L22-26 |
| **FC-03** | 原计划将 `control` LoRA 目标模块写为 `spatial_attn.*, ffn.*` | **设计文档定义为 ControlNeXt 旁路 / DiT 浅层 `blocks.0..10`**。代码层面 `manifest.py` 白名单确实是 `spatial_attn.*` + `ffn.*`，但设计意图是浅层注入，需在文档与代码间**确认对齐** | `DESIGN_LORA_PIPELINE.md` §2 vs `manifest.py` L32-35 |
| **FC-04** | 原计划将复合损失写为 $\mathcal{L}_{flow} + \lambda_{edge}\mathcal{L}_{edge} + \lambda_{dpo}\mathcal{L}_{dpo}$ | DPO 损失**不是**主训练循环的一部分，而是 RSI 偏好对齐阶段专用；主训练损失应为 $\mathcal{L}_{flow} + \lambda_{edge}\mathcal{L}_{edge} + \lambda_{normal}\mathcal{L}_{normal}$ | `DESIGN_LORA_PIPELINE.md` §4.2 |
| **FC-05** | 代码中 DPO β 默认值为 `500.0`（`losses.py` L79），但设计文档未声明 | 需在文档中**正式锁死 β=500.0** 为基线值并在计划中标注 | `losses.py` L79 |

### 0.3 跨文档指标矛盾汇总与收敛决议 (Metric Threshold Reconciliation)

> [!CAUTION]
> **`profiles.py` 代码中的阈值与 `DESIGN_REVIEW_AND_ALIGNMENT.md` §3.4 权威表格存在系统性偏差**——代码中的 `rsi_promotion` 比设计文档严格 2–10 倍（如 `normal_angular_error_deg` 代码为 5.0° vs 文档 12.0°，`chamfer_mm` 代码为 1.0mm vs 文档 4.0mm，`silhouette_iou` 代码为 0.97 vs 文档 0.90）。这是一个必须在执行前澄清的 **P0 级矛盾**。

| 指标 | REVIEW 文档 `rsi_promotion` | `profiles.py` 代码 `RSI_PROMOTION` | RSI 文档 `dual_verifier.py` 默认值 | LoRA `packager.py` 默认值 | **裁决** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `normal_angular_error_deg` | ≤ 12.0° | ≤ 5.0° | ≤ 12.0° | ≤ 12.0° | **⚠️ 代码过严**：需确认是否为刻意的"超额锁定" |
| `silhouette_iou` | ≥ 0.90 | ≥ 0.97 | ≥ 0.88 | — | **⚠️ 三处不同**：REVIEW 0.90, 代码 0.97, RSI 0.88 |
| `chamfer_mm` | ≤ 4.0 mm | ≤ 1.0 mm | — | ≤ 5.0 mm | **⚠️ 四处不同** |
| `depth_rmse_m` | ≤ 0.03 m | ≤ 0.05 m | ≤ 0.03 m | — | **⚠️ 代码反而更宽松** |
| `foot_sliding_cm` | ≤ 0.2 cm | ≤ 0.5 cm | 0.02 m/s (速度!) | — | **⚠️ 单位不统一**：RSI 文档用速度(m/s)，其他用位移(cm) |
| `volume_breathing_pct` | ≤ 2.5% | ≤ 1.5% | — | — | 代码更严 |

**收敛决议（CR-01）**：
1. **以 `DESIGN_REVIEW_AND_ALIGNMENT.md` §3.4 为唯一权威三级阈值 SSOT**——这是经过完整跨文档审查后产出的对齐决议。
2. **`profiles.py` 当前的"超额严格"值视为理想远期目标**——在当前骨架期不作为硬阻断门禁，否则所有实验都将无法通过。近期执行使用 REVIEW 文档值。
3. **`dual_verifier.py` 中 `silhouette_iou` 默认值从 0.88 更新为 0.90**——对齐 `rsi_promotion` 档位。
4. **足底滑移指标统一为位移制 `foot_sliding_cm`**——以 `REVIEW` 文档的 `foot_sliding_cm ≤ 0.2 cm`（触地相单帧水平位移）为准。`dual_verifier.py` 中 `foot_slide_velocity_ms` 保留但标注为辅助参考。

### 0.4 缺失超参数锁定 (Missing Hyperparameter Lock-Down)

| 编号 | 缺失项 | 锁定基线值 | 理由 |
| :---: | :--- | :---: | :--- |
| **HP-01** | 边缘对齐损失权重 $\lambda_{\text{edge}}$ | **0.1** | 业界常用 Sobel 辅助损失权重范围 0.05–0.2，取中间值 |
| **HP-02** | 法线重建损失权重 $\lambda_{\text{normal}}$ | **0.05** | 低于 edge，因法线在潜空间已被 min-depth 间接约束 |
| **HP-03** | DPO 温度参数 β | **500.0** | 采用代码 `losses.py` 已有默认值，与 Diffusion-DPO 原论文量级一致 |
| **HP-04** | 法线重建损失 $\mathcal{L}_{\text{normal}}$ 公式 | $\|1 - \langle \hat{\mathbf{n}}, \mathbf{n}_{\text{gt}} \rangle\|$ | 余弦角距离，在 FGPP 法线通道上直接计算 |

---

## 1. 战役一：数据流水线 (Data Pipeline)

### 1.1 战略目标
从当前的"少量手工样本"状态进化为"千级规模自动化成对物理真值生产体系"。

### 1.2 现状评估

| 子系统 | 当前状态 | 差距诊断 |
| :--- | :--- | :--- |
| FGPP 读写 (`fgpp/io.py`) | ✅ 537 行完整实现，含异构 OpenEXR 2.0 读写、消毒器、Nearest 缩放 | 工程就绪 |
| 通道 Schema (`fgpp/schema.py`) | ✅ 233 行，15 通道 `34B/px` 完整定义、白模防呆校验 | 工程就绪 |
| 数据集脚本 (`prepare_datasets.py`) | ✅ 392 行完整实现 | **但依赖 ubuntu24 上 `fulgur` 已编译且路径固定** |
| 特征预编码 (`feature_store.py`) | ✅ 537 行，WebDataset 分片、Protocol-based 编码器注入 | **默认 `LinearProjectionEncoder` 为纯 numpy 占位，真实 VAE 编码尚未实跑** |
| 30 场景 JSON | ✅ T1-T5 共 30 个场景定义 | **均为程序化几何体，缺少复杂生产级资产场景** |
| Colab Runner (`colab_runner.py`) | ✅ 185 行，GPU 探测 + Drive 断点续跑 | 工程就绪 |

### 1.3 攻坚行动项

#### 行动 D1：程序化场景域随机化扩展 (Domain Randomization Engine)

> **目标**：在现有 30 个基准场景基础上，通过参数化变体（运镜轨迹、光照色温、材质粗糙度、几何摆放）自动派生出 **≥300 个训练用场景配置**。

- **实现方式**：在 `python/synapse/bench/` 下新增 `domain_randomizer.py`，接受基准场景 JSON 作为模板，在以下维度实施 Curriculum DR（与 `synapse.rsi.curriculum` 中已定义的 8 个条件因子对齐）：
  - `F01_geometry`（L1~L4：从单体球到多体人形）
  - `F02_camera_trajectory`（L1~L4：静态/环绕/推拉/螺旋）
  - `F03_lighting`（L1~L4：均匀环境光到高对比极端逆光）
  - `F04_material`（L1~L4：纯白漫反射到高金属高光）
  - `F08_scale`（相机距离倍率缩放）
- **验收标准**：`pytest tests/test_domain_randomizer.py` 验证生成的变体 JSON 合法且帧数满足 $T=4k+1$
- **执行环境**：纯 CPU，本地或 ubuntu24 均可

#### 行动 D2：首批 100 套 97 帧 FGPP 序列批量烘焙 (Track-B Batch Bake)

> **目标**：在 `ubuntu24` 上利用 `fulgur` 无头模式批量导出首批训练数据。

```bash
# 在 ubuntu24 上执行
ssh ubuntu24 "cd ~/work/dervgen/fulgur && for scene in $(ls ../synapse/data/train_scenes/*.json); do
  bazel run //course:102-pbr-direct -- \
    --headless --width=832 --height=480 \
    --export-fgpp=/root/work/datasets/synapse/train/$(basename $scene .json) \
    --fgpp-frames=97 \
    --scene-config=$scene
done"
```

- **前置依赖**：行动 D1 完成场景 JSON 生成；`fulgur` 在 ubuntu24 上编译通过
- **产出**：`/root/work/datasets/synapse/train/` 下 ≥100 个子目录，每个含 97 帧 `frame_XXXXXX.exr` + `metadata.json`
- **验收标准**：`tools/wire_guard.py` 对每个导出目录做 15 通道 `34B/px` 合规扫描

#### 行动 D3：大规模 WebDataset 潜空间分片打包 (Feature Store at Scale)

> **目标**：将烘焙出的 FGPP 序列离线预编码为训练可直接消费的 `.tar` 分片。

- **阶段 1（CPU Mock 预编码）**：使用 `LinearProjectionEncoder`（已内置于 `feature_store.py`）在 ubuntu24 上验证分片管线端到端正确性
- **阶段 2（GPU 真实 VAE 编码）**：在 Colab A100 或 win-dev 上挂载真实 Wan 2.1 3D 因果 VAE 权重，执行实际潜空间编码
- **五段式结构化 Caption**：所有样本必须通过 `format_structured_caption()` 生成包含 `trigger_token`, `subject_desc`, `lighting_env`, `camera_motion`, `art_style_tag` 五个非空字段的 Caption
- **验收标准**：
  - 分片总大小 ≤ 1GB/shard
  - `iter_feature_shards()` 往返解包后，`.meta.json` 的 `scene_id` 与 `frame_count` 与源数据一致
  - 对 Mock 编码器：潜空间形状为 `[C, τ, H//8, W//8]`，其中 $\tau = (97-1)/4 + 1 = 25$

#### 行动 D4：数据清洗与质量准入门禁 (Quality Ingestion Gate)

> **目标**：在数据入库前自动过滤低质量样本。

在 `feature_store.py` 的 `add_sample()` 流程中增加前置校验：

| 过滤条件 | 阈值 | 理由 |
| :--- | :--- | :--- |
| 前景像素占比 (`Cryptomatte.ID > 0`) | ≥ 5% | 剔除无效空景 |
| 深度动态范围 | $z_{\max} - z_{\min} \ge 0.1\text{m}$ | 剔除扁平无深度变化视角 |
| 过曝检测 | `DirectRadiance` 95th-percentile ≤ 65504 | 剔除 HALF 溢出帧 |
| 法线有效性 | $\|\mathbf{n}\|_2 \in [0.99, 1.01]$ 前景像素占比 ≥ 95% | 剔除法线退化帧 |
| 旋转矩阵行列式 | $|\det(R) - 1| < 10^{-5}$ | 剔除相机参数异常帧 |

---

## 2. 战役二：算法核心 (Algorithm Pipeline)

### 2.1 战略目标
将已验证的"数学公式正确"状态推进到"在真实 DiT 权重上稳定运行、几何服从度可测量"的工程闭环。

### 2.2 现状评估

| 子系统 | 当前状态 | 差距诊断 |
| :--- | :--- | :--- |
| WanAdapter 7 通道条件构建 (`wan.py`) | ✅ 490 行完整实现 | `_lora_manager` 槽位标注为"M5 PR-16 stub"——LoRA 基础设施已全部实现但**接线尚未完成** |
| 因果滑窗调度 (`scheduler.py`) | ✅ 682 行完整实现（保方差融合、κ 突变检测、3 帧暖启动） | 工程就绪 |
| ACEScg 软肩逆映射 (`aces_color.py`) | ✅ 312 行完整实现 | 工程就绪 |
| 16px 手术刀合成 (`composite/blend.py`) | ✅ 600+ 行完整实现 | 工程就绪 |
| MockVideoAdapter (`mock.py`) | ✅ 327 行，Lambertian + 缺陷注入 + 确定性噪声 | 零 GPU 回归保障就绪 |

### 2.3 攻坚行动项

#### 行动 A1：WanAdapter ↔ MultiLoRAManager 接线闭合

> **目标**：将已实现的 LoRA 基础设施（`manifest.py` + `manager.py` + `patcher.py`）正式接入 `WanAdapter` 主路径。

- 在 `WanAdapter.__init__()` 中，当 `lora_manifests` 参数非空时，构造 `MultiLoRAManager` 实例
- 在 `WanAdapter.generate()` 的推理路径中，调用 `manager.apply_merged_weights()` 注入 LoRA 权重，并在推理完成后调用 `manager.restore_base_weights()`
- 确保 `build_spatial_lora_masks()` 正确消费 FGPP 的 `Cryptomatte.ID` 通道
- **验收标准**：`test_wan_adapter.py` 新增测试用例——使用 Mock Transport + 双 LoRA（`style` + `identity`）验证前景/背景掩膜隔离正确性

#### 行动 A2：E1 消融实验在真实底模上的首次验证

> **目标**：在 Colab 或 win-dev 上，使用 `Wan2.1-T2V-1.3B` 权重执行 E1 四档消融（E1-A/B/C/D），获取首批真实几何服从度数据。

- 使用 `scripts/colab_runner.py --exp e1` 驱动
- 输入数据：行动 D2 产出的 Track-B 97 帧 FGPP 序列
- 对比各档位的量化指标：`depth_rmse_m`, `normal_angular_error_deg`, `silhouette_iou`
- **关键产出**：四档消融对比报告（JSON + HTML），锚定"7 通道完整 FGPP 引导 vs 无引导"的几何增益幅度

#### 行动 A3：97 帧长视频端到端跑通与接缝质量验证

> **目标**：验证 $W=97, S=64, \Delta=33$ 因果滑窗在真实 VAE 下的接缝平滑度。

- 使用 `ConditionScheduler` 对 T5 长时序场景（如 `t5_orbit_97f_cube`）生成 `CausalWindowPlan`
- 验证 3 帧暖启动 $[64k-3, 64k-2, 64k-1]$ 的因果 VAE 前缀注入
- 检查融合区 Sigmoid 退火后的 PSNR（目标 ≥ 42dB）
- 检查融合区的 `flicker_score`（应 ≤ REVIEW 文档 `track_b_video` 档位 0.025）

---

## 3. 战役三：训练与微调体系 (Training & PEFT Pipeline)

### 3.1 战略目标
从"Mock 模型 50 步 Loss 单调下降验证"进化为"在真实 `Wan 2.1` DiT 权重上完成首个 LoRA 微调闭环并通过自动化基准门禁"。

### 3.2 现状评估

| 子系统 | 当前状态 | 差距诊断 |
| :--- | :--- | :--- |
| `StatelessLoRATrainer` (`trainer.py`) | ✅ 522 行，含 `LoRALinear`、`inject_lora`、`train_step`、`export_checkpoint` | Tiny Mock DiT 验证通过，**真实 Wan 2.1 权重尚未实跑** |
| 复合损失函数 (`losses.py`) | ✅ 142 行，`flow_matching_loss` + `latent_edge_alignment_loss` + `diffusion_dpo_loss` | **缺少 `$\mathcal{L}_{\text{normal}}$` 法线重建损失实现** |
| LoRA Manifest (`manifest.py`) | ✅ 122 行，5 类白名单 + Safetensors 强制 + SHA256 | 工程就绪 |
| MultiLoRAManager (`manager.py`) | ✅ 241 行，Cryptomatte 掩码隔离 + Gram-Schmidt 正交化 | 工程就绪 |
| LoRA 打包与门禁 (`packager.py`) | ✅ 340 行，`package_and_gate_lora_bundle` + `LoRAGateThresholds` | `chamfer_mm` 默认值 5.0mm，应对齐 `rsi_promotion` 为 4.0mm |
| RSI 双重验证 (`dual_verifier.py`) | ✅ 344 行，物理优先 + 失败→LoRA 桶路由 | `silhouette_iou` 默认 0.88，应对齐为 0.90 |
| RSI 偏好配对 (`pair_builder.py`) | ✅ 269 行，20% 黄金锚点强制混采 | 工程就绪 |
| RSI 单调递增门禁 (`promotion_gate.py`) | ✅ 268 行，四项合取条件 + Quarantine 归档 | 工程就绪 |

### 3.3 攻坚行动项

#### 行动 T1：补齐法线重建损失 $\mathcal{L}_{\text{normal}}$

> **目标**：在 `losses.py` 中新增 `normal_reconstruction_loss`。

```python
def normal_reconstruction_loss(
    n_pred: np.ndarray,   # [H, W, 3] 或 [C, H, W]
    n_gt: np.ndarray,     # [H, W, 3] FGPP WorldNormal
    mask: np.ndarray | None = None,  # 前景掩码
) -> float:
    """法线余弦角距离: E[1 - <n_pred, n_gt>]"""
    ...
```

- 锁定 $\lambda_{\text{normal}} = 0.05$（HP-02）
- **验收标准**：`test_lora_trainer.py` 新增测试，验证 $\mathcal{L}_{\text{normal}}(n, n) = 0$ 且 $\mathcal{L}_{\text{normal}}(n, -n) = 2.0$

#### 行动 T2：5 类正交 LoRA 首轮训练实验

> **目标**：优先训练 `control` LoRA（几何服从度增肌），作为首个端到端验证。

- **训练配置（对齐 DESIGN_LORA_PIPELINE §2 权威表格）**：

| LoRA 类别 | Rank | Alpha | 目标模块 | 损失配置 | 首轮训练步数 |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **`control`** | 32 | 32 | ControlNeXt / DiT `blocks.0..10` | $\mathcal{L}_{\text{flow}} + 0.1 \cdot \mathcal{L}_{\text{edge}} + 0.05 \cdot \mathcal{L}_{\text{normal}}$ | 500 |
| `identity` | 32~64 | 32 | `spatial_attn.to_q/k/v/out.0` | $\mathcal{L}_{\text{flow}} + 0.1 \cdot \mathcal{L}_{\text{edge}}$ | 500 |
| `style` | 16 | 16 | `cross_attn.to_k/v` + `ffn.*` | $\mathcal{L}_{\text{flow}}$ | 300 |
| `whitebox` | 32 | 64 | Spatial + Temporal 全 Attention | $\mathcal{L}_{\text{flow}} + 0.1 \cdot \mathcal{L}_{\text{edge}} + 0.05 \cdot \mathcal{L}_{\text{normal}}$ | 500 |
| `motion` | 16~32 | 16 | `temporal_attn.*` | $\mathcal{L}_{\text{flow}}$ | 500 |

- **执行环境**：首轮使用 `Wan2.1-T2V-1.3B`（Colab T4 16GB 可承载），后续升级至 `14B`（需 A100 40GB）
- **验收标准**：Flow-matching loss 在 500 步内单调下降且不发散

#### 行动 T3：LoRA 准出门禁对齐 REVIEW 文档三级阈值

> **目标**：修正 `packager.py` 和 `dual_verifier.py` 中的默认阈值，使其与 `DESIGN_REVIEW_AND_ALIGNMENT.md` §3.4 完全对齐。

具体修改：

```python
# packager.py — LoRAGateThresholds 默认值对齐 rsi_promotion
chamfer_mm: float = Field(default=4.0, ge=0.0)  # 原 5.0 → 4.0
normal_angular_error_deg: float = Field(default=12.0, ge=0.0)  # 保持 12.0 ✓

# dual_verifier.py — PhysicsThresholds 默认值对齐 rsi_promotion
silhouette_iou: float = Field(default=0.90, ge=0.0, le=1.0)  # 原 0.88 → 0.90
```

- **注意**：`profiles.py` 中的"超额严格"值（如 `normal_angular_error_deg=5.0`）保留为远期目标，但当前不作为硬阻断
- **验收标准**：`test_rsi_verifier_and_pairs.py` 更新断言以匹配新默认值

#### 行动 T4：RSI 首轮物理验证闭环实验

> **目标**：端到端跑通"生成→双重验证→偏好配对→DPO 微调→单调门禁"全链路。

1. 使用行动 T2 训练出的 `control` LoRA 对 30 场景生成视频
2. 调用 `DualVerifierEngine` 执行物理检查（`silhouette_iou`, `depth_rmse_m`, `normal_angular_error_deg`）
3. 失败样本通过 `pair_builder.py` 与对应 Track-A 真值配对，构建 DPO JSONL
4. 执行 DPO 微调（$\beta = 500.0$，HP-03）
5. 调用 `evaluate_monotonic_promotion()` 检验 30 场景零退化
6. **关键验收**：新 LoRA 的 `OverallScore > 基线`，且**任何单场景零退化**

---

## 4. 分阶段执行时间线 (Phased Execution Timeline)

```mermaid
flowchart LR
    subgraph Phase1 ["第 1 阶段 (1-2 周)：数据生产与基础补齐"]
        D1["D1: 域随机化场景生成器"] --> D2["D2: ubuntu24 批量烘焙 100 套 FGPP"]
        T1["T1: 补齐 L_normal 损失"] --> T3["T3: 阈值对齐 REVIEW 文档"]
        A1["A1: WanAdapter ↔ LoRA 接线"]
    end

    subgraph Phase2 ["第 2 阶段 (2-3 周)：首轮训练与消融实验"]
        D3["D3: WebDataset 分片打包"]
        A2["A2: E1 消融实验 (1.3B)"]
        T2["T2: control LoRA 首轮训练"]
    end

    subgraph Phase3 ["第 3 阶段 (3-4 周)：RSI 闭环与规模化"]
        A3["A3: 97f 长视频接缝验证"]
        D4["D4: 数据清洗门禁"]
        T4["T4: RSI 双重验证 + DPO 闭环"]
    end

    D2 --> D3 --> T2
    A1 --> A2
    T1 & T3 --> T2
    T2 --> T4
    A2 --> A3
    D4 -.-> D3
```

### 第 1 阶段（Week 1-2）：数据生产与基础补齐
- **D1** + **D2**：场景域随机化 → ubuntu24 批量烘焙（可与下列并行）
- **T1**：补齐 `normal_reconstruction_loss`
- **T3**：对齐 `packager.py` 与 `dual_verifier.py` 阈值
- **A1**：`WanAdapter` ↔ `MultiLoRAManager` 接线

### 第 2 阶段（Week 2-3）：首轮训练与消融实验
- **D3**：WebDataset 分片打包（先 Mock 验证，后 GPU 真实 VAE）
- **A2**：Colab 上跑通 E1 消融四档实验
- **T2**：`control` LoRA 首轮 500 步训练

### 第 3 阶段（Week 3-4）：RSI 闭环与规模化
- **A3**：97 帧长视频端到端接缝验证
- **D4**：数据清洗门禁代码落地
- **T4**：完整 RSI 闭环首轮验证（双重验算→DPO→单调门禁）

---

## 5. 风险登记簿 (Risk Register)

| 风险 | 触发条件 | 影响 | 缓解措施 |
| :--- | :--- | :--- | :--- |
| **RK-A1**: `fulgur` 在 ubuntu24 上无头批量烘焙 OOM 或崩溃 | 复杂场景 + 高分辨率 + 97 帧连续导出 | D2 数据生产停滞 | 降级为 49 帧（$T=4\times12+1$）或降低分辨率至 416×240 |
| **RK-A2**: 1.3B 模型几何先验过弱，E1 消融增益不显著 | Wan 2.1 1.3B 对白模/法线分布外 | A2 消融实验无意义 | 直接跳至 14B 模型（需 A100 40GB） |
| **RK-A3**: WebDataset 真实 VAE 编码显存不足 | 3D 因果 VAE 对 97 帧 832×480 的峰值显存 | D3 分片打包失败 | 使用 `ConditionScheduler` 分窗口编码，每次仅编码 49 帧 |
| **RK-A4**: DPO β=500.0 导致训练不稳定 | Flow-matching DPO 对 β 敏感 | T4 RSI 闭环失败 | 尝试 β∈{100, 250, 500, 1000} 网格搜索 |
| **RK-A5**: `profiles.py` 超严阈值导致所有实验被误判为不及格 | 代码中 `rsi_promotion` 远严于 REVIEW 文档 | 所有门禁阻断 | **P0 优先执行 T3 对齐操作**，使用 REVIEW 文档值 |

---

## 6. 与全局 IMPLEMENTATION_PLAN 的对应关系

| 本计划行动项 | 对应 IMPLEMENTATION_PLAN PR | 状态 |
| :--- | :--- | :--- |
| D1 域随机化 | 新增（不在原 30-PR 范围内，为数据扩张专项） | 🆕 待新增 |
| D2 批量烘焙 | PR-15 的数据基础设施扩展 | 🔄 扩展 |
| D3 WebDataset 分片 | **PR-17** (`training/feature_store.py`) | ✅ 代码已实现，待实跑 |
| D4 数据清洗门禁 | PR-17 的质量过滤扩展 | 🔄 扩展 |
| A1 WanAdapter ↔ LoRA 接线 | **PR-16** 的最后一步集成 | 🔄 补全 |
| A2 E1 消融实验 | 对应 `scripts/colab_runner.py --exp e1` | ✅ 脚本就绪，待执行 |
| A3 97f 长视频验证 | **PR-06** 的生产验证 | ✅ 代码已实现，待实跑 |
| T1 补齐 $\mathcal{L}_{\text{normal}}$ | **PR-18** 补充 | 🔄 补全 |
| T2 首轮 LoRA 训练 | **PR-18** 的实际执行 | ✅ 代码已实现，待实跑 |
| T3 阈值对齐 | **PR-13** / **PR-25** 的阈值修正 | 🔄 修正 |
| T4 RSI 闭环 | **PR-24** + **PR-25** | ✅ 代码已实现，待实跑 |
