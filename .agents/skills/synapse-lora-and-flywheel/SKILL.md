---
name: synapse-lora-and-flywheel
description: >-
  Synapse LoRA & PEFT 微调流水线、Cryptomatte 多实体空域掩码、3D 资产智能质检（synapse.inspect）与 RSI 物理驱动自演进飞轮指南。涵盖 FeatureStoreBuilder WebDataset 分片、LoRAManifest 安全校验、MultiLoRAManager 正交合并、Synapse-Bench 评测及 3D 模型评估脚本与 HTML 报告（三级门禁、SVG 雷达图）。在开发训练管线、质检路由、RSI 飞轮或模型评估时使用。
---

# LoRA 微调、3D 智能质检与 RSI 自演进飞轮指南

> 本技能涵盖 `synapse.training`、`synapse.adapters.lora`、`synapse.inspect`、`synapse.rsi` 与 `synapse.bench` 的核心工作流与安全规范。

---

## 一、WebDataset 潜变量分片与结构化 Caption (`synapse.training`)

### 1. 五段式确定性 Caption (`format_structured_caption`)
所有训练样本 Caption 必须包含 5 个非空结构化字段（任一为空即抛出 `ContractValidationError`）：
```python
from synapse.training import format_structured_caption

caption = format_structured_caption(
    trigger_token="sks_hero",
    subject_desc="armored knight with cape",
    lighting_env="dramatic rim light in dark cathedral",
    camera_motion="slow dolly in",
    art_style_tag="cinematic_pbr",
)
```

### 2. 离线潜变量预编码与 `.tar` 分片读写 (`FeatureStoreBuilder`)
将高成本的 3D 因果 VAE 编码与文本编码在训练前离线完成并打包为 WebDataset `.tar` 分片（包含 `.latent.npy`、`.prompt.npy`、`.fgpp_cond.npy`、`.meta.json`）：
```python
from synapse.training import FeatureStoreBuilder, iter_feature_shards

builder = FeatureStoreBuilder(latent_encoder=vae_enc, text_encoder=txt_enc)
builder.add_sample(
    sample_id="shot_001",
    video_frames=rgb_u8_frames,  # [T, H, W, 3] uint8
    caption=caption,
    fgpp_cond=control_tensor,    # [7, T, H, W] float32
)
shard_paths = builder.build_shards(output_dir)

for sample in iter_feature_shards(shard_paths):
    latent = sample["latent"]    # [C_latent, tau, H//8, W//8]
    prompt = sample["prompt"]    # [D]
```

---

## 二、LoRA 安全清单与 Cryptomatte 多实体空域掩码 (`synapse.adapters.lora`)

### 1. `LoRAManifest` 三重安全守卫 (`synapse.lora.v1.0`)
1. **严禁 Pickle 权重**：`adapter_path` 必须以 `.safetensors` 结尾（`.pt` / `.bin` 立即拒绝）。
2. **SHA256 強校验**：`adapter_sha256` 必须为 64 位小写十六进制哈希。
3. **分类目标模块白名单**：
   - `"style"`：仅允许 `cross_attn\.to_k`, `cross_attn\.to_v`, `ffn\..*`
   - `"identity"`：仅允许 `spatial_attn\..*`, `cross_attn\.to_k`, `cross_attn\.to_v`
   - `"control"` / `"whitebox"`：仅允许 `spatial_attn\..*`, `ffn\..*`
   - `"motion"`：仅允许 `temporal_attn\..*`

### 2. `MultiLoRAManager` 空域注意力掩码隔离
通过 `build_spatial_lora_masks` 将 `[T, H, W]` `uint32` `Cryptomatte.ID` 以 $8\times 8$ 块主导实体投票下采样至潜空间网格 `[T, H//8, W//8]`，确保前景身份 LoRA（激活于 $M_{\text{fg}}$）与背景风格 LoRA（激活于 $\neg M_{\text{fg}}$）零串扰（实测跨实体掩码泄漏率 `0.0 < 3%`）。

---

## 三、3D 建模智能质检 (`synapse.inspect`) 与 RSI 自演进飞轮 (`synapse.rsi`)

1. **Tier 0 确定性几何拓扑硬门禁 (`synapse.inspect.geometry_gate`)**：
   - 双轨架构互验：生产级 `csrc/geometry/` C++20 算子核（Bazel + `@aether//math`）与 `synapse.recon.geometry` Python Oracle；
   - 包含流形（2-Manifold）、水密闭合（Watertight）、简并面（Degenerate Faces）、自相交（Self-Intersections）与浮动碎岛过滤；
   - `validate_mesh_geometry(vertices, faces, require_watertight=...)` 输出结构化 `GeometricValidationResult`，前置硬阻断劣化网格进入下游神经评测。
2. **白模粗糙度防呆门禁（RFC 0044 §4.4）**：
   - 角色代理必须通过 `validate_whitebox_coarseness_floor(vertex_count, has_articulated_limbs, is_character=True)`（要求顶点数 $\ge 1500$ 且具备铰接四肢）。
   - `synapse.bench.generate_humanoid_proxy()` 生成包含 1991 顶点、14 个骨骼分段的程序化标准人形白模。
3. **三级质检分流与 15 分钟熔断**：
   - `L0`（全自动放行）$\to$ `L1`（算法自动自愈：法线重算、流形修复、反照率去高光裁剪）$\to$ `L2`（导出 `.inspection.tar.zst` 交由人工精修，单资产/单镜头超时 **15 分钟** 立即触发退出码 `40` 工程熔断降级）。
4. **RSI 闭环演进 (`synapse.rsi`)**：
   - 结合物理渲染一致性奖励（几何 Chamfer、材质能量守恒、骨骼骨长零漂移）与 `ReplayBuffer` 20% 历史黄金锚点混采，防止多轮自演进出现色彩过饱和或高频伪影坍缩。

---

## 四、3D 模型评估模块、CLI 与 HTML 报告 (`synapse.inspect.model_evaluator` & `eval_report`)

> 独立于 Visual Jev 判别流水线，提供零重型外部依赖的确定性 3D 模型拓扑、自穿插、参考网格几何精度与多视角渲染质量评估。

### 1. 统一 CLI 与脚本入口 (`synapse inspect evaluate` / `scripts/evaluate_3d_models.py`)
```bash
# 主 CLI 入口（同时兼容 python scripts/evaluate_3d_models.py ...）
synapse inspect evaluate \
    --input-dir /path/to/models \
    --output-dir /path/to/report \
    --reference-dir /path/to/ref_models \
    --profile TRACK_A_ASSET \
    --num-views 8 --resolution 256
```
> 兼容参数别名：`--gate-profile` 等价于 `--profile`，`--views` 等价于 `--num-views`。

**三级门禁档位** (`--profile` / `--gate-profile`，非法档位按 Fail-Closed 返回退出码 `10`)：
| 档位 | 适用场景 | 拓扑门禁 | `chamfer_mm` 阈值 |
| :--- | :--- | :--- | :---: |
| `TRACK_B_VIDEO` | 白模视频预演快速筛选 | `is_manifold` | `≤ 8.0 mm` |
| `TRACK_A_ASSET` | 生产 3D 资产准出 | `is_manifold` + `is_watertight` + 零退化面/自穿插 | `≤ 5.0 mm` |
| `RSI_PROMOTION` | LoRA/RSI 自演进模型晋升 | `is_manifold` + `is_watertight` + 零退化面/自穿插 | `≤ 4.0 mm` |

**退出码**：`0`=全部 PASS，`10`=参数或档位名校验失败，`20`=输入目录/模型缺失，`40`=存在模型门禁未达标（`"status": "gate_failed"`，HTML 与 JSON 报告仍完整生成）。

### 2. 评估报告数据模型 (`synapse.inspect.eval_report`)
```python
from synapse.inspect.eval_report import (
    TopologyResult,       # 拓扑检查结果 (vertices, faces, manifold, watertight, ...)
    ModelEvalResult,      # 单模型评估结果 (topology, screenshots, metrics, gate_results)
    EvalReportData,       # 完整报告数据 (models[], profile_name, timestamp)
    generate_eval_html_report,  # 生成自包含 HTML 报告 (内嵌 CSS/SVG 雷达图/Base64 截图)
)
```

### 3. 核心评估指标与正向归一化雷达图
- **无参考网格必评指标**：`is_manifold`、`is_watertight`、`num_degenerate_faces`、`self_intersecting_pairs`、`connected_components`、`boundary_edges`、`floating_island_faces`（由 `validate_mesh_geometry` 双轨几何核计算）；
- **提供 `--reference-dir` 同名网格时附加指标**：几何精度 `chamfer_mm`、`f_score_1mm`，以及多视角同位软光栅渲染画质 `psnr_db`、`ssim`、`ciede2000`、`lpips`；
- **雷达图正向归一化**：所有误差/缺陷计数指标严格映射为 `0 缺陷 → 1.0 满分`，并剔除 `vertex_count`/`face_count` 等非质量规模量，杜绝雷达图方向反转。

### 4. 测试覆盖
`tests/test_eval_report.py`、`tests/test_geometry_oracle_parity.py` 与 `tests/test_metrics_image_quality.py` 全面覆盖数据模型、C++/Python 双轨几何一致性、雷达图正向归一化、多视角画质指标与 CLI 确定性退出码。

> 完整交付清单与 Review 指南：[docs/REVIEW_3D_EVAL_DELIVERY.md](../../../docs/REVIEW_3D_EVAL_DELIVERY.md)
