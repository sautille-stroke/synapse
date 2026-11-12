---
name: synapse-lora-and-flywheel
description: >-
  Synapse LoRA & PEFT 微调流水线、Cryptomatte 多实体空域掩码、3D 资产智能质检（synapse.inspect）与 RSI 物理驱动自演进飞轮指南。涵盖 FeatureStoreBuilder WebDataset 分片、LoRAManifest 安全校验、MultiLoRAManager 正交合并及 Synapse-Bench 评测。在开发训练管线、质检路由或 RSI 飞轮时使用。
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

1. **白模粗糙度防呆门禁（RFC 0044 §4.4）**：
   - 角色代理必须通过 `validate_whitebox_coarseness_floor(vertex_count, has_articulated_limbs, is_character=True)`（要求顶点数 $\ge 1500$ 且具备铰接四肢）。
   - `synapse.bench.generate_humanoid_proxy()` 生成包含 1991 顶点、14 个骨骼分段的程序化标准人形白模。
2. **三级质检分流与 15 分钟熔断**：
   - `L0`（全自动放行）$\to$ `L1`（算法自动自愈：法线重算、流形修复、反照率去高光裁剪）$\to$ `L2`（导出 `.inspection.tar.zst` 交由人工精修，单资产/单镜头超时 **15 分钟** 立即触发退出码 `40` 工程熔断降级）。
3. **RSI 闭环演进 (`synapse.rsi`)**：
   - 结合物理渲染一致性奖励（几何 Chamfer、材质能量守恒、骨骼骨长零漂移）与 `ReplayBuffer` 20% 历史黄金锚点混采，防止多轮自演进出现色彩过饱和或高频伪影坍缩。
