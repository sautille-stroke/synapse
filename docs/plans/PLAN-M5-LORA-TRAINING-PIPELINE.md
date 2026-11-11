# 细分执行计划 3：Phase M5 合成数据潜空间缓存与五维正交解耦 LoRA 微调流水线 (`PR-16` ~ `PR-19`)

- **所属总计划**: [docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) (`v2.0` Horizon 2)
- **前置设计文档**: [docs/DESIGN_LORA_PIPELINE.md](../DESIGN_LORA_PIPELINE.md) · [docs/DESIGN_REVIEW_AND_ALIGNMENT.md](../DESIGN_REVIEW_AND_ALIGNMENT.md) (`RV-02`, `RV-03`)
- **覆盖范围**: `PR-16` ~ `PR-19`（共 4 个原子 PR，预估 5.5 人天）
- **核心原则**: 训练与推理严格解耦（重型 3D VAE / T5 离线预编码）、五维 LoRA 正交隔离（`style` / `identity` / `control` / `whitebox` / `motion`）、`Cryptomatte.ID` 前后景空间掩膜防串色、Safetensors 唯一安全格式（严禁 `.bin` pickle）。

---

<a id="pr-16"></a>
## 1. PR-16：LoRA 资产契约 Schema 与运行时 `MultiLoRAManager`（含 Cryptomatte 掩膜隔离与正交投影）

- **建议标题**: `feat(lora): add LoRAManifest v1.0 schema and Cryptomatte-aware MultiLoRAManager [model-tag]`
- **前置依赖**: `PR-07` (`WanAdapter`)
- **代码净增预算**: `~520` 行
- **涉及文件清单**:
  - 新增：`python/synapse/adapters/lora/__init__.py`, `python/synapse/adapters/lora/manifest.py`, `python/synapse/adapters/lora/manager.py`, `python/synapse/adapters/lora/patcher.py`, `tests/test_lora_manager.py`
  - 修改：`python/synapse/adapters/wan.py`

### 1.1 精确落地规格
1. **`adapters/lora/manifest.py` (`LoRAManifest`, `schema_version = "synapse.lora.v1.0"`)**：
   - 强类型校验 `category: Literal["style", "identity", "control", "whitebox", "motion"]`；
   - 校验每类 LoRA 允许挂载的 `target_modules` 白名单（例如 `style` 仅允许 Cross-Attention `to_k/to_v` 与 FFN，禁止篡改 `temporal_attn`；`motion` 仅允许 `temporal_attn.*`，从契约源头锁死正交性！）；
   - 拒绝加载任何 `.bin` / `.pt` pickle 权重路径，强制要求 `adapter_model.safetensors` 并校验文件 SHA256。
2. **`adapters/lora/manager.py` (`MultiLoRAManager`)**：
   - **快速线性合并与无损卸载 (`apply_merged_weights` / `restore_base_weights`)**：维护增量残差字典 $\Delta W = \sum_k s_k \frac{\alpha_k}{r_k} B_k A_k$，推理结束精确回退（支持浮点误差校验或原权重快照引用，杜绝多次切换累积数值漂移）；
   - **Cryptomatte 前后景潜空间隔离 (`build_spatial_lora_masks`)**：将 FGPP `Cryptomatte.ID` 通过 $8\times 8$ 主导实体下采样映射至潜空间网格 $[\tau, H/8, W/8]$：`identity` LoRA 仅在目标角色 ID 掩膜 $M_{\text{fg}}$ 内激活，`style` LoRA 默认在 $(1 - M_{\text{fg}})$ 或低频缩放因子下激活，防止角色皮肤被风格滤镜偏色；
   - **梯度冲突正交投影 (`orthogonalize_conflicting_loras`)**：当两个 LoRA 作用于同一线性层且扁平化余弦相似度 $|\cos\theta| > 0.25$ 时，采用 Gram-Schmidt 投影剔除次级 LoRA 在主角色 LoRA 子空间上的平行分量。
3. **`adapters/lora/patcher.py` & `WanAdapter` 接线**：
   - 为类 `nn.Linear` 模块提供可插拔的空间掩膜前向 Hook，并在 `WanAdapter` 中支持传入 `lora_bundles: list[tuple[Path, float]]`。

### 1.2 验证门禁
- 在 `ubuntu24` CPU 环境用小型合成 `nn.Module`（模拟 DiT Spatial/Temporal/Cross Attention 层）运行单测：1) 加载 2 个 LoRA 推理后再 `restore_base_weights()`，权重与初始值逐元素误差 `< 1e-7`；2) 开启 Cryptomatte 掩膜后，前景 Token 输出仅受 `identity` LoRA 影响，背景 Token 仅受 `style` LoRA 影响。

---

<a id="pr-17"></a>
## 2. PR-17：离线潜空间特征预编码与 `WebDataset` 分片缓存器 (`FeatureStore`)

- **建议标题**: `feat(training): implement offline 3D VAE/FGPP pre-encoding and WebDataset shard builder [model-tag]`
- **前置依赖**: `PR-16`
- **代码净增预算**: `~490` 行
- **涉及文件清单**:
  - 新增：`python/synapse/training/__init__.py`, `python/synapse/training/feature_store.py`, `tests/test_lora_feature_store.py`
  - 修改：`pyproject.toml`（新增可选组 `train = ["safetensors>=0.4.0", "pyyaml>=6.0"]`）

### 2.1 精确落地规格
1. **结构化合成打标器 (`format_structured_caption`)**：
   - 强制拼接五段式确定性描述：`"{trigger_token}, {subject_desc}, {lighting_env}, {camera_motion}, {art_style_tag}"`。
2. **离线编码协议 (`PreEncodeBackend` + `FeatureStoreBuilder`)**：
   - 抽象 `LatentEncoderProtocol` 与 `TextEncoderProtocol`（生产环境绑定 `Wan 2.1` 3D Causal VAE 与 UMT5-XXL，单测/CI 注入轻量确定性线性投影编码器）；
   - 调用 `synapse.adapters.scheduler.compute_latent_flow_operator` 与 $8\times 8$ **最小深度池化（Min-Depth Pooling）** + 相机空间法线重采样，生成与潜空间张量 `[16, τ, H/8, W/8]` 严格对齐的条件张量 `fgpp_cond`；
   - 按单分片上限（默认 `max_shard_bytes = 2 * 1024**3`）写出标准 POSIX `.tar` `WebDataset` 分片（`shard-00000.tar`），每条样本包含 `.latent.npy/.safetensors`、`.prompt.safetensors`、`.fgpp_cond.safetensors` 与 `.meta.json`（含源序列 SHA256），并提供流式读取迭代器 `iter_feature_shards()`。
3. **注册 `JobBundleWorker` 新阶段**：动态注册 `stage = "lora_preencode"`。

### 2.2 验证门禁
- 构造 3 组 17 帧合成 FGPP + 目标视频帧，打包为 2 个分片 `.tar` 后流式读回：校验张量维度 `[16, 5, H/8, W/8]` 精确匹配 $T=4k+1$ 法则，且所有样本哈希校验 100% 通过。

---

<a id="pr-18"></a>
## 3. PR-18：无状态五维正交 DiT PEFT 训练器（Flow-Matching + 几何边缘/法线辅助损失 + DPO）

- **建议标题**: `feat(training): add stateless DiT LoRA trainer with Flow-Matching, geometric edge, and DPO losses [model-tag]`
- **前置依赖**: `PR-17`
- **代码净增预算**: `~580` 行
- **涉及文件清单**:
  - 新增：`python/synapse/training/losses.py`, `python/synapse/training/trainer.py`, `tests/test_lora_trainer.py`

### 3.1 精确落地规格
1. **`training/losses.py` 复合训练目标**：
   - `flow_matching_loss(v_pred, x_1, x_0)`：标准 Rectified Flow 速度场均方误差 $\mathbb{E}\|v_\theta(x_t, t, c) - (x_1 - x_0)\|_2^2$；
   - `latent_edge_alignment_loss(z_0_hat, depth_latent_minpool)`：通过 2D Sobel 梯度算子约束预测去噪潜空间特征梯度与池化对数深度跳变边缘一致（专供 `control` 与 `whitebox` LoRA）；
   - `diffusion_dpo_loss(v_theta_w, v_theta_l, v_ref_w, v_ref_l, beta=500.0)`：面向 `Phase M7` RSI 自演进偏好对 $(x, y_{\text{chosen}}, y_{\text{rejected}})$ 的流匹配 DPO 损失函数！
2. **`training/trainer.py` (`StatelessLoRATrainer`)**：
   - 解析强类型 YAML/JSON 训练配置 `LoRATrainConfig`（指定 `category`, `rank`, `alpha`, `lr_spatial`, `lr_temporal`, `max_steps`, `grad_accum_steps`, `mixed_precision="bf16"`, `loss_mode="sft"|"dpo"`）；
   - 自动冻结基础骨干模型全部参数（断言 `requires_grad` 参数占比 `< 3%`），仅对目标类别白名单子层注入低秩矩阵 $A \sim \mathcal{N}(0, \sigma^2), B = \mathbf{0}$（保证第 `0` 步输出与基座 100% 恒等！）；
   - 注册 `JobBundleWorker` `stage = "lora_train"`，单次训练完毕写出检查点与损失曲线摘要后立即释放显存退出。

### 3.2 验证门禁
- 在 `ubuntu24` 使用微型 2 层 Toy DiT 模型（隐藏维 `64`）与合成潜空间对运行 40 步 SFT 及 20 步 DPO 训练：1) 验证第 0 步 $\Delta W \equiv 0$；2) 验证训练后目标损失下降 $> 60\%$ 且基础冻结权重无任何 bit 改变。

---

<a id="pr-19"></a>
## 4. PR-19：LoRA 自洽资产打包、`Synapse-Bench` 自动化退化门禁与 `synapse lora` CLI 集成

- **建议标题**: `feat(lora): add safetensors bundle packager, benchmark gatekeeper, and synapse lora CLI [model-tag]`
- **前置依赖**: `PR-15`, `PR-18`
- **代码净增预算**: `~440` 行
- **涉及文件清单**:
  - 新增：`python/synapse/training/packager.py`, `tests/test_lora_cli_gate.py`
  - 修改：`python/synapse/cli/__init__.py`

### 4.1 精确落地规格
1. **`training/packager.py` (`package_and_gate_lora_bundle`)**：
   - 将训练权重导出为 `adapter_model.safetensors`，自动生成完整 `lora_manifest.json`（记录训练集 SHA256、步数、最终 Loss、操作模型标识）与 `evaluation_report.json`；
   - 串联 `BenchPipelineRunner` 在指定黄金场景与负向泛化场景上评测：校验 `rsi_promotion` 档位（如 `chamfer_mm <= 5.0`, `normal_angular_error_deg <= 12.0`, `border_ring_ciede2000 <= 2.5` 及负向场景退化幅度 `< 8%`）；任一硬指标不达标立即抛出 `QualityGateError (exit_code=40)` 阻断入库。
2. **统一挂载 `synapse lora` CLI 子命令族**：
   - 完整落地 `synapse lora prepare-data`、`synapse lora train`、`synapse lora eval`、`synapse lora merge` 四个子命令（全部支持 `--json`）。

### 4.2 验证门禁
- 子进程运行 `synapse lora eval --json`：输入达标测试 Bundle 返回 `exit_code == 0`；输入人为注入偏色退化的 Bundle 精确返回 `exit_code == 40` 且输出非空 `agent_fix_hint`。
