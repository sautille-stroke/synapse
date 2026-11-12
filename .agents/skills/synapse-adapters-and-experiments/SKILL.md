---
name: synapse-adapters-and-experiments
description: >-
  Synapse 视频生成适配器、因果滑窗调度、ACEScg 风格锁、16px 手术刀合成以及 ubuntu24 + Google Colab 双阶段实验执行指南。涵盖 WanAdapter 7 通道条件构建、ConditionScheduler (W=97, S=64, overlap=33)、run_cpu_verification.py 无卡验证、workenv/google-colab 一站式部署与 colab_runner.py 断点续跑。在编写适配器代码或执行 E1-E5 实验时使用。
---

# 视频生成适配器与双阶段实验执行指南（`ubuntu24` + Google Colab）

> 本技能覆盖 `synapse.adapters`、`synapse.style`、`synapse.composite` 的核心算法契约，以及在无卡 `ubuntu24` 与 Google Colab GPU 节点上协同完成 E1–E5 实验的标准操作流程。

---

## 一、核心算法与 API 速查

### 1. 7 通道控制张量构建 (`WanAdapter.prepare_conditions`)
`WanAdapter` 将 `FGPPSequence` 转换为形状为 **`[7, T, H, W]`** (`float32`) 的空间控制张量 `payload.control_tensor`：
- **Channel 0**：对数归一化线性深度 $d_{\log} = \frac{\ln(\text{clip}(z, z_{\text{near}}, z_{\text{far}}) / z_{\text{near}})}{\ln(z_{\text{far}} / z_{\text{near}})} \in [0, 1]$
- **Channels 1–3**：世界空间单位法线 $(n_x, n_y, n_z)$（源自 `FGPPFrame.world_normal`，即 FGPP `WorldNormal.XYZ`）
- **Channels 4–5**：归一化帧间运动向量 $(v_x, v_y)$
- **Channel 6**：前后帧重投影遮挡掩码（Disocclusion Mask）

**E1 消融实验四档配置**：
- `E1-A`：仅 RGB 关键帧（全零控制通道 `[]`）
- `E1-B`：RGB + Log-Depth（激活通道 `[0]`）
- `E1-C`：RGB + Log-Depth + Normal（激活通道 `[0, 1, 2, 3]`）
- `E1-D`：完整 7 通道 FGPP 控制（激活通道 `[0, 1, 2, 3, 4, 5, 6]`）

### 2. 长时序因果滑窗与切镜重锚定 (`synapse.adapters.scheduler`)
- **窗口划分 (`CausalWindowPlan`)**：默认 `window_frames=97`（$\tau=25$ 潜空间步）、`stride_frames=64`（$\Delta\tau=16$）、`overlap_frames=33`（$\tau_{\text{overlap}}=9$）；对于窗口 $k \ge 1$，自动注入前序 3 帧 `[stride*k - 3 .. stride*k - 1]` 作为 3D 因果 VAE 暖启动前缀。
- **潜空间保方差融合 (`fuse_overlapping_latents`)**：
  $$z_{\text{fused}}(\tau') = \frac{(1 - \alpha(\tau')) z_{\text{prev}}(\tau') + \alpha(\tau') z_{\text{curr}}(\tau')}{\sqrt{(1 - \alpha(\tau'))^2 + \alpha(\tau')^2}}, \quad \alpha(\tau') = \sigma\!\left(\lambda\left(\frac{\tau'}{\Delta\tau_{\text{ov}} - 1} - 0.5\right)\right)$$
- **切镜/不连续检测 (`DiscontinuityDetector`)**：
  $$\kappa(t) = 0.4 \cdot \text{norm\_motion\_jump}(t) + 0.4 \cdot \text{normal\_cosine\_drop}(t) + 0.2 \cdot (1 - \text{mask\_iou}(t))$$
  当 $\kappa(t) > \tau_{\text{cut}}$（默认 `0.65`）时发出 `ReanchorRequest` 重置时序累积并强制注入物理锚点。

### 3. ACEScg 内蕴风格提取与 16px 手术刀合成 (`synapse.style` & `synapse.composite`)
1. **内蕴风格提取**：`desc = extract_style_fgpp_intrinsic(srgb_frames, fgpp_seq)`，分离反照率与光照色温/阴影对比度，并通过 `to_style_profile_dict(desc)` 导出 `style_profile.v2.0.0`。
2. **运动自适应手术刀掩码**：`alpha = generate_surgical_mask(mask_3d_new, mask_sam2_old, motion_vectors, r_base=6, feather_width=4.0)`。
3. **16px 边界环 CIE LAB 色彩匹配 + 拉普拉斯金字塔融合**：
   ```python
   matched_patch = border_ring_lab_match(patch_rgb, base_rgb, alpha > 0.5, ring_width_px=16)
   blended = laplacian_pyramid_blend(matched_patch, base_rgb, alpha, num_levels=4)
   ```

---

## 二、双阶段实验执行工作流

### 阶段 A：在 `ubuntu24` 上执行 E1–E5 全链路无卡基线验证
任何实验代码修改后，首先在 `ubuntu24` 上运行 [`scripts/run_cpu_verification.py`](file:///Volumes/work/dervgen/synapse/scripts/run_cpu_verification.py)，一次性验证 E1–E5 的张量形状、色差收敛、滑窗切镜检测、分片打包与 30 场景基准：

```bash
ssh ubuntu24 "cd ~/work/dervgen/synapse && /root/venv-synapse/bin/python scripts/run_cpu_verification.py"
```
验证报告自动输出至 [`data/manifests/ubuntu24_cpu_verification_report.json`](file:///Volumes/work/dervgen/synapse/data/manifests/ubuntu24_cpu_verification_report.json)。

### 阶段 B：在 Google Colab (T4 / L4 / A100) 上执行 GPU 实验

Synapse 在 [`workenv/google-colab/`](file:///Volumes/work/dervgen/synapse/workenv/google-colab/README.md) 中提供了完整的一站式部署套件：
- [`setup_colab.sh`](file:///Volumes/work/dervgen/synapse/workenv/google-colab/setup_colab.sh)：安装系统依赖（OpenEXR、GL、FFmpeg）与 `synapse[image,metrics,gpu,train]`
- [`models.json`](file:///Volumes/work/dervgen/synapse/workenv/google-colab/models.json)：模型矩阵（Wan2.1 1.3B/14B、VAE、UMT5、SAM 2.1）与 VRAM 硬件分级
- [`io_spec.md`](file:///Volumes/work/dervgen/synapse/workenv/google-colab/io_spec.md)：各环节详细输入输出契约字典（张量形状、通道定义、退出码）
- [`synapse_colab_workflow.ipynb`](file:///Volumes/work/dervgen/synapse/workenv/google-colab/synapse_colab_workflow.ipynb)：开箱即用 Jupyter Notebook

使用 [`scripts/colab_runner.py`](file:///Volumes/work/dervgen/synapse/scripts/colab_runner.py) 自动探测显存档位（$<35\text{ GB}$ 自动选用 `Wan2.1-T2V-1.3B` + CPU Offload，$\ge 35\text{ GB}$ 选用 `14B`）并将每个实验阶段的状态写入 Google Drive：

```bash
# 1. 一键配置 Colab Linux/CUDA 运行环境
chmod +x workenv/google-colab/setup_colab.sh
./workenv/google-colab/setup_colab.sh

# 2. 检查 Colab GPU 显存与推荐配置
python scripts/colab_runner.py --check-env

# 3. 解压数据包并带 Google Drive 断点续跑执行全部实验
python scripts/colab_runner.py \
  --bundle /content/synapse_colab_bundle_v1.tar.gz \
  --dataset-dir /content/datasets/synapse \
  --checkpoint-dir /content/drive/MyDrive/dervgen/work/$(whoami)/synapse/$(date +%F)/exp-all/checkpoints \
  --exp all
```

> **注意**：`colab_runner.py --exp all` 当前执行 **E1、E2、E4** 三个阶段。E3（97 帧长时序滑窗）和 E5（Synapse-Bench 30 场景基准）仅在 `ubuntu24` 上通过 `scripts/run_cpu_verification.py` 无卡验证；Colab GPU 版将在后续迭代中支持。
