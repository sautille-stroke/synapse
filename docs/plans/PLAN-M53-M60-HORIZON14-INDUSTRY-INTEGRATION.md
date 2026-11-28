---
type: Guide
title: "Horizon 14: 行业前沿全量纳入路线图 (M53–M60)"
tags: [synapse, horizon-14]
status: active
stale_after: 2028-06-01T00:00:00Z
---
# Horizon 14: 行业前沿全量纳入路线图 (M53–M60)

> **承接**: `v5.0.0` Horizon 13B 已封板
> **版本**: `v1.0` · **日期**: 2026-11-28
> **操作模型标识**: `[opus4.6]` / `[gemini4-hydrogen]`
> **当前状态**: ✅ **`Release v6.0.0` — M53–M60 全量合入 `main`，50 场景 `Synapse-Bench` 与全仓 CI 门禁全绿封板**
> **设计原则**: 整条流水线是护城河——不烧万卡训底模，做确定性物理与神经渲染的互操作立交桥

---

## 1. 战略总纲

将 2025–2026 年 9 大行业前沿技术全量纳入 synapse 流水线，每项外部模型/工具作为
流水线上一个可插拔 `*Transport` / `*Backend` 算子，配套 `Mock*` 回退保证零 GPU 日常回归。

### 纳入清单

| # | 技术 | 发布方 | 许可 | 对应里程碑 |
|---|------|--------|------|-----------|
| 1 | Wan2.1-VACE (统一视频编辑) | 阿里通义 | Apache 2.0 | M53 |
| 2 | Wan2.1 FLF2V (首尾帧控制) | 阿里通义 | Apache 2.0 | M53 |
| 3 | NVIDIA DiffusionRenderer (神经逆渲染) | NVIDIA | NVIDIA Open | M54 |
| 4 | TRELLIS.2 (O-Voxel 3D 生成) | Microsoft | MIT | M55 |
| 5 | SAM 2.1 (视频实例分割) | Meta FAIR | Apache 2.0 | M56 |
| 6 | Spatia (持久化空间记忆) | 学术 | — | M57 |
| 7 | Mask-Aware LoRA | 社区/学术 | — | M58 |
| 8 | QLoRA for Video DiTs | 社区/学术 | — | M58 |
| 9 | glTF KHR_gaussian_splatting | Khronos | Royalty-free | M59 |

---

## 2. 里程碑概览 (M53–M60, ~24 PRs)

| 里程碑 | 名称 | PRs | Wave | 核心模块 |
|--------|------|-----|------|---------|
| M53 | VACE + FLF2V | PR-216 ~ PR-218 | 1 | `synapse.adapters` |
| M54 | DiffusionRenderer | PR-219 ~ PR-221 | 1 | `synapse.bridge`, `synapse.recon` |
| M55 | TRELLIS.2 O-Voxel | PR-222 ~ PR-224 | 2 | `synapse.bridge`, `synapse.recon` |
| M56 | SAM 2.1 视频分割 | PR-225 ~ PR-227 | 1 | `synapse.bridge`, `synapse.composite` |
| M57 | Spatia 空间记忆 | PR-228 ~ PR-229 | 2 | `synapse.adapters.scheduler` |
| M58 | LoRA 进化 | PR-230 ~ PR-232 | 2 | `synapse.training` |
| M59 | glTF 3DGS 标准 | PR-233 ~ PR-234 | 3 | `synapse.recon` |
| M60 | Bench/文档/封板 | PR-235 ~ PR-237 | 3 | `docs/`, `synapse.bench` |

---

## 3. M53: VACE 统一条件注入与 FLF2V 切镜增强 (本 PR 落地)

### PR-216: VaceAdapter 基类与 VCU 条件融合

- **模块**: `synapse.adapters.vace`
- **核心交付**: 实现 Video Condition Unit (VCU) 统一接口
  - 将文本 T、参考帧序列 F、时空掩码 M 编码为 $V = [T; F; M]$ 条件张量
  - 新增 `ConditionMode` 枚举：`FGPP_7CH` (现有) | `VCU_UNIFIED`
  - `VaceAdapter` 继承 `VideoModelAdapter`，注册 ID `"vace"`
- **验收标准**:
  - `MockVideoAdapter` 兼容 VCU 路径
  - 与 `WanAdapter` 7ch 路径共存，通过 `ConditionMode` 枚举选择
  - 单测覆盖 VCU 张量形状断言

### PR-217: VACE R2V / V2V / MV2V 多任务路由

- **模块**: `synapse.adapters.vace`
- **核心交付**: 三种任务模式的调度路由与掩码构建
  - `VaceTaskMode.REFERENCE_TO_VIDEO` (R2V)
  - `VaceTaskMode.VIDEO_TO_VIDEO` (V2V)
  - `VaceTaskMode.MASKED_VIDEO_TO_VIDEO` (MV2V)
- **验收标准**: 单测覆盖三种模式的张量形状与 dtype 断言

### PR-218: FLF2V 首尾帧锚定与切镜重锚定增强

- **模块**: `synapse.adapters.scheduler`
- **核心交付**: `ConditionScheduler` 新增 FLF2V 模式
  - 切镜点提供前后锚定帧作为首尾约束
  - `ScheduledWindow` 新增 `anchor_first_frame` / `anchor_last_frame` 字段
- **验收标准**: 97f 因果滑窗 PSNR 回归不低于 baseline

---

## 4. 后续里程碑概要

> 详细 PR 拆解见批准的路线图 artifact。

### M54: NVIDIA DiffusionRenderer 神经逆渲染 Oracle (`#231`, `#238`)
- `DiffusionRendererTransport` G-buffer 映射至 FGPP 15 通道（含环境贴图着色重照明）
- `ReconPipeline` (`inverse_backend="diffusion_renderer" | "dual_track"`) 经典路径双轨互验与 `recon/pipeline/dual_track_agreement` 指标埋点
- `FeatureStoreBuilder.add_relight_augmented_samples()` >=8 光照变体合成增强

### M55: TRELLIS.2 O-Voxel 重建后端 (`#232`, `#238`)
- `Trellis2Transport` O-Voxel → polygon 解码与 PBR + Opacity 完整导出
- `TopologyPolicy` (`strict` | `open_surface_allowed` | `permissive`) 贯穿 `validate_mesh_topology`、`check_manifold` 与 `check_watertight`

### M56: SAM 2.1 视频分割 (`#235`, `#238`)
- `SAM2Transport` 时空实例分割桥接与 `fuse_sam2_cryptomatte()`
- `generate_surgical_mask()` / `generate_surgical_mask_sequence()` 切镜重启 + 掩码融合
- `MultiLoRAManager.build_spatial_lora_masks_from_sam2()` 与无 Cryptomatte 纯 SAM2 `manage_spatial_hooks()`

### M57: Spatia 持久化空间记忆 (`#236`, `#237`)
- `SpatialMemory` OpenGL Y-up 增量 3D 点云投影检索（96-bit 体素去重与驱逐同步）
- `ConditionScheduler` v2 + `WanAdapter.generate()` 空间记忆条件注入及切镜重置

### M58: LoRA 进化 (`#239`, `#241`)
- Mask-Aware 区域隔离 `FlowMatchingLoss` 与 FGPP `fgpp_cond` 前景掩码提取 (`train/mask_coverage`)
- `qlora.py`: Dettmers 16 级 NF4 量化 `QLoRALinear`（冻结 4-bit base + fp32 `lora_A/B`）
- `bucketing.py`: `FrameBucketSampler`（8px 对齐 + `4k+1` 帧桶）与 `--auto-tune` VRAM 自适应超参

### M59: glTF `KHR_gaussian_splatting` (`#240`)
- `recon/gltf_splat.py`: 符合 `KHR_gaussian_splatting` 的二进制 glTF 2.0 (`.glb`) 编码/解码
- `KHR_gaussian_splatting_compression_spz_2` 紧凑 SPZ v2 压缩（3 字节规范化四元数，`>=3x` 压缩比 & `PSNR>=50 dB`）
- `.ply` ↔ `.glb` / `.s3dgs` / `.npz` 双向转换器与 CLI `export-gltf-3dgs` / `convert-3dgs`

### M60: Bench/文档/封板 (`#242`, `#243`, `#244`)
- `Synapse-Bench` 扩展至 50 场景（新增 VACE、DiffusionRenderer、TRELLIS.2、SAM2、Spatia 覆盖场景）
- `INDUSTRY_TRACKING_AND_TRENDS.md` 更新为 `active` 与 Horizon 14 全量记录
- `wire_guard.py` 扩展 Horizon 14 模块与契约门禁校验 → **`v6.0.0` 封板**

---

## 5. 执行铁律

1. **Mock 回退保证零 GPU 日常回归**: 每个新 Transport 均实现 `Mock*Backend`
2. **防静默失败三问**: 每个 PR 通过存在问、接线问、执行问
3. **PR 净增 300–750 行**: 严格控量
4. **Server-side Zero-Python**: 所有新增后端均为 SDK 桥接 + 无状态 CLI Worker
5. **Git Worktree 隔离**: `/tmp/synapse-wt-<branch>` → PR → Squash Merge
