# RFC 0062 详细设计规范：3D 引导生成、可微逆向渲染与局部修补双向闭环

- **文档状态**: `Approved Technical Spec (v1.0)`
- **关联 RFC**: [RFC 0062: 3D 动画引导生成、可微逆向渲染与局部修补双向闭环设计规范](RFC-0062-INVERSE-RENDERING-AND-3D-NEURAL-CLOSED-LOOP.md)
- **关联实施计划**: [PLAN-M9-M12-HORIZON3-LANDING.md](plans/PLAN-M9-M12-HORIZON3-LANDING.md) (`M11: PR-38 ~ PR-40`)
- **依赖上层规约**: [RFC 0060](RFC-0060-REAL-VIDEO-TO-3D-DIGITAL-TWIN.md) · [RFC 0061](RFC-0061-LOCAL-REFINEMENT-AND-MULTI-MODEL-FUSION.md)
- **修订日期**: 2026-11-13

---

## 1. 设计目标与架构边界 (Scope & Boundaries)

本规范为 `RFC 0062` 的 Layer-B 详细工程设计文档，定义 `python/synapse/recon/loop/` 子模块的数据流、色彩辐射一致性算子、Causal VAE 相位约束、双轨可微优化器与 CLI/Worker 接线。

### 核心边界铁律
1. **职责正交**：`fulgur`（Layer 1）专职前向实时光栅化与 FGPP 导出，绝不掺入 AD 反向传播磁带；可微物理求解由 `Mitsuba 3`（`[invrender]` 可选 extra）或纯 NumPy 解析 UV 微分核承担。
2. **零隐藏重依赖**：`import synapse.recon.loop` 不得触发 `mitsuba` / `drjit` / `torch` 导入；仅当 `variant in {"cuda_ad_rgb", "llvm_ad_rgb"}` 且显式实例化 `MitsubaInverseOptimizer` 时延迟导入，缺失时抛出 `BackendUnavailableError`（退出码 `30`）并附 `"agent_fix_hint"`。
3. **无状态 Worker 约束**：通过 `JobBundle` Stage `inverse_render_loop` 或 `synapse recon closed-loop --json` 单步运行后退出，严禁常驻进程。

---

## 2. 闭环系统时序流与状态交互 (Sequence & Data Flow)

```mermaid
sequenceDiagram
    autonumber
    participant Guide as forward_guide.py<br/>(T=4k+1 & κ=1.0)
    participant Wan as WanAdapter / ConditionScheduler
    participant Rad as radiometric.py<br/>(ACEScg <-> sRGB)
    participant Opt as InverseOptimizer<br/>(Mitsuba3 / AnalyticCPU)
    participant Tex as texture_extractor.py
    participant Refine as synapse.refine (RFC 0061)

    Guide->>Wan: 注入 LookDev 预卷 (N_preroll ∈ {4,8}) + 无阴影 DirectRadiance + 正片帧
    Note over Wan: 在 t = N_preroll 处强制 κ=1.0 切镜重锚定<br/>隔离 5500K 棚光与正片戏剧光影
    Wan-->>Rad: 返回生成视频序列 V_srgb [T_total, H, W, 3]
    Rad->>Opt: 提交预卷/轻扫线性参考 I_target_lin + 正片 sRGB
    loop Iterations (t = 1..K)
        Opt->>Opt: 计算 L_lin(I_render, I_target_lin) + L_lpips(sRGB) + λ_tv‖∇T‖²
        Opt->>Opt: Adam / 解析梯度更新 Albedo, Roughness, Metallic UV 贴图
    end
    Opt-->>Tex: 输出最优 UV PBR 贴图与收敛日志
    Tex-->>Guide: 重渲染风格对齐 3D 序列 I_aligned
    Opt->>Refine: 将 I_aligned + V_srgb + 瑕疵掩码送入 RFC 0061 手术刀缝合
    Refine-->>Guide: 输出无解剖坍塌且色彩一致的最终序列
```

---

## 3. 模块结构与公开类型契约 (`python/synapse/recon/loop/`)

```
python/synapse/recon/loop/
├── __init__.py             # 延迟导出门面
├── types.py                # 数据契约: LookDevPrerollSpec, InverseRenderConfig, OptimizedMaterialResult, ClosedLoopReport
├── radiometric.py          # 场景线性 ACEScg <-> Display sRGB 双向变换（含 soft_knee_inverse_tonemap）
├── forward_guide.py        # 3D 因果 VAE 相位校验 (T=4k+1) + κ=1.0 切镜锚定计划生成
├── mitsuba_optimizer.py    # MitsubaInverseOptimizer (cuda_ad_rgb/llvm_ad_rgb) + AnalyticUVInverseOptimizer (analytic_cpu)
├── texture_extractor.py    # 多视角 UV 反投影加权、边界空洞修补 (Inpaint Seam) 与 OpenPBR 落盘
└── closed_loop_runner.py   # ClosedLoopRunner: 串联 Forward -> Gen -> Optimize -> Re-render -> Scalpel Patch
```

### 3.1 `types.py` 核心契约

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class LookDevPrerollSpec:
    """LookDev 标定预卷与光源轻扫规范。"""
    n_preroll: Literal[0, 4, 8] = 4          # 严格受限于 Wan 3D Causal VAE 4x 时域压缩步长
    n_light_sweep: int = 0                   # 0 或 4 帧切向光源扫描（必须保持总长 4k+1）
    studio_prompt: str = (
        "soft diffuse overcast studio lighting, shadowless neutral 5500K "
        "lookdev environment, pure ambient occlusion, clean albedo capture"
    )
    suppress_cast_shadows: bool = True

    def total_frames(self, n_shot: int) -> int:
        total = self.n_preroll + self.n_light_sweep + n_shot
        if (total - 1) % 4 != 0:
            raise ContractValidationError(
                f"Total frame count {total} violates Wan 3D Causal VAE T=4k+1 constraint",
                agent_fix_hint="Adjust n_shot or choose n_preroll in {0, 4, 8} so (T - 1) % 4 == 0.",
            )
        return total

@dataclass
class InverseRenderConfig:
    resolution: tuple[int, int] = (512, 512)
    spp: int = 16
    iterations: int = 100
    learning_rate: float = 0.03
    tv_weight: float = 0.005
    lambda_lpips: float = 0.1
    variant: Literal["analytic_cpu", "llvm_ad_rgb", "cuda_ad_rgb"] = "analytic_cpu"

@dataclass
class OptimizedMaterialResult:
    albedo_uv: np.ndarray        # [H_uv, W_uv, 3], float32 linear [0, 1]
    roughness_uv: np.ndarray     # [H_uv, W_uv], float32 in [0.04, 1.0]
    metallic_uv: np.ndarray      # [H_uv, W_uv], float32 in [0.0, 1.0]
    loss_history: list[float]
    psnr_before_db: float
    psnr_after_db: float
    ciede2000_after: float
```

---

## 4. 核心算法细节 (Algorithms)

### 4.1 场景线性与显示域双向变换 (`radiometric.py`)
1. **`srgb_to_linear_rgb(x: np.ndarray) -> np.ndarray`**：标准 IEC 61966-2-1 分段逆 EOTF（阈值 $0.04045$，线性段 `/12.92`，幂次段 `((x + 0.055) / 1.055) ** 2.4`）。
2. **`soft_knee_inverse_tonemap(x_lin_ldr: np.ndarray, shoulder: float = 0.85, max_radiance: float = 8.0) -> np.ndarray`**：当输入为经过 Filmic/ACES 色调映射的压缩 sRGB 视频时，对高于 `shoulder` 的高光膝部执行平滑可逆压缩还原 $L = \text{shoulder} + (1 - \text{shoulder}) \cdot \text{arctanh}(\dots)$，使 Mitsuba/Analytic 优化器能无偏拟合高动态镜面粗糙度 $\alpha$。
3. **`linear_to_srgb_rgb(x_lin: np.ndarray) -> np.ndarray`**：前向 EOTF 映射至 `[0, 1]` 显示域，供 `SSIM`、`CIEDE2000` 及 `LPIPS` 感知项计算。

### 4.2 3D Causal VAE 相位锁定与切镜防漏光 (`forward_guide.py`)
- **时序结构构建**：`build_active_generation_plan(n_shot, preroll_spec)` 生成长度为 `T_total = 4k + 1` 的帧级调度清单：
  - `[0 .. n_preroll - 1]`：均匀半球 5500K 漫射光 LookDev 序列，`cut_score = 0.0`，Prompt 替换为 `studio_prompt`；
  - `[n_preroll .. n_preroll + n_light_sweep - 1]`（可选）：固定视角点光源切向扫描，激发 GGX 高光以解耦 `Roughness`/`Metallic`；
  - `t = n_preroll + n_light_sweep`（正片首帧）：注入 **`cut_score = 1.0`**，迫使 `ConditionScheduler` 重置跨窗隐状态记忆，杜绝 5500K 棚拍基准光渗入戏剧光影正片。

### 4.3 双轨可微求解器 (`mitsuba_optimizer.py`)
1. **Track A — CPU 确定性解析代理 (`AnalyticUVInverseOptimizer`, 默认 `variant="analytic_cpu"`)**：
   - 输入已知 UV  visibility 缓冲 `uv_coord_map [T, H, W, 2]`、逐像素入射辐照 `irradiance_map [T, H, W, 3]` 与 BRDF 镜面基响应（由 `aether`/简化 GGX 近似提供）；
   - 通过像素-UV 双线性散射雅可比矩阵 $\frac{\partial \hat{I}_t(p)}{\partial T(u, v)}$ 直接计算梯度 $\nabla_{T} \mathcal{L}_1 + \lambda_{\text{tv}} \nabla_T \|\nabla T\|_2^2$，配合 Adam 更新（零 C++/GPU 依赖，`< 1.5s` 在 `ubuntu24` CPU 上稳定实现 `> 10 dB` PSNR 增益）。
2. **Track B — Mitsuba 3 光追反演 (`MitsubaInverseOptimizer`, `variant in {"llvm_ad_rgb", "cuda_ad_rgb"}`)**：
   - 延迟加载 `mitsuba` 与 `drjit`；缺失依赖时触发 `BackendUnavailableError` (exit `30`) 并提示 `pip install -e ".[invrender]"`。
   - 优先从 LookDev 预卷帧求解平滑 `albedo_uv`（锁定极低学习率衰减防过拟合），随后联合正片帧与 Light Sweep 帧微调 `roughness_uv` / `metallic_uv` 及低阶环境光球谐系数。

### 4.4 闭环联动 RFC 0061 手术刀修补 (`closed_loop_runner.py`)
1. 将反演得到的 OpenPBR 贴图注入 3D 渲染（Mock / `fulgur` CLI），生成色彩、光照与 AI 视频对齐的 `I_aligned`；
2. 对 `V_target` 中发生手/脸退化的帧，自动生成对称差异/解剖 ROI 掩码并调用 `synapse.refine.LocalRefinementPipeline`（`RFC 0061`），完成 $16\text{px}$ 手术刀缝合（接缝 `border_ring_ciede2000 <= 1.8/2.0`）。

---

## 5. 可选依赖与 CLI / Worker 契约

### 5.1 Optional Extra (`pyproject.toml`)
新增独立可选依赖组 `[invrender]`（`mitsuba>=3.5.0`, `drjit>=0.4.5`），不污染核心 `numpy`/`pydantic` 依赖树，亦允许在不安装 `[invrender]` 的 `ubuntu24` 与 CI 机器上完整运行 `AnalyticUVInverseOptimizer` 回归测试。

### 5.2 CLI 与 `JobBundle` 接线
- **CLI 命令**：
  ```bash
  synapse recon closed-loop \
    --mesh /workspace/inputs/character.obj \
    --cameras /workspace/inputs/cameras.json \
    --target-video /workspace/inputs/wan_out.npy \
    --variant analytic_cpu \
    --n-preroll 4 \
    --output-dir /workspace/outputs/loop_run \
    --json
  ```
- **JSON 输出契约**：符合全仓规范，包含 `"status": "ok"`, `"exit_code": 0`, `"trace_id"`, `"psnr_before_db"`, `"psnr_after_db"`, `"ciede2000_after"`, `"seam_ciede2000"`。
- **Worker Stage**：`BundleManifest` 支持 `stage: "inverse_render_loop"`。

---

## 6. 防御性矩阵与异常处理 (Error Contract)

| 异常场景 | 抛出异常类型 | ExitCode | `"agent_fix_hint"` 修复动作 |
|---|---|---|---|
| `T_total` 不满足 `4k + 1` 或 `n_preroll not in {0,4,8}` | `ContractValidationError` | `10` | 调整 `n_preroll` 为 `0/4/8` 或裁剪正片帧数 |
| 网格 / 相机轨迹 / 视频输入缺失或 SHA256 不匹配 | `AssetNotFoundError` | `20` | 校验输入路径与 `BundleManifest` SHA256 |
| 指定 `cuda_ad_rgb`/`llvm_ad_rgb` 但未装 `[invrender]` | `BackendUnavailableError` | `30` | 安装 `.[invrender]` 或切换 `--variant analytic_cpu` |
| 优化后 PSNR 提升 `< 10.0 dB` 或 `ciede2000 > 2.5` | `QualityGateError` | `40` | 开启 `n_preroll=4` 无阴影 LookDev 引导后重试 |

---

## 7. 实施路线 (`M11: PR-38 ~ PR-40`)

本设计由 **[PLAN-M9-M12-HORIZON3-LANDING.md](plans/PLAN-M9-M12-HORIZON3-LANDING.md) 的 M11 里程碑** 分 3 个 PR（`PR-38`, `PR-39`, `PR-40`）逐步落地并在 `ubuntu24` 完成无卡全绿验收。
