# Synapse (神经突触)

> **3D CG 渲染与神经视频生成互操作中枢 & 电影级低成本混合渲染管线**  
> Interoperability Hub between 3D CG Rendering and Generative Video Models & Hybrid Cinema Pipeline.

---

## 1. 核心定位与目标

当前 3D 渲染（确定性物理几何与材质）与生成式视频模型（潜空间时序像素扩散）存在根本性的表示鸿沟。  
**Synapse** 扮演连接两者的“神经突触”，以**极低成本实现电影级画面质量**：
- **3D 渲染**提供精确的时空几何骨架与局部物理修正；
- **视频生成模型**提供丰富的质感、色彩与细节补全；
- 两者深度互操作、互为先验，而非割裂单向替代。

---

## 2. 混合渲染管线流程（5 阶段）

```mermaid
flowchart LR
    A["① 分镜布景\n(WGP-style 代理场景)"] --> B["② 白模引导视频生成\n(FGPP 几何锁死生成)"]
    B --> C["③ 精细 3D 建模\n(Video→3D 重建与人工精修)"]
    C --> D["④ 局部重渲染与风格对齐\n(3D+SAM2 手术刀重绘)"]
    D --> E["⑤ 统一剪辑与成片\n(全流程色彩时序合成)"]

    style A fill:#e3f2fd,stroke:#1565c0
    style B fill:#fff3e0,stroke:#e65100
    style C fill:#e8f5e9,stroke:#2e7d32
    style D fill:#fce4ec,stroke:#c2185b
    style E fill:#f3e5f5,stroke:#6a1b9a
```

1. **分镜布景**：使用类 WGP/StageForge 轻量交互完成 3D 代理白模（Blocking Geometry）与运镜轨迹规划。
2. **白模引导生成**：通过 **FGPP**（深度/法线/光流）特征引导包注入视频模型（Wan 2.1 / Seedance / Veo），生成大体风格与动态对齐的视频。
3. **精细 3D 建模**：从生成的视频中提取多视角帧与光照，逆向重建高保真角色与道具资产并由人工介入精修。
4. **局部重渲染与风格对齐**：对视频中穿模、畸变或不合理的局部区域提取掩膜，结合风格描述符（Style Descriptor）用 3D 引擎局部重渲染后无缝融合。
5. **统一剪辑**：时序一致性检测、色调统一匹配，输出标准成片。

---

## 3. 四大核心互操作能力

- **能力 1：Video → 3D 建模**：多视角视频摄入 → 几何重建 → 材质反解（去光照） → 人工精修。
- **能力 2：3D ↔ 视频 双向布景辅助**：正向 3D 白模生成引导信号；逆向视频反解相机与场景空间布局。
- **能力 3：风格提取与对齐**：从生成视频提取色温、曝光、阴影软硬与材质质感描述符，直接映射驱动 3D 渲染引擎参数。
- **能力 4：局部手术刀重渲染**：3D 投影与 SAM2 联合掩膜，沿光流膨胀，风格一致性边缘环带融合。

---

## 4. 目录架构

```
synapse/
├── python/synapse/
│   ├── fgpp/         # FGPP (Feature Guidance Package Protocol) 引导信号 SDK
│   ├── bench/        # Synapse-Bench 标准评测场景、相机轨迹与程序化几何
│   ├── adapters/     # 视频生成模型统一适配器 (Wan 2.1 / Seedance / Veo2)
│   ├── pipeline/     # 5 阶段混合渲染生产管线编排调度
│   ├── style/        # 风格提取、风格描述符与渲染参数映射
│   ├── composite/    # 不合理区域检测、联合掩膜与手术刀合成
│   ├── metrics/      # 跨模态时序、保真度与往返度量
│   └── cli/          # 命令行工具 (synapse-bench 等)
├── data/
│   └── scenarios/    # 内置基准场景定义 JSON
└── tests/            # 完整单元测试套件
```

---

## 5. 快速上手

```bash
# 安装开发环境
pip install -e ".[dev]"

# 运行全套测试
pytest tests/ -v
```
