---
name: okf-knowledge-format
description: >-
  DervGen 生态基于 Google Open Knowledge Format (OKF) 的知识文档结构化管理规范。
  全量规范定义在 all-overview 仓库，本 Skill 为 synapse 仓库的快速参考。
  创建或修改知识文档时参阅 all-overview/devenv-skills/skills/okf-knowledge-format/SKILL.md 获取完整指南。
---

# OKF 知识文档格式快速参考 (synapse)

> 本仓库遵循 [Google Open Knowledge Format (OKF)](https://github.com/GoogleCloudPlatform/open-knowledge-format) 规范。
> 完整规范定义：[all-overview OKF Skill](../../../all-overview/devenv-skills/skills/okf-knowledge-format/SKILL.md)
> RFC：[RFC 0082](../../../all-overview/rfcs/0082-okf-knowledge-format-adoption.md)

## 快速规则

1. **所有知识文档必须包含 YAML frontmatter**，`type` 为唯一必填字段
2. **推荐字段**：`title`, `description`, `tags`, `status`, `stale_after`
3. **文件名使用 `kebab-case`**（如 `render-standards.md`）
4. **新增文档时更新对应目录的 `index.md`**

## 常用 Type

| Type | 说明 |
|---|---|
| `Guide` | 操作指南、API 文档、教程 |
| `Architecture` | 架构设计文档 |
| `Standard` | 编码/构建标准 |
| `RFC` | RFC 决议 |
| `Design` | 技术详细设计 |
| `Research` | 前沿研究 |
| `Implementation` | RFC 落地记录 |
| `Governance` | 治理规则 |

## Frontmatter 模板

```markdown
---
type: Guide
title: "文档标题"
description: "一句话摘要"
tags: [synapse, topic]
status: active
stale_after: 2027-11-24T00:00:00Z
---
# 文档标题
...
```
