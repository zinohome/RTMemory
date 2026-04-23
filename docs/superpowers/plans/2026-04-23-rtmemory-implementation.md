# RTMemory 实施计划总览

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

RTMemory 项目按子系统拆分为 7 个独立实施计划，每个计划产出可独立运行和测试的软件。

## 计划拆分

| # | 计划文件 | 内容 | 依赖 |
|---|---------|------|------|
| 1 | `01-foundation.md` | 项目骨架 + 数据库模型 + 配置系统 + Docker Compose | 无 |
| 2 | `02-llm-adapter.md` | LLM 适配层 + 嵌入服务 | 计划1 |
| 3 | `03-graph-engine.md` | 图谱引擎（实体/关系 CRUD + 矛盾处理 + 图遍历） | 计划1 |
| 4 | `04-extraction-pipeline.md` | 三层提取流水线 + 文档处理 | 计划2, 计划3 |
| 5 | `05-search-engine.md` | 混合搜索引擎（向量/图/关键词 + RRF + Profile Boost） | 计划3 |
| 6 | `06-profile-engine.md` | 画像计算引擎 + 置信度衰减 + 画像 API | 计划3, 计划5 |
| 7 | `07-sdk-and-integrations.md` | Python SDK + JS SDK + MCP Server + LangChain + Claude 集成 | 计划1-6 |

## 执行顺序

```
01 ──→ 02 ──→ 04
 │              │
 └──→ 03 ──→ 05 ──→ 06 ──→ 07
```

计划 02 和 03 可并行执行（都只依赖 01）。
计划 04 依赖 02+03，计划 05 依赖 03，计划 06 依赖 05，计划 07 依赖全部。

## 各计划产出

1. **Foundation**: 可启动的 FastAPI 服务 + PG 数据库 + 健康检查 API + 空间 CRUD
2. **LLM Adapter**: 可切换的 LLM/嵌入调用接口 + 单元测试
3. **Graph Engine**: 实体/关系的完整 CRUD + 矛盾处理 + 递归图遍历
4. **Extraction Pipeline**: 三层提取 + 文档处理 + 集成测试
5. **Search Engine**: 三通道搜索 + RRF 融合 + Profile Boost
6. **Profile Engine**: 画像计算 + 置信度衰减 + 缓存
7. **SDK & Integrations**: 可 pip/npm 安装的客户端包 + MCP/LangChain/Claude 集成