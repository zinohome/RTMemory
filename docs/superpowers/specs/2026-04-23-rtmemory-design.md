# RTMemory — 时序知识图谱驱动的 AI 记忆系统

## 1. 概述

RTMemory（Real-Time Memory）是一个完整的、开源的 AI 记忆与知识库系统，包含服务端和客户端 SDK。为 AI Agent/聊天机器人提供跨对话的持久记忆、结构化用户画像和混合搜索能力。

### 1.1 核心定位

为 AI Agent/聊天机器人提供：
- 持久记忆：跨对话记住用户信息和偏好
- 知识库：基于文档为用户提供专业回答
- 结构化画像：自动维护的用户上下文视图

### 1.2 与 supermemory 的对比

| 维度 | supermemory | RTMemory |
|------|-------------|----------|
| 记忆模型 | 扁平字符串 + 版本链 | 时序知识图谱（实体-关系-时间） |
| 用户画像 | static/dynamic 二分 + 字符串列表 | 图谱投射的结构化视图，带置信度 |
| 混合搜索 | 向量检索 + 记忆语义匹配 | 向量 + 图遍历 + 关键词三层搜索，RRF 融合 |
| 遗忘机制 | 二元标记 isForgotten + TTL | 置信度衰减曲线，渐进遗忘 |
| 矛盾处理 | 版本链覆盖 | 时间区间天然消解 + 置信度仲裁 |
| 开源程度 | 客户端开源，服务端闭源 | 全套开源（服务端 + SDK + 集成） |
| 部署 | Cloudflare Workers 绑定 | 纯自托管，Docker Compose 一键启动 |

### 1.3 三大核心创新

1. **时序知识图谱**：时间是第一公民，矛盾不再需要"覆盖"，而是时间线的自然演进
2. **画像即图谱投射**：不是存画像，而是从图谱中实时计算画像，永远一致
3. **置信度衰减遗忘**：不是二元遗忘，而是记忆的置信度随时间和引用频率渐变，像人脑一样

---

## 2. 技术选型

| 层级 | 技术 | 理由 |
|------|------|------|
| 服务端 | Python + FastAPI | AI/ML 生态天然契合，异步性能好 |
| 数据库 | PostgreSQL 17 + pgvector | 一站式：关系数据 + 向量搜索，运维最简 |
| 嵌入模型 | sentence-transformers（本地）/ OpenAI API | 配置切换，本地优先，中文用 bge-base-zh-v1.5 |
| LLM | OpenAI / Anthropic / Ollama | 多模型适配层，配置切换 |
| 文档提取 | PyMuPDF(PDF) + trafilatura(网页) | 自托管，零云依赖 |
| 部署 | Docker Compose | 单机一键启动 |
| Python SDK | httpx + pydantic | 异步 + 严格类型验证 |
| JS SDK | fetch + zod | 轻量 + 类型安全 |
| 任务队列 | arq (Redis) 或 asyncio BackgroundTasks | 轻量级异步任务 |

---

## 3. 整体架构

```
+---------------------------------------------------------+
|                    RTMemory Server                       |
|                     (FastAPI)                            |
|                                                          |
|  +-------------+  +-------------+  +-----------------+  |
|  |  API Layer   |  |  Auth &     |  |  SDK Layer     |  |
|  |  (REST +     |  |  Tenant Mgr |  |  (Python + JS) |  |
|  |   WebSocket) |  |             |  |                |  |
|  +------+-------+  +------+------+  +-------+--------+  |
|         |                 |                  |           |
|  +------+-----------------+------------------+--------+ |
|  |              Core Service Layer                     | |
|  |  +----------+ +-----------+ +------------------+   | |
|  |  |  Memory   | |  Profile  | |  Search Engine   |   | |
|  |  |  Engine   | |  Manager  | |  (Vector+Graph+  |   | |
|  |  |           | |           | |   Keyword)       |   | |
|  |  +----+------+ +-----+-----+ +--------+---------+   | |
|  |       |              |               |              | |
|  |  +----+--------------+---------------+---------+   | |
|  |  |         Temporal Knowledge Graph              |   | |
|  |  |    (Entities - Relations - Temporal Edges)     |   | |
|  |  +------------------+----------------------------+   | |
|  +---------------------+-------------------------------+ |
|                        |                                |
|  +---------------------+-------------------------------+ |
|  |            Processing Pipeline                      | |
|  |  +----------+ +----------+ +----------+            | |
|  |  |Extraction| | Embedding| | Document |            | |
|  |  |  Worker  | |  Worker  | |  Worker  |            | |
|  |  +----------+ +----------+ +----------+            | |
|  +---------------------+-------------------------------+ |
|                        |                                |
|  +---------------------+-------------------------------+ |
|  |           LLM Adapter Layer                        | |
|  |   OpenAI - Anthropic - Ollama - (configurable)     | |
|  +----------------------------------------------------+ |
+---------------------------------------------------------+
                          |
                   +------+-------+
                   |  PostgreSQL  |
                   |  + pgvector  |
                   +--------------+
```

设计原则：
- **单体模块化**：一个 FastAPI 进程，内部模块清晰分层
- **图谱即核心**：时序知识图谱是数据中心，其他模块基于它
- **异步流水线**：提取/嵌入/文档处理用后台任务队列
- **多模型可插拔**：LLM Adapter 层统一接口，配置切换

---

## 4. 核心数据模型 — 时序知识图谱

### 4.1 实体表 `entities`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| name | TEXT | 实体名称 |
| entity_type | ENUM | person, org, location, concept, project, technology |
| description | TEXT | 实体描述摘要 |
| embedding | VECTOR(1536) | 实体语义向量（维度取决于嵌入模型，1536 为 OpenAI 默认，本地模型可能为 768） |
| confidence | FLOAT | 整体置信度 [0,1] |
| org_id | UUID | 租户隔离 |
| space_id | UUID | 空间隔离 |
| created_at | TIMESTAMPTZ | 创建时间 |
| updated_at | TIMESTAMPTZ | 更新时间 |

### 4.2 关系边表 `relations`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| source_entity_id | UUID | 主体实体 |
| target_entity_id | UUID | 客体实体 |
| relation_type | TEXT | 关系类型（lives_in, prefers, works_at, knows...） |
| value | TEXT | 关系值/补充 |
| valid_from | TIMESTAMPTZ | 生效起始时间 |
| valid_to | TIMESTAMPTZ | 生效结束时间（NULL = 至今） |
| confidence | FLOAT | 关系置信度 [0,1] |
| is_current | BOOLEAN | 是否为当前有效关系 |
| source_count | INT | 来源次数 |
| embedding | VECTOR(1536) | 关系语义向量（维度同实体表） |
| org_id | UUID | 租户隔离 |
| space_id | UUID | 空间隔离 |
| created_at | TIMESTAMPTZ | 创建时间 |
| updated_at | TIMESTAMPTZ | 更新时间 |

### 4.3 记忆条目表 `memories`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| content | TEXT | 原始记忆文本 |
| custom_id | TEXT | 外部ID，用于去重 |
| memory_type | ENUM | fact, preference, status, inference |
| entity_id | UUID | 关联的主实体（可选） |
| relation_id | UUID | 关联的关系边（可选） |
| confidence | FLOAT | 置信度 [0,1] |
| decay_rate | FLOAT | 衰减速率 |
| is_forgotten | BOOLEAN | 是否已遗忘 |
| forget_at | TIMESTAMPTZ | 预计遗忘时间 |
| forget_reason | TEXT | 遗忘原因 |
| version | INT | 版本号 |
| parent_id | UUID | 父版本（更新链） |
| root_id | UUID | 根版本 |
| metadata | JSONB | 扩展元数据 |
| embedding | VECTOR(1536) | 记忆语义向量（维度同实体表） |
| org_id | UUID | 租户隔离 |
| space_id | UUID | 空间隔离 |
| created_at | TIMESTAMPTZ | 创建时间 |
| updated_at | TIMESTAMPTZ | 更新时间 |

### 4.4 文档表 `documents`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| title | TEXT | 标题 |
| content | TEXT | 原始内容 |
| doc_type | ENUM | text, pdf, webpage |
| url | TEXT | 来源 URL |
| status | ENUM | queued, extracting, chunking, embedding, done, failed |
| summary | TEXT | AI 摘要 |
| summary_embedding | VECTOR(1536) | 摘要向量（维度同实体表） |
| metadata | JSONB | 扩展元数据 |
| org_id | UUID | 租户隔离 |
| space_id | UUID | 空间隔离 |
| created_at | TIMESTAMPTZ | 创建时间 |
| updated_at | TIMESTAMPTZ | 更新时间 |

### 4.5 分块表 `chunks`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| document_id | UUID | 所属文档 |
| content | TEXT | 分块内容 |
| position | INT | 顺序位置 |
| embedding | VECTOR(1536) | 分块向量（维度同实体表） |
| created_at | TIMESTAMPTZ | 创建时间 |

### 4.6 记忆-文档溯源 `memory_sources`

| 字段 | 类型 | 说明 |
|------|------|------|
| memory_id | UUID | 记忆条目 |
| document_id | UUID | 来源文档 |
| chunk_id | UUID | 来源分块（可选） |
| relevance_score | FLOAT | 相关度 |

### 4.7 空间表 `spaces`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| name | TEXT | 空间名称 |
| description | TEXT | 描述 |
| org_id | UUID | 租户 |
| owner_id | UUID | 创建者 |
| container_tag | TEXT | 容器标签（兼容 supermemory 格式） |
| is_default | BOOLEAN | 是否为默认空间 |
| created_at | TIMESTAMPTZ | 创建时间 |
| updated_at | TIMESTAMPTZ | 更新时间 |

### 4.8 数据关系示意

```
Entity:张军 --[prefers(Python)]--> Entity:Python       (valid: 2023-, conf: 0.95)
     |
     |--[lives_in(上海)]--> Entity:上海                 (valid: 2020-2024, conf: 0.9, is_current=false)
     |--[lives_in(北京)]--> Entity:北京                 (valid: 2024-, conf: 0.95, is_current=true)
     |
     +--[works_at]--> Entity:Cloudflare                 (valid: 2023-, conf: 0.85)

Memory: "张军最近在研究知识图谱" -> entity_id=张军, confidence=0.8, decay_rate=0.01
Memory: "张军偏好 TypeScript"  -> relation_id=prefers(TS), confidence=0.9
```

矛盾处理核心机制：时间区间消解。同一关系在不同时间区间各有一条记录。搜索时 `is_current=true` 的自动返回。

---

## 5. 记忆提取流水线

### 5.1 三层提取架构

```
用户消息/对话片段
       |
       v
+-----------------+
|  Layer 1: 轻量判断  |  <- 极快，几乎零成本
|  这条消息是否包含   |
|  新事实/偏好变化？  |
+------+----------+
       | 是
       v
+-----------------+
|  Layer 2: 即时提取  |  <- LLM 结构化提取
|  提取实体+关系+记忆 |
|  处理矛盾，更新图谱  |
+------+----------+
       |
       v
+-----------------+
|  Layer 3: 深度扫描  |  <- 对话结束/累积N条后
|  捕捉隐含信息       |
|  跨消息关联推理      |
|  置信度再评估        |
+-----------------+
```

### 5.2 Layer 1 — 轻量判断

不调 LLM，用规则 + 小模型过滤 70-80% 的闲聊消息：

```python
class FactDetector:
    """判断消息是否包含新事实"""
    rules = [
        r"我(?:是|在|有|用|喜欢|偏好|搬到|换|改)",
        r"我们(?:用|选|决定|计划)",
        r"(?:推荐|建议|偏好|习惯)",
    ]

    def should_extract(self, message: str, context: list[str]) -> bool:
        if any(re.match(p, message) for p in self.rules):
            return True
        # 可选：小模型意图分类
        # 可选：消息与最近记忆的语义差异度 > 阈值
        return False
```

### 5.3 Layer 2 — 即时提取

用 LLM 做结构化提取，一次调用完成实体、关系、记忆和矛盾检测：

```
EXTRACTION_PROMPT:
从以下对话中提取实体、关系和记忆。

输出 JSON：
{
  "entities": [{"name": "张军", "type": "person", "description": "..."}],
  "relations": [{
    "source": "张军", "target": "北京",
    "relation": "lives_in", "valid_from": "2024-01",
    "confidence": 0.95
  }],
  "memories": [{
    "content": "张军搬到北京了",
    "type": "fact",
    "confidence": 0.9
  }],
  "contradictions": [{
    "new": "lives_in(北京)",
    "old": "lives_in(上海)",
    "resolution": "update"  // update | extend | ignore
  }]
}
```

矛盾处理流程：
1. 查询图谱：当前是否有冲突关系？
2. 无冲突 -> 直接插入
3. 有冲突 -> 设置旧关系 valid_to=now, is_current=false；插入新关系 valid_from=now
4. source_count += 1（被再次提及时增强置信度）

支持 `entity_context` 参数引导提取方向（对应 supermemory 的 entityContext）。

### 5.4 Layer 3 — 深度扫描

对话结束（或每 10 条消息）触发。捕捉隐含偏好、跨消息关联、状态变化、置信度调整。

### 5.5 文档处理流水线

```
文档输入 (text/pdf/url)
    -> [提取] 文本提取（PDF用PyMuPDF，网页用trafilatura）
    -> [分块] 语义分块（按段落/主题切分）
    -> [嵌入] 生成 chunk embedding + summary embedding
    -> [图谱] 从文档中提取实体和关系，汇入知识图谱
    -> [索引] 写入 documents / chunks / entities / relations
```

---

## 6. 混合搜索引擎

### 6.1 三层搜索架构

```
查询: "张军最近在用什么框架做前端？"
       |
       v
Query Processor: 查询改写(可选) + 实体识别
       |
  +----------+----------+
  v          v          v
向量搜索   图遍历      关键词
(pgvector) (CTE递归)  (tsvector全文)
  |          |          |
  +----------+----------+
       |
       v
Result Fusion: RRF (Reciprocal Rank Fusion)
       |
       v
Profile Boost: 用户相关记忆加权提升
       |
       v
    最终搜索结果
```

### 6.2 三个搜索通道

**通道 1 — 向量语义搜索（pgvector）**

在 memories + chunks + entities 上做向量搜索，统一归一化后融合。

**通道 2 — 图遍历搜索（递归 CTE）**

从识别出的实体出发，递归遍历关联关系（最多 3 跳），距离越近分数越高。

**通道 3 — 关键词全文搜索（PostgreSQL tsvector）**

对文档和记忆做全文检索，支持中文分词。

### 6.3 结果融合 — RRF

```python
def reciprocal_rank_fusion(results: dict[str, list], k: int = 60) -> list:
    scores = defaultdict(float)
    for channel, channel_results in results.items():
        for rank, result in enumerate(channel_results):
            scores[result.id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 6.4 Profile Boost

融合排序后，对与当前用户相关的结果做加权提升：关联用户实体的结果 x1.5，匹配用户偏好的结果 x1.2。

### 6.5 搜索参数

| 参数 | 类型 | 说明 |
|------|------|------|
| q | string | 查询文本 |
| space_id | string | 空间隔离 |
| user_id | string | 用于 Profile Boost |
| mode | enum | hybrid / memory_only / documents_only |
| channels | list | 可选通道：vector, graph, keyword |
| limit | int | 结果数量上限 |
| rerank | bool | 是否用 LLM 重排序 |
| include_profile | bool | 是否附带用户画像 |
| chunk_threshold | float | 分块选择阈值 [0,1] |
| document_threshold | float | 文档选择阈值 [0,1] |
| only_matching_chunks | bool | 是否只返回匹配块 |
| include_full_docs | bool | 是否包含完整文档内容 |
| include_summary | bool | 是否包含文档摘要 |
| filters | object | metadata AND/OR 过滤条件 |
| rewrite_query | bool | 是否改写查询（增加延迟约400ms） |

### 6.6 搜索响应结构

```json
{
  "results": [
    {
      "type": "memory",
      "content": "张军最近在用 Next.js 做前端项目",
      "score": 0.92,
      "source": "vector+graph",
      "entity": {"name": "张军", "type": "person"},
      "metadata": {}
    },
    {
      "type": "document_chunk",
      "content": "Next.js 15 引入了 Server Actions...",
      "score": 0.78,
      "source": "vector+keyword",
      "document": {"title": "Next.js 15 新特性", "url": "..."},
      "metadata": {}
    }
  ],
  "profile": {
    "identity": {"name": "张军", "location": "北京"},
    "preferences": {"stack": ["Python", "TypeScript"]},
    "current_status": {"focus": "知识图谱", "project": "RTMemory"}
  },
  "timing_ms": 45
}
```

---

## 7. 用户画像系统

### 7.1 画像 = 图谱投射

画像不是独立存储，而是从知识图谱中实时投射计算。保证画像和图谱永远一致。

### 7.2 四层画像模型

| 层级 | 内容 | 置信度 | 变化频率 |
|------|------|--------|---------|
| 身份层 (identity) | name, location, role, company | 高 (>0.8) | 极慢 |
| 偏好层 (preferences) | languages, tech_stack, style | 中 (0.5-0.8) | 偶尔 |
| 状态层 (current_status) | focus, project, mood | 低 (<0.5) | 高频 |
| 关系层 (relationships) | team, collaborators | 中 | 偶尔 |

每个属性带独立置信度值。

### 7.3 画像计算引擎

```python
class ProfileEngine:
    async def compute_profile(self, entity_id: str) -> UserProfile:
        # 1. 收集所有当前有效的关系
        current_relations = await self.get_current_relations(entity_id)
        # 2. 按关系类型分类投射到四层
        identity = self._project_identity(entity, current_relations)
        preferences = self._project_preferences(entity, current_relations)
        status = self._project_status(entity, current_relations)
        relationships = self._project_relationships(entity, current_relations)
        # 3. 收集近期动态记忆
        dynamic_memories = await self.get_recent_memories(entity_id, limit=10, min_confidence=0.3)
        # 4. 置信度映射
        confidence_map = self._build_confidence_map(current_relations)
        return UserProfile(...)
```

### 7.4 置信度衰减遗忘

```
C(t) = C0 * e^(-lambda * delta_days) * (1 + alpha * log(n+1))

C0    = 初始置信度
lambda = 衰减速率 (decay_rate)
delta_days = 距上次强化的天数
alpha = 引用增强系数
n     = 被引用/再提的次数
```

不同类型记忆的衰减参数：

| 类型 | decay_rate | 说明 |
|------|-----------|------|
| identity | 0.001 | 极慢衰减，几乎永久 |
| preference | 0.005 | 慢衰减，偏好较稳定 |
| status | 0.02 | 中等衰减，状态变化快 |
| inference | 0.05 | 快衰减，推断需要验证 |

遗忘阈值：置信度低于 0.1 时标记为 is_forgotten=true，不再出现在搜索和画像中，数据保留可追溯。

### 7.5 画像缓存

计算结果缓存（Redis 或内存），图谱变更时使缓存失效。未变更时直接返回缓存，实现约 50ms 响应。

### 7.6 画像 API

```
POST /v1/profile
{
  "entity_id": "ent_xxx",
  "space_id": "sp_xxx",
  "q": "前端框架",           // 可选：附带搜索结果
  "fresh": false             // 是否强制重新计算
}

响应:
{
  "profile": {
    "identity": {"name": "张军", "location": "北京", "role": "全栈工程师"},
    "preferences": {"stack": ["Python", "TypeScript"], "style": "简洁"},
    "current_status": {"focus": "知识图谱", "project": "RTMemory"},
    "relationships": {"team": ["李明", "王芳"]},
    "dynamic_memories": ["最近在研究时序知识图谱", "RTMemory 项目进行中"]
  },
  "confidence": {"location": 0.95, "stack": 0.85, "focus": 0.7},
  "search_results": [...],
  "computed_at": "2026-04-23T10:00:00Z",
  "timing_ms": 48
}
```

---

## 8. API 设计

### 8.1 路由总览

```
/v1/
  |-- memories/                    # 记忆管理
  |   |-- POST   /                 # 添加记忆（触发提取流水线）
  |   |-- GET    /                 # 列出记忆（分页+过滤+排序）
  |   |-- GET    /:id              # 获取单条记忆（含版本链）
  |   |-- PATCH  /:id              # 更新记忆
  |   |-- DELETE /:id              # 遗忘记忆（软删除）
  |   +-- POST   /forget           # 批量遗忘（支持内容匹配）
  |
  |-- entities/                    # 实体管理
  |   |-- GET    /                 # 列出实体
  |   |-- GET    /:id              # 获取实体+关联关系
  |   |-- PATCH  /:id              # 更新实体
  |   +-- DELETE /:id              # 删除实体
  |
  |-- relations/                   # 关系管理
  |   |-- GET    /                 # 列出关系（支持图遍历查询）
  |   |-- GET    /:id              # 获取关系详情
  |   |-- PATCH  /:id              # 更新关系
  |   +-- DELETE /:id              # 删除关系
  |
  |-- documents/                   # 文档管理
  |   |-- POST   /                 # 上传文档（text/pdf/url）
  |   |-- POST   /upload           # 文件上传（multipart）
  |   |-- GET    /                 # 列出文档（分页+状态过滤+排序）
  |   |-- GET    /:id              # 获取文档+关联记忆
  |   +-- DELETE /:id              # 删除文档
  |
  |-- search/                      # 混合搜索
  |   +-- POST   /                 # 统一搜索接口
  |
  |-- profile/                     # 用户画像
  |   +-- POST   /                 # 获取/计算画像
  |
  |-- conversations/               # 对话记忆（便捷入口）
  |   |-- POST   /                 # 提交对话片段（触发提取）
  |   +-- POST   /end              # 对话结束（触发深度扫描）
  |
  |-- spaces/                      # 空间管理
  |   |-- POST   /                 # 创建空间
  |   |-- GET    /                 # 列出空间
  |   |-- GET    /:id              # 空间详情
  |   +-- DELETE /:id              # 删除空间（含文档迁移选项）
  |
  +-- graph/                       # 图谱可视化查询
      |-- GET    /                 # 图谱概览
      +-- GET    /:entity_id       # 实体邻域子图
```

### 8.2 记忆添加 — 完整参数

```
POST /v1/memories/
{
  "content": "我刚搬到北京，现在在用 Next.js 做前端项目",
  "space_id": "sp_xxx",
  "user_id": "user_xxx",
  "custom_id": "ext_id_123",           // 外部ID，用于去重
  "entity_context": "这是张军的知识库",  // 引导提取的上下文
  "metadata": {"source": "slack"}       // 扩展元数据
}
```

### 8.3 记忆遗忘 — 支持内容匹配

```
POST /v1/memories/forget
{
  "memory_id": "mem_xxx",               // 按ID遗忘
  // 或
  "content_match": "住在上海",          // 按内容模糊匹配
  "reason": "信息过时"
}
```

### 8.4 记忆更新

```
PATCH /v1/memories/:id
{
  "content": "更新后的内容",
  "metadata": {"edited": true}
}
```

---

## 9. SDK 设计

### 9.1 Python SDK

```python
from rtmemory import RTMemoryClient

client = RTMemoryClient(
    base_url="http://localhost:8000",
    api_key="..."
)

# 记忆
result = await client.memories.add(
    content="我刚搬到北京",
    space_id="sp_xxx",
    custom_id="ext_123",
    entity_context="张军的知识库",
    metadata={"source": "slack"}
)

# 对话
await client.conversations.add(
    messages=[...],
    space_id="sp_xxx",
    user_id="user_xxx"
)
await client.conversations.end(conversation_id="conv_xxx", space_id="sp_xxx")

# 搜索
results = await client.search(
    q="张军最近在用什么框架？",
    space_id="sp_xxx",
    user_id="user_xxx",
    mode="hybrid",
    include_profile=True,
    chunk_threshold=0.5,
    include_full_docs=True,
    include_summary=True,
    filters={"AND": [{"key": "source", "value": "slack"}]}
)

# 画像
profile = await client.profile.get(
    entity_id="ent_xxx",
    space_id="sp_xxx",
    q="前端框架"
)

# 文档
doc = await client.documents.add(content="https://...", space_id="sp_xxx", title="技术指南")
doc = await client.documents.upload(file="./kb.pdf", space_id="sp_xxx")
docs = await client.documents.list(space_id="sp_xxx", status="done", sort="created_at", order="desc")

# 图谱
subgraph = await client.graph.get_neighborhood(entity_id="ent_xxx", depth=2)

# 空间
space = await client.spaces.create(name="项目知识库")
spaces = await client.spaces.list()

# 遗忘
await client.memories.forget(memory_id="mem_xxx", reason="用户要求删除")
await client.memories.forget(content_match="住在上海", reason="信息过时")

# 更新
await client.memories.update(memory_id="mem_xxx", content="更新后内容")
```

### 9.2 JavaScript SDK

```javascript
import { RTMemoryClient } from 'rtmemory';

const client = new RTMemoryClient({
  baseUrl: 'http://localhost:8000',
  apiKey: '...'
});

await client.memories.add({
  content: '我刚搬到北京',
  spaceId: 'sp_xxx',
  customId: 'ext_123',
  entityContext: '张军的知识库'
});

const results = await client.search({
  q: '张军的技术栈是什么？',
  spaceId: 'sp_xxx',
  userId: 'user_xxx',
  mode: 'hybrid',
  includeProfile: true
});

const profile = await client.profile.get({
  entityId: 'ent_xxx',
  spaceId: 'sp_xxx'
});
```

---

## 10. 集成层

### 10.1 MCP Server

支持 Cursor / Claude Desktop / Windsurf 等工具通过 MCP 协议直接接入 RTMemory。

```
MCP 工具列表:
- rtmemory_search        # 搜索记忆和知识库
- rtmemory_add           # 添加记忆
- rtmemory_profile       # 获取用户画像
- rtmemory_forget        # 遗忘记忆
- rtmemory_add_document  # 添加文档
- rtmemory_list_documents # 列出文档
```

通过 stdio 或 SSE 暴露，配置方式：
```
npx install-mcp@latest http://localhost:8000/mcp
```

### 10.2 LangChain 集成

```python
from rtmemory.integrations.langchain import RTMemoryTools

tools = RTMemoryTools(
    base_url="http://localhost:8000",
    api_key="...",
    space_id="sp_xxx",
    user_id="user_xxx"
)
# 生成 LangChain Tool 对象：search_memories, add_memory, get_profile, forget_memory, add_document, list_documents
```

### 10.3 Claude Code Memory 集成

```python
from rtmemory.integrations.claude import ClaudeMemoryAdapter

adapter = ClaudeMemoryAdapter(base_url="http://localhost:8000", api_key="...")
# 支持 view/create/str_replace/insert/delete/rename
# 映射到 RTMemory 的记忆和搜索操作
```

### 10.4 通用 LLM Agent 工具包

```python
from rtmemory.tools import get_memory_tools

tools = get_memory_tools(
    client=client,
    space_id="sp_xxx",
    user_id="user_xxx"
)
# 生成：search_memories, add_memory, get_profile, forget_memory, add_document
```

---

## 11. 部署

### 11.1 Docker Compose

```yaml
services:
  rtmemory-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rtmemory:secret@postgres:5432/rtmemory
      - LLM_PROVIDER=ollama
      - LLM_MODEL=qwen2.5:7b
      - LLM_BASE_URL=http://ollama:11434
      - EMBEDDING_PROVIDER=local
      - EMBEDDING_MODEL=bge-base-zh-v1.5
    depends_on:
      postgres:
        condition: service_healthy

  rtmemory-worker:
    build: .
    command: python -m rtmemory.worker
    environment:
      - DATABASE_URL=postgresql://rtmemory:secret@postgres:5432/rtmemory
      - LLM_PROVIDER=ollama
      - LLM_BASE_URL=http://ollama:11434
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: pgvector/pgvector:pg17
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=rtmemory
      - POSTGRES_USER=rtmemory
      - POSTGRES_PASSWORD=secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rtmemory"]

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"

volumes:
  pgdata:
  ollama_data:
```

### 11.2 LLM 配置

```yaml
llm:
  extraction:
    provider: ollama
    model: qwen2.5:7b
    base_url: http://localhost:11434
    temperature: 0.1
    max_tokens: 2048

  embedding:
    provider: local
    model: bge-base-zh-v1.5
    # provider: openai
    # model: text-embedding-3-small

  rerank:
    provider: openai
    model: gpt-4o-mini
```

---

## 12. 项目目录结构

```
rtmemory/
|-- server/                          # 服务端
|   |-- app/
|   |   |-- main.py                  # FastAPI 入口
|   |   |-- config.py                # 配置加载
|   |   |-- api/                     # API 路由
|   |   |   |-- memories.py
|   |   |   |-- entities.py
|   |   |   |-- relations.py
|   |   |   |-- documents.py
|   |   |   |-- search.py
|   |   |   |-- profile.py
|   |   |   |-- conversations.py
|   |   |   |-- spaces.py
|   |   |   +-- graph.py
|   |   |-- core/                    # 核心引擎
|   |   |   |-- graph_engine.py       # 图谱操作
|   |   |   |-- memory_engine.py      # 记忆管理
|   |   |   |-- profile_engine.py     # 画像计算
|   |   |   |-- search_engine.py     # 混合搜索
|   |   |   +-- llm_adapter.py       # 多模型适配
|   |   |-- extraction/              # 提取流水线
|   |   |   |-- fact_detector.py      # Layer 1 轻量判断
|   |   |   |-- extractor.py          # Layer 2 即时提取
|   |   |   |-- deep_scanner.py      # Layer 3 深度扫描
|   |   |   +-- document_processor.py # 文档处理
|   |   |-- db/                      # 数据库层
|   |   |   |-- models.py            # SQLAlchemy 模型
|   |   |   |-- migrations/          # Alembic 迁移
|   |   |   +-- vector.py            # pgvector 操作
|   |   +-- worker.py               # 后台任务入口
|   |-- Dockerfile
|   +-- pyproject.toml
|
|-- sdk-python/                      # Python SDK
|   |-- rtmemory/
|   |   |-- client.py
|   |   |-- memories.py
|   |   |-- search.py
|   |   |-- profile.py
|   |   |-- documents.py
|   |   |-- graph.py
|   |   |-- spaces.py
|   |   +-- tools.py                 # LLM Agent 工具
|   +-- pyproject.toml
|
|-- sdk-js/                          # JavaScript SDK
|   |-- src/
|   |   |-- client.ts
|   |   |-- memories.ts
|   |   |-- search.ts
|   |   |-- profile.ts
|   |   |-- documents.ts
|   |   +-- index.ts
|   +-- package.json
|
|-- integrations/                    # 集成层
|   |-- mcp-server/                  # MCP Server
|   |   |-- main.py
|   |   +-- pyproject.toml
|   |-- langchain/                   # LangChain 集成
|   |   +-- tools.py
|   +-- claude/                      # Claude Code 集成
|       +-- memory_adapter.py
|
|-- docker-compose.yml
|-- config.yaml
+-- README.md
```

### 核心依赖

```
# 服务端
fastapi, uvicorn
sqlalchemy + asyncpg          # ORM + 异步PG驱动
pgvector-python               # 向量操作
alembic                       # 数据库迁移
sentence-transformers         # 本地嵌入模型
pymupdf                       # PDF 提取
trafilatura                   # 网页提取
httpx                         # 异步 HTTP (LLM 调用)
arq                           # 轻量任务队列（基于Redis，可选）
pydantic                      # 数据验证

# Python SDK
httpx, pydantic

# JS SDK
fetch/axios, zod
```

---

## 13. supermemory 功能覆盖矩阵

| supermemory API | RTMemory 对应 | 状态 |
|-----------------|--------------|------|
| client.add() | memories.add() | 覆盖（含 customId, entityContext） |
| client.profile() | profile.get() | 超越（结构化分层 + 置信度） |
| client.search.execute() | search() | 覆盖（含全部搜索参数） |
| client.documents.add() | documents.add() | 超越（图谱提取） |
| client.documents.upload() | documents.upload() | 覆盖 |
| client.documents.list() | documents.list() | 覆盖（含 status/sort/order） |
| client.documents.delete() | documents.delete() | 覆盖 |
| client.memories.forget() | memories.forget() | 覆盖（含内容匹配） |
| client.memories.list() | memories.list() | 覆盖（含过滤/分页/排序） |
| client.memories.update() | memories.update() | 覆盖 |
| MCP Server | mcp-server/ | 覆盖 |
| Vercel AI SDK 工具 | tools.py | 覆盖 |
| LangChain 工具 | integrations/langchain/ | 覆盖 |
| Claude Memory Tool | integrations/claude/ | 覆盖 |
| Space/Project 管理 | spaces API | 覆盖 |
| Infinite Chat Proxy | — | 不需要（云端特性） |
| Connectors (Notion/GDrive) | — | 后续迭代 |