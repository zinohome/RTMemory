# RTMemory Search Engine — 三层混合搜索

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hybrid search engine combining vector similarity, graph traversal, and keyword matching with RRF fusion and profile boosting.

**Architecture:** Three parallel search channels → RRF fusion → Profile Boost → final results. Vector search via pgvector, graph traversal via recursive CTE, keyword via tsvector. Optional LLM-based query rewriting.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0 (async), pgvector, PostgreSQL tsvector

**Depends on:** GraphEngine (plan 03) for graph traversal queries, EmbeddingService (plan 02) for query embedding, LLM Adapter (plan 02) for optional query rewriting.

---

## File Map

| File | Purpose |
|------|---------|
| `server/app/schemas/search.py` | Pydantic request/response models for search |
| `server/app/core/search_channels.py` | Three search channel implementations (vector, graph, keyword) |
| `server/app/core/search_fusion.py` | RRF fusion + Profile Boost logic |
| `server/app/core/query_processor.py` | Query rewriting + entity recognition |
| `server/app/core/search_engine.py` | SearchEngine class — hybrid search orchestration |
| `server/app/api/search.py` | POST /v1/search/ route |
| `server/tests/test_search_channels.py` | Unit tests for each search channel |
| `server/tests/test_search_fusion.py` | Unit tests for RRF + Profile Boost |
| `server/tests/test_query_processor.py` | Unit tests for query processor |
| `server/tests/test_search_engine.py` | Integration test for full search pipeline |
| `server/tests/test_search_api.py` | API-level integration test |
| `server/tests/conftest.py` | Shared test fixtures (db session, seeded data) |

---

## Phase 1: Schemas & Test Infrastructure

### Step 1.1 — Create search Pydantic schemas

- [ ] **Write** `server/app/schemas/search.py`

```python
"""Pydantic schemas for the hybrid search API."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────

class SearchMode(str, Enum):
    hybrid = "hybrid"
    memory_only = "memory_only"
    documents_only = "documents_only"


class SearchChannel(str, Enum):
    vector = "vector"
    graph = "graph"
    keyword = "keyword"


class ResultType(str, Enum):
    memory = "memory"
    entity = "entity"
    document_chunk = "document_chunk"
    document = "document"


# ── Request ────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """POST /v1/search/ request body."""

    q: str = Field(..., min_length=1, max_length=2000, description="Search query text")
    space_id: uuid.UUID = Field(..., description="Space isolation scope")
    user_id: Optional[uuid.UUID] = Field(default=None, description="User ID for Profile Boost")
    mode: SearchMode = Field(default=SearchMode.hybrid, description="Search mode")
    channels: Optional[list[SearchChannel]] = Field(
        default=None,
        description="Channels to use; defaults to all three for hybrid",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results")
    rerank: bool = Field(default=False, description="Whether to LLM-rerank results")
    include_profile: bool = Field(default=False, description="Include user profile in response")
    chunk_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Chunk similarity threshold")
    document_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Document similarity threshold")
    only_matching_chunks: bool = Field(default=False, description="Only return matching chunks, not full docs")
    include_full_docs: bool = Field(default=False, description="Include full document content")
    include_summary: bool = Field(default=True, description="Include document summary")
    filters: Optional[dict[str, Any]] = Field(default=None, description="Metadata AND/OR filter conditions")
    rewrite_query: bool = Field(default=False, description="Rewrite query via LLM (adds ~400ms)")


# ── Result items ───────────────────────────────────────────────

class EntityBrief(BaseModel):
    name: str
    type: str


class DocumentBrief(BaseModel):
    id: uuid.UUID
    title: str
    url: Optional[str] = None


class SearchResultItem(BaseModel):
    type: ResultType
    id: uuid.UUID
    content: str
    score: float
    source: str = Field(default="", description="Which channels contributed, e.g. 'vector+graph'")
    entity: Optional[EntityBrief] = None
    document: Optional[DocumentBrief] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


# ── Profile (inline for search response) ───────────────────────

class SearchProfile(BaseModel):
    identity: Optional[dict[str, Any]] = None
    preferences: Optional[dict[str, Any]] = None
    current_status: Optional[dict[str, Any]] = None


# ── Timing ─────────────────────────────────────────────────────

class SearchTiming(BaseModel):
    total_ms: float
    vector_ms: Optional[float] = None
    graph_ms: Optional[float] = None
    keyword_ms: Optional[float] = None
    fusion_ms: Optional[float] = None
    profile_ms: Optional[float] = None
    rewrite_ms: Optional[float] = None


# ── Response ────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    results: list[SearchResultItem] = Field(default_factory=list)
    profile: Optional[SearchProfile] = None
    timing: SearchTiming
    query: str = Field(description="Original or rewritten query")
```

- [ ] **Run** schema validation smoke test

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "
from app.schemas.search import SearchRequest, SearchResponse, SearchMode, SearchChannel
req = SearchRequest(q='test', space_id='11111111-1111-1111-1111-111111111111')
assert req.mode == SearchMode.hybrid
assert req.limit == 20
assert req.channels is None
req2 = SearchRequest(q='test', space_id='11111111-1111-1111-1111-111111111111', channels=[SearchChannel.vector, SearchChannel.keyword])
assert len(req2.channels) == 2
print('Schema validation OK')
"
```

**Expected:** `Schema validation OK`

- [ ] **Commit:** `git add server/app/schemas/search.py && git commit -m "feat(search): add search Pydantic schemas — request, response, timing"`

---

### Step 1.2 — Create shared test fixtures for search

- [ ] **Write** `server/tests/conftest.py` (extend if exists; create if not)

```python
"""Shared test fixtures for RTMemory search tests."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.db.models import Base, Entity, Relation, Memory, Document, Chunk, Space


TEST_DATABASE_URL = "sqlite+aiosqlite:///test_rtmemory.db"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def sqlite_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(sqlite_engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(sqlite_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


# ── Seed data factory ──────────────────────────────────────────

def make_entity(
    name: str = "张军",
    entity_type: str = "person",
    space_id: uuid.UUID | None = None,
    org_id: uuid.UUID | None = None,
) -> dict:
    sid = space_id or uuid.uuid4()
    oid = org_id or uuid.uuid4()
    return {
        "id": uuid.uuid4(),
        "name": name,
        "entity_type": entity_type,
        "description": f"Entity: {name}",
        "confidence": 0.9,
        "org_id": oid,
        "space_id": sid,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def make_memory(
    content: str = "张军最近在用 Next.js 做前端项目",
    entity_id: uuid.UUID | None = None,
    space_id: uuid.UUID | None = None,
    org_id: uuid.UUID | None = None,
) -> dict:
    sid = space_id or uuid.uuid4()
    oid = org_id or uuid.uuid4()
    return {
        "id": uuid.uuid4(),
        "content": content,
        "memory_type": "fact",
        "entity_id": entity_id,
        "confidence": 0.85,
        "decay_rate": 0.01,
        "is_forgotten": False,
        "version": 1,
        "org_id": oid,
        "space_id": sid,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def make_document(
    title: str = "Next.js 15 新特性",
    content: str = "Next.js 15 引入了 Server Actions...",
    space_id: uuid.UUID | None = None,
    org_id: uuid.UUID | None = None,
) -> dict:
    sid = space_id or uuid.uuid4()
    oid = org_id or uuid.uuid4()
    return {
        "id": uuid.uuid4(),
        "title": title,
        "content": content,
        "doc_type": "text",
        "status": "done",
        "org_id": oid,
        "space_id": sid,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def make_chunk(
    document_id: uuid.UUID,
    content: str = "Next.js 15 引入了 Server Actions",
    position: int = 0,
) -> dict:
    return {
        "id": uuid.uuid4(),
        "document_id": document_id,
        "content": content,
        "position": position,
        "created_at": datetime.now(timezone.utc),
    }


# ── Fixtures that provide seeded data ──────────────────────────

@pytest_asyncio.fixture
async def seeded_search_data(db_session: AsyncSession):
    """Insert a small graph for search testing: user entity + related memories + docs."""
    space_id = uuid.uuid4()
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    user_entity = Entity(**make_entity(name="张军", entity_type="person", space_id=space_id, org_id=org_id))
    city_entity = Entity(**make_entity(name="北京", entity_type="location", space_id=space_id, org_id=org_id))
    tech_entity = Entity(**make_entity(name="Next.js", entity_type="technology", space_id=space_id, org_id=org_id))
    db_session.add_all([user_entity, city_entity, tech_entity])
    await db_session.commit()

    lives_in = Relation(
        id=uuid.uuid4(),
        source_entity_id=user_entity.id,
        target_entity_id=city_entity.id,
        relation_type="lives_in",
        is_current=True,
        confidence=0.95,
        org_id=org_id,
        space_id=space_id,
    )
    prefers = Relation(
        id=uuid.uuid4(),
        source_entity_id=user_entity.id,
        target_entity_id=tech_entity.id,
        relation_type="prefers",
        is_current=True,
        confidence=0.85,
        org_id=org_id,
        space_id=space_id,
    )
    db_session.add_all([lives_in, prefers])
    await db_session.commit()

    mem1 = Memory(**make_memory(content="张军最近在用 Next.js 做前端项目", entity_id=user_entity.id, space_id=space_id, org_id=org_id))
    mem2 = Memory(**make_memory(content="张军搬到了北京", entity_id=user_entity.id, space_id=space_id, org_id=org_id))
    db_session.add_all([mem1, mem2])
    await db_session.commit()

    doc = Document(**make_document(title="Next.js 15 新特性", content="Next.js 15 引入了 Server Actions", space_id=space_id, org_id=org_id))
    db_session.add(doc)
    await db_session.commit()

    chunk = Chunk(**make_chunk(document_id=doc.id, content="Next.js 15 引入了 Server Actions"))
    db_session.add(chunk)
    await db_session.commit()

    return {
        "space_id": space_id,
        "org_id": org_id,
        "user_id": user_id,
        "user_entity_id": user_entity.id,
        "city_entity_id": city_entity.id,
        "tech_entity_id": tech_entity.id,
        "memory_ids": [mem1.id, mem2.id],
        "document_id": doc.id,
        "chunk_id": chunk.id,
    }
```

- [ ] **Commit:** `git add server/tests/conftest.py && git commit -m "feat(search): add shared test fixtures with seeded search data"`

---

## Phase 2: Search Channels

### Step 2.1 — Write failing tests for search channels

- [ ] **Write** `server/tests/test_search_channels.py`

```python
"""Unit tests for the three search channels: vector, graph, keyword."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.search_channels import (
    VectorSearchChannel,
    GraphSearchChannel,
    KeywordSearchChannel,
    ChannelResult,
)


# ── ChannelResult dataclass ────────────────────────────────────

class TestChannelResult:
    def test_channel_result_creation(self):
        cr = ChannelResult(
            items=[{"id": "a", "score": 0.9}],
            channel="vector",
            timing_ms=12.5,
        )
        assert cr.channel == "vector"
        assert len(cr.items) == 1
        assert cr.timing_ms == 12.5


# ── Vector search channel ─────────────────────────────────────

class TestVectorSearchChannel:
    @pytest.mark.asyncio
    async def test_vector_search_memories(self):
        """Vector search across memories should produce ChannelResult with similarity scores."""
        channel = VectorSearchChannel(db_session=AsyncMock())

        mock_result = AsyncMock()
        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.type = "memory"
        row1.content = "张军最近在用 Next.js 做前端项目"
        row1.similarity = 0.92
        row1.entity_id = uuid.uuid4()
        row1.entity_name = "张军"
        row1.entity_type = "person"
        row1.metadata_ = None
        row1.created_at = None
        mock_result.fetchall.return_value = [row1]
        channel._session.execute = AsyncMock(return_value=mock_result)

        query_vec = [0.1] * 1536
        result = await channel.search(query_vec=query_vec, space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)

        assert isinstance(result, ChannelResult)
        assert result.channel == "vector"
        assert len(result.items) >= 1
        assert result.items[0]["score"] == 0.92
        assert result.timing_ms >= 0

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(self):
        """Vector search with no matches returns empty items list."""
        channel = VectorSearchChannel(db_session=AsyncMock())
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(query_vec=[0.0] * 1536, space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)

        assert len(result.items) == 0


# ── Graph traversal channel ────────────────────────────────────

class TestGraphSearchChannel:
    @pytest.mark.asyncio
    async def test_graph_search_finds_related_entities(self):
        """Graph search from a seed entity should traverse 3 hops and return related items."""
        channel = GraphSearchChannel(db_session=AsyncMock())

        seed_entity_id = uuid.uuid4()
        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.type = "entity"
        row1.content = "北京"
        row1.depth = 1
        row1.relation_type = "lives_in"
        row1.entity_name = "北京"
        row1.entity_type = "location"
        row1.confidence = 0.95
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = [row1]
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(seed_entity_ids=[seed_entity_id], space_id=uuid.uuid4(), org_id=uuid.uuid4(), max_depth=3, limit=10)

        assert result.channel == "graph"
        assert len(result.items) >= 1
        assert result.items[0]["depth"] == 1

    @pytest.mark.asyncio
    async def test_graph_search_no_seed_entities(self):
        """Graph search with no seed entities returns empty."""
        channel = GraphSearchChannel(db_session=AsyncMock())
        result = await channel.search(seed_entity_ids=[], space_id=uuid.uuid4(), org_id=uuid.uuid4(), max_depth=3, limit=10)
        assert len(result.items) == 0


# ── Keyword search channel ─────────────────────────────────────

class TestKeywordSearchChannel:
    @pytest.mark.asyncio
    async def test_keyword_search_finds_matches(self):
        """Keyword tsvector search should return rows ranked by ts_rank."""
        channel = KeywordSearchChannel(db_session=AsyncMock())

        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.type = "memory"
        row1.content = "张军最近在研究知识图谱"
        row1.rank = 0.3
        row1.entity_id = None
        row1.entity_name = None
        row1.entity_type = None
        row1.metadata_ = None
        row1.created_at = None
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = [row1]
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(query_text="知识图谱", space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)

        assert result.channel == "keyword"
        assert len(result.items) >= 1

    @pytest.mark.asyncio
    async def test_keyword_search_empty(self):
        """Keyword search with no matches returns empty."""
        channel = KeywordSearchChannel(db_session=AsyncMock())
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(query_text="不存在的词汇", space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)
        assert len(result.items) == 0
```

- [ ] **Run tests** (they will FAIL until implementation exists):

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_channels.py -x --tb=short 2>&1 | head -20
```

**Expected:** `ImportError` or `ModuleNotFoundError` — the module does not exist yet.

- [ ] **Commit:** `git add server/tests/test_search_channels.py && git commit -m "test(search): add failing tests for vector, graph, keyword channels"`

---

### Step 2.2 — Implement ChannelResult + VectorSearchChannel

- [ ] **Write** `server/app/core/search_channels.py`

```python
"""Three search channel implementations: vector, graph, keyword."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# ── Shared result type ──────────────────────────────────────────

@dataclass
class ChannelResult:
    """Result container for a single search channel."""

    items: list[dict[str, Any]] = field(default_factory=list)
    channel: str = ""
    timing_ms: float = 0.0


# ── Vector Search Channel ──────────────────────────────────────

class VectorSearchChannel:
    """Search across memories, chunks, and entities using pgvector cosine similarity.

    SQL pattern:
        1 - (embedding <=> $query_vec) AS similarity
    """

    def __init__(self, db_session: AsyncSession):
        self._session = db_session

    async def search(
        self,
        query_vec: list[float],
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        limit: int = 20,
        chunk_threshold: float = 0.0,
        document_threshold: float = 0.0,
    ) -> ChannelResult:
        """Run vector similarity search across memories, chunks, and entities."""
        t0 = time.perf_counter()
        items: list[dict[str, Any]] = []

        # ── Vector search across memories ───────────────────────
        mem_sql = text("""
            SELECT
                m.id,
                'memory' AS type,
                m.content,
                1 - (m.embedding <=> :query_vec) AS similarity,
                m.entity_id,
                e.name AS entity_name,
                e.entity_type AS entity_type,
                m.metadata AS metadata_,
                m.created_at
            FROM memories m
            LEFT JOIN entities e ON m.entity_id = e.id
            WHERE m.space_id = :space_id
              AND m.org_id = :org_id
              AND m.is_forgotten = false
              AND m.embedding IS NOT NULL
            ORDER BY m.embedding <=> :query_vec
            LIMIT :limit
        """)
        mem_result = await self._session.execute(
            mem_sql,
            {
                "query_vec": str(query_vec),
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in mem_result.fetchall():
            score = float(row.similarity) if row.similarity is not None else 0.0
            if score >= chunk_threshold:
                items.append({
                    "id": row.id,
                    "type": row.type,
                    "content": row.content,
                    "score": score,
                    "entity_id": row.entity_id,
                    "entity_name": row.entity_name,
                    "entity_type": row.entity_type,
                    "metadata": row.metadata_,
                    "created_at": row.created_at,
                })

        # ── Vector search across chunks ─────────────────────────
        chunk_sql = text("""
            SELECT
                c.id,
                'document_chunk' AS type,
                c.content,
                1 - (c.embedding <=> :query_vec) AS similarity,
                c.document_id,
                d.title AS doc_title,
                d.url AS doc_url,
                d.id AS doc_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.space_id = :space_id
              AND d.org_id = :org_id
              AND d.status = 'done'
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> :query_vec
            LIMIT :limit
        """)
        chunk_result = await self._session.execute(
            chunk_sql,
            {
                "query_vec": str(query_vec),
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in chunk_result.fetchall():
            score = float(row.similarity) if row.similarity is not None else 0.0
            if score >= chunk_threshold:
                items.append({
                    "id": row.id,
                    "type": row.type,
                    "content": row.content,
                    "score": score,
                    "document_id": row.document_id,
                    "document": {"id": row.doc_id, "title": row.doc_title, "url": row.doc_url},
                })

        # ── Vector search across entities ────────────────────────
        ent_sql = text("""
            SELECT
                e.id,
                'entity' AS type,
                e.name AS content,
                1 - (e.embedding <=> :query_vec) AS similarity,
                e.entity_type
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND e.embedding IS NOT NULL
            ORDER BY e.embedding <=> :query_vec
            LIMIT :limit
        """)
        ent_result = await self._session.execute(
            ent_sql,
            {
                "query_vec": str(query_vec),
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in ent_result.fetchall():
            score = float(row.similarity) if row.similarity is not None else 0.0
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": score,
                "entity_name": row.content,
                "entity_type": row.entity_type,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChannelResult(items=items, channel="vector", timing_ms=elapsed_ms)
```

- [ ] **Run** channel result tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_channels.py::TestChannelResult -xvs
```

**Expected:** `1 passed`

- [ ] **Commit:** `git add server/app/core/search_channels.py && git commit -m "feat(search): implement ChannelResult + VectorSearchChannel with pgvector cosine similarity"`

---

### Step 2.3 — Implement GraphSearchChannel

- [ ] **Append to** `server/app/core/search_channels.py`

```python
# ── Graph Traversal Channel ────────────────────────────────────

class GraphSearchChannel:
    """Search by traversing the knowledge graph from identified seed entities.

    Uses a recursive CTE to walk up to 3 hops from seed entities.
    Closer hops get higher scores: score = base_score / (depth + 1)
    """

    DEPTH_SCORES = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.3}

    def __init__(self, db_session: AsyncSession):
        self._session = db_session

    async def search(
        self,
        seed_entity_ids: list[uuid.UUID],
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        max_depth: int = 3,
        limit: int = 20,
    ) -> ChannelResult:
        """Traverse graph from seed entities using recursive CTE."""
        t0 = time.perf_counter()

        if not seed_entity_ids:
            return ChannelResult(items=[], channel="graph", timing_ms=0.0)

        items: list[dict[str, Any]] = []

        cte_sql = text("""
            WITH RECURSIVE graph_traverse AS (
                -- Base case: start from seed entities (depth 0)
                SELECT
                    e.id,
                    e.name,
                    e.entity_type,
                    e.description,
                    0 AS depth,
                    NULL::text AS relation_type,
                    e.confidence
                FROM entities e
                WHERE e.id = ANY(:seed_ids)
                  AND e.space_id = :space_id
                  AND e.org_id = :org_id

                UNION ALL

                -- Recursive case: follow outgoing relations (1 hop at a time)
                SELECT
                    target.id,
                    target.name,
                    target.entity_type,
                    target.description,
                    gt.depth + 1,
                    r.relation_type,
                    r.confidence
                FROM graph_traverse gt
                JOIN relations r ON r.source_entity_id = gt.id
                JOIN entities target ON r.target_entity_id = target.id
                WHERE r.is_current = true
                  AND r.space_id = :space_id
                  AND r.org_id = :org_id
                  AND gt.depth < :max_depth
            )
            SELECT DISTINCT ON (id)
                id,
                name,
                entity_type,
                description,
                depth,
                relation_type,
                confidence
            FROM graph_traverse
            ORDER BY id, depth ASC
            LIMIT :limit
        """)

        result = await self._session.execute(
            cte_sql,
            {
                "seed_ids": [str(eid) for eid in seed_entity_ids],
                "space_id": str(space_id),
                "org_id": str(org_id),
                "max_depth": max_depth,
                "limit": limit,
            },
        )

        for row in result.fetchall():
            depth = int(row.depth)
            base_score = self.DEPTH_SCORES.get(depth, 0.1)
            conf = float(row.confidence) if row.confidence is not None else 0.5
            score = base_score * conf

            items.append({
                "id": row.id,
                "type": "entity",
                "content": row.description or row.name,
                "score": score,
                "depth": depth,
                "relation_type": row.relation_type,
                "entity_name": row.name,
                "entity_type": row.entity_type,
            })

        # Also fetch memories attached to traversed entities
        if items:
            entity_ids = [item["id"] for item in items]
            mem_sql = text("""
                SELECT
                    m.id,
                    m.content,
                    m.confidence,
                    m.entity_id,
                    e.name AS entity_name,
                    e.entity_type AS entity_type,
                    m.metadata AS metadata_,
                    m.created_at
                FROM memories m
                LEFT JOIN entities e ON m.entity_id = e.id
                WHERE m.entity_id = ANY(:entity_ids)
                  AND m.space_id = :space_id
                  AND m.org_id = :org_id
                  AND m.is_forgotten = false
                ORDER BY m.confidence DESC
                LIMIT :limit
            """)
            mem_result = await self._session.execute(
                mem_sql,
                {
                    "entity_ids": [str(eid) for eid in entity_ids],
                    "space_id": str(space_id),
                    "org_id": str(org_id),
                    "limit": limit,
                },
            )
            for row in mem_result.fetchall():
                items.append({
                    "id": row.id,
                    "type": "memory",
                    "content": row.content,
                    "score": float(row.confidence) * 0.7,
                    "depth": None,
                    "entity_id": row.entity_id,
                    "entity_name": row.entity_name,
                    "entity_type": row.entity_type,
                    "metadata": row.metadata_,
                    "created_at": row.created_at,
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChannelResult(items=items, channel="graph", timing_ms=elapsed_ms)
```

- [ ] **Run** graph channel test:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_channels.py::TestGraphSearchChannel -xvs
```

**Expected:** `2 passed`

- [ ] **Commit:** `git add server/app/core/search_channels.py && git commit -m "feat(search): implement GraphSearchChannel with recursive CTE up to 3 hops"`

---

### Step 2.4 — Implement KeywordSearchChannel

- [ ] **Append to** `server/app/core/search_channels.py`

```python
# ── Keyword Full-Text Search Channel ────────────────────────────

class KeywordSearchChannel:
    """Full-text search using PostgreSQL tsvector with 'simple' config for Chinese.

    Uses to_tsvector('simple', ...) + to_tsquery('simple', ...) for Chinese support
    since 'simple' config does not stem and treats each token as-is.
    """

    def __init__(self, db_session: AsyncSession):
        self._session = db_session

    async def search(
        self,
        query_text: str,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        limit: int = 20,
    ) -> ChannelResult:
        """Run full-text search across memories and document chunks."""
        t0 = time.perf_counter()
        items: list[dict[str, Any]] = []

        # Convert query text to tsquery: split on whitespace, join with &
        tokens = query_text.strip().split()
        tsquery_str = " & ".join(tokens)

        # ── FTS on memories ────────────────────────────────────
        mem_sql = text("""
            SELECT
                m.id,
                'memory' AS type,
                m.content,
                ts_rank(
                    to_tsvector('simple', m.content),
                    to_tsquery('simple', :tsquery)
                ) AS rank,
                m.entity_id,
                e.name AS entity_name,
                e.entity_type AS entity_type,
                m.metadata AS metadata_,
                m.created_at
            FROM memories m
            LEFT JOIN entities e ON m.entity_id = e.id
            WHERE m.space_id = :space_id
              AND m.org_id = :org_id
              AND m.is_forgotten = false
              AND to_tsvector('simple', m.content) @@ to_tsquery('simple', :tsquery)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        mem_result = await self._session.execute(
            mem_sql,
            {
                "tsquery": tsquery_str,
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in mem_result.fetchall():
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": float(row.rank) if row.rank is not None else 0.0,
                "entity_id": row.entity_id,
                "entity_name": row.entity_name,
                "entity_type": row.entity_type,
                "metadata": row.metadata_,
                "created_at": row.created_at,
            })

        # ── FTS on chunks ───────────────────────────────────────
        chunk_sql = text("""
            SELECT
                c.id,
                'document_chunk' AS type,
                c.content,
                ts_rank(
                    to_tsvector('simple', c.content),
                    to_tsquery('simple', :tsquery)
                ) AS rank,
                c.document_id,
                d.title AS doc_title,
                d.url AS doc_url,
                d.id AS doc_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.space_id = :space_id
              AND d.org_id = :org_id
              AND d.status = 'done'
              AND to_tsvector('simple', c.content) @@ to_tsquery('simple', :tsquery)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        chunk_result = await self._session.execute(
            chunk_sql,
            {
                "tsquery": tsquery_str,
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in chunk_result.fetchall():
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": float(row.rank) if row.rank is not None else 0.0,
                "document_id": row.document_id,
                "document": {"id": row.doc_id, "title": row.doc_title, "url": row.doc_url},
            })

        # ── FTS on entities (name + description) ────────────────
        ent_sql = text("""
            SELECT
                e.id,
                'entity' AS type,
                COALESCE(e.description, e.name) AS content,
                ts_rank(
                    to_tsvector('simple', COALESCE(e.description, '') || ' ' || e.name),
                    to_tsquery('simple', :tsquery)
                ) AS rank,
                e.name AS entity_name,
                e.entity_type
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND (
                to_tsvector('simple', e.name) @@ to_tsquery('simple', :tsquery)
                OR to_tsvector('simple', COALESCE(e.description, '')) @@ to_tsquery('simple', :tsquery)
              )
            ORDER BY rank DESC
            LIMIT :limit
        """)
        ent_result = await self._session.execute(
            ent_sql,
            {
                "tsquery": tsquery_str,
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in ent_result.fetchall():
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": float(row.rank) if row.rank is not None else 0.0,
                "entity_name": row.entity_name,
                "entity_type": row.entity_type,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChannelResult(items=items, channel="keyword", timing_ms=elapsed_ms)
```

- [ ] **Run** all search channel tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_channels.py -xvs
```

**Expected:** All channel tests pass.

- [ ] **Commit:** `git add server/app/core/search_channels.py && git commit -m "feat(search): implement KeywordSearchChannel with tsvector full-text search"`

---

## Phase 3: RRF Fusion + Profile Boost

### Step 3.1 — Write failing tests for RRF fusion and Profile Boost

- [ ] **Write** `server/tests/test_search_fusion.py`

```python
"""Unit tests for RRF fusion and Profile Boost."""

from __future__ import annotations

import uuid

import pytest

from app.core.search_fusion import reciprocal_rank_fusion, apply_profile_boost, FusedResult


class TestReciprocalRankFusion:
    def test_rrf_basic_two_channels(self):
        """RRF with two channels: items appearing in both get higher scores."""
        vector_results = [
            {"id": uuid.uuid4(), "score": 0.95, "content": "a", "type": "memory"},
            {"id": uuid.uuid4(), "score": 0.80, "content": "b", "type": "memory"},
        ]
        keyword_results = [
            {"id": vector_results[0]["id"], "score": 0.5, "content": "a", "type": "memory"},
            {"id": uuid.uuid4(), "score": 0.4, "content": "c", "type": "entity"},
        ]

        channel_map = {"vector": vector_results, "keyword": keyword_results}
        fused = reciprocal_rank_fusion(channel_map, k=60)

        assert len(fused) == 3
        top_id = fused[0].id
        assert top_id == vector_results[0]["id"]

    def test_rrf_single_channel(self):
        """RRF with one channel still works — scores = 1/(k+rank+1)."""
        items = [
            {"id": uuid.uuid4(), "score": 0.9, "content": "a", "type": "memory"},
            {"id": uuid.uuid4(), "score": 0.5, "content": "b", "type": "entity"},
        ]
        fused = reciprocal_rank_fusion({"vector": items}, k=60)
        assert len(fused) == 2
        assert abs(fused[0].rrf_score - 1.0 / 61.0) < 1e-9

    def test_rrf_empty_channels(self):
        """RRF with no channels returns empty."""
        fused = reciprocal_rank_fusion({}, k=60)
        assert len(fused) == 0

    def test_rrf_k_parameter(self):
        """RRF score formula: score += 1/(k + rank + 1), k=60."""
        items = [{"id": uuid.uuid4(), "score": 0.9, "content": "x", "type": "memory"}]
        fused = reciprocal_rank_fusion({"vector": items}, k=60)
        assert abs(fused[0].rrf_score - 1.0 / 61.0) < 1e-9

    def test_rrf_k_60_three_channels(self):
        """Item in all 3 channels: score = 3 * 1/(60+0+1) when ranked first in each."""
        item_id = uuid.uuid4()
        ch = {
            "vector": [{"id": item_id, "score": 0.9, "content": "x", "type": "memory"}],
            "graph": [{"id": item_id, "score": 0.7, "content": "x", "type": "memory"}],
            "keyword": [{"id": item_id, "score": 0.5, "content": "x", "type": "memory"}],
        }
        fused = reciprocal_rank_fusion(ch, k=60)
        assert len(fused) == 1
        expected = 3.0 / 61.0
        assert abs(fused[0].rrf_score - expected) < 1e-9


class TestProfileBoost:
    def test_entity_match_boost(self):
        """Results matching user entity get x1.5 boost."""
        user_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=user_entity_id, source_channels=["vector"]),
            FusedResult(id=uuid.uuid4(), rrf_score=0.04, content="b", type="memory", entity_id=uuid.uuid4(), source_channels=["vector"]),
        ]

        boosted = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=[])
        assert boosted[0].boosted_score == pytest.approx(0.05 * 1.5, rel=1e-6)
        assert boosted[1].boosted_score == pytest.approx(0.04, rel=1e-6)

    def test_preference_match_boost(self):
        """Results matching user preferences get x1.2 boost."""
        pref_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=pref_entity_id, source_channels=["vector"]),
        ]

        boosted = apply_profile_boost(fused, user_entity_id=uuid.uuid4(), user_preference_entity_ids=[pref_entity_id])
        assert boosted[0].boosted_score == pytest.approx(0.05 * 1.2, rel=1e-6)

    def test_entity_and_preference_boost_stacks(self):
        """Entity x1.5 and preference x1.2 are multiplicative: 1.5 * 1.2 = 1.8."""
        user_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=user_entity_id, source_channels=["vector"]),
        ]

        boosted = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=[user_entity_id])
        assert boosted[0].boosted_score == pytest.approx(0.05 * 1.5 * 1.2, rel=1e-6)

    def test_no_boost_without_user(self):
        """No user_entity_id means no boost applied."""
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=uuid.uuid4(), source_channels=["vector"]),
        ]
        boosted = apply_profile_boost(fused, user_entity_id=None, user_preference_entity_ids=[])
        assert boosted[0].boosted_score == pytest.approx(0.05, rel=1e-6)

    def test_boosted_results_sorted(self):
        """Results should be re-sorted after profile boost."""
        user_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.03, content="low", type="memory", entity_id=user_entity_id, source_channels=["vector"]),
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="high", type="memory", entity_id=uuid.uuid4(), source_channels=["vector"]),
        ]
        boosted = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=[])
        assert boosted[0].boosted_score >= boosted[1].boosted_score
```

- [ ] **Run tests** (will fail — module not yet created):

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_fusion.py -x --tb=short 2>&1 | head -10
```

**Expected:** `ModuleNotFoundError`

- [ ] **Commit:** `git add server/tests/test_search_fusion.py && git commit -m "test(search): add failing tests for RRF fusion and Profile Boost"`

---

### Step 3.2 — Implement reciprocal_rank_fusion + apply_profile_boost

- [ ] **Write** `server/app/core/search_fusion.py`

```python
"""RRF (Reciprocal Rank Fusion) and Profile Boost for search result merging."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FusedResult:
    """A single search result after RRF fusion and optional profile boosting."""

    id: uuid.UUID
    rrf_score: float = 0.0
    boosted_score: float = 0.0
    content: str = ""
    type: str = ""
    entity_id: Optional[uuid.UUID] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    document_id: Optional[uuid.UUID] = None
    document: Optional[dict[str, Any]] = None
    source_channels: list[str] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[Any] = None
    depth: Optional[int] = None
    relation_type: Optional[str] = None


def reciprocal_rank_fusion(
    channel_results: dict[str, list[dict[str, Any]]],
    k: int = 60,
) -> list[FusedResult]:
    """Merge results from multiple search channels using Reciprocal Rank Fusion.

    Formula: score += 1 / (k + rank + 1)   where k=60, rank is 0-indexed.

    Args:
        channel_results: Map of channel name to list of result dicts.
            Each result must have at least "id" key.
        k: RRF constant (default 60).

    Returns:
        List of FusedResult sorted by rrf_score descending.
    """
    scores: dict[uuid.UUID, float] = defaultdict(float)
    item_data: dict[uuid.UUID, dict[str, Any]] = {}
    item_channels: dict[uuid.UUID, list[str]] = defaultdict(list)

    for channel_name, results in channel_results.items():
        for rank, item in enumerate(results):
            item_id = item["id"]
            scores[item_id] += 1.0 / (k + rank + 1)
            item_channels[item_id].append(channel_name)
            if item_id not in item_data:
                item_data[item_id] = item

    fused: list[FusedResult] = []
    for item_id, rrf_score in scores.items():
        data = item_data.get(item_id, {})
        fused.append(FusedResult(
            id=item_id,
            rrf_score=rrf_score,
            boosted_score=rrf_score,
            content=data.get("content", ""),
            type=data.get("type", ""),
            entity_id=data.get("entity_id"),
            entity_name=data.get("entity_name"),
            entity_type=data.get("entity_type"),
            document_id=data.get("document_id"),
            document=data.get("document"),
            source_channels=item_channels[item_id],
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            depth=data.get("depth"),
            relation_type=data.get("relation_type"),
        ))

    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused


# ── Profile Boost ────────────────────────────────────────────────

ENTITY_MATCH_BOOST = 1.5
PREFERENCE_MATCH_BOOST = 1.2


def apply_profile_boost(
    results: list[FusedResult],
    user_entity_id: Optional[uuid.UUID] = None,
    user_preference_entity_ids: list[uuid.UUID] | None = None,
) -> list[FusedResult]:
    """Apply profile-based score boosting to fused results.

    Rules (from spec):
    - Entity match:   if result.entity_id == user_entity_id -> x1.5
    - Preference match: if result.entity_id in user_preference_entity_ids -> x1.2
    - Both match: multiplicative -> x1.5 * x1.2 = x1.8
    """
    if user_preference_entity_ids is None:
        user_preference_entity_ids = []

    for result in results:
        boost = 1.0
        if user_entity_id is not None and result.entity_id == user_entity_id:
            boost *= ENTITY_MATCH_BOOST
        if result.entity_id in user_preference_entity_ids:
            boost *= PREFERENCE_MATCH_BOOST
        result.boosted_score = result.rrf_score * boost

    results.sort(key=lambda r: r.boosted_score, reverse=True)
    return results
```

- [ ] **Run** RRF and Profile Boost tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_fusion.py -xvs
```

**Expected:** All 10 tests pass.

- [ ] **Commit:** `git add server/app/core/search_fusion.py && git commit -m "feat(search): implement RRF fusion (k=60) and Profile Boost (entity x1.5, preference x1.2)"`

---

## Phase 4: Query Processor

### Step 4.1 — Write failing tests for query processor

- [ ] **Write** `server/tests/test_query_processor.py`

```python
"""Unit tests for query processor: entity recognition + optional LLM rewrite."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.query_processor import QueryProcessor, ProcessedQuery


class TestProcessedQuery:
    def test_processed_query_defaults(self):
        pq = ProcessedQuery(original="张军最近在用什么框架？", rewritten=None, entity_ids=[])
        assert pq.original == "张军最近在用什么框架？"
        assert pq.rewritten is None
        assert pq.entity_ids == []
        assert pq.effective_query == "张军最近在用什么框架？"

    def test_processed_query_with_rewrite(self):
        pq = ProcessedQuery(original="前端框架", rewritten="前端开发框架技术选型", entity_ids=[])
        assert pq.effective_query == "前端开发框架技术选型"


class TestQueryProcessor:
    @pytest.mark.asyncio
    async def test_entity_recognition_finds_known_entities(self):
        """Entity recognition should match query terms against entity names in the DB."""
        processor = QueryProcessor(db_session=AsyncMock())

        mock_result = AsyncMock()
        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.name = "张军"
        row1.entity_type = "person"
        row2 = MagicMock()
        row2.id = uuid.uuid4()
        row2.name = "Next.js"
        row2.entity_type = "technology"
        mock_result.fetchall.return_value = [row1, row2]
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("张军最近在用什么框架做前端？", space_id=uuid.uuid4(), org_id=uuid.uuid4())
        assert len(pq.entity_ids) == 2

    @pytest.mark.asyncio
    async def test_entity_recognition_no_match(self):
        """No matching entities returns empty entity_ids."""
        processor = QueryProcessor(db_session=AsyncMock())
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("无关查询", space_id=uuid.uuid4(), org_id=uuid.uuid4())
        assert len(pq.entity_ids) == 0

    @pytest.mark.asyncio
    async def test_query_rewrite_disabled_by_default(self):
        """Without rewrite_query=True, the query is not rewritten."""
        processor = QueryProcessor(db_session=AsyncMock(), llm_adapter=None)
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("测试查询", space_id=uuid.uuid4(), org_id=uuid.uuid4(), rewrite_query=False)
        assert pq.rewritten is None
        assert pq.effective_query == "测试查询"

    @pytest.mark.asyncio
    async def test_query_rewrite_calls_llm(self):
        """With rewrite_query=True and LLM adapter present, query is rewritten."""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "前端开发框架技术选型"

        processor = QueryProcessor(db_session=AsyncMock(), llm_adapter=mock_llm)
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = []
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("前端框架", space_id=uuid.uuid4(), org_id=uuid.uuid4(), rewrite_query=True)
        assert pq.rewritten == "前端开发框架技术选型"
        assert pq.effective_query == "前端开发框架技术选型"
        mock_llm.chat.assert_called_once()
```

- [ ] **Run** (will fail — module not yet created):

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_query_processor.py -x --tb=short 2>&1 | head -10
```

**Expected:** `ModuleNotFoundError`

- [ ] **Commit:** `git add server/tests/test_query_processor.py && git commit -m "test(search): add failing tests for query processor"`

---

### Step 4.2 — Implement QueryProcessor

- [ ] **Write** `server/app/core/query_processor.py`

```python
"""Query processor: entity recognition + optional LLM query rewrite."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ProcessedQuery:
    """Output of query processing."""

    original: str
    rewritten: Optional[str] = None
    entity_ids: list[uuid.UUID] = field(default_factory=list)

    @property
    def effective_query(self) -> str:
        """Return rewritten query if available, else original."""
        return self.rewritten if self.rewritten is not None else self.original


REWRITE_SYSTEM_PROMPT = """You are a search query optimizer. Rewrite the user's search query to be more specific and find better results. Keep the language the same (Chinese stays Chinese). Return ONLY the rewritten query, nothing else."""


class QueryProcessor:
    """Process search queries: recognize entities and optionally rewrite via LLM."""

    def __init__(
        self,
        db_session: AsyncSession,
        llm_adapter: Any | None = None,
    ):
        self._session = db_session
        self._llm_adapter = llm_adapter

    async def process(
        self,
        query: str,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        rewrite_query: bool = False,
    ) -> ProcessedQuery:
        """Process a search query: entity recognition + optional LLM rewrite.

        Steps:
        1. Extract candidate terms from the query (split on punctuation/whitespace)
        2. Match candidates against entity names in the DB
        3. Optionally rewrite the query via LLM for better search

        Args:
            query: Raw search query string.
            space_id: Space scope.
            org_id: Organization scope.
            rewrite_query: Whether to call LLM for query rewriting.

        Returns:
            ProcessedQuery with recognized entity IDs and optional rewrite.
        """
        result = ProcessedQuery(original=query)

        # ── Step 1: Entity recognition ──────────────────────────
        candidates = self._extract_candidates(query)

        if candidates:
            entity_ids = await self._recognize_entities(candidates, space_id, org_id)
            result.entity_ids = entity_ids

        # ── Step 2: Optional LLM rewrite ────────────────────────
        if rewrite_query and self._llm_adapter is not None:
            rewritten = await self._rewrite_via_llm(query)
            result.rewritten = rewritten

        return result

    def _extract_candidates(self, query: str) -> list[str]:
        """Extract candidate entity name tokens from the query.

        Strategy: Split on common Chinese/English punctuation and whitespace,
        filter tokens >= 2 chars, deduplicate while preserving order.
        """
        tokens = re.split(r'[，。！？、；：""''（）\(\)\[\]\{\}\s,.\-!?;:]+', query)
        candidates = []
        seen = set()
        for token in tokens:
            token = token.strip()
            if len(token) >= 2 and token not in seen:
                candidates.append(token)
                seen.add(token)
        return candidates

    async def _recognize_entities(
        self,
        candidates: list[str],
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        """Match candidate tokens against entity names in the DB.

        Uses ILIKE for fuzzy matching — handles partial name matches.
        """
        entity_ids: list[uuid.UUID] = []

        conditions = []
        params: dict[str, Any] = {
            "space_id": str(space_id),
            "org_id": str(org_id),
        }
        for i, candidate in enumerate(candidates):
            param_name = f"c{i}"
            conditions.append(f"e.name ILIKE :{param_name}")
            params[param_name] = f"%{candidate}%"

        if not conditions:
            return entity_ids

        where_clause = " OR ".join(conditions)
        sql = text(f"""
            SELECT e.id, e.name, e.entity_type
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND ({where_clause})
        """)

        db_result = await self._session.execute(sql, params)
        for row in db_result.fetchall():
            entity_ids.append(row.id)

        return entity_ids

    async def _rewrite_via_llm(self, query: str) -> str:
        """Call LLM adapter to rewrite the query for better search results."""
        response = await self._llm_adapter.chat(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original query: {query}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.strip() if isinstance(response, str) else str(response).strip()
```

- [ ] **Run** query processor tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_query_processor.py -xvs
```

**Expected:** All 6 tests pass.

- [ ] **Commit:** `git add server/app/core/query_processor.py && git commit -m "feat(search): implement QueryProcessor with entity recognition and optional LLM rewrite"`

---

## Phase 5: SearchEngine Orchestrator

### Step 5.1 — Write failing tests for SearchEngine

- [ ] **Write** `server/tests/test_search_engine.py`

```python
"""Integration tests for the SearchEngine orchestrator."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.search_engine import SearchEngine
from app.core.search_channels import ChannelResult
from app.core.search_fusion import FusedResult
from app.schemas.search import SearchRequest, SearchMode, SearchChannel


class TestSearchEngine:
    @pytest.mark.asyncio
    async def test_hybrid_search_calls_all_channels(self):
        """Hybrid mode with default channels should call vector, graph, and keyword."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )

        engine._embedding_service.embed.return_value = [0.1] * 1536

        with patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)) as mock_vec, \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)) as mock_graph, \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)) as mock_kw:

            space_id = uuid.uuid4()
            request = SearchRequest(q="测试查询", space_id=space_id)
            response = await engine.search(request, org_id=uuid.uuid4())

            mock_vec.assert_called_once()
            mock_graph.assert_called_once()
            mock_kw.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_only_mode_skips_document_channels(self):
        """memory_only mode should not search chunks/documents via keyword."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [0.1] * 1536

        with patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)), \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)), \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)):

            request = SearchRequest(q="测试", space_id=uuid.uuid4(), mode=SearchMode.memory_only)
            response = await engine.search(request, org_id=uuid.uuid4())

    @pytest.mark.asyncio
    async def test_search_includes_timing(self):
        """Search response must include timing breakdown per channel."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [0.1] * 1536

        with patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=12.0)), \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=5.0)), \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=8.0)):

            request = SearchRequest(q="测试", space_id=uuid.uuid4())
            response = await engine.search(request, org_id=uuid.uuid4())
            assert response.timing.total_ms > 0
            assert response.timing.vector_ms is not None

    @pytest.mark.asyncio
    async def test_channel_filter_vector_only(self):
        """Requesting only the vector channel should skip graph and keyword."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [0.1] * 1536

        with patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)) as mock_vec, \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)) as mock_graph, \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)) as mock_kw:

            request = SearchRequest(q="测试", space_id=uuid.uuid4(), channels=[SearchChannel.vector])
            response = await engine.search(request, org_id=uuid.uuid4())

            mock_vec.assert_called_once()
            mock_graph.assert_not_called()
            mock_kw.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_boost_applied_when_user_id(self):
        """When user_id is provided, profile boost should be applied."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [0.1] * 1536

        item_id = uuid.uuid4()
        entity_id = uuid.uuid4()
        fused_result = FusedResult(
            id=item_id,
            rrf_score=0.05,
            content="test",
            type="memory",
            entity_id=entity_id,
            source_channels=["vector"],
        )

        with patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)), \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)), \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)), \
             patch.object(engine, "_get_user_entity_id", return_value=entity_id) as mock_get_user, \
             patch.object(engine, "_get_user_preference_ids", return_value=[]) as mock_get_prefs, \
             patch("app.core.search_engine.reciprocal_rank_fusion", return_value=[fused_result]) as mock_rrf, \
             patch("app.core.search_engine.apply_profile_boost", return_value=[fused_result]) as mock_boost:

            request = SearchRequest(q="测试", space_id=uuid.uuid4(), user_id=uuid.uuid4())
            response = await engine.search(request, org_id=uuid.uuid4())

            mock_boost.assert_called_once()
```

- [ ] **Commit:** `git add server/tests/test_search_engine.py && git commit -m "test(search): add failing integration tests for SearchEngine orchestrator"`

---

### Step 5.2 — Implement SearchEngine class

- [ ] **Write** `server/app/core/search_engine.py`

```python
"""SearchEngine — hybrid search orchestration.

Combines vector search, graph traversal, and keyword search channels,
fuses them with RRF, and applies profile boosting.

Architecture:
    Query -> QueryProcessor -> [Vector, Graph, Keyword] -> RRF -> Profile Boost -> Results
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.search_channels import (
    ChannelResult,
    VectorSearchChannel,
    GraphSearchChannel,
    KeywordSearchChannel,
)
from app.core.search_fusion import (
    FusedResult,
    reciprocal_rank_fusion,
    apply_profile_boost,
)
from app.core.query_processor import QueryProcessor
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTiming,
    SearchProfile,
    SearchMode,
    SearchChannel,
    ResultType,
    EntityBrief,
    DocumentBrief,
)


class SearchEngine:
    """Hybrid search engine combining vector, graph, and keyword channels.

    Usage:
        engine = SearchEngine(db_session=session, embedding_service=embed_svc, llm_adapter=llm)
        response = await engine.search(request, org_id=org_id)
    """

    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: Any,
        llm_adapter: Any | None = None,
    ):
        self._session = db_session
        self._embedding_service = embedding_service
        self._llm_adapter = llm_adapter

        self._vector_channel = VectorSearchChannel(db_session)
        self._graph_channel = GraphSearchChannel(db_session)
        self._keyword_channel = KeywordSearchChannel(db_session)
        self._query_processor = QueryProcessor(db_session, llm_adapter)

    async def search(
        self,
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> SearchResponse:
        """Execute a hybrid search request.

        Pipeline:
        1. Process query (entity recognition + optional rewrite)
        2. Run search channels in parallel
        3. Fuse results with RRF
        4. Apply profile boost if user_id provided
        5. Build response with timing
        """
        t0 = time.perf_counter()

        # ── Step 1: Query processing ────────────────────────────
        rewrite_t0 = time.perf_counter()
        processed = await self._query_processor.process(
            query=request.q,
            space_id=request.space_id,
            org_id=org_id,
            rewrite_query=request.rewrite_query,
        )
        rewrite_ms = (time.perf_counter() - rewrite_t0) * 1000

        # ── Step 2: Determine active channels ───────────────────
        active_channels = self._resolve_channels(request)

        # ── Step 3: Get query embedding for vector search ───────
        query_vec: list[float] | None = None
        if SearchChannel.vector in active_channels:
            query_vec = await self._embedding_service.embed(processed.effective_query)

        # ── Step 4: Run channels in parallel ────────────────────
        channel_tasks: dict[str, asyncio.Task] = {}

        if SearchChannel.vector in active_channels and query_vec is not None:
            channel_tasks["vector"] = asyncio.create_task(
                self._run_vector_search(query_vec, request, org_id)
            )

        if SearchChannel.graph in active_channels:
            channel_tasks["graph"] = asyncio.create_task(
                self._run_graph_search(processed.entity_ids, request, org_id)
            )

        if SearchChannel.keyword in active_channels:
            channel_tasks["keyword"] = asyncio.create_task(
                self._run_keyword_search(processed.effective_query, request, org_id)
            )

        # Gather all channel results
        channel_results: dict[str, ChannelResult] = {}
        if channel_tasks:
            done = await asyncio.gather(*channel_tasks.values(), return_exceptions=True)
            for ch_name, result in zip(channel_tasks.keys(), done):
                if isinstance(result, Exception):
                    channel_results[ch_name] = ChannelResult(items=[], channel=ch_name, timing_ms=0.0)
                else:
                    channel_results[ch_name] = result

        # ── Step 5: RRF fusion ──────────────────────────────────
        fusion_t0 = time.perf_counter()
        channel_item_map: dict[str, list[dict[str, Any]]] = {}
        for ch_name, ch_result in channel_results.items():
            channel_item_map[ch_name] = ch_result.items

        channel_item_map = self._filter_by_mode(channel_item_map, request.mode)
        fused = reciprocal_rank_fusion(channel_item_map, k=60)
        fusion_ms = (time.perf_counter() - fusion_t0) * 1000

        # ── Step 6: Profile Boost ───────────────────────────────
        profile_t0 = time.perf_counter()
        profile: SearchProfile | None = None
        if request.user_id is not None:
            user_entity_id = await self._get_user_entity_id(request.user_id, request.space_id, org_id)
            pref_ids = await self._get_user_preference_ids(user_entity_id, request.space_id, org_id) if user_entity_id else []
            fused = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=pref_ids)

            if request.include_profile and user_entity_id:
                profile = await self._build_profile_snapshot(user_entity_id, request.space_id, org_id)
        profile_ms = (time.perf_counter() - profile_t0) * 1000

        # ── Step 7: Build response ──────────────────────────────
        results = self._build_result_items(fused, request)

        total_ms = (time.perf_counter() - t0) * 1000
        timing = SearchTiming(
            total_ms=round(total_ms, 2),
            vector_ms=round(channel_results.get("vector", ChannelResult()).timing_ms, 2) if "vector" in channel_results else None,
            graph_ms=round(channel_results.get("graph", ChannelResult()).timing_ms, 2) if "graph" in channel_results else None,
            keyword_ms=round(channel_results.get("keyword", ChannelResult()).timing_ms, 2) if "keyword" in channel_results else None,
            fusion_ms=round(fusion_ms, 2),
            profile_ms=round(profile_ms, 2),
            rewrite_ms=round(rewrite_ms, 2) if request.rewrite_query else None,
        )

        return SearchResponse(
            results=results[:request.limit],
            profile=profile,
            timing=timing,
            query=processed.effective_query,
        )

    # ── Channel runners ─────────────────────────────────────────

    async def _run_vector_search(
        self,
        query_vec: list[float],
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> ChannelResult:
        """Run vector similarity search."""
        return await self._vector_channel.search(
            query_vec=query_vec,
            space_id=request.space_id,
            org_id=org_id,
            limit=request.limit,
            chunk_threshold=request.chunk_threshold,
            document_threshold=request.document_threshold,
        )

    async def _run_graph_search(
        self,
        seed_entity_ids: list[uuid.UUID],
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> ChannelResult:
        """Run graph traversal search."""
        return await self._graph_channel.search(
            seed_entity_ids=seed_entity_ids,
            space_id=request.space_id,
            org_id=org_id,
            max_depth=3,
            limit=request.limit,
        )

    async def _run_keyword_search(
        self,
        query_text: str,
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> ChannelResult:
        """Run keyword full-text search."""
        return await self._keyword_channel.search(
            query_text=query_text,
            space_id=request.space_id,
            org_id=org_id,
            limit=request.limit,
        )

    # ── Channel resolution ──────────────────────────────────────

    def _resolve_channels(self, request: SearchRequest) -> set[SearchChannel]:
        """Determine which channels to run based on request parameters."""
        if request.channels is not None:
            return set(request.channels)

        if request.mode == SearchMode.hybrid:
            return {SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword}
        elif request.mode == SearchMode.memory_only:
            return {SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword}
        elif request.mode == SearchMode.documents_only:
            return {SearchChannel.vector, SearchChannel.keyword}
        return {SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword}

    def _filter_by_mode(
        self,
        channel_item_map: dict[str, list[dict[str, Any]]],
        mode: SearchMode,
    ) -> dict[str, list[dict[str, Any]]]:
        """Filter channel results by search mode (memory_only vs documents_only)."""
        if mode == SearchMode.memory_only:
            filtered: dict[str, list[dict[str, Any]]] = {}
            for ch, items in channel_item_map.items():
                filtered[ch] = [i for i in items if i.get("type") in ("memory", "entity")]
            return filtered
        elif mode == SearchMode.documents_only:
            filtered = {}
            for ch, items in channel_item_map.items():
                filtered[ch] = [i for i in items if i.get("type") in ("document_chunk", "document")]
            return filtered
        return channel_item_map

    # ── Profile helpers ─────────────────────────────────────────

    async def _get_user_entity_id(
        self,
        user_id: uuid.UUID,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> uuid.UUID | None:
        """Look up the entity associated with a user_id.

        Convention: entity metadata contains user_id field.
        """
        sql = text("""
            SELECT e.id
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND e.metadata->>'user_id' = :user_id
            LIMIT 1
        """)
        result = await self._session.execute(sql, {
            "space_id": str(space_id),
            "org_id": str(org_id),
            "user_id": str(user_id),
        })
        row = result.fetchone()
        return row.id if row else None

    async def _get_user_preference_ids(
        self,
        user_entity_id: uuid.UUID,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        """Get entity IDs that the user entity has 'prefers' relations to."""
        sql = text("""
            SELECT r.target_entity_id
            FROM relations r
            WHERE r.source_entity_id = :entity_id
              AND r.relation_type = 'prefers'
              AND r.is_current = true
              AND r.space_id = :space_id
              AND r.org_id = :org_id
        """)
        result = await self._session.execute(sql, {
            "entity_id": str(user_entity_id),
            "space_id": str(space_id),
            "org_id": str(org_id),
        })
        return [row.target_entity_id for row in result.fetchall()]

    async def _build_profile_snapshot(
        self,
        user_entity_id: uuid.UUID,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> SearchProfile:
        """Build a lightweight profile snapshot for the search response.

        Simplified version — full profile computation is in ProfileEngine.
        """
        identity_sql = text("""
            SELECT r.relation_type, t.name AS target_name, t.entity_type
            FROM relations r
            JOIN entities t ON r.target_entity_id = t.id
            WHERE r.source_entity_id = :entity_id
              AND r.is_current = true
              AND r.space_id = :space_id
              AND r.org_id = :org_id
        """)
        result = await self._session.execute(identity_sql, {
            "entity_id": str(user_entity_id),
            "space_id": str(space_id),
            "org_id": str(org_id),
        })

        identity: dict[str, Any] = {}
        preferences: dict[str, Any] = {}
        current_status: dict[str, Any] = {}

        for row in result.fetchall():
            rel_type = row.relation_type
            target_name = row.target_name
            if rel_type in ("lives_in", "works_at", "role"):
                identity[rel_type] = target_name
            elif rel_type == "prefers":
                etype = row.entity_type or "other"
                preferences.setdefault(etype, []).append(target_name)
            else:
                current_status[rel_type] = target_name

        return SearchProfile(
            identity=identity if identity else None,
            preferences=preferences if preferences else None,
            current_status=current_status if current_status else None,
        )

    # ── Result builder ─────────────────────────────────────────

    def _build_result_items(
        self,
        fused: list[FusedResult],
        request: SearchRequest,
    ) -> list[SearchResultItem]:
        """Convert FusedResult list to API SearchResultItem list."""
        items: list[SearchResultItem] = []
        for fr in fused:
            entity_brief = None
            if fr.entity_name:
                entity_brief = EntityBrief(name=fr.entity_name, type=fr.entity_type or "")

            doc_brief = None
            if fr.document:
                doc_brief = DocumentBrief(
                    id=fr.document.get("id", fr.document_id or uuid.UUID(int=0)),
                    title=fr.document.get("title", ""),
                    url=fr.document.get("url"),
                )

            try:
                result_type = ResultType(fr.type)
            except ValueError:
                result_type = ResultType.memory

            items.append(SearchResultItem(
                type=result_type,
                id=fr.id,
                content=fr.content,
                score=round(fr.boosted_score, 6),
                source="+".join(fr.source_channels),
                entity=entity_brief,
                document=doc_brief,
                metadata=fr.metadata,
                created_at=fr.created_at,
            ))
        return items
```

- [ ] **Run** SearchEngine tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_engine.py -xvs
```

**Expected:** All 5 tests pass.

- [ ] **Commit:** `git add server/app/core/search_engine.py && git commit -m "feat(search): implement SearchEngine orchestrator with parallel channels, RRF, and profile boost"`

---

## Phase 6: API Route

### Step 6.1 — Write failing tests for search API endpoint

- [ ] **Write** `server/tests/test_search_api.py`

```python
"""API-level integration tests for POST /v1/search/."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.search import SearchResponse


@pytest.fixture
def search_request_payload():
    return {
        "q": "张军最近在用什么框架？",
        "space_id": str(uuid.uuid4()),
        "mode": "hybrid",
        "limit": 10,
        "include_profile": False,
        "rewrite_query": False,
    }


class TestSearchAPI:
    @pytest.mark.asyncio
    async def test_search_endpoint_returns_200(self, search_request_payload):
        """POST /v1/search/ should return 200 with valid payload."""
        mock_response = SearchResponse(
            results=[],
            timing={"total_ms": 0.0, "vector_ms": None, "graph_ms": None, "keyword_ms": None, "fusion_ms": None, "profile_ms": None, "rewrite_ms": None},
            query="张军最近在用什么框架？",
        )

        with patch("app.api.search.SearchEngine") as MockEngine:
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            MockEngine.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/search/", json=search_request_payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "timing" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_search_endpoint_validates_required_fields(self):
        """POST /v1/search/ without required 'q' should return 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/search/", json={"space_id": str(uuid.uuid4())})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_endpoint_with_channels(self, search_request_payload):
        """POST /v1/search/ with channels parameter should be accepted."""
        search_request_payload["channels"] = ["vector", "keyword"]
        mock_response = SearchResponse(
            results=[],
            timing={"total_ms": 0.0, "vector_ms": 1.0, "graph_ms": None, "keyword_ms": 2.0, "fusion_ms": 0.1, "profile_ms": None, "rewrite_ms": None},
            query="张军最近在用什么框架？",
        )

        with patch("app.api.search.SearchEngine") as MockEngine:
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            MockEngine.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/search/", json=search_request_payload)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_endpoint_with_filters(self, search_request_payload):
        """POST /v1/search/ with metadata filters should be accepted."""
        search_request_payload["filters"] = {"AND": [{"key": "source", "value": "slack"}]}
        mock_response = SearchResponse(
            results=[],
            timing={"total_ms": 0.0, "vector_ms": None, "graph_ms": None, "keyword_ms": None, "fusion_ms": None, "profile_ms": None, "rewrite_ms": None},
            query="张军最近在用什么框架？",
        )

        with patch("app.api.search.SearchEngine") as MockEngine:
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            MockEngine.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/search/", json=search_request_payload)

        assert response.status_code == 200
```

- [ ] **Commit:** `git add server/tests/test_search_api.py && git commit -m "test(search): add API-level integration tests for POST /v1/search/"`

---

### Step 6.2 — Implement POST /v1/search/ route

- [ ] **Write** `server/app/api/search.py`

```python
"""POST /v1/search/ — unified hybrid search endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.search_engine import SearchEngine
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/v1/search", tags=["search"])


def _get_embedding_service():
    """Dependency injection for embedding service.

    Uses app.state.embedding_service if set (initialized at startup),
    otherwise creates one from config settings.
    """
    from app.config import get_settings
    settings = get_settings()
    from app.core.embedding.service import EmbeddingService
    return EmbeddingService.from_config(settings)


def _get_llm_adapter():
    """Dependency injection for LLM adapter (optional).

    Returns None if no LLM is configured.
    """
    try:
        from app.config import get_settings
        settings = get_settings()
        from app.core.llm.adapter import LLMAdapter
        return LLMAdapter.from_config(settings)
    except Exception:
        return None


@router.post("/", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    embedding_service=Depends(_get_embedding_service),
    llm_adapter=Depends(_get_llm_adapter),
) -> SearchResponse:
    """Execute a hybrid search across memories, entities, and documents.

    Pipeline: QueryProcessor -> [Vector, Graph, Keyword] -> RRF -> Profile Boost -> Results

    The search combines three channels:
    - **Vector**: pgvector cosine similarity across memories, chunks, entities
    - **Graph**: Recursive CTE from identified entities (up to 3 hops)
    - **Keyword**: PostgreSQL tsvector full-text search (simple config for Chinese)

    Results are fused via Reciprocal Rank Fusion (k=60) and optionally
    boosted by user profile (entity match x1.5, preference match x1.2).
    """
    org_id = await _resolve_org_id(db, request.space_id)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Space {request.space_id} not found",
        )

    engine = SearchEngine(
        db_session=db,
        embedding_service=embedding_service,
        llm_adapter=llm_adapter,
    )

    response = await engine.search(request, org_id=org_id)
    return response


async def _resolve_org_id(db: AsyncSession, space_id: uuid.UUID) -> uuid.UUID | None:
    """Look up the org_id for a given space."""
    from sqlalchemy import text
    result = await db.execute(
        text("SELECT org_id FROM spaces WHERE id = :space_id"),
        {"space_id": str(space_id)},
    )
    row = result.fetchone()
    return row.org_id if row else None
```

- [ ] **Register the router in** `server/app/main.py` (add if not present):

```python
from app.api.search import router as search_router
app.include_router(search_router)
```

- [ ] **Run** API tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_api.py -xvs
```

**Expected:** All 4 API tests pass.

- [ ] **Commit:** `git add server/app/api/search.py && git commit -m "feat(search): implement POST /v1/search/ route with SearchEngine integration"`

---

## Phase 7: Metadata Filters & Edge Cases

### Step 7.1 — Add metadata filter support to search channels

- [ ] **Add** filter method to `VectorSearchChannel` in `server/app/core/search_channels.py`

```python
    def _build_metadata_filter_clause(self, filters: dict | None) -> tuple[str, dict]:
        """Build SQL WHERE clause from metadata filter dict.

        Supports AND/OR nested structure:
        {"AND": [{"key": "source", "value": "slack"}]}
        {"OR": [{"key": "source", "value": "slack"}, {"key": "source", "value": "email"}]}

        Returns (sql_fragment, params_dict).
        """
        if not filters:
            return "", {}

        conditions = []
        params = {}
        op = "AND"

        if "AND" in filters:
            items = filters["AND"]
            op = "AND"
        elif "OR" in filters:
            items = filters["OR"]
            op = "OR"
        else:
            items = [filters]
            op = "AND"

        for i, f in enumerate(items):
            key = f.get("key", "")
            value = f.get("value", "")
            param_key = f"mf_{key}_{i}"
            conditions.append(f"metadata->>:{param_key} = :{param_key}_val")
            params[param_key] = key
            params[f"{param_key}_val"] = str(value)

        if not conditions:
            return "", {}

        clause = f" {op} ".join(conditions)
        return f"AND ({clause})", params
```

- [ ] **Add** filter tests to `server/tests/test_search_channels.py`

```python
class TestMetadataFilter:
    def test_build_and_filter(self):
        channel = VectorSearchChannel(db_session=AsyncMock())
        clause, params = channel._build_metadata_filter_clause({"AND": [{"key": "source", "value": "slack"}]})
        assert "AND" in clause
        assert "metadata" in clause

    def test_build_or_filter(self):
        channel = VectorSearchChannel(db_session=AsyncMock())
        clause, params = channel._build_metadata_filter_clause({"OR": [{"key": "source", "value": "slack"}, {"key": "source", "value": "email"}]})
        assert "OR" in clause

    def test_empty_filter(self):
        channel = VectorSearchChannel(db_session=AsyncMock())
        clause, params = channel._build_metadata_filter_clause(None)
        assert clause == ""
        assert params == {}
```

- [ ] **Run** filter tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_channels.py::TestMetadataFilter -xvs
```

**Expected:** All 3 filter tests pass.

- [ ] **Commit:** `git add server/app/core/search_channels.py server/tests/test_search_channels.py && git commit -m "feat(search): add metadata filter builder for AND/OR conditions"`

---

### Step 7.2 — Handle documents_only mode and result assembly

- [ ] **Add** document assembly method to SearchEngine in `server/app/core/search_engine.py`

```python
    async def _assemble_document_results(
        self,
        chunk_items: list[dict[str, Any]],
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """Assemble chunks into document-level results when include_full_docs=True.

        If only_matching_chunks=True, return chunks as-is.
        If include_full_docs=True, fetch full documents and attach their chunks.
        """
        if request.only_matching_chunks or not request.include_full_docs:
            return chunk_items

        doc_ids = list({item["document_id"] for item in chunk_items if "document_id" in item})
        if not doc_ids:
            return chunk_items

        sql = text("""
            SELECT d.id, d.title, d.content, d.url, d.summary, d.metadata
            FROM documents d
            WHERE d.id = ANY(:doc_ids)
              AND d.space_id = :space_id
              AND d.org_id = :org_id
        """)
        result = await self._session.execute(sql, {
            "doc_ids": [str(did) for did in doc_ids],
            "space_id": str(request.space_id),
            "org_id": str(org_id),
        })

        doc_map: dict[uuid.UUID, dict] = {}
        for row in result.fetchall():
            doc_map[row.id] = {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "url": row.url,
                "summary": row.summary if request.include_summary else None,
                "metadata": row.metadata,
            }

        assembled = []
        for item in chunk_items:
            doc_id = item.get("document_id")
            if doc_id and doc_id in doc_map:
                item["document"] = doc_map[doc_id]
            assembled.append(item)

        return assembled
```

- [ ] **Commit:** `git add server/app/core/search_engine.py && git commit -m "feat(search): add document assembly for include_full_docs and only_matching_chunks modes"`

---

## Phase 8: End-to-End Validation

### Step 8.1 — Run full test suite for search module

- [ ] **Run** all search-related tests:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_search_channels.py tests/test_search_fusion.py tests/test_query_processor.py tests/test_search_engine.py tests/test_search_api.py -v --tb=short
```

**Expected:** All tests pass (approximately 20 tests total).

### Step 8.2 — Verify schema round-trip

- [ ] **Run** schema serialization check:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "
from app.schemas.search import SearchRequest, SearchResponse, SearchTiming, SearchResultItem, SearchMode, SearchChannel, ResultType, EntityBrief, DocumentBrief, SearchProfile
import uuid, json

req = SearchRequest(
    q='张军最近在用什么框架做前端？',
    space_id=uuid.uuid4(),
    user_id=uuid.uuid4(),
    mode=SearchMode.hybrid,
    channels=[SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword],
    limit=50,
    rerank=True,
    include_profile=True,
    chunk_threshold=0.3,
    document_threshold=0.2,
    only_matching_chunks=False,
    include_full_docs=True,
    include_summary=True,
    filters={'AND': [{'key': 'source', 'value': 'slack'}]},
    rewrite_query=True,
)
req_json = req.model_dump_json()
req2 = SearchRequest.model_validate_json(req_json)
assert req2.q == req.q
assert req2.mode == SearchMode.hybrid
assert len(req2.channels) == 3
assert req2.chunk_threshold == 0.3
assert req2.filters['AND'][0]['key'] == 'source'
print('Request round-trip OK')

timing = SearchTiming(total_ms=45.0, vector_ms=12.0, graph_ms=5.0, keyword_ms=8.0, fusion_ms=0.5, profile_ms=2.0, rewrite_ms=15.0)
item = SearchResultItem(
    type=ResultType.memory,
    id=uuid.uuid4(),
    content='张军最近在用 Next.js 做前端项目',
    score=0.92,
    source='vector+graph',
    entity=EntityBrief(name='张军', type='person'),
)
profile = SearchProfile(identity={'name': '张军', 'location': '北京'}, preferences={'stack': ['Python', 'TypeScript']}, current_status={'focus': '知识图谱'})
resp = SearchResponse(results=[item], profile=profile, timing=timing, query='张军最近在用什么框架做前端？')
resp_json = resp.model_dump_json()
resp2 = SearchResponse.model_validate_json(resp_json)
assert len(resp2.results) == 1
assert resp2.results[0].score == 0.92
assert resp2.profile.identity['location'] == '北京'
assert resp2.timing.total_ms == 45.0
print('Response round-trip OK')
print('All schema validation passed!')
"
```

**Expected:**
```
Request round-trip OK
Response round-trip OK
All schema validation passed!
```

- [ ] **Final commit:** `git add -A && git commit -m "feat(search): complete search engine — all channels, RRF fusion, profile boost, API route"`

---

## Summary of Deliverables

| Component | File | Key Details |
|-----------|------|-------------|
| Schemas | `server/app/schemas/search.py` | SearchRequest with all 17 params, SearchResponse with timing |
| Vector Channel | `server/app/core/search_channels.py` | `1 - (embedding <=> $query_vec) AS similarity` on memories, chunks, entities |
| Graph Channel | `server/app/core/search_channels.py` | Recursive CTE from seed entities, max 3 hops, depth-based scoring |
| Keyword Channel | `server/app/core/search_channels.py` | `to_tsvector('simple', ...) @@ to_tsquery('simple', ...)` for Chinese |
| RRF Fusion | `server/app/core/search_fusion.py` | `score += 1/(k + rank + 1)`, k=60, across all channels |
| Profile Boost | `server/app/core/search_fusion.py` | Entity match x1.5, preference match x1.2, multiplicative x1.8 |
| Query Processor | `server/app/core/query_processor.py` | Entity recognition via ILIKE, optional LLM rewrite |
| SearchEngine | `server/app/core/search_engine.py` | Parallel channels, RRF, Profile Boost, timing breakdown, document assembly |
| API Route | `server/app/api/search.py` | `POST /v1/search/` with all search parameters |
| Tests | `server/tests/test_search_*.py` | 5 test files covering channels, fusion, processor, engine, API |

### Key Formulas (as specified)

- **Vector similarity:** `1 - (embedding <=> $query_vec) AS similarity`
- **RRF fusion:** `score += 1/(k + rank + 1)` where `k=60`
- **Profile Boost:** entity match `x1.5`, preference match `x1.2`, both multiplicative `x1.8`
- **Graph depth scores:** depth 0 -> 1.0, 1 -> 0.8, 2 -> 0.5, 3 -> 0.3
- **Keyword search:** `to_tsvector('simple', content) @@ to_tsquery('simple', $tsquery)`