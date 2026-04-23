# RTMemory Graph Engine -- 时序知识图谱核心

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the temporal knowledge graph engine -- entity/relation/memory CRUD with contradiction handling, version chains, and graph traversal.

**Architecture:** GraphEngine as central service layer. Temporal relations use valid_from/valid_to time intervals. Contradictions resolved by closing old relations and opening new ones. Memory updates create version chains.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0 (async), FastAPI, PostgreSQL, pgvector

**Depends on:** Plan 01 (foundation: project skeleton, DB models, config, Docker Compose)

---

## File Map

| File | Purpose |
|------|---------|
| `server/app/db/models.py` | SQLAlchemy models (created in plan 01, extended here if needed) |
| `server/app/db/session.py` | Async session factory (created in plan 01) |
| `server/app/core/graph_engine.py` | GraphEngine class -- all graph operations |
| `server/app/api/entities.py` | `/v1/entities/` REST routes |
| `server/app/api/relations.py` | `/v1/relations/` REST routes |
| `server/app/api/memories.py` | `/v1/memories/` REST routes |
| `server/app/schemas/graph.py` | Pydantic request/response schemas |
| `server/tests/test_graph_engine.py` | GraphEngine unit tests |
| `server/tests/test_api_entities.py` | Entity API integration tests |
| `server/tests/test_api_relations.py` | Relation API integration tests |
| `server/tests/test_api_memories.py` | Memory API integration tests |
| `server/tests/conftest.py` | Test fixtures (async DB session, test client) |

---

## Phase 1: Schemas & Test Infrastructure

### Step 1.1 -- Create Pydantic schemas for entities, relations, memories

- [ ] **Write** `server/app/schemas/graph.py`

```python
"""Pydantic schemas for graph engine request/response models."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Entity ──────────────────────────────────────────────────────

class EntityType(str, Enum):
    person = "person"
    org = "org"
    location = "location"
    concept = "concept"
    project = "project"
    technology = "technology"


class EntityCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    entity_type: EntityType
    description: str = Field(default="", max_length=2000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    org_id: uuid.UUID
    space_id: uuid.UUID


class EntityUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    entity_type: Optional[EntityType] = None
    description: Optional[str] = Field(default=None, max_length=2000)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EntityOut(BaseModel):
    id: uuid.UUID
    name: str
    entity_type: EntityType
    description: str
    confidence: float
    org_id: uuid.UUID
    space_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EntityListOut(BaseModel):
    items: list[EntityOut]
    total: int
    offset: int
    limit: int


# ── Relation ───────────────────────────────────────────────────

class RelationCreate(BaseModel):
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    relation_type: str = Field(..., min_length=1, max_length=200)
    value: str = Field(default="", max_length=2000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    org_id: uuid.UUID
    space_id: uuid.UUID


class RelationUpdate(BaseModel):
    relation_type: Optional[str] = Field(default=None, min_length=1, max_length=200)
    value: Optional[str] = Field(default=None, max_length=2000)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_current: Optional[bool] = None


class RelationOut(BaseModel):
    id: uuid.UUID
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    relation_type: str
    value: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    confidence: float
    is_current: bool
    source_count: int
    org_id: uuid.UUID
    space_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RelationListOut(BaseModel):
    items: list[RelationOut]
    total: int
    offset: int
    limit: int


# ── Memory ─────────────────────────────────────────────────────

class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    custom_id: Optional[str] = Field(default=None, max_length=500)
    memory_type: MemoryType = Field(default=MemoryType.fact)
    entity_id: Optional[uuid.UUID] = None
    relation_id: Optional[uuid.UUID] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[dict] = None
    org_id: uuid.UUID
    space_id: uuid.UUID
    document_ids: Optional[list[uuid.UUID]] = None


class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(default=None, min_length=1, max_length=10000)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    decay_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[dict] = None


class MemoryForget(BaseModel):
    forget_reason: str = Field(default="", max_length=2000)


class MemoryOut(BaseModel):
    id: uuid.UUID
    content: str
    custom_id: Optional[str] = None
    memory_type: MemoryType
    entity_id: Optional[uuid.UUID] = None
    relation_id: Optional[uuid.UUID] = None
    confidence: float
    decay_rate: Optional[float] = None
    is_forgotten: bool
    forget_at: Optional[datetime] = None
    forget_reason: Optional[str] = None
    version: int
    parent_id: Optional[uuid.UUID] = None
    root_id: Optional[uuid.UUID] = None
    metadata: Optional[dict] = None
    org_id: uuid.UUID
    space_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MemoryVersionChainOut(BaseModel):
    current: MemoryOut
    versions: list[MemoryOut]


class MemoryListOut(BaseModel):
    items: list[MemoryOut]
    total: int
    offset: int
    limit: int


# ── Graph Traversal ────────────────────────────────────────────

class GraphTraversalParams(BaseModel):
    entity_id: uuid.UUID
    max_hops: int = Field(default=3, ge=1, le=10)
    relation_types: Optional[list[str]] = None
    direction: str = Field(default="both", pattern="^(outgoing|incoming|both)$")


class TraversedRelationOut(BaseModel):
    relation: RelationOut
    hop: int
    direction: str  # "outgoing" or "incoming"


class GraphTraversalOut(BaseModel):
    start_entity_id: uuid.UUID
    entities: list[EntityOut]
    relations: list[TraversedRelationOut]
    max_hops: int


# ── Memory-Document Source ─────────────────────────────────────

class MemorySourceCreate(BaseModel):
    memory_id: uuid.UUID
    document_id: uuid.UUID
    chunk_id: Optional[uuid.UUID] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MemorySourceOut(BaseModel):
    memory_id: uuid.UUID
    document_id: uuid.UUID
    chunk_id: Optional[uuid.UUID] = None
    relevance_score: float

    model_config = {"from_attributes": True}
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from app.schemas.graph import EntityCreate, RelationCreate, MemoryCreate; print('schemas OK')"`
**Expected:** `schemas OK`

**Commit:** `git add server/app/schemas/graph.py && git commit -m "Add Pydantic schemas for entity, relation, memory, traversal, and source models"`

---

### Step 1.2 -- Create test conftest with async DB session fixture

- [ ] **Write** `server/tests/conftest.py`

```python
"""Test configuration and fixtures for RTMemory server tests."""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base
from app.db.session import get_session


# Use a separate test database. Set env var or default to local test DB.
TEST_DATABASE_URL = (
    "postgresql+asyncpg://rtmemory_test:rtmemory_test@localhost:5432/rtmemory_test"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create a test database engine and drop all tables after session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test, rolled back after."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def db_session_committed(
    test_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session that commits (for tests that need persisted data)."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
        # Clean up: delete all rows in reverse dependency order
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()


def make_test_org_id() -> uuid.UUID:
    return uuid.uuid4()


def make_test_space_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest_asyncio.fixture
def org_id() -> uuid.UUID:
    return make_test_org_id()


@pytest_asyncio.fixture
def space_id() -> uuid.UUID:
    return make_test_space_id()
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from tests.conftest import TEST_DATABASE_URL; print('conftest OK')"`
**Expected:** `conftest OK`

**Commit:** `git add server/tests/conftest.py && git commit -m "Add test conftest with async DB session fixtures"`

---

### Step 1.3 -- Create tests/__init__.py and ensure test runner works

- [ ] **Write** `server/tests/__init__.py`

```python
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest --collect-only 2>&1 | head -5`
**Expected:** No import errors; test collection runs (may find 0 tests initially).

**Commit:** `git add server/tests/__init__.py && git commit -m "Add tests package init"`

---

## Phase 2: GraphEngine Core -- Entity Operations

### Step 2.1 -- Write failing tests for entity CRUD

- [ ] **Write** `server/tests/test_graph_engine.py` -- entity section only

```python
"""Unit tests for GraphEngine entity operations."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.schemas.graph import EntityCreate, EntityUpdate


@pytest_asyncio.fixture
async def engine(db_session: AsyncSession) -> GraphEngine:
    return GraphEngine(db_session)


class TestEntityCreate:
    async def test_create_entity_returns_entity_out(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = EntityCreate(
            name="Zhang Jun",
            entity_type="person",
            description="Software engineer",
            confidence=0.9,
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_entity(data)
        assert result.name == "Zhang Jun"
        assert result.entity_type.value == "person"
        assert result.confidence == 0.9
        assert result.id is not None

    async def test_create_entity_generates_uuid_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = EntityCreate(
            name="Beijing",
            entity_type="location",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_entity(data)
        assert isinstance(result.id, uuid.UUID)

    async def test_create_entity_sets_timestamps(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = EntityCreate(
            name="Python",
            entity_type="technology",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_entity(data)
        assert result.created_at is not None
        assert result.updated_at is not None


class TestEntityGet:
    async def test_get_entity_by_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = EntityCreate(
            name="Zhang Jun",
            entity_type="person",
            org_id=org_id,
            space_id=space_id,
        )
        created = await engine.create_entity(data)
        fetched = await engine.get_entity(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Zhang Jun"

    async def test_get_entity_not_found_returns_none(self, engine: GraphEngine):
        result = await engine.get_entity(uuid.uuid4())
        assert result is None


class TestEntityList:
    async def test_list_entities_empty(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        result = await engine.list_entities(org_id=org_id, space_id=space_id)
        assert result.total == 0
        assert result.items == []

    async def test_list_entities_returns_created(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_entity(EntityCreate(name="B", entity_type="org", org_id=org_id, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id)
        assert result.total == 2

    async def test_list_entities_filters_by_org(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        other_org = uuid.uuid4()
        await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_entity(EntityCreate(name="B", entity_type="person", org_id=other_org, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id)
        assert result.total == 1

    async def test_list_entities_pagination(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        for i in range(5):
            await engine.create_entity(EntityCreate(name=f"E{i}", entity_type="person", org_id=org_id, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id, limit=2, offset=0)
        assert len(result.items) == 2
        assert result.total == 5

    async def test_list_entities_filter_by_type(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_entity(EntityCreate(name="B", entity_type="technology", org_id=org_id, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id, entity_type="person")
        assert result.total == 1
        assert result.items[0].entity_type.value == "person"


class TestEntityUpdate:
    async def test_update_entity_name(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_entity(EntityCreate(name="Old", entity_type="person", org_id=org_id, space_id=space_id))
        updated = await engine.update_entity(created.id, EntityUpdate(name="New"))
        assert updated.name == "New"

    async def test_update_entity_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Entity not found"):
            await engine.update_entity(uuid.uuid4(), EntityUpdate(name="X"))


class TestEntityDelete:
    async def test_delete_entity(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_entity(EntityCreate(name="ToDelete", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.delete_entity(created.id)
        fetched = await engine.get_entity(created.id)
        assert fetched is None

    async def test_delete_entity_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Entity not found"):
            await engine.delete_entity(uuid.uuid4())
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -x -q 2>&1 | tail -5`
**Expected:** FAIL -- `ModuleNotFoundError: No module named 'app.core.graph_engine'`

**Commit:** `git add server/tests/test_graph_engine.py && git commit -m "Add failing entity CRUD tests for GraphEngine"`

---

### Step 2.2 -- Implement GraphEngine entity CRUD methods

- [ ] **Write** `server/app/core/__init__.py`

```python
```

- [ ] **Write** `server/app/core/graph_engine.py` -- entity methods only

```python
"""GraphEngine -- central data access layer for the temporal knowledge graph."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Entity, Memory, MemorySource, Relation
from app.schemas.graph import (
    EntityCreate,
    EntityOut,
    EntityUpdate,
    EntityListOut,
    EntityType,
    GraphTraversalOut,
    GraphTraversalParams,
    MemoryCreate,
    MemoryForget,
    MemoryListOut,
    MemoryOut,
    MemorySourceCreate,
    MemorySourceOut,
    MemoryUpdate,
    MemoryVersionChainOut,
    RelationCreate,
    RelationListOut,
    RelationOut,
    RelationUpdate,
    TraversedRelationOut,
)


class GraphEngine:
    """Central data access layer for all knowledge graph operations.

    Receives an async SQLAlchemy session via dependency injection.
    All methods are async and operate on the database through this session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Entity CRUD ────────────────────────────────────────────

    async def create_entity(self, data: EntityCreate) -> EntityOut:
        """Create a new entity."""
        now = datetime.now(timezone.utc)
        entity = Entity(
            id=uuid.uuid4(),
            name=data.name,
            entity_type=data.entity_type.value,
            description=data.description,
            confidence=data.confidence,
            org_id=data.org_id,
            space_id=data.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(entity)
        await self.session.flush()
        return EntityOut.model_validate(entity)

    async def get_entity(self, entity_id: uuid.UUID) -> Optional[EntityOut]:
        """Get an entity by ID. Returns None if not found."""
        stmt = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            return None
        return EntityOut.model_validate(entity)

    async def list_entities(
        self,
        *,
        org_id: uuid.UUID,
        space_id: Optional[uuid.UUID] = None,
        entity_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> EntityListOut:
        """List entities with optional filters and pagination."""
        stmt = select(Entity).where(Entity.org_id == org_id)
        count_stmt = select(func.count()).select_from(Entity).where(Entity.org_id == org_id)

        if space_id is not None:
            stmt = stmt.where(Entity.space_id == space_id)
            count_stmt = count_stmt.where(Entity.space_id == space_id)
        if entity_type is not None:
            stmt = stmt.where(Entity.entity_type == entity_type)
            count_stmt = count_stmt.where(Entity.entity_type == entity_type)

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Entity.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        entities = result.scalars().all()

        return EntityListOut(
            items=[EntityOut.model_validate(e) for e in entities],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def update_entity(
        self, entity_id: uuid.UUID, data: EntityUpdate
    ) -> EntityOut:
        """Update an entity's mutable fields."""
        stmt = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            raise ValueError("Entity not found")

        update_data = data.model_dump(exclude_unset=True)
        if update_data:
            # Convert entity_type enum to string if present
            if "entity_type" in update_data and update_data["entity_type"] is not None:
                update_data["entity_type"] = update_data["entity_type"].value
            update_data["updated_at"] = datetime.now(timezone.utc)
            for key, value in update_data.items():
                setattr(entity, key, value)
            await self.session.flush()

        return EntityOut.model_validate(entity)

    async def delete_entity(self, entity_id: uuid.UUID) -> None:
        """Hard-delete an entity by ID."""
        stmt = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            raise ValueError("Entity not found")

        await self.session.delete(entity)
        await self.session.flush()
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py::TestEntityCreate -x -v 2>&1 | tail -10`
**Expected:** All 3 entity create tests PASS

**Commit:** `git add server/app/core/__init__.py server/app/core/graph_engine.py && git commit -m "Implement GraphEngine entity CRUD methods"`

---

### Step 2.3 -- Run and verify all entity CRUD tests pass

- [ ] **Run** all entity tests

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -x -v 2>&1`
**Expected:** All entity tests (create, get, list, update, delete) pass

**Commit:** (none needed if all pass -- already committed)

---

## Phase 3: GraphEngine Core -- Relation Operations with Temporal Logic

### Step 3.1 -- Write failing tests for relation CRUD and contradiction handling

- [ ] **Append** to `server/tests/test_graph_engine.py` -- relation section

```python
# ── Relation Tests ─────────────────────────────────────────────

from app.schemas.graph import RelationCreate, RelationUpdate


@pytest_asyncio.fixture
async def two_entities(engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
    """Create two entities for relation tests."""
    e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
    e2 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))
    return e1, e2


class TestRelationCreate:
    async def test_create_relation_returns_relation_out(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        data = RelationCreate(
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relation_type="lives_in",
            value="",
            confidence=0.9,
            org_id=e1.org_id,
            space_id=e1.space_id,
        )
        result = await engine.create_relation(data)
        assert result.relation_type == "lives_in"
        assert result.source_entity_id == e1.id
        assert result.target_entity_id == e2.id
        assert result.is_current is True
        assert result.valid_from is not None
        assert result.valid_to is None
        assert result.source_count == 1

    async def test_create_relation_sets_temporal_defaults(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        data = RelationCreate(
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relation_type="works_at",
            org_id=e1.org_id,
            space_id=e1.space_id,
        )
        result = await engine.create_relation(data)
        assert result.is_current is True
        assert result.valid_to is None
        assert result.source_count == 1


class TestRelationGet:
    async def test_get_relation_by_id(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        created = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        fetched = await engine.get_relation(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_relation_not_found_returns_none(self, engine: GraphEngine):
        result = await engine.get_relation(uuid.uuid4())
        assert result is None


class TestRelationList:
    async def test_list_relations_empty(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        result = await engine.list_relations(org_id=org_id, space_id=space_id)
        assert result.total == 0

    async def test_list_relations_filters_current_only(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        result_all = await engine.list_relations(org_id=e1.org_id, space_id=e1.space_id)
        result_current = await engine.list_relations(org_id=e1.org_id, space_id=e1.space_id, is_current=True)
        assert result_all.total == 1
        assert result_current.total == 1

    async def test_list_relations_filter_by_source_entity(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        result = await engine.list_relations(org_id=e1.org_id, space_id=e1.space_id, source_entity_id=e1.id)
        assert result.total == 1


class TestRelationUpdate:
    async def test_update_relation_value(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        created = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        updated = await engine.update_relation(created.id, RelationUpdate(value="Haidian district"))
        assert updated.value == "Haidian district"


class TestRelationDelete:
    async def test_delete_relation(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        created = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        await engine.delete_relation(created.id)
        fetched = await engine.get_relation(created.id)
        assert fetched is None


class TestRelationContradiction:
    async def test_contradiction_closes_old_and_opens_new(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        """When adding a new relation with same source+type that conflicts, close old, insert new."""
        e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="Shanghai", entity_type="location", org_id=org_id, space_id=space_id))
        e3 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))

        # Create first relation: Zhang Jun lives_in Shanghai
        old_rel = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        assert old_rel.is_current is True
        assert old_rel.valid_to is None

        # Create contradicting relation: Zhang Jun lives_in Beijing
        new_rel = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e3.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        assert new_rel.is_current is True
        assert new_rel.valid_to is not None  # This test checks we handle contradiction
        assert new_rel.source_count == 1

        # Reload old relation -- should be closed
        refreshed_old = await engine.get_relation(old_rel.id)
        assert refreshed_old.is_current is False
        assert refreshed_old.valid_to is not None

    async def test_reaffirm_same_relation_increments_source_count(self, engine: GraphEngine, two_entities):
        """When adding a relation that matches an existing current one (same source+type+target), increment source_count."""
        e1, e2 = two_entities
        # Create initial relation
        rel1 = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        assert rel1.source_count == 1

        # Reaffirm: same source entity, same type, same target
        rel2 = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        # The existing relation should have source_count incremented
        refreshed = await engine.get_relation(rel1.id)
        assert refreshed.source_count == 2
        assert refreshed.is_current is True
        # rel2 should be the same relation (updated), not a new one
        assert rel2.id == rel1.id

    async def test_contradiction_with_different_target_closes_old(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        """Different target for same source+type triggers contradiction resolution."""
        e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="Python", entity_type="technology", org_id=org_id, space_id=space_id))
        e3 = await engine.create_entity(EntityCreate(name="TypeScript", entity_type="technology", org_id=org_id, space_id=space_id))

        old = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="prefers", org_id=org_id, space_id=space_id,
        ))
        new = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e3.id,
            relation_type="prefers", org_id=org_id, space_id=space_id,
        ))

        refreshed_old = await engine.get_relation(old.id)
        assert refreshed_old.is_current is False
        assert refreshed_old.valid_to is not None

        refreshed_new = await engine.get_relation(new.id)
        assert refreshed_new.is_current is True
        assert refreshed_new.valid_to is None
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Relation" -x -v 2>&1 | tail -10`
**Expected:** FAIL -- `create_relation` not yet implemented

**Commit:** `git add server/tests/test_graph_engine.py && git commit -m "Add failing relation CRUD and contradiction tests"`

---

### Step 3.2 -- Implement relation CRUD with contradiction handling in GraphEngine

- [ ] **Edit** `server/app/core/graph_engine.py` -- add relation methods

Add these methods to the `GraphEngine` class:

```python
    # ── Relation CRUD ──────────────────────────────────────────

    async def create_relation(self, data: RelationCreate) -> RelationOut:
        """Create a new relation with temporal defaults.

        Handles contradictions: if a current relation with the same
        source_entity_id + relation_type already exists:
          - Same target => reaffirm: increment source_count on existing
          - Different target => contradiction: close old, insert new
        """
        now = datetime.now(timezone.utc)

        # Check for existing current relation with same source + type
        stmt = select(Relation).where(
            Relation.source_entity_id == data.source_entity_id,
            Relation.relation_type == data.relation_type,
            Relation.is_current == True,
            Relation.org_id == data.org_id,
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            if existing.target_entity_id == data.target_entity_id:
                # Reaffirm: same source + type + target -> increment source_count
                existing.source_count += 1
                existing.updated_at = now
                existing.confidence = max(existing.confidence, data.confidence)
                await self.session.flush()
                return RelationOut.model_validate(existing)
            else:
                # Contradiction: different target -> close old, insert new
                existing.is_current = False
                existing.valid_to = now
                existing.updated_at = now
                await self.session.flush()

        relation = Relation(
            id=uuid.uuid4(),
            source_entity_id=data.source_entity_id,
            target_entity_id=data.target_entity_id,
            relation_type=data.relation_type,
            value=data.value,
            valid_from=now,
            valid_to=None,
            confidence=data.confidence,
            is_current=True,
            source_count=1,
            org_id=data.org_id,
            space_id=data.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(relation)
        await self.session.flush()
        return RelationOut.model_validate(relation)

    async def get_relation(self, relation_id: uuid.UUID) -> Optional[RelationOut]:
        """Get a relation by ID. Returns None if not found."""
        stmt = select(Relation).where(Relation.id == relation_id)
        result = await self.session.execute(stmt)
        relation = result.scalar_one_or_none()
        if relation is None:
            return None
        return RelationOut.model_validate(relation)

    async def list_relations(
        self,
        *,
        org_id: uuid.UUID,
        space_id: Optional[uuid.UUID] = None,
        source_entity_id: Optional[uuid.UUID] = None,
        target_entity_id: Optional[uuid.UUID] = None,
        relation_type: Optional[str] = None,
        is_current: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RelationListOut:
        """List relations with optional filters and pagination."""
        stmt = select(Relation).where(Relation.org_id == org_id)
        count_stmt = select(func.count()).select_from(Relation).where(Relation.org_id == org_id)

        if space_id is not None:
            stmt = stmt.where(Relation.space_id == space_id)
            count_stmt = count_stmt.where(Relation.space_id == space_id)
        if source_entity_id is not None:
            stmt = stmt.where(Relation.source_entity_id == source_entity_id)
            count_stmt = count_stmt.where(Relation.source_entity_id == source_entity_id)
        if target_entity_id is not None:
            stmt = stmt.where(Relation.target_entity_id == target_entity_id)
            count_stmt = count_stmt.where(Relation.target_entity_id == target_entity_id)
        if relation_type is not None:
            stmt = stmt.where(Relation.relation_type == relation_type)
            count_stmt = count_stmt.where(Relation.relation_type == relation_type)
        if is_current is not None:
            stmt = stmt.where(Relation.is_current == is_current)
            count_stmt = count_stmt.where(Relation.is_current == is_current)

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Relation.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        relations = result.scalars().all()

        return RelationListOut(
            items=[RelationOut.model_validate(r) for r in relations],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def update_relation(
        self, relation_id: uuid.UUID, data: RelationUpdate
    ) -> RelationOut:
        """Update a relation's mutable fields (value, confidence, is_current)."""
        stmt = select(Relation).where(Relation.id == relation_id)
        result = await self.session.execute(stmt)
        relation = result.scalar_one_or_none()
        if relation is None:
            raise ValueError("Relation not found")

        update_data = data.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc)
            for key, value in update_data.items():
                setattr(relation, key, value)
            await self.session.flush()

        return RelationOut.model_validate(relation)

    async def delete_relation(self, relation_id: uuid.UUID) -> None:
        """Hard-delete a relation by ID."""
        stmt = select(Relation).where(Relation.id == relation_id)
        result = await self.session.execute(stmt)
        relation = result.scalar_one_or_none()
        if relation is None:
            raise ValueError("Relation not found")

        await self.session.delete(relation)
        await self.session.flush()
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Relation" -x -v 2>&1 | tail -15`
**Expected:** All relation CRUD + contradiction tests PASS

**Commit:** `git add server/app/core/graph_engine.py && git commit -m "Implement relation CRUD with contradiction handling in GraphEngine"`

---

### Step 3.3 -- Run all entity + relation tests together

- [ ] **Run** full test suite for graph engine

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -v 2>&1`
**Expected:** All entity and relation tests pass

**Commit:** (none needed if all pass)

---

## Phase 4: GraphEngine Core -- Graph Traversal

### Step 4.1 -- Write failing tests for graph traversal

- [ ] **Append** to `server/tests/test_graph_engine.py` -- traversal section

```python
# ── Graph Traversal Tests ─────────────────────────────────────

from app.schemas.graph import GraphTraversalParams


class TestGraphTraversal:
    async def test_traverse_one_hop(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=3,
        ))
        assert result.start_entity_id == e1.id
        assert len(result.relations) == 1
        assert result.relations[0].hop == 1

    async def test_traverse_two_hops(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))
        e3 = await engine.create_entity(EntityCreate(name="China", entity_type="location", org_id=org_id, space_id=space_id))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        await engine.create_relation(RelationCreate(
            source_entity_id=e2.id, target_entity_id=e3.id,
            relation_type="part_of", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=3,
        ))
        assert len(result.relations) == 2
        hops = {r.hop for r in result.relations}
        assert 1 in hops
        assert 2 in hops

    async def test_traverse_respects_max_hops(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="A", entity_type="concept", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="B", entity_type="concept", org_id=org_id, space_id=space_id))
        e3 = await engine.create_entity(EntityCreate(name="C", entity_type="concept", org_id=org_id, space_id=space_id))
        e4 = await engine.create_entity(EntityCreate(name="D", entity_type="concept", org_id=org_id, space_id=space_id))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="knows", org_id=org_id, space_id=space_id,
        ))
        await engine.create_relation(RelationCreate(
            source_entity_id=e2.id, target_entity_id=e3.id,
            relation_type="knows", org_id=org_id, space_id=space_id,
        ))
        await engine.create_relation(RelationCreate(
            source_entity_id=e3.id, target_entity_id=e4.id,
            relation_type="knows", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=2,
        ))
        max_hop_found = max(r.hop for r in result.relations) if result.relations else 0
        assert max_hop_found <= 2

    async def test_traverse_only_current_relations(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="Shanghai", entity_type="location", org_id=org_id, space_id=space_id))
        e3 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))
        # Create old relation (will be closed by contradiction)
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        # Create new contradicting relation (closes old)
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e3.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=3,
        ))
        # Should only include current relations
        for tr in result.relations:
            assert tr.relation.is_current is True

    async def test_traverse_filter_by_relation_type(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))
        e3 = await engine.create_entity(EntityCreate(name="Python", entity_type="technology", org_id=org_id, space_id=space_id))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e3.id,
            relation_type="prefers", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=3, relation_types=["lives_in"],
        ))
        assert len(result.relations) == 1
        assert result.relations[0].relation.relation_type == "lives_in"

    async def test_traverse_outgoing_direction(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="B", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="knows", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=3, direction="outgoing",
        ))
        assert len(result.relations) == 1
        assert result.relations[0].direction == "outgoing"

    async def test_traverse_incoming_direction(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        e2 = await engine.create_entity(EntityCreate(name="B", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="knows", org_id=org_id, space_id=space_id,
        ))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e2.id, max_hops=3, direction="incoming",
        ))
        assert len(result.relations) == 1
        assert result.relations[0].direction == "incoming"

    async def test_traverse_no_relations(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        e1 = await engine.create_entity(EntityCreate(name="Lonely", entity_type="person", org_id=org_id, space_id=space_id))

        result = await engine.traverse_graph(GraphTraversalParams(
            entity_id=e1.id, max_hops=3,
        ))
        assert len(result.relations) == 0
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Traversal" -x -v 2>&1 | tail -5`
**Expected:** FAIL -- `traverse_graph` not yet implemented

**Commit:** `git add server/tests/test_graph_engine.py && git commit -m "Add failing graph traversal tests"`

---

### Step 4.2 -- Implement graph traversal via recursive CTE in GraphEngine

- [ ] **Edit** `server/app/core/graph_engine.py` -- add traversal method

Add this method to the `GraphEngine` class:

```python
    # ── Graph Traversal ────────────────────────────────────────

    async def traverse_graph(self, params: GraphTraversalParams) -> GraphTraversalOut:
        """Traverse the knowledge graph starting from an entity using recursive CTE.

        Traverses up to max_hops levels, following current relations only.
        Supports direction filtering (outgoing, incoming, both) and
        relation type filtering.
        """
        # Build the recursive CTE using raw SQL for PostgreSQL
        direction = params.direction
        relation_types = params.relation_types

        # Determine which side of the relation to follow
        if direction == "outgoing":
            src_col = "source_entity_id"
            tgt_col = "target_entity_id"
        elif direction == "incoming":
            src_col = "target_entity_id"
            tgt_col = "source_entity_id"
        else:
            # "both" -- we'll union outgoing and incoming
            src_col = None
            tgt_col = None

        # Build CTE for both directions
        if direction == "both":
            sql = self._build_both_direction_cte(params.max_hops, relation_types)
        else:
            sql = self._build_single_direction_cte(
                src_col, tgt_col, params.max_hops, relation_types
            )

        # Execute the CTE query
        result = await self.session.execute(
            sql, {"start_entity_id": params.entity_id}
        )
        rows = result.fetchall()

        # Collect unique entity IDs and relation data
        entity_ids = set()
        entity_ids.add(params.entity_id)
        traversed_relations = []
        seen_relation_ids = set()

        for row in rows:
            rel_id, src_eid, tgt_eid, rel_type, value, valid_from, valid_to, confidence, is_current, source_count, org_id_r, space_id_r, created_at, updated_at, hop, direction_label = row

            entity_ids.add(src_eid)
            entity_ids.add(tgt_eid)

            if rel_id not in seen_relation_ids:
                seen_relation_ids.add(rel_id)
                traversed_relations.append(
                    TraversedRelationOut(
                        relation=RelationOut(
                            id=rel_id,
                            source_entity_id=src_eid,
                            target_entity_id=tgt_eid,
                            relation_type=rel_type,
                            value=value or "",
                            valid_from=valid_from,
                            valid_to=valid_to,
                            confidence=confidence,
                            is_current=is_current,
                            source_count=source_count,
                            org_id=org_id_r,
                            space_id=space_id_r,
                            created_at=created_at,
                            updated_at=updated_at,
                        ),
                        hop=hop,
                        direction=direction_label,
                    )
                )

        # Fetch all discovered entities
        entities = []
        if entity_ids:
            stmt = select(Entity).where(Entity.id.in_(entity_ids))
            ent_result = await self.session.execute(stmt)
            entities = [EntityOut.model_validate(e) for e in ent_result.scalars().all()]

        return GraphTraversalOut(
            start_entity_id=params.entity_id,
            entities=entities,
            relations=traversed_relations,
            max_hops=params.max_hops,
        )

    def _build_single_direction_cte(
        self, src_col: str, tgt_col: str, max_hops: int, relation_types: list[str] | None
    ) -> str:
        """Build recursive CTE SQL for a single traversal direction."""
        type_filter = ""
        if relation_types:
            type_list = ",".join(f"'{rt}'" for rt in relation_types)
            type_filter = f"AND r.relation_type IN ({type_list})"

        return f"""
        WITH RECURSIVE graph_traverse AS (
            -- Base case: direct relations from start entity
            SELECT
                r.id AS rel_id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                1 AS hop,
                '{src_col}' AS direction
            FROM relations r
            WHERE r.{src_col} = :start_entity_id
              AND r.is_current = true
              {type_filter}

            UNION ALL

            -- Recursive case: follow edges from discovered entities
            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                gt.hop + 1,
                '{src_col}' AS direction
            FROM relations r
            JOIN graph_traverse gt ON r.{src_col} = gt.{tgt_col}
            WHERE r.is_current = true
              AND gt.hop < {max_hops}
              {type_filter}
        )
        SELECT * FROM graph_traverse ORDER BY hop
        """

    def _build_both_direction_cte(
        self, max_hops: int, relation_types: list[str] | None
    ) -> str:
        """Build recursive CTE SQL for both traversal directions."""
        type_filter = ""
        if relation_types:
            type_list = ",".join(f"'{rt}'" for rt in relation_types)
            type_filter = f"AND r.relation_type IN ({type_list})"

        return f"""
        WITH RECURSIVE graph_traverse AS (
            -- Base case: outgoing relations from start entity
            SELECT
                r.id AS rel_id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                1 AS hop,
                'outgoing' AS direction
            FROM relations r
            WHERE r.source_entity_id = :start_entity_id
              AND r.is_current = true
              {type_filter}

            UNION ALL

            -- Base case: incoming relations to start entity
            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                1 AS hop,
                'incoming' AS direction
            FROM relations r
            WHERE r.target_entity_id = :start_entity_id
              AND r.is_current = true
              {type_filter}

            UNION ALL

            -- Recursive: follow outgoing edges
            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                gt.hop + 1,
                'outgoing' AS direction
            FROM relations r
            JOIN graph_traverse gt ON r.source_entity_id = gt.target_entity_id
            WHERE r.is_current = true
              AND gt.hop < {max_hops}
              AND gt.direction = 'outgoing'
              {type_filter}

            UNION ALL

            -- Recursive: follow incoming edges
            SELECT
                r.id,
                r.source_entity_id,
                r.target_entity_id,
                r.relation_type,
                r.value,
                r.valid_from,
                r.valid_to,
                r.confidence,
                r.is_current,
                r.source_count,
                r.org_id,
                r.space_id,
                r.created_at,
                r.updated_at,
                gt.hop + 1,
                'incoming' AS direction
            FROM relations r
            JOIN graph_traverse gt ON r.target_entity_id = gt.source_entity_id
            WHERE r.is_current = true
              AND gt.hop < {max_hops}
              AND gt.direction = 'incoming'
              {type_filter}
        )
        SELECT DISTINCT ON (rel_id) * FROM graph_traverse ORDER BY rel_id, hop
        """
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Traversal" -x -v 2>&1 | tail -15`
**Expected:** All graph traversal tests PASS

**Commit:** `git add server/app/core/graph_engine.py && git commit -m "Implement graph traversal via recursive CTE in GraphEngine"`

---

### Step 4.3 -- Run all graph engine tests so far

- [ ] **Run** all graph engine tests

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -v 2>&1`
**Expected:** All entity, relation, and traversal tests pass

---

## Phase 5: GraphEngine Core -- Memory Operations with Version Chains

### Step 5.1 -- Write failing tests for memory CRUD and version chains

- [ ] **Append** to `server/tests/test_graph_engine.py` -- memory section

```python
# ── Memory Tests ───────────────────────────────────────────────

from app.schemas.graph import MemoryCreate, MemoryUpdate, MemoryForget


class TestMemoryCreate:
    async def test_create_memory_returns_memory_out(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = MemoryCreate(
            content="Zhang Jun moved to Beijing",
            memory_type="fact",
            confidence=0.9,
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_memory(data)
        assert result.content == "Zhang Jun moved to Beijing"
        assert result.memory_type.value == "fact"
        assert result.is_forgotten is False
        assert result.version == 1
        assert result.parent_id is None
        assert result.root_id == result.id  # First version: root is self

    async def test_create_memory_with_custom_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = MemoryCreate(
            content="Test",
            custom_id="ext_123",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_memory(data)
        assert result.custom_id == "ext_123"

    async def test_create_memory_with_entity_link(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        entity = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
        data = MemoryCreate(
            content="Zhang Jun prefers Python",
            memory_type="preference",
            entity_id=entity.id,
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_memory(data)
        assert result.entity_id == entity.id

    async def test_create_memory_sets_decay_rate_by_type(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = MemoryCreate(
            content="Test fact",
            memory_type="fact",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_memory(data)
        assert result.decay_rate is not None
        # fact type should have slow decay
        assert result.decay_rate == 0.005


class TestMemoryGet:
    async def test_get_memory_by_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Hello", org_id=org_id, space_id=space_id,
        ))
        fetched = await engine.get_memory(created.id)
        assert fetched is not None
        assert fetched.content == "Hello"

    async def test_get_memory_not_found(self, engine: GraphEngine):
        result = await engine.get_memory(uuid.uuid4())
        assert result is None

    async def test_get_memory_with_version_chain(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Version 1", org_id=org_id, space_id=space_id,
        ))
        updated = await engine.update_memory(created.id, MemoryUpdate(content="Version 2"))
        chain = await engine.get_memory_version_chain(updated.id)
        assert chain.current.version == 2
        assert len(chain.versions) == 2
        version_numbers = [v.version for v in chain.versions]
        assert 1 in version_numbers
        assert 2 in version_numbers


class TestMemoryList:
    async def test_list_memories_empty(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        result = await engine.list_memories(org_id=org_id, space_id=space_id)
        assert result.total == 0

    async def test_list_memories_excludes_forgotten_by_default(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Forget me", org_id=org_id, space_id=space_id,
        ))
        await engine.forget_memory(created.id, MemoryForget(forget_reason="user request"))
        result = await engine.list_memories(org_id=org_id, space_id=space_id)
        assert result.total == 0

    async def test_list_memories_includes_forgotten_when_requested(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Forget me", org_id=org_id, space_id=space_id,
        ))
        await engine.forget_memory(created.id, MemoryForget(forget_reason="user request"))
        result = await engine.list_memories(org_id=org_id, space_id=space_id, include_forgotten=True)
        assert result.total == 1

    async def test_list_memories_filter_by_type(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        await engine.create_memory(MemoryCreate(
            content="Fact", memory_type="fact", org_id=org_id, space_id=space_id,
        ))
        await engine.create_memory(MemoryCreate(
            content="Preference", memory_type="preference", org_id=org_id, space_id=space_id,
        ))
        result = await engine.list_memories(org_id=org_id, space_id=space_id, memory_type="fact")
        assert result.total == 1
        assert result.items[0].memory_type.value == "fact"

    async def test_list_memories_only_latest_versions(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        """By default, list should return only the latest version of each memory."""
        v1 = await engine.create_memory(MemoryCreate(
            content="V1", org_id=org_id, space_id=space_id,
        ))
        v2 = await engine.update_memory(v1.id, MemoryUpdate(content="V2"))
        result = await engine.list_memories(org_id=org_id, space_id=space_id)
        # Should only return v2 (the latest version)
        assert result.total == 1
        assert result.items[0].version == 2


class TestMemoryUpdate:
    async def test_update_creates_new_version(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        v1 = await engine.create_memory(MemoryCreate(
            content="Original", org_id=org_id, space_id=space_id,
        ))
        v2 = await engine.update_memory(v1.id, MemoryUpdate(content="Updated"))
        assert v2.version == 2
        assert v2.parent_id == v1.id
        assert v2.root_id == v1.id  # root stays the same
        assert v2.content == "Updated"

    async def test_update_chain_three_versions(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        v1 = await engine.create_memory(MemoryCreate(
            content="V1", org_id=org_id, space_id=space_id,
        ))
        v2 = await engine.update_memory(v1.id, MemoryUpdate(content="V2"))
        v3 = await engine.update_memory(v2.id, MemoryUpdate(content="V3"))
        assert v3.version == 3
        assert v3.parent_id == v2.id
        assert v3.root_id == v1.id

    async def test_update_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Memory not found"):
            await engine.update_memory(uuid.uuid4(), MemoryUpdate(content="X"))


class TestMemoryForget:
    async def test_forget_sets_is_forgotten(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Forget me", org_id=org_id, space_id=space_id,
        ))
        result = await engine.forget_memory(created.id, MemoryForget(forget_reason="user request"))
        assert result.is_forgotten is True
        assert result.forget_reason == "user request"
        assert result.forget_at is not None

    async def test_forget_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Memory not found"):
            await engine.forget_memory(uuid.uuid4(), MemoryForget(forget_reason="gone"))
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Memory" -x -v 2>&1 | tail -5`
**Expected:** FAIL -- `create_memory` not yet implemented

**Commit:** `git add server/tests/test_graph_engine.py && git commit -m "Add failing memory CRUD, version chain, and forget tests"`

---

### Step 5.2 -- Implement memory CRUD with version chains in GraphEngine

- [ ] **Edit** `server/app/core/graph_engine.py` -- add memory methods

Add these methods and the decay rate lookup to the `GraphEngine` class:

```python
    # ── Decay rate defaults by memory type ─────────────────────

    DECAY_RATES: dict[str, float] = {
        "fact": 0.005,
        "preference": 0.005,
        "status": 0.02,
        "inference": 0.05,
    }

    # ── Memory CRUD ────────────────────────────────────────────

    async def create_memory(self, data: MemoryCreate) -> MemoryOut:
        """Create a new memory. Sets version=1, root_id=self, parent_id=None."""
        now = datetime.now(timezone.utc)
        memory_id = uuid.uuid4()
        decay_rate = data.decay_rate
        if decay_rate is None:
            decay_rate = self.DECAY_RATES.get(data.memory_type.value, 0.01)

        memory = Memory(
            id=memory_id,
            content=data.content,
            custom_id=data.custom_id,
            memory_type=data.memory_type.value,
            entity_id=data.entity_id,
            relation_id=data.relation_id,
            confidence=data.confidence,
            decay_rate=decay_rate,
            is_forgotten=False,
            forget_at=None,
            forget_reason=None,
            version=1,
            parent_id=None,
            root_id=memory_id,  # First version: root is self
            metadata=data.metadata,
            org_id=data.org_id,
            space_id=data.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(memory)
        await self.session.flush()

        # Create memory-source links if document_ids provided
        if data.document_ids:
            for doc_id in data.document_ids:
                source = MemorySource(
                    memory_id=memory_id,
                    document_id=doc_id,
                    relevance_score=0.0,
                )
                self.session.add(source)
            await self.session.flush()

        return MemoryOut.model_validate(memory)

    async def get_memory(self, memory_id: uuid.UUID) -> Optional[MemoryOut]:
        """Get a memory by ID. Returns None if not found."""
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            return None
        return MemoryOut.model_validate(memory)

    async def get_memory_version_chain(
        self, memory_id: uuid.UUID
    ) -> MemoryVersionChainOut:
        """Get a memory with its full version chain.

        Given any memory ID in the chain, finds the root and returns
        all versions ordered by version number.
        """
        # First, find the memory to get its root_id
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            raise ValueError("Memory not found")

        root_id = memory.root_id

        # Fetch all versions in the chain
        chain_stmt = (
            select(Memory)
            .where(Memory.root_id == root_id)
            .order_by(Memory.version.asc())
        )
        chain_result = await self.session.execute(chain_stmt)
        versions = [MemoryOut.model_validate(m) for m in chain_result.scalars().all()]

        # Current = highest version
        current = versions[-1] if versions else MemoryOut.model_validate(memory)

        return MemoryVersionChainOut(current=current, versions=versions)

    async def list_memories(
        self,
        *,
        org_id: uuid.UUID,
        space_id: Optional[uuid.UUID] = None,
        memory_type: Optional[str] = None,
        entity_id: Optional[uuid.UUID] = None,
        include_forgotten: bool = False,
        latest_versions_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> MemoryListOut:
        """List memories with optional filters and pagination.

        By default excludes forgotten memories and returns only the
        latest version of each memory chain.
        """
        stmt = select(Memory).where(Memory.org_id == org_id)
        count_stmt = select(func.count()).select_from(Memory).where(Memory.org_id == org_id)

        if space_id is not None:
            stmt = stmt.where(Memory.space_id == space_id)
            count_stmt = count_stmt.where(Memory.space_id == space_id)
        if not include_forgotten:
            stmt = stmt.where(Memory.is_forgotten == False)
            count_stmt = count_stmt.where(Memory.is_forgotten == False)
        if memory_type is not None:
            stmt = stmt.where(Memory.memory_type == memory_type)
            count_stmt = count_stmt.where(Memory.memory_type == memory_type)
        if entity_id is not None:
            stmt = stmt.where(Memory.entity_id == entity_id)
            count_stmt = count_stmt.where(Memory.entity_id == entity_id)
        if latest_versions_only:
            # Only include the latest version: where id is NOT a parent_id of another memory
            subq = select(Memory.parent_id).where(Memory.parent_id.isnot(None))
            stmt = stmt.where(Memory.id.notin_(subq))
            count_stmt = count_stmt.where(Memory.id.notin_(subq))

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Memory.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        memories = result.scalars().all()

        return MemoryListOut(
            items=[MemoryOut.model_validate(m) for m in memories],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def update_memory(
        self, memory_id: uuid.UUID, data: MemoryUpdate
    ) -> MemoryOut:
        """Update a memory by creating a new version.

        Instead of mutating the existing row, creates a new Memory row
        with version+1, parent_id pointing to the old memory, and the
        same root_id.
        """
        now = datetime.now(timezone.utc)

        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        old_memory = result.scalar_one_or_none()
        if old_memory is None:
            raise ValueError("Memory not found")

        new_version = old_memory.version + 1
        new_id = uuid.uuid4()

        # Build new content/metadata from update data
        new_content = data.content if data.content is not None else old_memory.content
        new_confidence = data.confidence if data.confidence is not None else old_memory.confidence
        new_decay_rate = data.decay_rate if data.decay_rate is not None else old_memory.decay_rate
        new_metadata = data.metadata if data.metadata is not None else old_memory.metadata

        new_memory = Memory(
            id=new_id,
            content=new_content,
            custom_id=old_memory.custom_id,
            memory_type=old_memory.memory_type,
            entity_id=old_memory.entity_id,
            relation_id=old_memory.relation_id,
            confidence=new_confidence,
            decay_rate=new_decay_rate,
            is_forgotten=False,
            forget_at=None,
            forget_reason=None,
            version=new_version,
            parent_id=memory_id,
            root_id=old_memory.root_id,
            metadata=new_metadata,
            org_id=old_memory.org_id,
            space_id=old_memory.space_id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(new_memory)
        await self.session.flush()

        # Copy memory sources from old to new
        source_stmt = select(MemorySource).where(MemorySource.memory_id == memory_id)
        source_result = await self.session.execute(source_stmt)
        for source in source_result.scalars().all():
            new_source = MemorySource(
                memory_id=new_id,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                relevance_score=source.relevance_score,
            )
            self.session.add(new_source)
        await self.session.flush()

        return MemoryOut.model_validate(new_memory)

    async def forget_memory(
        self, memory_id: uuid.UUID, data: MemoryForget
    ) -> MemoryOut:
        """Soft-delete a memory by setting is_forgotten=true.

        Sets forget_reason and forget_at timestamp.
        """
        now = datetime.now(timezone.utc)

        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            raise ValueError("Memory not found")

        memory.is_forgotten = True
        memory.forget_reason = data.forget_reason
        memory.forget_at = now
        memory.updated_at = now
        await self.session.flush()

        return MemoryOut.model_validate(memory)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Memory" -x -v 2>&1 | tail -15`
**Expected:** All memory CRUD, version chain, and forget tests PASS

**Commit:** `git add server/app/core/graph_engine.py && git commit -m "Implement memory CRUD with version chains and soft-delete forget"`

---

### Step 5.3 -- Run all graph engine tests together

- [ ] **Run** complete graph engine test suite

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -v 2>&1`
**Expected:** All entity, relation, traversal, and memory tests pass

---

## Phase 6: GraphEngine Core -- Memory-Document Source Tracking

### Step 6.1 -- Write failing tests for memory sources

- [ ] **Append** to `server/tests/test_graph_engine.py` -- source section

```python
# ── Memory Source Tests ────────────────────────────────────────

from app.schemas.graph import MemorySourceCreate


class TestMemorySource:
    async def test_add_memory_source(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        memory = await engine.create_memory(MemoryCreate(
            content="From doc", org_id=org_id, space_id=space_id,
        ))
        doc_id = uuid.uuid4()
        source = await engine.add_memory_source(MemorySourceCreate(
            memory_id=memory.id,
            document_id=doc_id,
            relevance_score=0.85,
        ))
        assert source.memory_id == memory.id
        assert source.document_id == doc_id
        assert source.relevance_score == 0.85

    async def test_get_memory_sources(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        memory = await engine.create_memory(MemoryCreate(
            content="From doc", org_id=org_id, space_id=space_id,
        ))
        doc_id = uuid.uuid4()
        await engine.add_memory_source(MemorySourceCreate(
            memory_id=memory.id,
            document_id=doc_id,
            relevance_score=0.7,
        ))
        sources = await engine.get_memory_sources(memory.id)
        assert len(sources) == 1
        assert sources[0].document_id == doc_id

    async def test_create_memory_with_document_ids(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        doc_id = uuid.uuid4()
        memory = await engine.create_memory(MemoryCreate(
            content="From doc",
            org_id=org_id,
            space_id=space_id,
            document_ids=[doc_id],
        ))
        sources = await engine.get_memory_sources(memory.id)
        assert len(sources) == 1
        assert sources[0].document_id == doc_id
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Source" -x -v 2>&1 | tail -5`
**Expected:** FAIL -- `add_memory_source` not yet implemented

**Commit:** `git add server/tests/test_graph_engine.py && git commit -m "Add failing memory source tracking tests"`

---

### Step 6.2 -- Implement memory source methods in GraphEngine

- [ ] **Edit** `server/app/core/graph_engine.py` -- add source methods

Add these methods to the `GraphEngine` class:

```python
    # ── Memory-Document Source Tracking ────────────────────────

    async def add_memory_source(self, data: MemorySourceCreate) -> MemorySourceOut:
        """Link a memory to its source document (and optionally chunk)."""
        source = MemorySource(
            memory_id=data.memory_id,
            document_id=data.document_id,
            chunk_id=data.chunk_id,
            relevance_score=data.relevance_score,
        )
        self.session.add(source)
        await self.session.flush()
        return MemorySourceOut.model_validate(source)

    async def get_memory_sources(self, memory_id: uuid.UUID) -> list[MemorySourceOut]:
        """Get all source links for a memory."""
        stmt = select(MemorySource).where(MemorySource.memory_id == memory_id)
        result = await self.session.execute(stmt)
        return [MemorySourceOut.model_validate(s) for s in result.scalars().all()]
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -k "Source" -x -v 2>&1 | tail -5`
**Expected:** All memory source tests PASS

**Commit:** `git add server/app/core/graph_engine.py && git commit -m "Implement memory-document source tracking in GraphEngine"`

---

### Step 6.3 -- Run all graph engine tests

- [ ] **Run** complete test suite

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_graph_engine.py -v 2>&1`
**Expected:** All tests pass (entity, relation, traversal, memory, source)

---

## Phase 7: API Routes -- Entities

### Step 7.1 -- Write failing tests for entity API routes

- [ ] **Write** `server/tests/test_api_entities.py`

```python
"""Integration tests for /v1/entities/ API routes."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _entity_create_payload(org_id: uuid.UUID, space_id: uuid.UUID) -> dict:
    return {
        "name": "Zhang Jun",
        "entity_type": "person",
        "description": "Software engineer",
        "confidence": 0.9,
        "org_id": str(org_id),
        "space_id": str(space_id),
    }


class TestEntityAPI:
    @pytest.mark.asyncio
    async def test_create_entity(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        resp = await client.post("/v1/entities/", json=_entity_create_payload(org_id, space_id))
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Zhang Jun"
        assert data["entity_type"] == "person"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_get_entity(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/entities/", json=_entity_create_payload(org_id, space_id))
        entity_id = create_resp.json()["id"]
        resp = await client.get(f"/v1/entities/{entity_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == entity_id

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, client: AsyncClient):
        resp = await client.get(f"/v1/entities/{uuid.uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_entities(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        await client.post("/v1/entities/", json=_entity_create_payload(org_id, space_id))
        resp = await client.get("/v1/entities/", params={"org_id": str(org_id), "space_id": str(space_id)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_update_entity(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/entities/", json=_entity_create_payload(org_id, space_id))
        entity_id = create_resp.json()["id"]
        resp = await client.patch(f"/v1/entities/{entity_id}", json={"name": "Updated Name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_entity(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/entities/", json=_entity_create_payload(org_id, space_id))
        entity_id = create_resp.json()["id"]
        resp = await client.delete(f"/v1/entities/{entity_id}")
        assert resp.status_code == 204
        # Verify deleted
        get_resp = await client.get(f"/v1/entities/{entity_id}")
        assert get_resp.status_code == 404
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_api_entities.py -x -v 2>&1 | tail -5`
**Expected:** FAIL -- route not yet registered

**Commit:** `git add server/tests/test_api_entities.py && git commit -m "Add failing entity API integration tests"`

---

### Step 7.2 -- Implement entity API routes

- [ ] **Write** `server/app/api/entities.py`

```python
"""API routes for /v1/entities/."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    EntityCreate,
    EntityListOut,
    EntityOut,
    EntityUpdate,
)

router = APIRouter(prefix="/v1/entities", tags=["entities"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


@router.post("/", response_model=EntityOut, status_code=201)
async def create_entity(
    data: EntityCreate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Create a new entity."""
    result = await engine.create_entity(data)
    await engine.session.commit()
    return result


@router.get("/", response_model=EntityListOut)
async def list_entities(
    org_id: uuid.UUID = Query(...),
    space_id: Optional[uuid.UUID] = Query(default=None),
    entity_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: GraphEngine = Depends(_get_engine),
):
    """List entities with optional filters and pagination."""
    result = await engine.list_entities(
        org_id=org_id,
        space_id=space_id,
        entity_type=entity_type,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/{entity_id}", response_model=EntityOut)
async def get_entity(
    entity_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get an entity by ID."""
    result = await engine.get_entity(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return result


@router.patch("/{entity_id}", response_model=EntityOut)
async def update_entity(
    entity_id: uuid.UUID,
    data: EntityUpdate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Update an entity."""
    try:
        result = await engine.update_entity(entity_id, data)
        await engine.session.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{entity_id}", status_code=204)
async def delete_entity(
    entity_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Delete an entity."""
    try:
        await engine.delete_entity(entity_id)
        await engine.session.commit()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from app.api.entities import router; print('entities router OK')"`
**Expected:** `entities router OK`

**Commit:** `git add server/app/api/entities.py && git commit -m "Implement /v1/entities/ API routes"`

---

### Step 7.3 -- Register entity routes in FastAPI app and run API tests

- [ ] **Edit** `server/app/main.py` to include the entity router

In `server/app/main.py`, add:

```python
from app.api.entities import router as entities_router
app.include_router(entities_router)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_api_entities.py -x -v 2>&1 | tail -15`
**Expected:** All entity API tests PASS

**Commit:** `git add server/app/main.py && git commit -m "Register entity API routes in FastAPI app"`

---

## Phase 8: API Routes -- Relations

### Step 8.1 -- Write failing tests for relation API routes

- [ ] **Write** `server/tests/test_api_relations.py`

```python
"""Integration tests for /v1/relations/ API routes."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _create_two_entities(client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID) -> tuple[str, str]:
    e1 = (await client.post("/v1/entities/", json={
        "name": "Zhang Jun", "entity_type": "person", "org_id": str(org_id), "space_id": str(space_id),
    })).json()
    e2 = (await client.post("/v1/entities/", json={
        "name": "Beijing", "entity_type": "location", "org_id": str(org_id), "space_id": str(space_id),
    })).json()
    return e1["id"], e2["id"]


class TestRelationAPI:
    @pytest.mark.asyncio
    async def test_create_relation(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        e1_id, e2_id = await _create_two_entities(client, org_id, space_id)
        resp = await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e2_id,
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["relation_type"] == "lives_in"
        assert data["is_current"] is True

    @pytest.mark.asyncio
    async def test_get_relation(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        e1_id, e2_id = await _create_two_entities(client, org_id, space_id)
        create_resp = await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e2_id,
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })
        rel_id = create_resp.json()["id"]
        resp = await client.get(f"/v1/relations/{rel_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == rel_id

    @pytest.mark.asyncio
    async def test_list_relations(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        e1_id, e2_id = await _create_two_entities(client, org_id, space_id)
        await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e2_id,
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })
        resp = await client.get("/v1/relations/", params={"org_id": str(org_id), "space_id": str(space_id)})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_update_relation(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        e1_id, e2_id = await _create_two_entities(client, org_id, space_id)
        create_resp = await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e2_id,
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })
        rel_id = create_resp.json()["id"]
        resp = await client.patch(f"/v1/relations/{rel_id}", json={"value": "Chaoyang district"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "Chaoyang district"

    @pytest.mark.asyncio
    async def test_delete_relation(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        e1_id, e2_id = await _create_two_entities(client, org_id, space_id)
        create_resp = await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e2_id,
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })
        rel_id = create_resp.json()["id"]
        resp = await client.delete(f"/v1/relations/{rel_id}")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_contradiction_via_api(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        e1_id, e2_id = await _create_two_entities(client, org_id, space_id)
        e3 = (await client.post("/v1/entities/", json={
            "name": "Shanghai", "entity_type": "location", "org_id": str(org_id), "space_id": str(space_id),
        })).json()

        old_resp = await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e2_id,
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })
        new_resp = await client.post("/v1/relations/", json={
            "source_entity_id": e1_id,
            "target_entity_id": e3["id"],
            "relation_type": "lives_in",
            "org_id": str(org_id),
            "space_id": str(space_id),
        })

        # Old relation should be closed
        old_id = old_resp.json()["id"]
        old_refreshed = (await client.get(f"/v1/relations/{old_id}")).json()
        assert old_refreshed["is_current"] is False
        assert old_refreshed["valid_to"] is not None

        # New relation should be current
        new_data = new_resp.json()
        assert new_data["is_current"] is True
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_api_relations.py -x -v 2>&1 | tail -5`
**Expected:** FAIL -- route not yet registered

**Commit:** `git add server/tests/test_api_relations.py && git commit -m "Add failing relation API integration tests"`

---

### Step 8.2 -- Implement relation API routes

- [ ] **Write** `server/app/api/relations.py`

```python
"""API routes for /v1/relations/."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    RelationCreate,
    RelationListOut,
    RelationOut,
    RelationUpdate,
)

router = APIRouter(prefix="/v1/relations", tags=["relations"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


@router.post("/", response_model=RelationOut, status_code=201)
async def create_relation(
    data: RelationCreate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Create a new relation. Handles contradiction automatically."""
    result = await engine.create_relation(data)
    await engine.session.commit()
    return result


@router.get("/", response_model=RelationListOut)
async def list_relations(
    org_id: uuid.UUID = Query(...),
    space_id: Optional[uuid.UUID] = Query(default=None),
    source_entity_id: Optional[uuid.UUID] = Query(default=None),
    target_entity_id: Optional[uuid.UUID] = Query(default=None),
    relation_type: Optional[str] = Query(default=None),
    is_current: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: GraphEngine = Depends(_get_engine),
):
    """List relations with optional filters and pagination."""
    result = await engine.list_relations(
        org_id=org_id,
        space_id=space_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        is_current=is_current,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/{relation_id}", response_model=RelationOut)
async def get_relation(
    relation_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get a relation by ID."""
    result = await engine.get_relation(relation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Relation not found")
    return result


@router.patch("/{relation_id}", response_model=RelationOut)
async def update_relation(
    relation_id: uuid.UUID,
    data: RelationUpdate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Update a relation."""
    try:
        result = await engine.update_relation(relation_id, data)
        await engine.session.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{relation_id}", status_code=204)
async def delete_relation(
    relation_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Delete a relation."""
    try:
        await engine.delete_relation(relation_id)
        await engine.session.commit()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Edit** `server/app/main.py` to register the relation router

```python
from app.api.relations import router as relations_router
app.include_router(relations_router)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_api_relations.py -x -v 2>&1 | tail -15`
**Expected:** All relation API tests PASS

**Commit:** `git add server/app/api/relations.py server/app/main.py && git commit -m "Implement /v1/relations/ API routes with contradiction handling"`

---

## Phase 9: API Routes -- Memories

### Step 9.1 -- Write failing tests for memory API routes

- [ ] **Write** `server/tests/test_api_memories.py`

```python
"""Integration tests for /v1/memories/ API routes."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _memory_create_payload(org_id: uuid.UUID, space_id: uuid.UUID) -> dict:
    return {
        "content": "Zhang Jun moved to Beijing",
        "memory_type": "fact",
        "confidence": 0.9,
        "org_id": str(org_id),
        "space_id": str(space_id),
    }


class TestMemoryAPI:
    @pytest.mark.asyncio
    async def test_create_memory(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        resp = await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        assert resp.status_code == 201
        data = resp.json()
        assert data["content"] == "Zhang Jun moved to Beijing"
        assert data["version"] == 1
        assert data["is_forgotten"] is False

    @pytest.mark.asyncio
    async def test_get_memory(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        mem_id = create_resp.json()["id"]
        resp = await client.get(f"/v1/memories/{mem_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == mem_id

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, client: AsyncClient):
        resp = await client.get(f"/v1/memories/{uuid.uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_memories(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        resp = await client.get("/v1/memories/", params={"org_id": str(org_id), "space_id": str(space_id)})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_update_memory_creates_new_version(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        mem_id = create_resp.json()["id"]
        resp = await client.patch(f"/v1/memories/{mem_id}", json={"content": "Updated content"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Updated content"
        assert data["version"] == 2
        assert data["parent_id"] == mem_id

    @pytest.mark.asyncio
    async def test_forget_memory(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        mem_id = create_resp.json()["id"]
        resp = await client.delete(f"/v1/memories/{mem_id}", json={"forget_reason": "user request"})
        assert resp.status_code == 200
        assert resp.json()["is_forgotten"] is True

    @pytest.mark.asyncio
    async def test_get_memory_version_chain(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        mem_id = create_resp.json()["id"]
        await client.patch(f"/v1/memories/{mem_id}", json={"content": "V2"})
        resp = await client.get(f"/v1/memories/{mem_id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current"]["version"] == 2
        assert len(data["versions"]) == 2

    @pytest.mark.asyncio
    async def test_list_memories_excludes_forgotten(self, client: AsyncClient, org_id: uuid.UUID, space_id: uuid.UUID):
        create_resp = await client.post("/v1/memories/", json=_memory_create_payload(org_id, space_id))
        mem_id = create_resp.json()["id"]
        await client.delete(f"/v1/memories/{mem_id}", json={"forget_reason": "gone"})
        resp = await client.get("/v1/memories/", params={"org_id": str(org_id), "space_id": str(space_id)})
        assert resp.json()["total"] == 0
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_api_memories.py -x -v 2>&1 | tail -5`
**Expected:** FAIL -- route not yet registered

**Commit:** `git add server/tests/test_api_memories.py && git commit -m "Add failing memory API integration tests"`

---

### Step 9.2 -- Implement memory API routes

- [ ] **Write** `server/app/api/memories.py`

```python
"""API routes for /v1/memories/."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    MemoryCreate,
    MemoryForget,
    MemoryListOut,
    MemoryOut,
    MemoryUpdate,
    MemoryVersionChainOut,
)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


@router.post("/", response_model=MemoryOut, status_code=201)
async def create_memory(
    data: MemoryCreate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Create a new memory."""
    result = await engine.create_memory(data)
    await engine.session.commit()
    return result


@router.get("/", response_model=MemoryListOut)
async def list_memories(
    org_id: uuid.UUID = Query(...),
    space_id: Optional[uuid.UUID] = Query(default=None),
    memory_type: Optional[str] = Query(default=None),
    entity_id: Optional[uuid.UUID] = Query(default=None),
    include_forgotten: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: GraphEngine = Depends(_get_engine),
):
    """List memories with optional filters and pagination."""
    result = await engine.list_memories(
        org_id=org_id,
        space_id=space_id,
        memory_type=memory_type,
        entity_id=entity_id,
        include_forgotten=include_forgotten,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/{memory_id}", response_model=MemoryOut)
async def get_memory(
    memory_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get a memory by ID."""
    result = await engine.get_memory(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@router.get("/{memory_id}/versions", response_model=MemoryVersionChainOut)
async def get_memory_versions(
    memory_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get the full version chain for a memory."""
    try:
        result = await engine.get_memory_version_chain(memory_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: uuid.UUID,
    data: MemoryUpdate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Update a memory (creates a new version in the chain)."""
    try:
        result = await engine.update_memory(memory_id, data)
        await engine.session.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{memory_id}", response_model=MemoryOut)
async def forget_memory(
    memory_id: uuid.UUID,
    data: MemoryForget = Body(default=MemoryForget()),
    engine: GraphEngine = Depends(_get_engine),
):
    """Soft-delete (forget) a memory."""
    try:
        result = await engine.forget_memory(memory_id, data)
        await engine.session.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Edit** `server/app/main.py` to register the memory router

```python
from app.api.memories import router as memories_router
app.include_router(memories_router)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_api_memories.py -x -v 2>&1 | tail -15`
**Expected:** All memory API tests PASS

**Commit:** `git add server/app/api/memories.py server/app/main.py && git commit -m "Implement /v1/memories/ API routes with version chain and forget"`

---

## Phase 10: End-to-End Validation

### Step 10.1 -- Write end-to-end scenario test

- [ ] **Write** `server/tests/test_e2e_graph.py`

```python
"""End-to-end scenario test: full contradiction + version chain + traversal flow."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.schemas.graph import (
    EntityCreate,
    GraphTraversalParams,
    MemoryCreate,
    MemoryForget,
    MemoryUpdate,
    RelationCreate,
)


@pytest_asyncio.fixture
async def engine(db_session_committed: AsyncSession) -> GraphEngine:
    return GraphEngine(db_session_committed)


class TestE2EContractionAndTraversal:
    @pytest.mark.asyncio
    async def test_full_scenario(self, engine: GraphEngine):
        org_id = uuid.uuid4()
        space_id = uuid.uuid4()

        # 1. Create entities
        zhang = await engine.create_entity(EntityCreate(
            name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id,
        ))
        shanghai = await engine.create_entity(EntityCreate(
            name="Shanghai", entity_type="location", org_id=org_id, space_id=space_id,
        ))
        beijing = await engine.create_entity(EntityCreate(
            name="Beijing", entity_type="location", org_id=org_id, space_id=space_id,
        ))
        python = await engine.create_entity(EntityCreate(
            name="Python", entity_type="technology", org_id=org_id, space_id=space_id,
        ))

        # 2. Zhang Jun lives in Shanghai
        rel_sh = await engine.create_relation(RelationCreate(
            source_entity_id=zhang.id, target_entity_id=shanghai.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        assert rel_sh.is_current is True

        # 3. Zhang Jun prefers Python
        rel_pref = await engine.create_relation(RelationCreate(
            source_entity_id=zhang.id, target_entity_id=python.id,
            relation_type="prefers", org_id=org_id, space_id=space_id,
        ))
        assert rel_pref.is_current is True

        # 4. Contradiction: Zhang Jun moves to Beijing
        rel_bj = await engine.create_relation(RelationCreate(
            source_entity_id=zhang.id, target_entity_id=beijing.id,
            relation_type="lives_in", org_id=org_id, space_id=space_id,
        ))
        assert rel_bj.is_current is True

        # 5. Verify Shanghai relation is closed
        sh_refreshed = await engine.get_relation(rel_sh.id)
        assert sh_refreshed.is_current is False
        assert sh_refreshed.valid_to is not None

        # 6. Reaffirm Python preference
        rel_pref2 = await engine.create_relation(RelationCreate(
            source_entity_id=zhang.id, target_entity_id=python.id,
            relation_type="prefers", org_id=org_id, space_id=space_id,
        ))
        assert rel_pref2.id == rel_pref.id  # Same relation, not a new one
        pref_refreshed = await engine.get_relation(rel_pref.id)
        assert pref_refreshed.source_count == 2

        # 7. Create a memory about moving
        mem1 = await engine.create_memory(MemoryCreate(
            content="Zhang Jun moved from Shanghai to Beijing",
            memory_type="fact",
            entity_id=zhang.id,
            org_id=org_id,
            space_id=space_id,
        ))
        assert mem1.version == 1
        assert mem1.root_id == mem1.id

        # 8. Update the memory (create version 2)
        mem2 = await engine.update_memory(mem1.id, MemoryUpdate(
            content="Zhang Jun moved from Shanghai to Beijing in 2024",
        ))
        assert mem2.version == 2
        assert mem2.parent_id == mem1.id
        assert mem2.root_id == mem1.id

        # 9. Get version chain
        chain = await engine.get_memory_version_chain(mem2.id)
        assert chain.current.version == 2
        assert len(chain.versions) == 2

        # 10. Traverse graph from Zhang Jun
        traversal = await engine.traverse_graph(GraphTraversalParams(
            entity_id=zhang.id, max_hops=2,
        ))
        assert len(traversal.relations) >= 2  # At least lives_in + prefers
        # All traversed relations should be current
        for tr in traversal.relations:
            assert tr.relation.is_current is True

        # 11. Forget the memory
        forgotten = await engine.forget_memory(mem2.id, MemoryForget(forget_reason="outdated"))
        assert forgotten.is_forgotten is True

        # 12. List memories should exclude forgotten
        mem_list = await engine.list_memories(org_id=org_id, space_id=space_id)
        assert mem_list.total == 0
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_e2e_graph.py -x -v 2>&1 | tail -10`
**Expected:** PASS (if DB is set up), or FAIL with connection error (acceptable -- DB needs to exist)

**Commit:** `git add server/tests/test_e2e_graph.py && git commit -m "Add end-to-end scenario test for contradiction, version chain, and traversal"`

---

### Step 10.2 -- Run complete test suite

- [ ] **Run** all tests

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/ -v 2>&1`
**Expected:** All tests pass

**Commit:** (none needed if all pass)

---

### Step 10.3 -- Final commit with full graph engine

- [ ] **Verify** all files present

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory/server && find . -name "*.py" -not -path "./.venv/*" | sort`
**Expected output:**
```
./app/__init__.py
./app/api/entities.py
./app/api/memories.py
./app/api/relations.py
./app/core/__init__.py
./app/core/graph_engine.py
./app/db/models.py
./app/db/session.py
./app/main.py
./app/schemas/__init__.py
./app/schemas/graph.py
./tests/__init__.py
./tests/conftest.py
./test_api_entities.py
./test_api_memories.py
./test_api_relations.py
./test_e2e_graph.py
./test_graph_engine.py
```

**Commit:** `git add -A && git commit -m "Complete Graph Engine: entity/relation/memory CRUD, contradiction handling, version chains, traversal, and API routes"`

---

## Summary

| Phase | Deliverable | Key Behavior |
|-------|-------------|-------------|
| 1 | Schemas + test infra | Pydantic models, async test fixtures |
| 2 | Entity CRUD | create/get/list/update/delete entities |
| 3 | Relation CRUD + contradiction | Temporal relations; same target = reaffirm, different target = close old + insert new |
| 4 | Graph traversal | Recursive CTE up to N hops, direction + type filtering |
| 5 | Memory CRUD + version chains | Create with auto decay_rate, update creates new version, forget = soft delete |
| 6 | Memory-document sources | Link memories to documents via memory_sources |
| 7 | Entity API routes | POST/GET/PATCH/DELETE /v1/entities/ |
| 8 | Relation API routes | POST/GET/PATCH/DELETE /v1/relations/ with contradiction auto-handled |
| 9 | Memory API routes | POST/GET/PATCH/DELETE /v1/memories/ with version chain endpoint |
| 10 | E2E validation | Full scenario: contradiction + reaffirm + version chain + traversal |

**Contradiction algorithm (core):**
```
create_relation(source, type, target):
  existing = find relation where source_entity_id=source AND relation_type=type AND is_current=true
  if existing:
    if existing.target == target:
      # Same relation reaffirmed
      existing.source_count += 1
      return existing
    else:
      # Contradiction: close old, insert new
      existing.is_current = false
      existing.valid_to = now
      insert new relation(source, type, target, valid_from=now, is_current=true, source_count=1)
  else:
    insert new relation(source, type, target, valid_from=now, is_current=true, source_count=1)
```

**Memory version chain:**
```
create_memory()  -> version=1, root_id=self.id, parent_id=null
update_memory()  -> version=old.version+1, root_id=old.root_id, parent_id=old.id
forget_memory()  -> is_forgotten=true, forget_at=now, forget_reason=provided
```