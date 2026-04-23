"""Unit tests for GraphEngine entity, relation, memory, and source operations.

Uses an in-memory SQLite database with custom type adapters for pgvector/JSONB columns.
Graph traversal (recursive CTE) tests require PostgreSQL and are marked accordingly.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.graph_engine import GraphEngine
from app.schemas.graph import (
    EntityCreate,
    EntityUpdate,
    RelationCreate,
    RelationUpdate,
    MemoryCreate,
    MemoryUpdate,
    MemoryForget,
    MemorySourceCreate,
)


# ── Fixtures ───────────────────────────────────────────────────

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


def _create_sqlite_tables(connection):
    """Create SQLite-compatible tables matching the real schema.
    pgvector columns are TEXT; JSONB columns are TEXT; UUID columns are CHAR(36).
    """
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS spaces (
            id CHAR(36) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            org_id CHAR(36) NOT NULL,
            owner_id CHAR(36),
            container_tag VARCHAR(255),
            is_default BOOLEAN NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS entities (
            id CHAR(36) PRIMARY KEY,
            name VARCHAR(512) NOT NULL,
            entity_type VARCHAR(50) NOT NULL DEFAULT 'person',
            description TEXT,
            embedding TEXT,
            confidence FLOAT NOT NULL DEFAULT 1.0,
            org_id CHAR(36) NOT NULL,
            space_id CHAR(36) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS relations (
            id CHAR(36) PRIMARY KEY,
            source_entity_id CHAR(36) NOT NULL,
            target_entity_id CHAR(36) NOT NULL,
            relation_type VARCHAR(255) NOT NULL,
            value TEXT,
            valid_from DATETIME NOT NULL,
            valid_to DATETIME,
            confidence FLOAT NOT NULL DEFAULT 1.0,
            is_current BOOLEAN NOT NULL DEFAULT 1,
            source_count INTEGER NOT NULL DEFAULT 1,
            embedding TEXT,
            org_id CHAR(36) NOT NULL,
            space_id CHAR(36) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS memories (
            id CHAR(36) PRIMARY KEY,
            content TEXT NOT NULL,
            custom_id VARCHAR(512),
            memory_type VARCHAR(50) NOT NULL DEFAULT 'fact',
            entity_id CHAR(36),
            relation_id CHAR(36),
            confidence FLOAT NOT NULL DEFAULT 1.0,
            decay_rate FLOAT NOT NULL DEFAULT 0.02,
            is_forgotten BOOLEAN NOT NULL DEFAULT 0,
            forget_at DATETIME,
            forget_reason TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            parent_id CHAR(36),
            root_id CHAR(36),
            metadata TEXT,
            embedding TEXT,
            org_id CHAR(36) NOT NULL,
            space_id CHAR(36) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
            id CHAR(36) PRIMARY KEY,
            title VARCHAR(512) NOT NULL,
            content TEXT,
            doc_type VARCHAR(50) NOT NULL DEFAULT 'text',
            url TEXT,
            status VARCHAR(50) NOT NULL DEFAULT 'queued',
            summary TEXT,
            summary_embedding TEXT,
            metadata TEXT,
            org_id CHAR(36) NOT NULL,
            space_id CHAR(36) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS chunks (
            id CHAR(36) PRIMARY KEY,
            document_id CHAR(36) NOT NULL,
            content TEXT NOT NULL,
            position INTEGER NOT NULL,
            embedding TEXT,
            created_at DATETIME NOT NULL
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS memory_sources (
            memory_id CHAR(36) NOT NULL,
            document_id CHAR(36) NOT NULL,
            chunk_id CHAR(36),
            relevance_score FLOAT NOT NULL DEFAULT 0.0,
            PRIMARY KEY (memory_id, document_id)
        )
    """))


def _drop_sqlite_tables(connection):
    """Drop all test tables."""
    for table in ["memory_sources", "chunks", "documents", "memories", "relations", "entities", "spaces"]:
        connection.execute(text(f"DROP TABLE IF EXISTS {table}"))


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh in-memory SQLite session for each test."""
    engine = create_async_engine(SQLITE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(_create_sqlite_tables)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(_drop_sqlite_tables)
    await engine.dispose()


@pytest_asyncio.fixture
async def engine(db_session: AsyncSession) -> GraphEngine:
    return GraphEngine(db_session)


@pytest_asyncio.fixture
def org_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest_asyncio.fixture
def space_id() -> uuid.UUID:
    return uuid.uuid4()


# ── Entity Tests ───────────────────────────────────────────────

class TestEntityCreate:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_create_entity_generates_uuid_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = EntityCreate(
            name="Beijing",
            entity_type="location",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_entity(data)
        assert isinstance(result.id, uuid.UUID)

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_get_entity_not_found_returns_none(self, engine: GraphEngine):
        result = await engine.get_entity(uuid.uuid4())
        assert result is None


class TestEntityList:
    @pytest.mark.asyncio
    async def test_list_entities_empty(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        result = await engine.list_entities(org_id=org_id, space_id=space_id)
        assert result.total == 0
        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_entities_returns_created(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_entity(EntityCreate(name="B", entity_type="org", org_id=org_id, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id)
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_list_entities_filters_by_org(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        other_org = uuid.uuid4()
        await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_entity(EntityCreate(name="B", entity_type="person", org_id=other_org, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id)
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_list_entities_pagination(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        for i in range(5):
            await engine.create_entity(EntityCreate(name=f"E{i}", entity_type="person", org_id=org_id, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id, limit=2, offset=0)
        assert len(result.items) == 2
        assert result.total == 5

    @pytest.mark.asyncio
    async def test_list_entities_filter_by_type(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        await engine.create_entity(EntityCreate(name="A", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.create_entity(EntityCreate(name="B", entity_type="technology", org_id=org_id, space_id=space_id))
        result = await engine.list_entities(org_id=org_id, space_id=space_id, entity_type="person")
        assert result.total == 1
        assert result.items[0].entity_type.value == "person"


class TestEntityUpdate:
    @pytest.mark.asyncio
    async def test_update_entity_name(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_entity(EntityCreate(name="Old", entity_type="person", org_id=org_id, space_id=space_id))
        updated = await engine.update_entity(created.id, EntityUpdate(name="New"))
        assert updated.name == "New"

    @pytest.mark.asyncio
    async def test_update_entity_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Entity not found"):
            await engine.update_entity(uuid.uuid4(), EntityUpdate(name="X"))


class TestEntityDelete:
    @pytest.mark.asyncio
    async def test_delete_entity(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_entity(EntityCreate(name="ToDelete", entity_type="person", org_id=org_id, space_id=space_id))
        await engine.delete_entity(created.id)
        fetched = await engine.get_entity(created.id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_delete_entity_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Entity not found"):
            await engine.delete_entity(uuid.uuid4())


# ── Relation Tests ─────────────────────────────────────────────

@pytest_asyncio.fixture
async def two_entities(engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
    """Create two entities for relation tests."""
    e1 = await engine.create_entity(EntityCreate(name="Zhang Jun", entity_type="person", org_id=org_id, space_id=space_id))
    e2 = await engine.create_entity(EntityCreate(name="Beijing", entity_type="location", org_id=org_id, space_id=space_id))
    return e1, e2


class TestRelationCreate:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_get_relation_by_id(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        created = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        fetched = await engine.get_relation(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_relation_not_found_returns_none(self, engine: GraphEngine):
        result = await engine.get_relation(uuid.uuid4())
        assert result is None


class TestRelationList:
    @pytest.mark.asyncio
    async def test_list_relations_empty(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        result = await engine.list_relations(org_id=org_id, space_id=space_id)
        assert result.total == 0

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_list_relations_filter_by_source_entity(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        result = await engine.list_relations(org_id=e1.org_id, space_id=e1.space_id, source_entity_id=e1.id)
        assert result.total == 1


class TestRelationUpdate:
    @pytest.mark.asyncio
    async def test_update_relation_value(self, engine: GraphEngine, two_entities):
        e1, e2 = two_entities
        created = await engine.create_relation(RelationCreate(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="lives_in", org_id=e1.org_id, space_id=e1.space_id,
        ))
        updated = await engine.update_relation(created.id, RelationUpdate(value="Haidian district"))
        assert updated.value == "Haidian district"


class TestRelationDelete:
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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
        assert new_rel.source_count == 1

        # Reload old relation -- should be closed
        refreshed_old = await engine.get_relation(old_rel.id)
        assert refreshed_old.is_current is False
        assert refreshed_old.valid_to is not None

    @pytest.mark.asyncio
    async def test_reaffirm_same_relation_increments_source_count(self, engine: GraphEngine, two_entities):
        """When adding a relation that matches an existing current one (same source+type+target), increment source_count."""
        e1, e2 = two_entities
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
        refreshed = await engine.get_relation(rel1.id)
        assert refreshed.source_count == 2
        assert refreshed.is_current is True
        assert rel2.id == rel1.id

    @pytest.mark.asyncio
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


# ── Memory Tests ───────────────────────────────────────────────

class TestMemoryCreate:
    @pytest.mark.asyncio
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
        assert result.root_id == result.id

    @pytest.mark.asyncio
    async def test_create_memory_with_custom_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = MemoryCreate(
            content="Test",
            custom_id="ext_123",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_memory(data)
        assert result.custom_id == "ext_123"

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_create_memory_sets_decay_rate_by_type(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        data = MemoryCreate(
            content="Test fact",
            memory_type="fact",
            org_id=org_id,
            space_id=space_id,
        )
        result = await engine.create_memory(data)
        assert result.decay_rate is not None
        assert result.decay_rate == 0.005


class TestMemoryGet:
    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Hello", org_id=org_id, space_id=space_id,
        ))
        fetched = await engine.get_memory(created.id)
        assert fetched is not None
        assert fetched.content == "Hello"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, engine: GraphEngine):
        result = await engine.get_memory(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_list_memories_empty(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        result = await engine.list_memories(org_id=org_id, space_id=space_id)
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_list_memories_excludes_forgotten_by_default(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Forget me", org_id=org_id, space_id=space_id,
        ))
        await engine.forget_memory(created.id, MemoryForget(forget_reason="user request"))
        result = await engine.list_memories(org_id=org_id, space_id=space_id)
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_list_memories_includes_forgotten_when_requested(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Forget me", org_id=org_id, space_id=space_id,
        ))
        await engine.forget_memory(created.id, MemoryForget(forget_reason="user request"))
        result = await engine.list_memories(org_id=org_id, space_id=space_id, include_forgotten=True)
        assert result.total == 1

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_list_memories_only_latest_versions(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        """By default, list should return only the latest version of each memory."""
        v1 = await engine.create_memory(MemoryCreate(
            content="V1", org_id=org_id, space_id=space_id,
        ))
        v2 = await engine.update_memory(v1.id, MemoryUpdate(content="V2"))
        result = await engine.list_memories(org_id=org_id, space_id=space_id)
        assert result.total == 1
        assert result.items[0].version == 2


class TestMemoryUpdate:
    @pytest.mark.asyncio
    async def test_update_creates_new_version(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        v1 = await engine.create_memory(MemoryCreate(
            content="Original", org_id=org_id, space_id=space_id,
        ))
        v2 = await engine.update_memory(v1.id, MemoryUpdate(content="Updated"))
        assert v2.version == 2
        assert v2.parent_id == v1.id
        assert v2.root_id == v1.id
        assert v2.content == "Updated"

    @pytest.mark.asyncio
    async def test_update_chain_three_versions(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        v1 = await engine.create_memory(MemoryCreate(
            content="V1", org_id=org_id, space_id=space_id,
        ))
        v2 = await engine.update_memory(v1.id, MemoryUpdate(content="V2"))
        v3 = await engine.update_memory(v2.id, MemoryUpdate(content="V3"))
        assert v3.version == 3
        assert v3.parent_id == v2.id
        assert v3.root_id == v1.id

    @pytest.mark.asyncio
    async def test_update_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Memory not found"):
            await engine.update_memory(uuid.uuid4(), MemoryUpdate(content="X"))


class TestMemoryForget:
    @pytest.mark.asyncio
    async def test_forget_sets_is_forgotten(self, engine: GraphEngine, org_id: uuid.UUID, space_id: uuid.UUID):
        created = await engine.create_memory(MemoryCreate(
            content="Forget me", org_id=org_id, space_id=space_id,
        ))
        result = await engine.forget_memory(created.id, MemoryForget(forget_reason="user request"))
        assert result.is_forgotten is True
        assert result.forget_reason == "user request"
        assert result.forget_at is not None

    @pytest.mark.asyncio
    async def test_forget_not_found_raises(self, engine: GraphEngine):
        with pytest.raises(ValueError, match="Memory not found"):
            await engine.forget_memory(uuid.uuid4(), MemoryForget(forget_reason="gone"))


# ── Memory Source Tests ────────────────────────────────────────

class TestMemorySource:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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