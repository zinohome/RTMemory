"""Tests for SQLAlchemy database models."""

import uuid
from datetime import datetime, timezone

import pytest

from app.db.base import Base
from app.db.models import (
    Space,
    Entity,
    Relation,
    Memory,
    Document,
    Chunk,
    MemorySource,
    EntityType,
    MemoryType,
    DocType,
    DocStatus,
)


class TestSpaceModel:
    def test_space_creation(self):
        space = Space(
            name="Test Space",
            description="A test space",
            org_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
        )
        assert space.name == "Test Space"
        assert space.is_default is False
        assert space.container_tag is None
        assert isinstance(space.id, uuid.UUID)

    def test_space_defaults(self):
        space = Space(name="Default")
        assert space.is_default is False


class TestEntityModel:
    def test_entity_creation(self):
        entity = Entity(
            name="张军",
            entity_type=EntityType.person,
            description="A software engineer",
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert entity.name == "张军"
        assert entity.entity_type == EntityType.person
        assert entity.confidence == 1.0

    def test_entity_types(self):
        assert EntityType.person == "person"
        assert EntityType.org == "org"
        assert EntityType.location == "location"
        assert EntityType.concept == "concept"
        assert EntityType.project == "project"
        assert EntityType.technology == "technology"


class TestRelationModel:
    def test_relation_creation(self):
        src = uuid.uuid4()
        tgt = uuid.uuid4()
        relation = Relation(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="lives_in",
            value="北京",
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert relation.relation_type == "lives_in"
        assert relation.is_current is True
        assert relation.valid_to is None
        assert relation.source_count == 1

    def test_relation_temporal_defaults(self):
        relation = Relation(
            source_entity_id=uuid.uuid4(),
            target_entity_id=uuid.uuid4(),
            relation_type="works_at",
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert relation.is_current is True
        assert relation.confidence == 1.0
        assert relation.source_count == 1
        assert relation.valid_to is None


class TestMemoryModel:
    def test_memory_creation(self):
        memory = Memory(
            content="张军最近在研究知识图谱",
            memory_type=MemoryType.fact,
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert memory.content == "张军最近在研究知识图谱"
        assert memory.memory_type == MemoryType.fact
        assert memory.is_forgotten is False
        assert memory.version == 1
        assert memory.confidence == 1.0

    def test_memory_types(self):
        assert MemoryType.fact == "fact"
        assert MemoryType.preference == "preference"
        assert MemoryType.status == "status"
        assert MemoryType.inference == "inference"

    def test_memory_decay_defaults(self):
        memory = Memory(
            content="test",
            memory_type=MemoryType.fact,
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert memory.decay_rate == 0.02
        assert memory.is_forgotten is False
        assert memory.forget_at is None


class TestDocumentModel:
    def test_document_creation(self):
        doc = Document(
            title="Next.js Guide",
            content="Full guide content here...",
            doc_type=DocType.text,
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert doc.title == "Next.js Guide"
        assert doc.doc_type == DocType.text
        assert doc.status == DocStatus.queued

    def test_document_types(self):
        assert DocType.text == "text"
        assert DocType.pdf == "pdf"
        assert DocType.webpage == "webpage"

    def test_document_status_flow(self):
        assert DocStatus.queued == "queued"
        assert DocStatus.extracting == "extracting"
        assert DocStatus.chunking == "chunking"
        assert DocStatus.embedding == "embedding"
        assert DocStatus.done == "done"
        assert DocStatus.failed == "failed"


class TestChunkModel:
    def test_chunk_creation(self):
        chunk = Chunk(
            document_id=uuid.uuid4(),
            content="A chunk of text",
            position=0,
        )
        assert chunk.content == "A chunk of text"
        assert chunk.position == 0


class TestMemorySourceModel:
    def test_memory_source_creation(self):
        ms = MemorySource(
            memory_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            relevance_score=0.85,
        )
        assert ms.relevance_score == 0.85