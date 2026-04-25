"""SQLAlchemy ORM models for all 7 RTMemory tables.

Tables:
  1. spaces       — Isolation boundary for all data
  2. entities     — Nodes in the temporal knowledge graph
  3. relations    — Temporal edges between entities
  4. memories     — Discrete memory entries with decay
  5. documents    — Uploaded documents for knowledge base
  6. chunks       — Segments of documents
  7. memory_sources — Traceability: memory -> document/chunk provenance
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db.base import Base


# ---------------------------------------------------------------------------
# Enum-like string types
# ---------------------------------------------------------------------------

class EntityType(str):
    """Entity type enum."""
    person = "person"
    org = "org"
    location = "location"
    concept = "concept"
    project = "project"
    technology = "technology"


class MemoryType(str):
    """Memory type enum."""
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class DocType(str):
    """Document type enum."""
    text = "text"
    pdf = "pdf"
    webpage = "webpage"


class DocStatus(str):
    """Document processing status enum."""
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    done = "done"
    failed = "failed"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _vector_dimension() -> int:
    """Read vector dimension from config. Falls back to 768 if config not ready."""
    try:
        return get_settings().embedding.vector_dimension
    except Exception:
        return 768


# ---------------------------------------------------------------------------
# 1. Spaces
# ---------------------------------------------------------------------------

class Space(Base):
    __tablename__ = "spaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    owner_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, default=None
    )
    container_tag: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, default=None
    )
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "id" not in kwargs:
            self.id = uuid.uuid4()
        if "is_default" not in kwargs:
            self.is_default = False
        if "created_at" not in kwargs:
            self.created_at = _utcnow()
        if "updated_at" not in kwargs:
            self.updated_at = _utcnow()

    # Relationships
    entities: Mapped[list["Entity"]] = relationship(
        "Entity", back_populates="space", passive_deletes=True,
        foreign_keys="Entity.space_id",
    )
    relations: Mapped[list["Relation"]] = relationship(
        "Relation", back_populates="space", passive_deletes=True,
        foreign_keys="Relation.space_id",
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="space", passive_deletes=True,
        foreign_keys="Memory.space_id",
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="space", passive_deletes=True,
        foreign_keys="Document.space_id",
    )

    def __repr__(self) -> str:
        return f"<Space id={self.id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# 2. Entities
# ---------------------------------------------------------------------------

class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=EntityType.person
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "id" not in kwargs:
            self.id = uuid.uuid4()
        if "entity_type" not in kwargs:
            self.entity_type = EntityType.person
        if "confidence" not in kwargs:
            self.confidence = 1.0
        if "created_at" not in kwargs:
            self.created_at = _utcnow()
        if "updated_at" not in kwargs:
            self.updated_at = _utcnow()

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="entities", foreign_keys=[space_id])
    source_relations: Mapped[list["Relation"]] = relationship(
        "Relation",
        foreign_keys="Relation.source_entity_id",
        back_populates="source_entity",
        passive_deletes=True,
    )
    target_relations: Mapped[list["Relation"]] = relationship(
        "Relation",
        foreign_keys="Relation.target_entity_id",
        back_populates="target_entity",
        passive_deletes=True,
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="entity", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_entities_space_id", "space_id"),
        Index("ix_entities_org_id", "org_id"),
        Index("ix_entities_entity_type", "entity_type"),
    )

    def __repr__(self) -> str:
        return f"<Entity id={self.id} name={self.name!r} type={self.entity_type}>"


# ---------------------------------------------------------------------------
# 3. Relations (temporal edges)
# ---------------------------------------------------------------------------

class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    valid_to: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="relations")
    source_entity: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="source_relations"
    )
    target_entity: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="target_relations"
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="relation", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_relations_source_entity_id", "source_entity_id"),
        Index("ix_relations_target_entity_id", "target_entity_id"),
        Index("ix_relations_space_id", "space_id"),
        Index("ix_relations_relation_type", "relation_type"),
        Index("ix_relations_is_current", "is_current"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "id" not in kwargs:
            self.id = uuid.uuid4()
        if "confidence" not in kwargs:
            self.confidence = 1.0
        if "is_current" not in kwargs:
            self.is_current = True
        if "source_count" not in kwargs:
            self.source_count = 1
        if "valid_from" not in kwargs:
            self.valid_from = _utcnow()
        if "created_at" not in kwargs:
            self.created_at = _utcnow()
        if "updated_at" not in kwargs:
            self.updated_at = _utcnow()


# ---------------------------------------------------------------------------
# 4. Memories
# ---------------------------------------------------------------------------

class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    custom_id: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True, default=None
    )
    memory_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=MemoryType.fact
    )
    entity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    relation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("relations.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    decay_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    is_forgotten: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    forget_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    forget_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
    root_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True, default=None
    )
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="memories")
    entity: Mapped[Optional["Entity"]] = relationship("Entity", back_populates="memories")
    relation: Mapped[Optional["Relation"]] = relationship("Relation", back_populates="memories")

    __table_args__ = (
        Index("ix_memories_space_id", "space_id"),
        Index("ix_memories_entity_id", "entity_id"),
        Index("ix_memories_memory_type", "memory_type"),
        Index("ix_memories_is_forgotten", "is_forgotten"),
        Index("ix_memories_custom_id", "custom_id"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "id" not in kwargs:
            self.id = uuid.uuid4()
        if "memory_type" not in kwargs:
            self.memory_type = MemoryType.fact
        if "confidence" not in kwargs:
            self.confidence = 1.0
        if "decay_rate" not in kwargs:
            self.decay_rate = 0.02
        if "is_forgotten" not in kwargs:
            self.is_forgotten = False
        if "version" not in kwargs:
            self.version = 1
        if "created_at" not in kwargs:
            self.created_at = _utcnow()
        if "updated_at" not in kwargs:
            self.updated_at = _utcnow()


# ---------------------------------------------------------------------------
# 5. Documents
# ---------------------------------------------------------------------------

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    doc_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocType.text
    )
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocStatus.queued
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    summary_embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True, default=None
    )
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_documents_space_id", "space_id"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_doc_type", "doc_type"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "id" not in kwargs:
            self.id = uuid.uuid4()
        if "doc_type" not in kwargs:
            self.doc_type = DocType.text
        if "status" not in kwargs:
            self.status = DocStatus.queued
        if "created_at" not in kwargs:
            self.created_at = _utcnow()
        if "updated_at" not in kwargs:
            self.updated_at = _utcnow()

    def __repr__(self) -> str:
        return f"<Document id={self.id} title={self.title!r} status={self.status}>"


# ---------------------------------------------------------------------------
# 6. Chunks
# ---------------------------------------------------------------------------

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "id" not in kwargs:
            self.id = uuid.uuid4()
        if "created_at" not in kwargs:
            self.created_at = _utcnow()

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} pos={self.position}>"


# ---------------------------------------------------------------------------
# 7. Memory Sources (traceability)
# ---------------------------------------------------------------------------

class MemorySource(Base):
    __tablename__ = "memory_sources"

    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_memory_sources_memory_id", "memory_id"),
        Index("ix_memory_sources_document_id", "document_id"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "relevance_score" not in kwargs:
            self.relevance_score = 0.0

    def __repr__(self) -> str:
        return f"<MemorySource memory={self.memory_id} doc={self.document_id} score={self.relevance_score}>"