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
    metadata: Optional[dict] = Field(default=None, validation_alias="metadata_")
    org_id: uuid.UUID
    space_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


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


# ── Graph Neighborhood (for visualization) ──────────────────────

class GraphNeighborhoodOut(BaseModel):
    center: EntityOut
    entities: list[EntityOut]
    relations: list[RelationOut]
    depth: int