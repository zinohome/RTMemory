"""Pydantic models for RTMemory API request and response objects."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class SearchMode(str, Enum):
    hybrid = "hybrid"
    memory_only = "memory_only"
    documents_only = "documents_only"


class DocumentStatus(str, Enum):
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    done = "done"
    failed = "failed"


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


# ── Memories ─────────────────────────────────────────────────────────────

class Memory(BaseModel):
    id: str
    content: str
    custom_id: str | None = None
    memory_type: MemoryType | None = None
    entity_id: str | None = None
    relation_id: str | None = None
    confidence: float = 1.0
    decay_rate: float = 0.01
    is_forgotten: bool = False
    forget_reason: str | None = None
    version: int = 1
    parent_id: str | None = None
    root_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    space_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryAddRequest(BaseModel):
    content: str
    space_id: str
    user_id: str | None = None
    custom_id: str | None = None
    entity_context: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryAddResponse(BaseModel):
    id: str
    content: str
    custom_id: str | None = None
    confidence: float = 1.0
    entity_id: str | None = None
    relation_ids: list[str] = Field(default_factory=list)
    memory_type: MemoryType | None = None
    created_at: datetime | None = None


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryForgetRequest(BaseModel):
    memory_id: str | None = None
    content_match: str | None = None
    reason: str | None = None


class MemoryListResponse(BaseModel):
    items: list[Memory] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 20


# ── Search ───────────────────────────────────────────────────────────────

class SearchFilter(BaseModel):
    key: str
    value: Any
    operator: str = "eq"


class SearchFilterGroup(BaseModel):
    AND: list[SearchFilter | SearchFilterGroup] | None = None
    OR: list[SearchFilter | SearchFilterGroup] | None = None


class SearchRequest(BaseModel):
    q: str
    space_id: str | None = None
    user_id: str | None = None
    mode: SearchMode = SearchMode.hybrid
    channels: list[str] | None = None
    limit: int = 10
    include_profile: bool = False
    chunk_threshold: float = 0.0
    document_threshold: float = 0.0
    only_matching_chunks: bool = False
    include_full_docs: bool = False
    include_summary: bool = False
    filters: SearchFilterGroup | None = None
    rewrite_query: bool = False
    rerank: bool = False


class SearchResultEntity(BaseModel):
    name: str
    type: str | None = None


class SearchResultDocument(BaseModel):
    title: str | None = None
    url: str | None = None


class SearchResult(BaseModel):
    type: str  # "memory" | "document_chunk" | "entity"
    content: str
    score: float
    source: str  # e.g. "vector+graph"
    entity: SearchResultEntity | None = None
    document: SearchResultDocument | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfileData(BaseModel):
    identity: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    current_status: dict[str, Any] = Field(default_factory=dict)
    relationships: dict[str, Any] = Field(default_factory=dict)
    dynamic_memories: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    profile: ProfileData | None = None
    timing_ms: int = 0


# ── Profile ──────────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    entity_id: str
    space_id: str
    q: str | None = None
    fresh: bool = False


class ProfileConfidence(BaseModel):
    """Confidence values keyed by profile attribute."""
    pass


class ProfileResponse(BaseModel):
    profile: ProfileData
    confidence: dict[str, float] = Field(default_factory=dict)
    search_results: list[SearchResult] = Field(default_factory=list)
    computed_at: datetime | None = None
    timing_ms: int = 0


# ── Documents ────────────────────────────────────────────────────────────

class Document(BaseModel):
    id: str
    title: str | None = None
    content: str | None = None
    doc_type: str | None = None
    url: str | None = None
    status: DocumentStatus = DocumentStatus.queued
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    space_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentAddRequest(BaseModel):
    content: str
    space_id: str
    title: str | None = None


class DocumentListResponse(BaseModel):
    items: list[Document] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 20


# ── Conversations ────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ConversationAddRequest(BaseModel):
    messages: list[ConversationMessage]
    space_id: str
    user_id: str | None = None


class ConversationAddResponse(BaseModel):
    id: str
    memory_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class ConversationEndRequest(BaseModel):
    conversation_id: str
    space_id: str


# ── Spaces ───────────────────────────────────────────────────────────────

class Space(BaseModel):
    id: str
    name: str
    description: str | None = None
    org_id: str | None = None
    owner_id: str | None = None
    container_tag: str | None = None
    is_default: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SpaceCreateRequest(BaseModel):
    name: str
    description: str | None = None


class SpaceListResponse(BaseModel):
    items: list[Space] = Field(default_factory=list)
    total: int = 0


# ── Graph ────────────────────────────────────────────────────────────────

class GraphEntity(BaseModel):
    id: str
    name: str
    entity_type: str | None = None
    description: str | None = None
    confidence: float = 1.0


class GraphRelation(BaseModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    value: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 1.0
    is_current: bool = True


class GraphNeighborhood(BaseModel):
    center: GraphEntity
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    depth: int = 1