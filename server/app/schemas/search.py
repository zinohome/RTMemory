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
    chunk_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Chunk similarity threshold"
    )
    document_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Document similarity threshold"
    )
    only_matching_chunks: bool = Field(
        default=False, description="Only return matching chunks, not full docs"
    )
    include_full_docs: bool = Field(
        default=False, description="Include full document content"
    )
    include_summary: bool = Field(
        default=True, description="Include document summary"
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None, description="Metadata AND/OR filter conditions"
    )
    rewrite_query: bool = Field(
        default=False, description="Rewrite query via LLM (adds ~400ms)"
    )


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