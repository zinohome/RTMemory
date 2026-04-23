"""Pydantic schemas for the extraction pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Extraction result types ──────────────────────────────────────

class EntityType(str, Enum):
    person = "person"
    org = "org"
    location = "location"
    concept = "concept"
    project = "project"
    technology = "technology"


class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class ContradictionResolution(str, Enum):
    update = "update"
    extend = "extend"
    ignore = "ignore"


# ── Layer 2 extraction output ────────────────────────────────────

class ExtractedEntity(BaseModel):
    """A single entity extracted from conversation."""
    name: str = Field(..., min_length=1, max_length=500)
    type: EntityType = Field(default=EntityType.person)
    description: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    """A single relation extracted from conversation."""
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    value: str = Field(default="")
    valid_from: Optional[str] = Field(default=None, description="ISO date or partial like 2024-01")
    valid_to: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedMemory(BaseModel):
    """A single memory extracted from conversation."""
    content: str = Field(..., min_length=1)
    type: MemoryType = Field(default=MemoryType.fact)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    entity_name: Optional[str] = Field(default=None, description="Name of the primary entity this memory is about")


class ExtractedContradiction(BaseModel):
    """A detected contradiction between new and existing knowledge."""
    new: str = Field(..., min_length=1, description="New relation or fact, e.g. lives_in(Beijing)")
    old: str = Field(..., min_length=1, description="Old relation or fact, e.g. lives_in(Shanghai)")
    resolution: ContradictionResolution = Field(default=ContradictionResolution.update)


class ExtractionResult(BaseModel):
    """Complete structured output from Layer 2 extraction."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    memories: list[ExtractedMemory] = Field(default_factory=list)
    contradictions: list[ExtractedContradiction] = Field(default_factory=list)


# ── Deep scan result (Layer 3) ──────────────────────────────────

class ConfidenceAdjustment(BaseModel):
    """An adjustment to confidence of an existing memory/relation."""
    target_type: str = Field(..., description="memory or relation")
    target_id: uuid.UUID = Field(...)
    old_confidence: float = Field(...)
    new_confidence: float = Field(...)
    reason: str = Field(default="")


class DeepScanResult(BaseModel):
    """Result from Layer 3 deep scan — richer than single extraction."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    memories: list[ExtractedMemory] = Field(default_factory=list)
    contradictions: list[ExtractedContradiction] = Field(default_factory=list)
    confidence_adjustments: list[ConfidenceAdjustment] = Field(default_factory=list)


# ── Conversation API schemas ────────────────────────────────────

class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class ConversationSubmitRequest(BaseModel):
    """Request body for POST /v1/conversations/."""
    messages: list[ConversationMessage] = Field(..., min_length=1)
    space_id: uuid.UUID
    user_id: Optional[str] = Field(default=None)
    entity_context: Optional[str] = Field(default=None, description="Context hint to guide extraction")
    metadata: Optional[dict] = Field(default=None)


class ConversationSubmitResponse(BaseModel):
    """Response for conversation submission."""
    conversation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    extracted: ExtractionResult = Field(default_factory=ExtractionResult)
    skipped: bool = Field(default=False, description="True if FactDetector filtered all messages")
    message_count: int = Field(...)


class ConversationEndRequest(BaseModel):
    """Request body for POST /v1/conversations/end."""
    conversation_id: uuid.UUID
    space_id: uuid.UUID
    user_id: Optional[str] = Field(default=None)


class ConversationEndResponse(BaseModel):
    """Response for conversation end (deep scan)."""
    conversation_id: uuid.UUID
    deep_scan_result: DeepScanResult = Field(default_factory=DeepScanResult)
    message_count: int = Field(default=0)


# ── Document API schemas ────────────────────────────────────────

class DocumentType(str, Enum):
    text = "text"
    pdf = "pdf"
    webpage = "webpage"


class DocumentStatus(str, Enum):
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    done = "done"
    failed = "failed"


class DocumentCreateRequest(BaseModel):
    """Request body for POST /v1/documents/."""
    title: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None, description="Raw text content (for doc_type=text)")
    url: Optional[str] = Field(default=None, description="URL to fetch (for doc_type=webpage)")
    doc_type: DocumentType = Field(default=DocumentType.text)
    space_id: uuid.UUID
    metadata: Optional[dict] = Field(default=None)


class DocumentUploadResponse(BaseModel):
    """Response for document upload/creation."""
    id: uuid.UUID
    title: Optional[str] = None
    doc_type: DocumentType
    status: DocumentStatus = Field(default=DocumentStatus.queued)
    space_id: uuid.UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now())


class DocumentOut(BaseModel):
    """Full document output."""
    id: uuid.UUID
    title: Optional[str] = None
    doc_type: DocumentType
    url: Optional[str] = None
    status: DocumentStatus
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    space_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated document list."""
    items: list[DocumentOut] = Field(default_factory=list)
    total: int = Field(default=0)
    offset: int = Field(default=0)
    limit: int = Field(default=20)