"""Profile data models — four-layer profile structure with confidence."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Decay rate constants per memory type ──

class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


DECAY_RATES: dict[MemoryType, float] = {
    MemoryType.fact: 0.001,
    MemoryType.preference: 0.005,
    MemoryType.status: 0.02,
    MemoryType.inference: 0.05,
}

FORGETTING_THRESHOLD: float = 0.1
REFERENCE_BOOST_ALPHA: float = 0.1


# ── Four-layer profile models ──

class IdentityLayer(BaseModel):
    """High confidence (>0.8), changes very slowly."""
    name: str | None = None
    location: str | None = None
    role: str | None = None
    company: str | None = None


class PreferencesLayer(BaseModel):
    """Medium confidence (0.5-0.8), changes occasionally."""
    languages: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    style: str | None = None


class CurrentStatusLayer(BaseModel):
    """Low confidence (<0.5), changes frequently."""
    focus: str | None = None
    project: str | None = None
    mood: str | None = None


class RelationshipsLayer(BaseModel):
    """Medium confidence, changes occasionally."""
    team: list[str] = Field(default_factory=list)
    collaborators: list[str] = Field(default_factory=list)


class ProfileData(BaseModel):
    """Complete profile assembled from four layers + dynamic memories."""
    identity: IdentityLayer = Field(default_factory=IdentityLayer)
    preferences: PreferencesLayer = Field(default_factory=PreferencesLayer)
    current_status: CurrentStatusLayer = Field(default_factory=CurrentStatusLayer)
    relationships: RelationshipsLayer = Field(default_factory=RelationshipsLayer)
    dynamic_memories: list[str] = Field(default_factory=list)


class ConfidenceMap(BaseModel):
    """Per-field confidence values from decay computation."""
    # identity fields
    name: float | None = None
    location: float | None = None
    role: float | None = None
    company: float | None = None
    # preferences fields
    languages: float | None = None
    stack: float | None = None
    style: float | None = None
    # current_status fields
    focus: float | None = None
    project: float | None = None
    mood: float | None = None
    # relationships fields
    team: float | None = None
    collaborators: float | None = None


class ProfileResponse(BaseModel):
    """Full profile API response."""
    profile: ProfileData
    confidence: ConfidenceMap
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: datetime
    timing_ms: float


class ProfileRequest(BaseModel):
    """Profile API request."""
    entity_id: str
    space_id: str
    q: str | None = None
    fresh: bool = False