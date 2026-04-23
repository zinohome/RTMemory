# RTMemory Profile Engine — 画像计算与置信度衰减

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the profile engine that computes structured user profiles from the knowledge graph, with confidence decay and caching.

**Architecture:** Profile = graph projection. Four-layer model (identity/preferences/status/relationships) mapped from relations. Confidence decays exponentially with time, boosted by references. Cached in-memory with graph-change invalidation.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0 (async), FastAPI, math (exp, log)

**Depends on:** GraphEngine (plan 03), SearchEngine (plan 05)

---

## Task 1: Profile Pydantic Models — Four-Layer Profile Structure

- [ ] **1.1 Create profile models file with all four layers + confidence map + decay constants**

**File:** `server/app/core/profile_models.py`

```python
"""Profile data models — four-layer profile structure with confidence."""
from __future__ import annotations

from dataclasses import dataclass, field
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
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -c "from server.app.core.profile_models import ProfileRequest, ProfileResponse, DECAY_RATES; print('Models OK:', list(DECAY_RATES.keys()))"`

**Expected:** `Models OK: [<MemoryType.fact: 'fact'>, <MemoryType.preference: 'preference'>, <MemoryType.status: 'status'>, <MemoryType.inference: 'inference'>]`

- [ ] **1.2 Write tests for profile models — serialization and defaults**

**File:** `server/tests/test_profile_models.py`

```python
"""Tests for profile Pydantic models."""
import pytest
from datetime import datetime

from server.app.core.profile_models import (
    IdentityLayer,
    PreferencesLayer,
    CurrentStatusLayer,
    RelationshipsLayer,
    ProfileData,
    ConfidenceMap,
    ProfileResponse,
    ProfileRequest,
    DECAY_RATES,
    FORGETTING_THRESHOLD,
    MemoryType,
)


class TestProfileModels:
    def test_identity_layer_defaults(self):
        layer = IdentityLayer()
        assert layer.name is None
        assert layer.location is None
        assert layer.role is None
        assert layer.company is None

    def test_identity_layer_with_values(self):
        layer = IdentityLayer(name="张军", location="北京", role="全栈工程师", company="ReTone")
        assert layer.name == "张军"
        assert layer.location == "北京"

    def test_preferences_layer_defaults(self):
        layer = PreferencesLayer()
        assert layer.languages == []
        assert layer.stack == []
        assert layer.style is None

    def test_current_status_layer_defaults(self):
        layer = CurrentStatusLayer()
        assert layer.focus is None
        assert layer.project is None
        assert layer.mood is None

    def test_relationships_layer_defaults(self):
        layer = RelationshipsLayer()
        assert layer.team == []
        assert layer.collaborators == []

    def test_profile_data_assembly(self):
        data = ProfileData(
            identity=IdentityLayer(name="张军", location="北京"),
            preferences=PreferencesLayer(stack=["Python", "TypeScript"], style="简洁"),
            current_status=CurrentStatusLayer(focus="知识图谱", project="RTMemory"),
            relationships=RelationshipsLayer(team=["李明", "王芳"]),
            dynamic_memories=["最近在研究时序知识图谱"],
        )
        assert data.identity.name == "张军"
        assert data.preferences.stack == ["Python", "TypeScript"]
        assert data.current_status.focus == "知识图谱"
        assert data.relationships.team == ["李明", "王芳"]
        assert data.dynamic_memories == ["最近在研究时序知识图谱"]

    def test_confidence_map_defaults(self):
        cm = ConfidenceMap()
        assert cm.location is None
        assert cm.stack is None
        assert cm.focus is None

    def test_profile_request_required_fields(self):
        req = ProfileRequest(entity_id="ent_xxx", space_id="sp_xxx")
        assert req.entity_id == "ent_xxx"
        assert req.q is None
        assert req.fresh is False

    def test_profile_request_with_optional(self):
        req = ProfileRequest(entity_id="ent_xxx", space_id="sp_xxx", q="前端框架", fresh=True)
        assert req.q == "前端框架"
        assert req.fresh is True

    def test_profile_response_serialization(self):
        resp = ProfileResponse(
            profile=ProfileData(identity=IdentityLayer(name="张军")),
            confidence=ConfidenceMap(location=0.95),
            search_results=[],
            computed_at=datetime(2026, 4, 23, 10, 0, 0),
            timing_ms=48.0,
        )
        d = resp.model_dump()
        assert d["profile"]["identity"]["name"] == "张军"
        assert d["confidence"]["location"] == 0.95
        assert d["timing_ms"] == 48.0

    def test_decay_rates_values(self):
        assert DECAY_RATES[MemoryType.fact] == 0.001
        assert DECAY_RATES[MemoryType.preference] == 0.005
        assert DECAY_RATES[MemoryType.status] == 0.02
        assert DECAY_RATES[MemoryType.inference] == 0.05

    def test_forgetting_threshold(self):
        assert FORGETTING_THRESHOLD == 0.1
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_models.py -v`

**Expected:** All 11 tests pass.

- [ ] **1.3 Commit profile models**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/profile_models.py server/tests/test_profile_models.py
git commit -m "Add Profile Pydantic models — four-layer structure, confidence map, decay constants"
```

---

## Task 2: Confidence Decay — Exponential Decay with Reference Boost

- [ ] **2.1 Write failing tests for confidence decay computation**

**File:** `server/tests/test_confidence_decay.py`

```python
"""Tests for confidence decay formula and forgetting logic."""
import pytest
from datetime import datetime, timezone, timedelta
from math import exp, log

from server.app.core.confidence_decay import (
    compute_decay,
    compute_memory_confidence,
    is_forgotten,
    DECAY_RATES,
    FORGETTING_THRESHOLD,
    REFERENCE_BOOST_ALPHA,
    MemoryType,
)


class TestComputeDecay:
    """C(t) = C0 * e^(-λ * Δt) * (1 + α * log(n+1))"""

    def test_no_decay_zero_days(self):
        """Δt=0 → no decay, only reference boost."""
        result = compute_decay(c0=0.9, decay_rate=0.001, delta_days=0, ref_count=0)
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_decay_identity_memory(self):
        """Identity decay_rate=0.001, after 365 days."""
        result = compute_decay(c0=0.9, decay_rate=0.001, delta_days=365, ref_count=0)
        expected = 0.9 * exp(-0.001 * 365)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result > 0.6  # identity decays slowly

    def test_decay_status_memory(self):
        """Status decay_rate=0.02, after 30 days."""
        result = compute_decay(c0=0.8, decay_rate=0.02, delta_days=30, ref_count=0)
        expected = 0.8 * exp(-0.02 * 30)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result < 0.5  # status decays faster

    def test_reference_boost_no_refs(self):
        """n=0 → boost factor = 1.0."""
        result = compute_decay(c0=1.0, decay_rate=0.0, delta_days=0, ref_count=0)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_reference_boost_with_refs(self):
        """n=5 → boost factor > 1.0."""
        result_no_ref = compute_decay(c0=0.8, decay_rate=0.01, delta_days=10, ref_count=0)
        result_with_ref = compute_decay(c0=0.8, decay_rate=0.01, delta_days=10, ref_count=5)
        assert result_with_ref > result_no_ref

    def test_reference_boost_formula(self):
        """Verify exact boost: (1 + α * log(n+1))."""
        alpha = REFERENCE_BOOST_ALPHA
        n = 9
        boost = 1 + alpha * log(n + 1)
        result = compute_decay(c0=1.0, decay_rate=0.0, delta_days=0, ref_count=n)
        assert result == pytest.approx(boost, abs=1e-6)

    def test_decay_clamped_to_zero(self):
        """Confidence never goes below 0."""
        result = compute_decay(c0=0.01, decay_rate=1.0, delta_days=1000, ref_count=0)
        assert result >= 0.0

    def test_decay_never_exceeds_one(self):
        """Even with massive reference boost, confidence capped at 1.0."""
        result = compute_decay(c0=0.9, decay_rate=0.0, delta_days=0, ref_count=1000)
        assert result <= 1.0


class TestIsForgotten:
    def test_above_threshold(self):
        assert is_forgotten(0.5) is False

    def test_below_threshold(self):
        assert is_forgotten(0.05) is True

    def test_at_threshold(self):
        """Exactly at threshold → not forgotten (boundary)."""
        assert is_forgotten(FORGETTING_THRESHOLD) is False

    def test_just_below_threshold(self):
        assert is_forgotten(0.099) is True


class TestComputeMemoryConfidence:
    """Integration: decay a memory object given its timestamps and type."""

    def test_fresh_memory_high_confidence(self):
        """Memory created 1 day ago, fact type → confidence barely changes."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.9,
            memory_type=MemoryType.fact,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result > 0.89

    def test_old_status_memory_low_confidence(self):
        """Status memory from 60 days ago → confidence very low."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.8,
            memory_type=MemoryType.status,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result < 0.3

    def test_old_identity_still_remembered(self):
        """Identity memory from 180 days ago, no refs → still above threshold."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2025, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.95,
            memory_type=MemoryType.fact,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result > FORGETTING_THRESHOLD

    def test_inference_memory_forgets_fast(self):
        """Inference memory, 90 days old → likely forgotten."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.6,
            memory_type=MemoryType.inference,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result < FORGETTING_THRESHOLD

    def test_referenced_memory_survives(self):
        """Old inference memory but referenced 5 times → boosted above threshold."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
        result_no_ref = compute_memory_confidence(
            initial_confidence=0.6,
            memory_type=MemoryType.inference,
            created_at=created,
            now=now,
            ref_count=0,
        )
        result_with_ref = compute_memory_confidence(
            initial_confidence=0.6,
            memory_type=MemoryType.inference,
            created_at=created,
            now=now,
            ref_count=5,
        )
        assert result_with_ref > result_no_ref
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_confidence_decay.py -v 2>&1 | head -30`

**Expected:** Import error — module does not exist yet. Tests FAIL.

- [ ] **2.2 Implement confidence decay module**

**File:** `server/app/core/confidence_decay.py`

```python
"""Confidence decay — exponential decay with reference boost.

Formula: C(t) = C0 * e^(-λ * Δt) * (1 + α * log(n+1))

Where:
  C0     = initial confidence
  λ      = decay rate (per day)
  Δt     = days since last reinforcement
  α      = reference boost coefficient (0.1)
  n      = number of times referenced / re-mentioned
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import exp, log

from server.app.core.profile_models import (
    DECAY_RATES,
    FORGETTING_THRESHOLD,
    REFERENCE_BOOST_ALPHA,
    MemoryType,
)


def compute_decay(
    c0: float,
    decay_rate: float,
    delta_days: float,
    ref_count: int = 0,
) -> float:
    """Compute decayed confidence.

    C(t) = C0 * e^(-λ * Δt) * (1 + α * log(n+1))

    Result clamped to [0.0, 1.0].
    """
    decay_factor = exp(-decay_rate * delta_days)
    boost_factor = 1.0 + REFERENCE_BOOST_ALPHA * log(ref_count + 1)
    raw = c0 * decay_factor * boost_factor
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, raw))


def is_forgotten(confidence: float) -> bool:
    """Check if confidence has fallen below forgetting threshold."""
    return confidence < FORGETTING_THRESHOLD


def compute_memory_confidence(
    initial_confidence: float,
    memory_type: MemoryType,
    created_at: datetime,
    now: datetime,
    ref_count: int = 0,
) -> float:
    """Compute current confidence for a memory based on its type and age.

    Uses the decay_rate for the memory type to determine how fast
    confidence falls off. Computed at read time (not via background job).
    """
    delta_days = (now - created_at).total_seconds() / 86400.0
    decay_rate = DECAY_RATES[memory_type]
    return compute_decay(
        c0=initial_confidence,
        decay_rate=decay_rate,
        delta_days=delta_days,
        ref_count=ref_count,
    )
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_confidence_decay.py -v`

**Expected:** All 13 tests pass.

- [ ] **2.3 Commit confidence decay**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/confidence_decay.py server/tests/test_confidence_decay.py
git commit -m "Add confidence decay module — exponential decay with reference boost"
```

---

## Task 3: Profile Projection — Relation-Type to Profile-Field Mapping

- [ ] **3.1 Write failing tests for profile projection**

**File:** `server/tests/test_profile_projection.py`

```python
"""Tests for profile projection — mapping relations to profile fields."""
import pytest
from datetime import datetime, timezone

from server.app.core.profile_projection import (
    ProfileProjectionConfig,
    default_projection_config,
    project_relations,
)
from server.app.core.profile_models import (
    ProfileData,
    IdentityLayer,
    PreferencesLayer,
    CurrentStatusLayer,
    RelationshipsLayer,
    ConfidenceMap,
)


# Minimal relation stub for testing (avoids DB dependency)
class FakeRelation:
    """Lightweight relation stub for projection tests."""
    def __init__(
        self,
        relation_type: str,
        target_name: str,
        value: str | None = None,
        confidence: float = 0.9,
        is_current: bool = True,
        updated_at: datetime | None = None,
        source_count: int = 0,
    ):
        self.relation_type = relation_type
        self.target_name = target_name
        self.value = value
        self.confidence = confidence
        self.is_current = is_current
        self.updated_at = updated_at or datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        self.source_count = source_count


class TestProfileProjectionConfig:
    def test_default_config_has_all_mappings(self):
        config = default_projection_config()
        assert "lives_in" in config.relation_map
        assert "works_at" in config.relation_map
        assert "has_role" in config.relation_map
        assert "prefers" in config.relation_map
        assert "current_focus" in config.relation_map
        assert "knows" in config.relation_map

    def test_default_mapping_identity(self):
        config = default_projection_config()
        assert config.relation_map["lives_in"] == "identity.location"
        assert config.relation_map["works_at"] == "identity.company"
        assert config.relation_map["has_role"] == "identity.role"

    def test_default_mapping_preferences(self):
        config = default_projection_config()
        assert config.relation_map["prefers"] == "preferences.stack"

    def test_default_mapping_status(self):
        config = default_projection_config()
        assert config.relation_map["current_focus"] == "current_status.focus"

    def test_default_mapping_relationships(self):
        config = default_projection_config()
        assert config.relation_map["knows"] == "relationships.team"


class TestProjectRelations:
    def test_empty_relations_returns_empty_profile(self):
        profile, confidence = project_relations([], "ent_xxx")
        assert profile.identity.name is None
        assert profile.identity.location is None
        assert profile.preferences.stack == []
        assert profile.relationships.team == []

    def test_single_lives_in(self):
        rels = [FakeRelation("lives_in", "北京", confidence=0.95)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.identity.location == "北京"
        assert confidence.location == pytest.approx(0.95, abs=1e-6)

    def test_multiple_lives_in_takes_highest_confidence(self):
        rels = [
            FakeRelation("lives_in", "上海", confidence=0.7, is_current=False),
            FakeRelation("lives_in", "北京", confidence=0.95, is_current=True),
        ]
        profile, confidence = project_relations(rels, "ent_xxx")
        # Should use is_current=true one
        assert profile.identity.location == "北京"

    def test_works_at(self):
        rels = [FakeRelation("works_at", "ReTone", confidence=0.85)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.identity.company == "ReTone"
        assert confidence.company == pytest.approx(0.85, abs=1e-6)

    def test_has_role(self):
        rels = [FakeRelation("has_role", "全栈工程师", confidence=0.88)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.identity.role == "全栈工程师"

    def test_prefers_accumulates_stack(self):
        rels = [
            FakeRelation("prefers", "Python", confidence=0.9),
            FakeRelation("prefers", "TypeScript", confidence=0.85),
        ]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert "Python" in profile.preferences.stack
        assert "TypeScript" in profile.preferences.stack

    def test_prefers_with_value_subfield_style(self):
        """prefers relation with value='style' maps to preferences.style."""
        rels = [FakeRelation("prefers", "简洁", value="style", confidence=0.7)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.preferences.style == "简洁"

    def test_prefers_with_value_subfield_languages(self):
        """prefers relation with value='languages' maps to preferences.languages."""
        rels = [
            FakeRelation("prefers", "中文", value="languages", confidence=0.8),
            FakeRelation("prefers", "English", value="languages", confidence=0.9),
        ]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert "中文" in profile.preferences.languages
        assert "English" in profile.preferences.languages

    def test_current_focus(self):
        rels = [FakeRelation("current_focus", "知识图谱", confidence=0.7)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.current_status.focus == "知识图谱"
        assert confidence.focus == pytest.approx(0.7, abs=1e-6)

    def test_knows_accumulates_team(self):
        rels = [
            FakeRelation("knows", "李明", confidence=0.6),
            FakeRelation("knows", "王芳", confidence=0.55),
        ]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert "李明" in profile.relationships.team
        assert "王芳" in profile.relationships.team

    def test_knows_with_value_collaborators(self):
        """knows relation with value='collaborators' maps to relationships.collaborators."""
        rels = [FakeRelation("knows", "赵四", value="collaborators", confidence=0.5)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert "赵四" in profile.relationships.collaborators

    def test_unknown_relation_type_ignored(self):
        """Relations with unmapped types are silently ignored."""
        rels = [FakeRelation("unknown_type", "something", confidence=0.9)]
        profile, confidence = project_relations(rels, "ent_xxx")
        # Should not crash, profile remains empty
        assert profile.identity.name is None

    def test_entity_name_from_entity_id_passthrough(self):
        """Entity name is set from the passed-in entity_name arg."""
        rels = [FakeRelation("lives_in", "北京", confidence=0.95)]
        profile, confidence = project_relations(rels, "ent_xxx", entity_name="张军")
        assert profile.identity.name == "张军"

    def test_full_profile_projection(self):
        """Comprehensive test with multiple relation types."""
        rels = [
            FakeRelation("lives_in", "北京", confidence=0.95),
            FakeRelation("works_at", "ReTone", confidence=0.85),
            FakeRelation("has_role", "全栈工程师", confidence=0.88),
            FakeRelation("prefers", "Python", confidence=0.9),
            FakeRelation("prefers", "TypeScript", confidence=0.85),
            FakeRelation("prefers", "简洁", value="style", confidence=0.7),
            FakeRelation("current_focus", "知识图谱", confidence=0.7),
            FakeRelation("knows", "李明", confidence=0.6),
            FakeRelation("knows", "王芳", confidence=0.55),
        ]
        profile, confidence = project_relations(rels, "ent_xxx", entity_name="张军")
        assert profile.identity.name == "张军"
        assert profile.identity.location == "北京"
        assert profile.identity.company == "ReTone"
        assert profile.identity.role == "全栈工程师"
        assert "Python" in profile.preferences.stack
        assert "TypeScript" in profile.preferences.stack
        assert profile.preferences.style == "简洁"
        assert profile.current_status.focus == "知识图谱"
        assert "李明" in profile.relationships.team
        assert "王芳" in profile.relationships.team

    def test_non_current_relations_filtered(self):
        """Only is_current=true relations should be used for identity/status."""
        rels = [
            FakeRelation("lives_in", "上海", confidence=0.7, is_current=False),
            FakeRelation("lives_in", "北京", confidence=0.95, is_current=True),
        ]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.identity.location == "北京"

    def test_only_current_false_all_filtered(self):
        """When all relations are is_current=false, profile fields stay None."""
        rels = [FakeRelation("lives_in", "上海", confidence=0.7, is_current=False)]
        profile, confidence = project_relations(rels, "ent_xxx")
        assert profile.identity.location is None
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_projection.py -v 2>&1 | head -30`

**Expected:** Import error — module does not exist. Tests FAIL.

- [ ] **3.2 Implement profile projection module**

**File:** `server/app/core/profile_projection.py`

```python
"""Profile projection — map relation types to profile fields.

The projection config maps relation_type strings to profile field paths
like "identity.location". The `prefers` and `knows` relation types use
the relation's `value` field as a sub-field discriminator (e.g.,
prefers+value="style" → preferences.style, knows+value="collaborators"
→ relationships.collaborators).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from server.app.core.profile_models import (
    ConfidenceMap,
    CurrentStatusLayer,
    IdentityLayer,
    PreferencesLayer,
    ProfileData,
    RelationshipsLayer,
)


# ── Configurable projection mapping ──

DEFAULT_RELATION_MAP: dict[str, str] = {
    "lives_in": "identity.location",
    "works_at": "identity.company",
    "has_role": "identity.role",
    "prefers": "preferences.stack",       # default sub-field; value field overrides
    "current_focus": "current_status.focus",
    "current_project": "current_status.project",
    "current_mood": "current_status.mood",
    "knows": "relationships.team",         # default sub-field; value field overrides
}

# Sub-field overrides: when relation.value matches, remap to this path
PREFERS_SUBFIELDS: dict[str, str] = {
    "style": "preferences.style",
    "languages": "preferences.languages",
    "stack": "preferences.stack",
}

KNOWS_SUBFIELDS: dict[str, str] = {
    "collaborators": "relationships.collaborators",
    "team": "relationships.team",
}


@dataclass
class ProfileProjectionConfig:
    """Configurable mapping from relation_type → profile field path."""
    relation_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RELATION_MAP))
    prefers_subfields: dict[str, str] = field(default_factory=lambda: dict(PREFERS_SUBFIELDS))
    knows_subfields: dict[str, str] = field(default_factory=lambda: dict(KNOWS_SUBFIELDS))


def default_projection_config() -> ProfileProjectionConfig:
    """Return the default projection configuration."""
    return ProfileProjectionConfig()


class RelationStub(Protocol):
    """Protocol for relation objects used in projection."""
    relation_type: str
    target_name: str
    value: str | None
    confidence: float
    is_current: bool
    updated_at: datetime
    source_count: int


def _resolve_field_path(
    relation_type: str,
    value: str | None,
    config: ProfileProjectionConfig,
) -> str | None:
    """Resolve the full profile field path for a relation.

    Handles sub-field overrides for `prefers` and `knows` via the
    relation's value field.
    """
    if relation_type == "prefers" and value:
        subfield = config.prefers_subfields.get(value)
        if subfield:
            return subfield

    if relation_type == "knows" and value:
        subfield = config.knows_subfields.get(value)
        if subfield:
            return subfield

    return config.relation_map.get(relation_type)


def _is_identity_field(field_path: str) -> bool:
    return field_path.startswith("identity.")


def _is_status_field(field_path: str) -> bool:
    return field_path.startswith("current_status.")


def project_relations(
    relations: list[Any],
    entity_id: str,
    entity_name: str | None = None,
    config: ProfileProjectionConfig | None = None,
) -> tuple[ProfileData, ConfidenceMap]:
    """Project a list of relations into a four-layer profile.

    For identity and current_status layers (scalar fields), only
    is_current=True relations are used, and highest-confidence wins.

    For preferences and relationships (list fields), all relations
    are accumulated regardless of is_current.

    Returns (ProfileData, ConfidenceMap).
    """
    if config is None:
        config = default_projection_config()

    identity = IdentityLayer(name=entity_name)
    preferences = PreferencesLayer()
    current_status = CurrentStatusLayer()
    relationships = RelationshipsLayer()
    confidence = ConfidenceMap()

    # Track best scalar values for identity/status
    # key=field_path, value=(target_name, confidence)
    best_scalar: dict[str, tuple[str, float]] = {}

    # Track list accumulators for preferences/relationships
    # key=field_path, value=list of (target_name, confidence)
    list_accum: dict[str, list[tuple[str, float]]] = {}

    for rel in relations:
        field_path = _resolve_field_path(rel.relation_type, rel.value, config)
        if field_path is None:
            continue  # unknown relation type, skip

        is_scalar = _is_identity_field(field_path) or _is_status_field(field_path)

        if is_scalar:
            # For scalar fields, only use is_current=True
            if not rel.is_current:
                continue
            existing = best_scalar.get(field_path)
            if existing is None or rel.confidence > existing[1]:
                best_scalar[field_path] = (rel.target_name, rel.confidence)
        else:
            # For list fields, accumulate all (even non-current)
            if field_path not in list_accum:
                list_accum[field_path] = []
            list_accum[field_path].append((rel.target_name, rel.confidence))

    # ── Populate identity from best_scalar ──
    for field_path, (value, conf) in best_scalar.items():
        _set_scalar_field(identity, current_status, field_path, value)
        _set_confidence_field(confidence, field_path, conf)

    # ── Populate preferences and relationships from list_accum ──
    for field_path, items in list_accum.items():
        # Sort by confidence descending for deterministic ordering
        items_sorted = sorted(items, key=lambda x: -x[1])
        values = [name for name, _ in items_sorted]
        best_conf = items_sorted[0][1] if items_sorted else 0.0
        _set_list_field(preferences, relationships, field_path, values)
        _set_confidence_field(confidence, field_path, best_conf)

    profile = ProfileData(
        identity=identity,
        preferences=preferences,
        current_status=current_status,
        relationships=relationships,
    )
    return profile, confidence


def _set_scalar_field(
    identity: IdentityLayer,
    current_status: CurrentStatusLayer,
    field_path: str,
    value: str,
) -> None:
    """Set a scalar field on identity or current_status."""
    mapping = {
        "identity.location": lambda v: setattr(identity, "location", v),
        "identity.company": lambda v: setattr(identity, "company", v),
        "identity.role": lambda v: setattr(identity, "role", v),
        "current_status.focus": lambda v: setattr(current_status, "focus", v),
        "current_status.project": lambda v: setattr(current_status, "project", v),
        "current_status.mood": lambda v: setattr(current_status, "mood", v),
    }
    setter = mapping.get(field_path)
    if setter:
        setter(value)


def _set_list_field(
    preferences: PreferencesLayer,
    relationships: RelationshipsLayer,
    field_path: str,
    values: list[str],
) -> None:
    """Set a list field on preferences or relationships."""
    mapping = {
        "preferences.stack": lambda v: setattr(preferences, "stack", v),
        "preferences.languages": lambda v: setattr(preferences, "languages", v),
        "preferences.style": lambda v: setattr(preferences, "style", v[0] if v else None),
        "relationships.team": lambda v: setattr(relationships, "team", v),
        "relationships.collaborators": lambda v: setattr(relationships, "collaborators", v),
    }
    setter = mapping.get(field_path)
    if setter:
        setter(values)


def _set_confidence_field(
    confidence: ConfidenceMap,
    field_path: str,
    value: float,
) -> None:
    """Set a confidence value for a profile field."""
    # Extract the leaf field name from path like "identity.location" → "location"
    field_name = field_path.split(".")[-1]
    if hasattr(confidence, field_name):
        setattr(confidence, field_name, value)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_projection.py -v`

**Expected:** All 16 tests pass.

- [ ] **3.3 Commit profile projection**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/profile_projection.py server/tests/test_profile_projection.py
git commit -m "Add profile projection module — configurable relation-type to profile-field mapping"
```

---

## Task 4: Profile Cache — In-Memory Cache with Graph-Change Invalidation

- [ ] **4.1 Write failing tests for profile cache**

**File:** `server/tests/test_profile_cache.py`

```python
"""Tests for profile cache with graph-change invalidation."""
import pytest
import time
from datetime import datetime, timezone

from server.app.core.profile_cache import ProfileCache
from server.app.core.profile_models import (
    ProfileData,
    IdentityLayer,
    ConfidenceMap,
)


class TestProfileCache:
    def _make_profile(self, name: str = "张军") -> ProfileData:
        return ProfileData(identity=IdentityLayer(name=name))

    def _make_confidence(self, loc: float = 0.9) -> ConfidenceMap:
        return ConfidenceMap(location=loc)

    def test_cache_miss_returns_none(self):
        cache = ProfileCache()
        result = cache.get("ent_xxx", "sp_xxx")
        assert result is None

    def test_cache_put_and_get(self):
        cache = ProfileCache()
        profile = self._make_profile()
        confidence = self._make_confidence()
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_xxx", "sp_xxx", profile, confidence, now, 48.0)
        result = cache.get("ent_xxx", "sp_xxx")
        assert result is not None
        assert result[0].identity.name == "张军"
        assert result[1].location == 0.9
        assert result[2] == now
        assert result[3] == 48.0

    def test_cache_different_entities_isolated(self):
        cache = ProfileCache()
        p1 = self._make_profile("张军")
        p2 = self._make_profile("李明")
        c1 = self._make_confidence(0.9)
        c2 = self._make_confidence(0.7)
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_1", "sp_xxx", p1, c1, now, 10.0)
        cache.put("ent_2", "sp_xxx", p2, c2, now, 20.0)
        r1 = cache.get("ent_1", "sp_xxx")
        r2 = cache.get("ent_2", "sp_xxx")
        assert r1[0].identity.name == "张军"
        assert r2[0].identity.name == "李明"

    def test_cache_different_spaces_isolated(self):
        cache = ProfileCache()
        p1 = self._make_profile("张军")
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_xxx", "sp_1", p1, self._make_confidence(), now, 10.0)
        r2 = cache.get("ent_xxx", "sp_2")
        assert r2 is None

    def test_invalidate_entity(self):
        cache = ProfileCache()
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_xxx", "sp_xxx", self._make_profile(), self._make_confidence(), now, 10.0)
        cache.invalidate("ent_xxx", "sp_xxx")
        result = cache.get("ent_xxx", "sp_xxx")
        assert result is None

    def test_invalidate_does_not_affect_other_entities(self):
        cache = ProfileCache()
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_1", "sp_xxx", self._make_profile("A"), self._make_confidence(), now, 10.0)
        cache.put("ent_2", "sp_xxx", self._make_profile("B"), self._make_confidence(), now, 10.0)
        cache.invalidate("ent_1", "sp_xxx")
        assert cache.get("ent_1", "sp_xxx") is None
        assert cache.get("ent_2", "sp_xxx") is not None

    def test_invalidate_all_for_space(self):
        cache = ProfileCache()
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_1", "sp_xxx", self._make_profile("A"), self._make_confidence(), now, 10.0)
        cache.put("ent_2", "sp_xxx", self._make_profile("B"), self._make_confidence(), now, 10.0)
        cache.put("ent_1", "sp_other", self._make_profile("C"), self._make_confidence(), now, 10.0)
        cache.invalidate_space("sp_xxx")
        assert cache.get("ent_1", "sp_xxx") is None
        assert cache.get("ent_2", "sp_xxx") is None
        assert cache.get("ent_1", "sp_other") is not None

    def test_overwrite_updates_cache(self):
        cache = ProfileCache()
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        cache.put("ent_xxx", "sp_xxx", self._make_profile("old"), self._make_confidence(0.5), now, 10.0)
        p_new = self._make_profile("new")
        c_new = self._make_confidence(0.8)
        cache.put("ent_xxx", "sp_xxx", p_new, c_new, now, 15.0)
        result = cache.get("ent_xxx", "sp_xxx")
        assert result[0].identity.name == "new"
        assert result[1].location == 0.8

    def test_cache_size_tracking(self):
        cache = ProfileCache()
        now = datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc)
        assert cache.size() == 0
        cache.put("ent_1", "sp_xxx", self._make_profile(), self._make_confidence(), now, 10.0)
        assert cache.size() == 1
        cache.put("ent_2", "sp_xxx", self._make_profile(), self._make_confidence(), now, 10.0)
        assert cache.size() == 2
        cache.invalidate("ent_1", "sp_xxx")
        assert cache.size() == 1
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_cache.py -v 2>&1 | head -20`

**Expected:** Import error — module does not exist. Tests FAIL.

- [ ] **4.2 Implement profile cache**

**File:** `server/app/core/profile_cache.py`

```python
"""Profile cache — in-memory cache with graph-change invalidation.

Cache key = (entity_id, space_id). Invalidated when any entity/relation/
memory for that entity changes. No TTL — pure invalidation-driven.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from server.app.core.profile_models import ConfidenceMap, ProfileData


class ProfileCache:
    """In-memory profile cache.

    Stores computed profiles keyed by (entity_id, space_id).
    Invalidated explicitly when the underlying graph data changes.
    """

    def __init__(self) -> None:
        # key: (entity_id, space_id)
        # value: (profile_data, confidence_map, computed_at, timing_ms)
        self._store: dict[tuple[str, str], tuple[ProfileData, ConfidenceMap, datetime, float]] = {}

    def get(
        self, entity_id: str, space_id: str
    ) -> tuple[ProfileData, ConfidenceMap, datetime, float] | None:
        """Return cached profile or None on miss."""
        return self._store.get((entity_id, space_id))

    def put(
        self,
        entity_id: str,
        space_id: str,
        profile: ProfileData,
        confidence: ConfidenceMap,
        computed_at: datetime,
        timing_ms: float,
    ) -> None:
        """Store a computed profile in the cache."""
        self._store[(entity_id, space_id)] = (profile, confidence, computed_at, timing_ms)

    def invalidate(self, entity_id: str, space_id: str) -> None:
        """Invalidate cached profile for a specific entity+space."""
        self._store.pop((entity_id, space_id), None)

    def invalidate_space(self, space_id: str) -> None:
        """Invalidate all cached profiles for a space."""
        keys_to_remove = [k for k in self._store if k[1] == space_id]
        for k in keys_to_remove:
            del self._store[k]

    def size(self) -> int:
        """Return number of cached profiles."""
        return len(self._store)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._store.clear()
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_cache.py -v`

**Expected:** All 10 tests pass.

- [ ] **4.3 Commit profile cache**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/profile_cache.py server/tests/test_profile_cache.py
git commit -m "Add profile cache — in-memory with graph-change invalidation"
```

---

## Task 5: ProfileEngine — Core Engine Combining Graph Reads, Projection, Decay, and Cache

- [ ] **5.1 Write failing tests for ProfileEngine**

**File:** `server/tests/test_profile_engine.py`

```python
"""Tests for ProfileEngine — the core orchestration class."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from server.app.core.profile_engine import ProfileEngine
from server.app.core.profile_models import (
    ProfileData,
    IdentityLayer,
    ConfidenceMap,
    MemoryType,
    FORGETTING_THRESHOLD,
)


# ── Fake GraphEngine stubs ──

class FakeEntity:
    def __init__(self, id: str, name: str, entity_type: str = "person"):
        self.id = id
        self.name = name
        self.entity_type = entity_type


class FakeRelation:
    def __init__(
        self,
        relation_type: str,
        target_entity: FakeEntity,
        value: str | None = None,
        confidence: float = 0.9,
        is_current: bool = True,
        source_count: int = 0,
        updated_at: datetime | None = None,
    ):
        self.relation_type = relation_type
        self.target_entity = target_entity
        self.target_name = target_entity.name
        self.value = value
        self.confidence = confidence
        self.is_current = is_current
        self.source_count = source_count
        self.updated_at = updated_at or datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


class FakeMemory:
    def __init__(
        self,
        id: str,
        content: str,
        memory_type: str = "fact",
        confidence: float = 0.8,
        decay_rate: float = 0.001,
        is_forgotten: bool = False,
        entity_id: str | None = None,
        created_at: datetime | None = None,
        source_count: int = 0,
    ):
        self.id = id
        self.content = content
        self.memory_type = memory_type
        self.confidence = confidence
        self.decay_rate = decay_rate
        self.is_forgotten = is_forgotten
        self.entity_id = entity_id
        self.created_at = created_at or datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        self.source_count = source_count


def make_fake_graph_engine(
    entity: FakeEntity | None = None,
    relations: list[FakeRelation] | None = None,
    memories: list[FakeMemory] | None = None,
):
    """Create a fake GraphEngine with async stubs."""
    ge = AsyncMock()
    ge.get_entity = AsyncMock(return_value=entity)
    ge.get_current_relations = AsyncMock(return_value=relations or [])
    ge.get_recent_memories = AsyncMock(return_value=memories or [])
    return ge


def make_fake_search_engine(results: list | None = None):
    se = AsyncMock()
    se.search = AsyncMock(return_value=results or [])
    return se


class TestProfileEngineCompute:
    @pytest.mark.asyncio
    async def test_compute_empty_entity(self):
        ge = make_fake_graph_engine(entity=None)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_missing", space_id="sp_xxx")
        assert result.profile.identity.name is None
        assert result.confidence.name is None

    @pytest.mark.asyncio
    async def test_compute_basic_profile(self):
        entity = FakeEntity("ent_1", "张军")
        relations = [
            FakeRelation("lives_in", FakeEntity("e2", "北京"), confidence=0.95),
            FakeRelation("works_at", FakeEntity("e3", "ReTone"), confidence=0.85),
            FakeRelation("prefers", FakeEntity("e4", "Python"), confidence=0.9),
        ]
        ge = make_fake_graph_engine(entity=entity, relations=relations)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert result.profile.identity.name == "张军"
        assert result.profile.identity.location == "北京"
        assert result.profile.identity.company == "ReTone"
        assert "Python" in result.profile.preferences.stack

    @pytest.mark.asyncio
    async def test_compute_dynamic_memories(self):
        entity = FakeEntity("ent_1", "张军")
        memories = [
            FakeMemory("m1", "最近在研究时序知识图谱", "status", confidence=0.7),
            FakeMemory("m2", "RTMemory 项目进行中", "status", confidence=0.6),
        ]
        ge = make_fake_graph_engine(entity=entity, memories=memories)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert "最近在研究时序知识图谱" in result.profile.dynamic_memories

    @pytest.mark.asyncio
    async def test_compute_filters_forgotten_memories(self):
        entity = FakeEntity("ent_1", "张军")
        memories = [
            FakeMemory("m1", "可见的记忆", "fact", confidence=0.8, is_forgotten=False),
            FakeMemory("m2", "已遗忘的记忆", "fact", confidence=0.05, is_forgotten=True),
        ]
        ge = make_fake_graph_engine(entity=entity, memories=memories)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert "可见的记忆" in result.profile.dynamic_memories
        assert "已遗忘的记忆" not in result.profile.dynamic_memories

    @pytest.mark.asyncio
    async def test_compute_with_search_query(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        search_results = [{"type": "memory", "content": "Next.js 15", "score": 0.9}]
        se = make_fake_search_engine(results=search_results)
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(
            entity_id="ent_1", space_id="sp_xxx", q="前端框架"
        )
        assert len(result.search_results) == 1
        assert result.search_results[0]["content"] == "Next.js 15"

    @pytest.mark.asyncio
    async def test_compute_timing_recorded(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert result.timing_ms >= 0
        assert result.computed_at is not None

    @pytest.mark.asyncio
    async def test_compute_fresh_bypasses_cache(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        # First call populates cache
        await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        # Second call with fresh=True should call graph_engine again
        await engine.compute(entity_id="ent_1", space_id="sp_xxx", fresh=True)
        # graph_engine.get_entity should have been called at least twice
        assert ge.get_entity.call_count >= 2

    @pytest.mark.asyncio
    async def test_compute_uses_cache(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        # First call populates cache
        result1 = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        # Second call should return cached result
        result2 = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        # Should not have called graph_engine again
        assert ge.get_entity.call_count == 1
        assert result1.profile.identity.name == result2.profile.identity.name

    @pytest.mark.asyncio
    async def test_invalidate_on_graph_change(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        # Simulate graph change
        engine.invalidate_cache("ent_1", "sp_xxx")
        # Next call should recompute
        await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert ge.get_entity.call_count == 2


class TestProfileEngineConfidenceDecay:
    @pytest.mark.asyncio
    async def test_dynamic_memories_sorted_by_confidence(self):
        entity = FakeEntity("ent_1", "张军")
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        memories = [
            FakeMemory(
                "m1", "低置信度记忆", "status",
                confidence=0.4,
                created_at=now - timedelta(days=30),
            ),
            FakeMemory(
                "m2", "高置信度记忆", "status",
                confidence=0.9,
                created_at=now - timedelta(days=1),
            ),
            FakeMemory(
                "m3", "中等置信度记忆", "status",
                confidence=0.7,
                created_at=now - timedelta(days=10),
            ),
        ]
        ge = make_fake_graph_engine(entity=entity, memories=memories)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        # Should be sorted by decayed confidence descending
        assert result.profile.dynamic_memories[0] == "高置信度记忆"

    @pytest.mark.asyncio
    async def test_memories_below_threshold_excluded(self):
        """Memories whose decayed confidence < 0.1 should not appear."""
        entity = FakeEntity("ent_1", "张军")
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        memories = [
            FakeMemory(
                "m1", "极老推断", "inference",
                confidence=0.5,
                created_at=now - timedelta(days=365),  # very old inference
            ),
            FakeMemory(
                "m2", "新鲜事实", "fact",
                confidence=0.9,
                created_at=now - timedelta(days=1),
            ),
        ]
        ge = make_fake_graph_engine(entity=entity, memories=memories)
        se = make_fake_search_engine()
        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        # The very old inference should have decayed below threshold
        assert "新鲜事实" in result.profile.dynamic_memories
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_engine.py -v 2>&1 | head -20`

**Expected:** Import error — module does not exist. Tests FAIL.

- [ ] **5.2 Implement ProfileEngine**

**File:** `server/app/core/profile_engine.py`

```python
"""ProfileEngine — computes user profiles from the knowledge graph.

Orchestrates: graph reads → projection → confidence decay → cache.
Profile is NOT stored — it's computed from the graph on demand and cached.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Protocol

from server.app.core.confidence_decay import (
    compute_memory_confidence,
    is_forgotten,
)
from server.app.core.profile_cache import ProfileCache
from server.app.core.profile_models import (
    ConfidenceMap,
    ProfileData,
    ProfileResponse,
    MemoryType,
    FORGETTING_THRESHOLD,
)
from server.app.core.profile_projection import project_relations


class GraphEngineProtocol(Protocol):
    async def get_entity(self, entity_id: str) -> Any | None: ...
    async def get_current_relations(self, entity_id: str, space_id: str) -> list[Any]: ...
    async def get_recent_memories(
        self, entity_id: str, space_id: str, limit: int = 10
    ) -> list[Any]: ...


class SearchEngineProtocol(Protocol):
    async def search(
        self, q: str, space_id: str, entity_id: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]: ...


class ProfileEngine:
    """Computes structured user profiles from the knowledge graph.

    Flow:
      1. Check cache (unless fresh=True)
      2. Read entity + current relations + recent memories from GraphEngine
      3. Project relations → four-layer profile
      4. Compute confidence decay for dynamic memories
      5. Optionally search via SearchEngine if q is provided
      6. Cache result and return

    Cache invalidation: call invalidate_cache() when graph data changes.
    """

    def __init__(
        self,
        graph_engine: GraphEngineProtocol,
        search_engine: SearchEngineProtocol | None = None,
        cache: ProfileCache | None = None,
    ) -> None:
        self._graph_engine = graph_engine
        self._search_engine = search_engine
        self._cache = cache or ProfileCache()

    async def compute(
        self,
        entity_id: str,
        space_id: str,
        q: str | None = None,
        fresh: bool = False,
    ) -> ProfileResponse:
        """Compute and return the profile for an entity.

        Args:
            entity_id: The entity to compute profile for.
            space_id: Space isolation.
            q: Optional search query — triggers search and attaches results.
            fresh: If True, bypass cache and recompute.
        """
        start = time.monotonic()

        # 1. Check cache
        if not fresh:
            cached = self._cache.get(entity_id, space_id)
            if cached is not None:
                profile, confidence, computed_at, timing_ms = cached
                # Even cached results may need search results attached
                search_results = await self._maybe_search(q, entity_id, space_id)
                return ProfileResponse(
                    profile=profile,
                    confidence=confidence,
                    search_results=search_results,
                    computed_at=computed_at,
                    timing_ms=timing_ms,
                )

        # 2. Read from graph
        entity = await self._graph_engine.get_entity(entity_id)
        relations = await self._graph_engine.get_current_relations(entity_id, space_id)
        recent_memories = await self._graph_engine.get_recent_memories(
            entity_id, space_id, limit=10
        )

        # 3. Project relations → profile
        entity_name = entity.name if entity else None
        profile, confidence = project_relations(
            relations, entity_id, entity_name=entity_name
        )

        # 4. Compute dynamic memories with confidence decay
        profile.dynamic_memories = await self._compute_dynamic_memories(
            recent_memories
        )

        # 5. Optional search
        search_results = await self._maybe_search(q, entity_id, space_id)

        # 6. Build response
        elapsed = (time.monotonic() - start) * 1000.0
        now = datetime.now(timezone.utc)

        response = ProfileResponse(
            profile=profile,
            confidence=confidence,
            search_results=search_results,
            computed_at=now,
            timing_ms=round(elapsed, 2),
        )

        # 7. Cache result
        self._cache.put(
            entity_id, space_id, profile, confidence, now, round(elapsed, 2)
        )

        return response

    async def _compute_dynamic_memories(
        self, memories: list[Any]
    ) -> list[str]:
        """Filter, decay, and sort dynamic memories by confidence.

        - Excludes is_forgotten=True memories
        - Computes decayed confidence at read time
        - Excludes memories below forgetting threshold
        - Sorts by decayed confidence descending
        - Returns content strings
        """
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, str]] = []

        for mem in memories:
            # Skip explicitly forgotten
            if getattr(mem, "is_forgotten", False):
                continue

            # Compute decayed confidence
            mem_type_str = getattr(mem, "memory_type", "fact")
            try:
                mem_type = MemoryType(mem_type_str)
            except ValueError:
                mem_type = MemoryType.inference  # default for unknown types

            initial_conf = getattr(mem, "confidence", 0.5)
            created_at = getattr(mem, "created_at", now)
            ref_count = getattr(mem, "source_count", 0)

            decayed = compute_memory_confidence(
                initial_confidence=initial_conf,
                memory_type=mem_type,
                created_at=created_at,
                now=now,
                ref_count=ref_count,
            )

            # Skip if below forgetting threshold
            if is_forgotten(decayed):
                continue

            scored.append((decayed, mem.content))

        # Sort by decayed confidence descending
        scored.sort(key=lambda x: -x[0])
        return [content for _, content in scored]

    async def _maybe_search(
        self, q: str | None, entity_id: str, space_id: str
    ) -> list[dict[str, Any]]:
        """Run search if query is provided and search engine is available."""
        if not q or not self._search_engine:
            return []
        try:
            return await self._search_engine.search(
                q=q, space_id=space_id, entity_id=entity_id, limit=5
            )
        except Exception:
            # Search failure should not break profile computation
            return []

    def invalidate_cache(self, entity_id: str, space_id: str) -> None:
        """Invalidate cached profile — call when graph data changes."""
        self._cache.invalidate(entity_id, space_id)

    def invalidate_cache_space(self, space_id: str) -> None:
        """Invalidate all cached profiles for a space."""
        self._cache.invalidate_space(space_id)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_engine.py -v`

**Expected:** All 10 tests pass.

- [ ] **5.3 Commit ProfileEngine**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/profile_engine.py server/tests/test_profile_engine.py
git commit -m "Add ProfileEngine — orchestrates graph reads, projection, decay, and cache"
```

---

## Task 6: Profile API Route — POST /v1/profile/

- [ ] **6.1 Write failing tests for profile API route**

**File:** `server/tests/test_profile_api.py`

```python
"""Tests for POST /v1/profile API route."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from server.app.core.profile_models import (
    ProfileData,
    IdentityLayer,
    ConfidenceMap,
    ProfileResponse,
)


class FakeEntity:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name


class FakeRelation:
    def __init__(
        self,
        relation_type: str,
        target_entity: FakeEntity,
        value: str | None = None,
        confidence: float = 0.9,
        is_current: bool = True,
        source_count: int = 0,
        updated_at: datetime | None = None,
    ):
        self.relation_type = relation_type
        self.target_entity = target_entity
        self.target_name = target_entity.name
        self.value = value
        self.confidence = confidence
        self.is_current = is_current
        self.source_count = source_count
        self.updated_at = updated_at or datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


class FakeMemory:
    def __init__(
        self,
        id: str,
        content: str,
        memory_type: str = "fact",
        confidence: float = 0.8,
        is_forgotten: bool = False,
        entity_id: str | None = None,
        created_at: datetime | None = None,
        source_count: int = 0,
    ):
        self.id = id
        self.content = content
        self.memory_type = memory_type
        self.confidence = confidence
        self.is_forgotten = is_forgotten
        self.entity_id = entity_id
        self.created_at = created_at or datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        self.source_count = source_count


def make_fake_graph_engine(entity, relations=None, memories=None):
    ge = AsyncMock()
    ge.get_entity = AsyncMock(return_value=entity)
    ge.get_current_relations = AsyncMock(return_value=relations or [])
    ge.get_recent_memories = AsyncMock(return_value=memories or [])
    return ge


def make_fake_search_engine(results=None):
    se = AsyncMock()
    se.search = AsyncMock(return_value=results or [])
    return se


def create_test_app(ge, se=None):
    from fastapi import FastAPI
    from server.app.api.profile import create_profile_router

    app = FastAPI()
    router = create_profile_router()
    app.include_router(router, prefix="/v1")

    # Override the engine dependency
    from server.app.api.profile import get_profile_engine
    from server.app.core.profile_engine import ProfileEngine

    engine = ProfileEngine(graph_engine=ge, search_engine=se)
    app.dependency_overrides[get_profile_engine] = lambda: engine
    return app


class TestProfileAPI:
    def test_profile_basic(self):
        entity = FakeEntity("ent_1", "张军")
        relations = [
            FakeRelation("lives_in", FakeEntity("e2", "北京"), confidence=0.95),
            FakeRelation("has_role", FakeEntity("e3", "全栈工程师"), confidence=0.88),
        ]
        ge = make_fake_graph_engine(entity=entity, relations=relations)
        app = create_test_app(ge)
        client = TestClient(app)

        resp = client.post("/v1/profile", json={
            "entity_id": "ent_1",
            "space_id": "sp_xxx",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["profile"]["identity"]["name"] == "张军"
        assert data["profile"]["identity"]["location"] == "北京"
        assert data["profile"]["identity"]["role"] == "全栈工程师"
        assert data["confidence"]["location"] == 0.95
        assert "computed_at" in data
        assert "timing_ms" in data

    def test_profile_with_search(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        se = make_fake_search_engine(results=[
            {"type": "memory", "content": "Next.js 15", "score": 0.9}
        ])
        app = create_test_app(ge, se)
        client = TestClient(app)

        resp = client.post("/v1/profile", json={
            "entity_id": "ent_1",
            "space_id": "sp_xxx",
            "q": "前端框架",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["search_results"]) == 1
        assert data["search_results"][0]["content"] == "Next.js 15"

    def test_profile_fresh_flag(self):
        entity = FakeEntity("ent_1", "张军")
        ge = make_fake_graph_engine(entity=entity)
        app = create_test_app(ge)
        client = TestClient(app)

        # First call
        resp1 = client.post("/v1/profile", json={
            "entity_id": "ent_1",
            "space_id": "sp_xxx",
        })
        assert resp1.status_code == 200

        # Second call with fresh=True
        resp2 = client.post("/v1/profile", json={
            "entity_id": "ent_1",
            "space_id": "sp_xxx",
            "fresh": True,
        })
        assert resp2.status_code == 200

        # Both should succeed
        assert resp1.json()["profile"]["identity"]["name"] == "张军"
        assert resp2.json()["profile"]["identity"]["name"] == "张军"

    def test_profile_entity_not_found(self):
        ge = make_fake_graph_engine(entity=None)
        app = create_test_app(ge)
        client = TestClient(app)

        resp = client.post("/v1/profile", json={
            "entity_id": "ent_missing",
            "space_id": "sp_xxx",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should return empty profile, not error
        assert data["profile"]["identity"]["name"] is None

    def test_profile_missing_entity_id_returns_422(self):
        ge = make_fake_graph_engine(entity=None)
        app = create_test_app(ge)
        client = TestClient(app)

        resp = client.post("/v1/profile", json={
            "space_id": "sp_xxx",
        })
        assert resp.status_code == 422

    def test_profile_with_dynamic_memories(self):
        entity = FakeEntity("ent_1", "张军")
        memories = [
            FakeMemory("m1", "最近在研究时序知识图谱", "status", confidence=0.7),
            FakeMemory("m2", "RTMemory 项目进行中", "status", confidence=0.6),
        ]
        ge = make_fake_graph_engine(entity=entity, memories=memories)
        app = create_test_app(ge)
        client = TestClient(app)

        resp = client.post("/v1/profile", json={
            "entity_id": "ent_1",
            "space_id": "sp_xxx",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["profile"]["dynamic_memories"]) >= 1

    def test_profile_full_response_shape(self):
        """Validate full response matches the spec shape."""
        entity = FakeEntity("ent_1", "张军")
        relations = [
            FakeRelation("lives_in", FakeEntity("e2", "北京"), confidence=0.95),
            FakeRelation("works_at", FakeEntity("e3", "ReTone"), confidence=0.85),
            FakeRelation("has_role", FakeEntity("e4", "全栈工程师"), confidence=0.88),
            FakeRelation("prefers", FakeEntity("e5", "Python"), confidence=0.9),
            FakeRelation("prefers", FakeEntity("e6", "TypeScript"), confidence=0.85),
            FakeRelation("prefers", FakeEntity("e7", "简洁"), value="style", confidence=0.7),
            FakeRelation("current_focus", FakeEntity("e8", "知识图谱"), confidence=0.7),
            FakeRelation("current_project", FakeEntity("e9", "RTMemory"), confidence=0.65),
            FakeRelation("knows", FakeEntity("e10", "李明"), confidence=0.6),
            FakeRelation("knows", FakeEntity("e11", "王芳"), confidence=0.55),
        ]
        ge = make_fake_graph_engine(entity=entity, relations=relations)
        app = create_test_app(ge)
        client = TestClient(app)

        resp = client.post("/v1/profile", json={
            "entity_id": "ent_1",
            "space_id": "sp_xxx",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Verify all four layers present
        assert "identity" in data["profile"]
        assert "preferences" in data["profile"]
        assert "current_status" in data["profile"]
        assert "relationships" in data["profile"]
        assert "dynamic_memories" in data["profile"]
        assert "confidence" in data
        assert "computed_at" in data
        assert "timing_ms" in data
        # Spot-check values
        assert data["profile"]["identity"]["name"] == "张军"
        assert data["profile"]["identity"]["location"] == "北京"
        assert data["profile"]["identity"]["role"] == "全栈工程师"
        assert "Python" in data["profile"]["preferences"]["stack"]
        assert data["profile"]["preferences"]["style"] == "简洁"
        assert data["profile"]["current_status"]["focus"] == "知识图谱"
        assert "李明" in data["profile"]["relationships"]["team"]
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_api.py -v 2>&1 | head -20`

**Expected:** Import error — module does not exist. Tests FAIL.

- [ ] **6.2 Implement profile API route**

**File:** `server/app/api/profile.py`

```python
"""Profile API route — POST /v1/profile.

Computes user profiles from the knowledge graph on demand.
Profile is NOT stored — computed and cached.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from server.app.core.profile_engine import ProfileEngine
from server.app.core.profile_models import ProfileRequest, ProfileResponse


# ── Dependency injection placeholder ──
# In production, these are wired in main.py with real engine instances.
# For tests, override via app.dependency_overrides.

_profile_engine: ProfileEngine | None = None


def set_profile_engine(engine: ProfileEngine) -> None:
    """Set the global ProfileEngine instance (called during app startup)."""
    global _profile_engine
    _profile_engine = engine


def get_profile_engine() -> ProfileEngine:
    """FastAPI dependency — returns the configured ProfileEngine."""
    if _profile_engine is None:
        raise HTTPException(status_code=503, detail="ProfileEngine not initialized")
    return _profile_engine


def create_profile_router() -> APIRouter:
    """Create and return the profile API router."""
    router = APIRouter(tags=["profile"])

    @router.post("/profile", response_model=ProfileResponse)
    async def get_profile(
        request: ProfileRequest,
        engine: ProfileEngine = Depends(get_profile_engine),
    ) -> ProfileResponse:
        """Compute and return the user profile for an entity.

        The profile is computed from the knowledge graph on demand and
        cached in memory. Use `fresh=True` to bypass the cache.

        If `q` is provided, a search is triggered and results are
        attached to the response.
        """
        result = await engine.compute(
            entity_id=request.entity_id,
            space_id=request.space_id,
            q=request.q,
            fresh=request.fresh,
        )
        return result

    return router
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_api.py -v`

**Expected:** All 7 tests pass.

- [ ] **6.3 Commit profile API route**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/api/profile.py server/tests/test_profile_api.py
git commit -m "Add POST /v1/profile API route — compute-on-demand with cache and search"
```

---

## Task 7: Wire ProfileEngine into App Startup and Invalidation Hooks

- [ ] **7.1 Write failing tests for app wiring**

**File:** `server/tests/test_profile_wiring.py`

```python
"""Tests for ProfileEngine wiring into FastAPI app startup."""
import pytest
from unittest.mock import AsyncMock

from server.app.core.profile_engine import ProfileEngine
from server.app.core.profile_cache import ProfileCache


class TestProfileEngineWiring:
    def test_profile_engine_requires_graph_engine(self):
        """ProfileEngine must have a graph_engine."""
        with pytest.raises(TypeError):
            ProfileEngine()  # missing required arg

    def test_profile_engine_optional_search_engine(self):
        """ProfileEngine works without a search engine."""
        ge = AsyncMock()
        engine = ProfileEngine(graph_engine=ge, search_engine=None)
        assert engine._search_engine is None

    def test_profile_engine_uses_provided_cache(self):
        """ProfileEngine uses the injected cache instance."""
        ge = AsyncMock()
        cache = ProfileCache()
        engine = ProfileEngine(graph_engine=ge, cache=cache)
        assert engine._cache is cache

    def test_profile_engine_creates_default_cache(self):
        """ProfileEngine creates its own cache if none provided."""
        ge = AsyncMock()
        engine = ProfileEngine(graph_engine=ge)
        assert engine._cache is not None
        assert isinstance(engine._cache, ProfileCache)

    def test_invalidate_cache_delegates(self):
        """ProfileEngine.invalidate_cache delegates to the cache."""
        ge = AsyncMock()
        cache = ProfileCache()
        engine = ProfileEngine(graph_engine=ge, cache=cache)
        # Manually put something in cache
        from server.app.core.profile_models import ProfileData, IdentityLayer, ConfidenceMap
        from datetime import datetime, timezone
        profile = ProfileData(identity=IdentityLayer(name="张军"))
        confidence = ConfidenceMap(location=0.9)
        cache.put("ent_1", "sp_xxx", profile, confidence, datetime(2026, 4, 23, tzinfo=timezone.utc), 10.0)
        assert cache.size() == 1
        engine.invalidate_cache("ent_1", "sp_xxx")
        assert cache.size() == 0

    def test_invalidate_cache_space_delegates(self):
        """ProfileEngine.invalidate_cache_space delegates to the cache."""
        ge = AsyncMock()
        cache = ProfileCache()
        engine = ProfileEngine(graph_engine=ge, cache=cache)
        from server.app.core.profile_models import ProfileData, IdentityLayer, ConfidenceMap
        from datetime import datetime, timezone
        profile = ProfileData(identity=IdentityLayer(name="张军"))
        confidence = ConfidenceMap(location=0.9)
        cache.put("ent_1", "sp_xxx", profile, confidence, datetime(2026, 4, 23, tzinfo=timezone.utc), 10.0)
        cache.put("ent_2", "sp_xxx", profile, confidence, datetime(2026, 4, 23, tzinfo=timezone.utc), 10.0)
        engine.invalidate_cache_space("sp_xxx")
        assert cache.size() == 0
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_wiring.py -v`

**Expected:** All 6 tests pass immediately (no external deps needed).

- [ ] **7.2 Add graph-change invalidation hook to ProfileEngine**

Add a method that can be called from GraphEngine when data changes. Edit `server/app/core/profile_engine.py` to add a convenience method for batch invalidation.

**File:** `server/app/core/profile_engine.py` — add after `invalidate_cache_space`:

```python
    async def on_graph_change(
        self,
        entity_ids: list[str],
        space_id: str,
    ) -> None:
        """Hook called when graph data changes.

        Invalidates cached profiles for all affected entities.
        Intended to be called by GraphEngine after entity/relation/memory
        mutations.

        Args:
            entity_ids: List of entity IDs whose data changed.
            space_id: The space the changes belong to.
        """
        for eid in entity_ids:
            self.invalidate_cache(eid, space_id)
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_wiring.py server/tests/test_profile_engine.py -v`

**Expected:** All 16 tests pass.

- [ ] **7.3 Commit wiring and invalidation hook**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/profile_engine.py server/tests/test_profile_wiring.py
git commit -m "Add graph-change invalidation hook and wiring tests for ProfileEngine"
```

---

## Task 8: Integration Test — End-to-End Profile Computation

- [ ] **8.1 Write integration test exercising the full pipeline**

**File:** `server/tests/test_profile_integration.py`

```python
"""Integration test — full profile computation pipeline."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from server.app.core.profile_engine import ProfileEngine
from server.app.core.profile_models import (
    ProfileData,
    IdentityLayer,
    PreferencesLayer,
    CurrentStatusLayer,
    RelationshipsLayer,
    ConfidenceMap,
    ProfileResponse,
    MemoryType,
    FORGETTING_THRESHOLD,
)
from server.app.core.confidence_decay import compute_decay, is_forgotten
from server.app.core.profile_cache import ProfileCache


class FakeEntity:
    def __init__(self, id: str, name: str, entity_type: str = "person"):
        self.id = id
        self.name = name
        self.entity_type = entity_type


class FakeRelation:
    def __init__(
        self,
        relation_type: str,
        target_entity: FakeEntity,
        value: str | None = None,
        confidence: float = 0.9,
        is_current: bool = True,
        source_count: int = 0,
        updated_at: datetime | None = None,
    ):
        self.relation_type = relation_type
        self.target_entity = target_entity
        self.target_name = target_entity.name
        self.value = value
        self.confidence = confidence
        self.is_current = is_current
        self.source_count = source_count
        self.updated_at = updated_at or datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


class FakeMemory:
    def __init__(
        self,
        id: str,
        content: str,
        memory_type: str = "fact",
        confidence: float = 0.8,
        decay_rate: float = 0.001,
        is_forgotten: bool = False,
        entity_id: str | None = None,
        created_at: datetime | None = None,
        source_count: int = 0,
    ):
        self.id = id
        self.content = content
        self.memory_type = memory_type
        self.confidence = confidence
        self.decay_rate = decay_rate
        self.is_forgotten = is_forgotten
        self.entity_id = entity_id
        self.created_at = created_at or datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        self.source_count = source_count


class TestFullProfilePipeline:
    """End-to-end test: graph data → profile computation → API response."""

    @pytest.mark.asyncio
    async def test_spec_example_profile(self):
        """Reproduce the spec example: 张军's full profile."""
        entity = FakeEntity("ent_1", "张军")
        relations = [
            FakeRelation("lives_in", FakeEntity("e2", "北京"), confidence=0.95),
            FakeRelation("works_at", FakeEntity("e3", "ReTone"), confidence=0.85),
            FakeRelation("has_role", FakeEntity("e4", "全栈工程师"), confidence=0.88),
            FakeRelation("prefers", FakeEntity("e5", "Python"), confidence=0.9),
            FakeRelation("prefers", FakeEntity("e6", "TypeScript"), confidence=0.85),
            FakeRelation("prefers", FakeEntity("e7", "简洁"), value="style", confidence=0.7),
            FakeRelation("current_focus", FakeEntity("e8", "知识图谱"), confidence=0.7),
            FakeRelation("current_project", FakeEntity("e9", "RTMemory"), confidence=0.65),
            FakeRelation("knows", FakeEntity("e10", "李明"), confidence=0.6),
            FakeRelation("knows", FakeEntity("e11", "王芳"), confidence=0.55),
        ]
        memories = [
            FakeMemory("m1", "最近在研究时序知识图谱", "status", confidence=0.7),
            FakeMemory("m2", "RTMemory 项目进行中", "status", confidence=0.6),
        ]

        ge = AsyncMock()
        ge.get_entity = AsyncMock(return_value=entity)
        ge.get_current_relations = AsyncMock(return_value=relations)
        ge.get_recent_memories = AsyncMock(return_value=memories)

        se = AsyncMock()
        se.search = AsyncMock(return_value=[])

        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")

        # Validate four layers
        assert result.profile.identity.name == "张军"
        assert result.profile.identity.location == "北京"
        assert result.profile.identity.role == "全栈工程师"
        assert result.profile.identity.company == "ReTone"

        assert "Python" in result.profile.preferences.stack
        assert "TypeScript" in result.profile.preferences.stack
        assert result.profile.preferences.style == "简洁"

        assert result.profile.current_status.focus == "知识图谱"
        assert result.profile.current_status.project == "RTMemory"

        assert "李明" in result.profile.relationships.team
        assert "王芳" in result.profile.relationships.team

        # Dynamic memories present
        assert len(result.profile.dynamic_memories) >= 1

        # Confidence map populated
        assert result.confidence.location is not None
        assert result.confidence.location == pytest.approx(0.95, abs=1e-6)

        # Timing and computed_at present
        assert result.timing_ms >= 0
        assert result.computed_at is not None

    @pytest.mark.asyncio
    async def test_confidence_decay_across_profile(self):
        """Verify confidence decay is applied consistently."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        entity = FakeEntity("ent_1", "张军")

        # An old inference memory
        memories = [
            FakeMemory(
                "m1", "旧推断", "inference",
                confidence=0.5,
                created_at=now - timedelta(days=180),
            ),
            FakeMemory(
                "m2", "新鲜事实", "fact",
                confidence=0.9,
                created_at=now - timedelta(days=1),
            ),
        ]

        ge = AsyncMock()
        ge.get_entity = AsyncMock(return_value=entity)
        ge.get_current_relations = AsyncMock(return_value=[])
        ge.get_recent_memories = AsyncMock(return_value=memories)

        engine = ProfileEngine(graph_engine=ge, search_engine=None)
        result = await engine.compute(entity_id="ent_1", space_id="sp_xxx")

        # Fresh fact should appear first (higher decayed confidence)
        if len(result.profile.dynamic_memories) >= 2:
            assert result.profile.dynamic_memories[0] == "新鲜事实"

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_graph_change(self):
        """After graph mutation, cache is invalidated and profile recomputes."""
        entity = FakeEntity("ent_1", "张军")
        relations_v1 = [
            FakeRelation("lives_in", FakeEntity("e2", "北京"), confidence=0.95),
        ]
        relations_v2 = [
            FakeRelation("lives_in", FakeEntity("e2", "上海"), confidence=0.9),
        ]

        ge = AsyncMock()
        ge.get_entity = AsyncMock(return_value=entity)
        ge.get_current_relations = AsyncMock(side_effect=[relations_v1, relations_v2])
        ge.get_recent_memories = AsyncMock(return_value=[])

        engine = ProfileEngine(graph_engine=ge, search_engine=None)

        # First compute
        r1 = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert r1.profile.identity.location == "北京"

        # Simulate graph change
        await engine.on_graph_change(["ent_1"], "sp_xxx")

        # Second compute — should re-read from graph
        r2 = await engine.compute(entity_id="ent_1", space_id="sp_xxx")
        assert r2.profile.identity.location == "上海"

    @pytest.mark.asyncio
    async def test_profile_without_search_engine(self):
        """Profile works fine when no search engine is configured."""
        entity = FakeEntity("ent_1", "张军")

        ge = AsyncMock()
        ge.get_entity = AsyncMock(return_value=entity)
        ge.get_current_relations = AsyncMock(return_value=[])
        ge.get_recent_memories = AsyncMock(return_value=[])

        engine = ProfileEngine(graph_engine=ge, search_engine=None)
        result = await engine.compute(
            entity_id="ent_1", space_id="sp_xxx", q="前端框架"
        )
        # No crash, search_results empty
        assert result.search_results == []

    @pytest.mark.asyncio
    async def test_profile_with_search_failure(self):
        """Search engine failure does not break profile computation."""
        entity = FakeEntity("ent_1", "张军")

        ge = AsyncMock()
        ge.get_entity = AsyncMock(return_value=entity)
        ge.get_current_relations = AsyncMock(return_value=[])
        ge.get_recent_memories = AsyncMock(return_value=[])

        se = AsyncMock()
        se.search = AsyncMock(side_effect=RuntimeError("search down"))

        engine = ProfileEngine(graph_engine=ge, search_engine=se)
        result = await engine.compute(
            entity_id="ent_1", space_id="sp_xxx", q="前端框架"
        )
        # Profile should still be returned
        assert result.profile.identity.name == "张军"
        assert result.search_results == []

    @pytest.mark.asyncio
    async def test_shared_cache_across_engine_instances(self):
        """Multiple ProfileEngine instances sharing the same cache."""
        entity = FakeEntity("ent_1", "张军")

        ge = AsyncMock()
        ge.get_entity = AsyncMock(return_value=entity)
        ge.get_current_relations = AsyncMock(return_value=[])
        ge.get_recent_memories = AsyncMock(return_value=[])

        shared_cache = ProfileCache()
        engine1 = ProfileEngine(graph_engine=ge, search_engine=None, cache=shared_cache)
        engine2 = ProfileEngine(graph_engine=ge, search_engine=None, cache=shared_cache)

        # Compute with engine1
        await engine1.compute(entity_id="ent_1", space_id="sp_xxx")

        # Engine2 should hit the cache
        r2 = await engine2.compute(entity_id="ent_1", space_id="sp_xxx")
        assert r2.profile.identity.name == "张军"
        # get_entity should only have been called once (engine2 hit cache)
        assert ge.get_entity.call_count == 1
```

**Run:** `cd /home/ubuntu/ReToneProjects/RTMemory && python -m pytest server/tests/test_profile_integration.py -v`

**Expected:** All 6 tests pass.

- [ ] **8.2 Commit integration tests**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/tests/test_profile_integration.py
git commit -m "Add end-to-end integration tests for profile computation pipeline"
```

---

## Task 9: Run Full Test Suite and Final Verification

- [ ] **9.1 Run all profile-related tests**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
python -m pytest server/tests/test_profile_models.py server/tests/test_confidence_decay.py server/tests/test_profile_projection.py server/tests/test_profile_cache.py server/tests/test_profile_engine.py server/tests/test_profile_api.py server/tests/test_profile_wiring.py server/tests/test_profile_integration.py -v
```

**Expected:** All tests pass (approximately 67 tests total).

- [ ] **9.2 Verify all files exist**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
ls -la server/app/core/profile_models.py
ls -la server/app/core/confidence_decay.py
ls -la server/app/core/profile_projection.py
ls -la server/app/core/profile_cache.py
ls -la server/app/core/profile_engine.py
ls -la server/app/api/profile.py
ls -la server/tests/test_profile_models.py
ls -la server/tests/test_confidence_decay.py
ls -la server/tests/test_profile_projection.py
ls -la server/tests/test_profile_cache.py
ls -la server/tests/test_profile_engine.py
ls -la server/tests/test_profile_api.py
ls -la server/tests/test_profile_wiring.py
ls -la server/tests/test_profile_integration.py
```

**Expected:** All 14 files exist.

- [ ] **9.3 Final commit with verification tag**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add -A
git commit -m "Complete Profile Engine implementation — four-layer model, confidence decay, caching, API"
```

---

## Summary of Deliverables

| File | Purpose |
|------|---------|
| `server/app/core/profile_models.py` | Pydantic models: IdentityLayer, PreferencesLayer, CurrentStatusLayer, RelationshipsLayer, ProfileData, ConfidenceMap, ProfileRequest, ProfileResponse, decay constants |
| `server/app/core/confidence_decay.py` | Confidence decay formula: `C(t) = C0 * e^(-λ * Δt) * (1 + α * log(n+1))`, forgetting threshold check |
| `server/app/core/profile_projection.py` | Configurable relation_type → profile_field mapping, sub-field overrides for prefers/knows |
| `server/app/core/profile_cache.py` | In-memory cache with (entity_id, space_id) keys, invalidation on graph change |
| `server/app/core/profile_engine.py` | Core engine: cache → graph read → projection → decay → search → cache. `on_graph_change()` hook |
| `server/app/api/profile.py` | POST /v1/profile route with FastAPI dependency injection |
| `server/tests/test_profile_models.py` | 11 tests for model serialization and defaults |
| `server/tests/test_confidence_decay.py` | 13 tests for decay formula and forgetting |
| `server/tests/test_profile_projection.py` | 16 tests for projection mapping |
| `server/tests/test_profile_cache.py` | 10 tests for cache operations |
| `server/tests/test_profile_engine.py` | 10 tests for engine orchestration |
| `server/tests/test_profile_api.py` | 7 tests for API route |
| `server/tests/test_profile_wiring.py` | 6 tests for wiring and invalidation |
| `server/tests/test_profile_integration.py` | 6 end-to-end integration tests |

**Total: ~79 tests across 8 test files, 6 source files**