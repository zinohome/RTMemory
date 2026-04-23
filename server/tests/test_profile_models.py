"""Tests for profile Pydantic models."""
import pytest
from datetime import datetime

from app.core.profile_models import (
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