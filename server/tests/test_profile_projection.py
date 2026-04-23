"""Tests for profile projection — mapping relations to profile fields."""
import pytest
from datetime import datetime, timezone

from app.core.profile_projection import (
    ProfileProjectionConfig,
    default_projection_config,
    project_relations,
)
from app.core.profile_models import (
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