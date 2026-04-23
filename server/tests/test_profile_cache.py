"""Tests for profile cache with graph-change invalidation."""
import pytest
from datetime import datetime, timezone

from app.core.profile_cache import ProfileCache
from app.core.profile_models import (
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