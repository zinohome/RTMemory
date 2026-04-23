"""Unit tests for RRF fusion and Profile Boost."""

from __future__ import annotations

import uuid

import pytest

from app.core.search_fusion import reciprocal_rank_fusion, apply_profile_boost, FusedResult


class TestReciprocalRankFusion:
    def test_rrf_basic_two_channels(self):
        """RRF with two channels: items appearing in both get higher scores."""
        vector_results = [
            {"id": uuid.uuid4(), "score": 0.95, "content": "a", "type": "memory"},
            {"id": uuid.uuid4(), "score": 0.80, "content": "b", "type": "memory"},
        ]
        keyword_results = [
            {"id": vector_results[0]["id"], "score": 0.5, "content": "a", "type": "memory"},
            {"id": uuid.uuid4(), "score": 0.4, "content": "c", "type": "entity"},
        ]

        channel_map = {"vector": vector_results, "keyword": keyword_results}
        fused = reciprocal_rank_fusion(channel_map, k=60)

        assert len(fused) == 3
        top_id = fused[0].id
        assert top_id == vector_results[0]["id"]

    def test_rrf_single_channel(self):
        """RRF with one channel still works — scores = 1/(k+rank+1)."""
        items = [
            {"id": uuid.uuid4(), "score": 0.9, "content": "a", "type": "memory"},
            {"id": uuid.uuid4(), "score": 0.5, "content": "b", "type": "entity"},
        ]
        fused = reciprocal_rank_fusion({"vector": items}, k=60)
        assert len(fused) == 2
        assert abs(fused[0].rrf_score - 1.0 / 61.0) < 1e-9

    def test_rrf_empty_channels(self):
        """RRF with no channels returns empty."""
        fused = reciprocal_rank_fusion({}, k=60)
        assert len(fused) == 0

    def test_rrf_k_parameter(self):
        """RRF score formula: score += 1/(k + rank + 1), k=60."""
        items = [{"id": uuid.uuid4(), "score": 0.9, "content": "x", "type": "memory"}]
        fused = reciprocal_rank_fusion({"vector": items}, k=60)
        assert abs(fused[0].rrf_score - 1.0 / 61.0) < 1e-9

    def test_rrf_k_60_three_channels(self):
        """Item in all 3 channels: score = 3 * 1/(60+0+1) when ranked first in each."""
        item_id = uuid.uuid4()
        ch = {
            "vector": [{"id": item_id, "score": 0.9, "content": "x", "type": "memory"}],
            "graph": [{"id": item_id, "score": 0.7, "content": "x", "type": "memory"}],
            "keyword": [{"id": item_id, "score": 0.5, "content": "x", "type": "memory"}],
        }
        fused = reciprocal_rank_fusion(ch, k=60)
        assert len(fused) == 1
        expected = 3.0 / 61.0
        assert abs(fused[0].rrf_score - expected) < 1e-9


class TestProfileBoost:
    def test_entity_match_boost(self):
        """Results matching user entity get x1.5 boost."""
        user_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=user_entity_id, source_channels=["vector"]),
            FusedResult(id=uuid.uuid4(), rrf_score=0.04, content="b", type="memory", entity_id=uuid.uuid4(), source_channels=["vector"]),
        ]

        boosted = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=[])
        assert boosted[0].boosted_score == pytest.approx(0.05 * 1.5, rel=1e-6)
        assert boosted[1].boosted_score == pytest.approx(0.04, rel=1e-6)

    def test_preference_match_boost(self):
        """Results matching user preferences get x1.2 boost."""
        pref_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=pref_entity_id, source_channels=["vector"]),
        ]

        boosted = apply_profile_boost(fused, user_entity_id=uuid.uuid4(), user_preference_entity_ids=[pref_entity_id])
        assert boosted[0].boosted_score == pytest.approx(0.05 * 1.2, rel=1e-6)

    def test_entity_and_preference_boost_stacks(self):
        """Entity x1.5 and preference x1.2 are multiplicative: 1.5 * 1.2 = 1.8."""
        user_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=user_entity_id, source_channels=["vector"]),
        ]

        boosted = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=[user_entity_id])
        assert boosted[0].boosted_score == pytest.approx(0.05 * 1.5 * 1.2, rel=1e-6)

    def test_no_boost_without_user(self):
        """No user_entity_id means no boost applied."""
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="a", type="memory", entity_id=uuid.uuid4(), source_channels=["vector"]),
        ]
        boosted = apply_profile_boost(fused, user_entity_id=None, user_preference_entity_ids=[])
        assert boosted[0].boosted_score == pytest.approx(0.05, rel=1e-6)

    def test_boosted_results_sorted(self):
        """Results should be re-sorted after profile boost."""
        user_entity_id = uuid.uuid4()
        fused = [
            FusedResult(id=uuid.uuid4(), rrf_score=0.03, content="low", type="memory", entity_id=user_entity_id, source_channels=["vector"]),
            FusedResult(id=uuid.uuid4(), rrf_score=0.05, content="high", type="memory", entity_id=uuid.uuid4(), source_channels=["vector"]),
        ]
        boosted = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=[])
        assert boosted[0].boosted_score >= boosted[1].boosted_score