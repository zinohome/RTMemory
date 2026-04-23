"""Unit tests for the three search channels: vector, graph, keyword."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.search_channels import (
    VectorSearchChannel,
    GraphSearchChannel,
    KeywordSearchChannel,
    ChannelResult,
)


# ── ChannelResult dataclass ────────────────────────────────────


class TestChannelResult:
    def test_channel_result_creation(self):
        cr = ChannelResult(
            items=[{"id": "a", "score": 0.9}],
            channel="vector",
            timing_ms=12.5,
        )
        assert cr.channel == "vector"
        assert len(cr.items) == 1
        assert cr.timing_ms == 12.5


# ── Vector search channel ─────────────────────────────────────


class TestVectorSearchChannel:
    @pytest.mark.asyncio
    async def test_vector_search_memories(self):
        """Vector search across memories should produce ChannelResult with similarity scores."""
        channel = VectorSearchChannel(db_session=AsyncMock())

        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.type = "memory"
        row1.content = "test content"
        row1.similarity = 0.92
        row1.entity_id = uuid.uuid4()
        row1.entity_name = "test entity"
        row1.entity_type = "person"
        row1.metadata_ = None
        row1.created_at = None

        # MagicMock (sync) for result since fetchall() is sync in SQLAlchemy
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [row1]
        channel._session.execute = AsyncMock(return_value=mock_result)

        query_vec = [0.1] * 768
        result = await channel.search(query_vec=query_vec, space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)

        assert isinstance(result, ChannelResult)
        assert result.channel == "vector"
        assert len(result.items) >= 1
        assert result.items[0]["score"] == 0.92
        assert result.timing_ms >= 0

    @pytest.mark.asyncio
    async def test_vector_search_empty_results(self):
        """Vector search with no matches returns empty items list."""
        channel = VectorSearchChannel(db_session=AsyncMock())
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(query_vec=[0.0] * 768, space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)

        assert len(result.items) == 0


# ── Graph traversal channel ────────────────────────────────────


class TestGraphSearchChannel:
    @pytest.mark.asyncio
    async def test_graph_search_finds_related_entities(self):
        """Graph search from a seed entity should traverse and return related items."""
        channel = GraphSearchChannel(db_session=AsyncMock())

        seed_entity_id = uuid.uuid4()
        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.name = "Beijing"
        row1.entity_type = "location"
        row1.description = "Capital of China"
        row1.depth = 1
        row1.relation_type = "lives_in"
        row1.confidence = 0.95
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [row1]
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(seed_entity_ids=[seed_entity_id], space_id=uuid.uuid4(), org_id=uuid.uuid4(), max_depth=3, limit=10)

        assert result.channel == "graph"
        assert len(result.items) >= 1
        assert result.items[0]["depth"] == 1

    @pytest.mark.asyncio
    async def test_graph_search_no_seed_entities(self):
        """Graph search with no seed entities returns empty."""
        channel = GraphSearchChannel(db_session=AsyncMock())
        result = await channel.search(seed_entity_ids=[], space_id=uuid.uuid4(), org_id=uuid.uuid4(), max_depth=3, limit=10)
        assert len(result.items) == 0


# ── Keyword search channel ─────────────────────────────────────


class TestKeywordSearchChannel:
    @pytest.mark.asyncio
    async def test_keyword_search_finds_matches(self):
        """Keyword tsvector search should return rows ranked by ts_rank."""
        channel = KeywordSearchChannel(db_session=AsyncMock())

        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.type = "memory"
        row1.content = "test content"
        row1.rank = 0.3
        row1.entity_id = None
        row1.entity_name = None
        row1.entity_type = None
        row1.metadata_ = None
        row1.created_at = None
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [row1]
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(query_text="test", space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)

        assert result.channel == "keyword"
        assert len(result.items) >= 1

    @pytest.mark.asyncio
    async def test_keyword_search_empty(self):
        """Keyword search with no matches returns empty."""
        channel = KeywordSearchChannel(db_session=AsyncMock())
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        channel._session.execute = AsyncMock(return_value=mock_result)

        result = await channel.search(query_text="nonexistent", space_id=uuid.uuid4(), org_id=uuid.uuid4(), limit=10)
        assert len(result.items) == 0


# ── Metadata filter builder ────────────────────────────────────


class TestMetadataFilter:
    def test_build_and_filter(self):
        channel = VectorSearchChannel(db_session=AsyncMock())
        clause, params = channel._build_metadata_filter_clause({"AND": [{"key": "source", "value": "slack"}]})
        assert "AND" in clause
        assert "metadata" in clause

    def test_build_or_filter(self):
        channel = VectorSearchChannel(db_session=AsyncMock())
        clause, params = channel._build_metadata_filter_clause({"OR": [{"key": "source", "value": "slack"}, {"key": "source", "value": "email"}]})
        assert "OR" in clause

    def test_empty_filter(self):
        channel = VectorSearchChannel(db_session=AsyncMock())
        clause, params = channel._build_metadata_filter_clause(None)
        assert clause == ""
        assert params == {}