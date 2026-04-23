"""Integration tests for the SearchEngine orchestrator."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.search_engine import SearchEngine
from app.core.search_channels import ChannelResult
from app.core.search_fusion import FusedResult
from app.core.query_processor import ProcessedQuery
from app.schemas.search import SearchRequest, SearchMode, SearchChannel


def _make_processed_query(query="test", entity_ids=None):
    """Create a ProcessedQuery for mocking."""
    return ProcessedQuery(
        original=query,
        rewritten=None,
        entity_ids=entity_ids or [],
    )


class TestSearchEngine:
    @pytest.mark.asyncio
    async def test_hybrid_search_calls_all_channels(self):
        """Hybrid mode with default channels should call vector, graph, and keyword."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [[0.1] * 768]

        processed = _make_processed_query("test query")
        with patch.object(engine, "_query_processor") as mock_qp, \
             patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)) as mock_vec, \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)) as mock_graph, \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)) as mock_kw:
            mock_qp.process = AsyncMock(return_value=processed)

            request = SearchRequest(q="test query", space_id=uuid.uuid4())
            response = await engine.search(request, org_id=uuid.uuid4())

            mock_vec.assert_called_once()
            mock_graph.assert_called_once()
            mock_kw.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_only_mode_skips_document_channels(self):
        """memory_only mode should filter results to memory and entity types only."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [[0.1] * 768]

        processed = _make_processed_query("test")
        with patch.object(engine, "_query_processor") as mock_qp, \
             patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)), \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)), \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)):
            mock_qp.process = AsyncMock(return_value=processed)

            request = SearchRequest(q="test", space_id=uuid.uuid4(), mode=SearchMode.memory_only)
            response = await engine.search(request, org_id=uuid.uuid4())

    @pytest.mark.asyncio
    async def test_search_includes_timing(self):
        """Search response must include timing breakdown per channel."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [[0.1] * 768]

        processed = _make_processed_query("test")
        with patch.object(engine, "_query_processor") as mock_qp, \
             patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=12.0)), \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=5.0)), \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=8.0)):
            mock_qp.process = AsyncMock(return_value=processed)

            request = SearchRequest(q="test", space_id=uuid.uuid4())
            response = await engine.search(request, org_id=uuid.uuid4())
            assert response.timing.total_ms > 0
            assert response.timing.vector_ms is not None

    @pytest.mark.asyncio
    async def test_channel_filter_vector_only(self):
        """Requesting only the vector channel should skip graph and keyword."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [[0.1] * 768]

        processed = _make_processed_query("test")
        with patch.object(engine, "_query_processor") as mock_qp, \
             patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)) as mock_vec, \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)) as mock_graph, \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)) as mock_kw:
            mock_qp.process = AsyncMock(return_value=processed)

            request = SearchRequest(q="test", space_id=uuid.uuid4(), channels=[SearchChannel.vector])
            response = await engine.search(request, org_id=uuid.uuid4())

            mock_vec.assert_called_once()
            mock_graph.assert_not_called()
            mock_kw.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_boost_applied_when_user_id(self):
        """When user_id is provided, profile boost should be applied."""
        engine = SearchEngine(
            db_session=AsyncMock(),
            embedding_service=AsyncMock(),
            llm_adapter=None,
        )
        engine._embedding_service.embed.return_value = [[0.1] * 768]

        item_id = uuid.uuid4()
        entity_id = uuid.uuid4()
        fused_result = FusedResult(
            id=item_id,
            rrf_score=0.05,
            content="test",
            type="memory",
            entity_id=entity_id,
            source_channels=["vector"],
        )

        processed = _make_processed_query("test")
        with patch.object(engine, "_query_processor") as mock_qp, \
             patch.object(engine, "_run_vector_search", return_value=ChannelResult(items=[], channel="vector", timing_ms=1.0)), \
             patch.object(engine, "_run_graph_search", return_value=ChannelResult(items=[], channel="graph", timing_ms=1.0)), \
             patch.object(engine, "_run_keyword_search", return_value=ChannelResult(items=[], channel="keyword", timing_ms=1.0)), \
             patch.object(engine, "_get_user_entity_id", return_value=entity_id) as mock_get_user, \
             patch.object(engine, "_get_user_preference_ids", return_value=[]) as mock_get_prefs, \
             patch("app.core.search_engine.reciprocal_rank_fusion", return_value=[fused_result]) as mock_rrf, \
             patch("app.core.search_engine.apply_profile_boost", return_value=[fused_result]) as mock_boost:
            mock_qp.process = AsyncMock(return_value=processed)

            request = SearchRequest(q="test", space_id=uuid.uuid4(), user_id=uuid.uuid4())
            response = await engine.search(request, org_id=uuid.uuid4())

            mock_boost.assert_called_once()