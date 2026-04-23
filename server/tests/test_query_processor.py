"""Unit tests for query processor: entity recognition + optional LLM rewrite."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.query_processor import QueryProcessor, ProcessedQuery


class TestProcessedQuery:
    def test_processed_query_defaults(self):
        pq = ProcessedQuery(original="test query", rewritten=None, entity_ids=[])
        assert pq.original == "test query"
        assert pq.rewritten is None
        assert pq.entity_ids == []
        assert pq.effective_query == "test query"

    def test_processed_query_with_rewrite(self):
        pq = ProcessedQuery(original="short query", rewritten="expanded rewritten query", entity_ids=[])
        assert pq.effective_query == "expanded rewritten query"


class TestQueryProcessor:
    @pytest.mark.asyncio
    async def test_entity_recognition_finds_known_entities(self):
        """Entity recognition should match query terms against entity names in the DB."""
        processor = QueryProcessor(db_session=AsyncMock())

        row1 = MagicMock()
        row1.id = uuid.uuid4()
        row1.name = "test entity"
        row1.entity_type = "person"
        row2 = MagicMock()
        row2.id = uuid.uuid4()
        row2.name = "Next.js"
        row2.entity_type = "technology"

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [row1, row2]
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("test entity query", space_id=uuid.uuid4(), org_id=uuid.uuid4())
        assert len(pq.entity_ids) == 2

    @pytest.mark.asyncio
    async def test_entity_recognition_no_match(self):
        """No matching entities returns empty entity_ids."""
        processor = QueryProcessor(db_session=AsyncMock())
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("unrelated query", space_id=uuid.uuid4(), org_id=uuid.uuid4())
        assert len(pq.entity_ids) == 0

    @pytest.mark.asyncio
    async def test_query_rewrite_disabled_by_default(self):
        """Without rewrite_query=True, the query is not rewritten."""
        processor = QueryProcessor(db_session=AsyncMock(), llm_adapter=None)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("test query", space_id=uuid.uuid4(), org_id=uuid.uuid4(), rewrite_query=False)
        assert pq.rewritten is None
        assert pq.effective_query == "test query"

    @pytest.mark.asyncio
    async def test_query_rewrite_calls_llm(self):
        """With rewrite_query=True and LLM adapter present, query is rewritten."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "expanded rewritten query"

        processor = QueryProcessor(db_session=AsyncMock(), llm_adapter=mock_llm)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        processor._session.execute = AsyncMock(return_value=mock_result)

        pq = await processor.process("short query", space_id=uuid.uuid4(), org_id=uuid.uuid4(), rewrite_query=True)
        assert pq.rewritten == "expanded rewritten query"
        assert pq.effective_query == "expanded rewritten query"
        mock_llm.complete.assert_called_once()

    def test_extract_candidates(self):
        """Test that candidate extraction works with Chinese text."""
        processor = QueryProcessor(db_session=AsyncMock())
        candidates = processor._extract_candidates("张军最近在用 Next.js 做前端项目")
        assert len(candidates) > 0
        # Should contain multi-char tokens
        assert any("张军" in c or "Next" in c for c in candidates)

    def test_extract_candidates_punctuation(self):
        """Test that punctuation is stripped from candidates."""
        processor = QueryProcessor(db_session=AsyncMock())
        candidates = processor._extract_candidates("hello, world! how are you?")
        # Short single-char tokens should be filtered out (need >= 2 chars)
        for c in candidates:
            assert len(c) >= 2