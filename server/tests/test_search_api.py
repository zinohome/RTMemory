"""API-level integration tests for POST /v1/search/."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.search import SearchResponse, SearchTiming


@pytest.fixture
def search_request_payload():
    return {
        "q": "test query",
        "space_id": str(uuid.uuid4()),
        "mode": "hybrid",
        "limit": 10,
        "include_profile": False,
        "rewrite_query": False,
    }


class TestSearchAPI:
    @pytest.mark.asyncio
    async def test_search_endpoint_returns_200(self, search_request_payload):
        """POST /v1/search/ should return 200 with valid payload."""
        org_id = uuid.uuid4()
        mock_response = SearchResponse(
            results=[],
            timing=SearchTiming(total_ms=0.0),
            query="test query",
        )

        mock_embedding = AsyncMock()
        mock_llm = None

        with patch("app.api.search.SearchEngine") as MockEngine, \
             patch("app.api.search._get_embedding_service", return_value=mock_embedding), \
             patch("app.api.search._get_llm_adapter", return_value=mock_llm), \
             patch("app.api.search._resolve_org_id", return_value=org_id):
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            MockEngine.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/search/", json=search_request_payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "timing" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_search_endpoint_validates_required_fields(self):
        """POST /v1/search/ without required 'q' should return 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/search/", json={"space_id": str(uuid.uuid4())})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_endpoint_with_channels(self, search_request_payload):
        """POST /v1/search/ with channels parameter should be accepted."""
        org_id = uuid.uuid4()
        search_request_payload["channels"] = ["vector", "keyword"]
        mock_response = SearchResponse(
            results=[],
            timing=SearchTiming(total_ms=0.0, vector_ms=1.0, graph_ms=None, keyword_ms=2.0, fusion_ms=0.1),
            query="test query",
        )

        mock_embedding = AsyncMock()
        mock_llm = None

        with patch("app.api.search.SearchEngine") as MockEngine, \
             patch("app.api.search._get_embedding_service", return_value=mock_embedding), \
             patch("app.api.search._get_llm_adapter", return_value=mock_llm), \
             patch("app.api.search._resolve_org_id", return_value=org_id):
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            MockEngine.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/search/", json=search_request_payload)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_endpoint_with_filters(self, search_request_payload):
        """POST /v1/search/ with metadata filters should be accepted."""
        org_id = uuid.uuid4()
        search_request_payload["filters"] = {"AND": [{"key": "source", "value": "slack"}]}
        mock_response = SearchResponse(
            results=[],
            timing=SearchTiming(total_ms=0.0),
            query="test query",
        )

        mock_embedding = AsyncMock()
        mock_llm = None

        with patch("app.api.search.SearchEngine") as MockEngine, \
             patch("app.api.search._get_embedding_service", return_value=mock_embedding), \
             patch("app.api.search._get_llm_adapter", return_value=mock_llm), \
             patch("app.api.search._resolve_org_id", return_value=org_id):
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            MockEngine.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/search/", json=search_request_payload)

        assert response.status_code == 200