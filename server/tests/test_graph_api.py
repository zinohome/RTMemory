"""API integration tests for /v1/graph/neighborhood — graph visualization endpoint."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.graph import (
    EntityOut,
    EntityType,
    RelationOut,
    TraversedRelationOut,
    GraphTraversalOut,
)


def _make_traversal_out(center_id, other_id, space_id):
    """Build a proper GraphTraversalOut for testing."""
    now = datetime.now(timezone.utc)

    center_entity = EntityOut(
        id=center_id, name="Alice", entity_type=EntityType.person,
        description="A person", confidence=1.0,
        org_id=uuid.uuid4(), space_id=space_id,
        created_at=now, updated_at=now,
    )
    other_entity = EntityOut(
        id=other_id, name="Bob", entity_type=EntityType.person,
        description="Another person", confidence=0.9,
        org_id=uuid.uuid4(), space_id=space_id,
        created_at=now, updated_at=now,
    )
    relation = RelationOut(
        id=uuid.uuid4(), source_entity_id=center_id, target_entity_id=other_id,
        relation_type="knows", value="friend", valid_from=now, valid_to=None,
        confidence=0.85, is_current=True, source_count=1,
        org_id=uuid.uuid4(), space_id=space_id, created_at=now, updated_at=now,
    )
    traversed = TraversedRelationOut(relation=relation, hop=1, direction="outgoing")

    return GraphTraversalOut(
        start_entity_id=center_id,
        entities=[center_entity, other_entity],
        relations=[traversed],
        max_hops=2,
    )


class TestGraphNeighborhoodAPI:
    """Tests for GET /v1/graph/neighborhood."""

    @pytest.mark.asyncio
    async def test_neighborhood_returns_graph_data(self):
        """GET /v1/graph/neighborhood should return nodes and edges in D3 format."""
        entity_id = uuid.uuid4()
        other_id = uuid.uuid4()
        space_id = uuid.uuid4()

        mock_engine = AsyncMock()
        mock_engine.traverse_graph.return_value = _make_traversal_out(
            entity_id, other_id, space_id
        )

        # Mock the _get_engine dependency to return our mock engine
        async def _override_engine():
            return mock_engine

        from app.api.graph import _get_engine as _graph_engine_dep
        app.dependency_overrides[_graph_engine_dep] = _override_engine

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/graph/neighborhood",
                    params={"entity_id": str(entity_id), "max_hops": 2},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["center"] == str(entity_id)
        assert len(data["nodes"]) == 2
        assert data["nodes"][0]["label"] == "Alice"
        assert data["nodes"][1]["label"] == "Bob"
        assert len(data["edges"]) == 1
        assert data["edges"][0]["label"] == "knows"
        assert data["edges"][0]["source"] == str(entity_id)
        assert data["edges"][0]["target"] == str(other_id)
        assert data["maxHops"] == 2

    @pytest.mark.asyncio
    async def test_neighborhood_entity_not_found(self):
        """GET /v1/graph/neighborhood with non-existent entity should return 404."""
        entity_id = uuid.uuid4()
        mock_engine = AsyncMock()
        mock_engine.traverse_graph.side_effect = ValueError("Entity not found")

        async def _override_engine():
            return mock_engine

        from app.api.graph import _get_engine as _graph_engine_dep
        app.dependency_overrides[_graph_engine_dep] = _override_engine

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/graph/neighborhood",
                    params={"entity_id": str(entity_id)},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_neighborhood_invalid_entity_id(self):
        """GET /v1/graph/neighborhood with invalid UUID should return 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/v1/graph/neighborhood",
                params={"entity_id": "not-a-uuid"},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_neighborhood_max_hops_bounds(self):
        """GET /v1/graph/neighborhood with max_hops > 10 should return 422."""
        entity_id = uuid.uuid4()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/v1/graph/neighborhood",
                params={"entity_id": str(entity_id), "max_hops": 15},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_neighborhood_invalid_direction(self):
        """GET /v1/graph/neighborhood with invalid direction should return 400."""
        entity_id = uuid.uuid4()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/v1/graph/neighborhood",
                params={"entity_id": str(entity_id), "direction": "invalid"},
            )
        assert response.status_code == 400