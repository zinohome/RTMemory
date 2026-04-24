"""End-to-end API smoke tests — verify all routes are registered and respond correctly."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    """Provide an async HTTP client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestRootEndpoints:
    """Test root and health endpoints."""

    async def test_root(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RTMemory"
        assert "version" in data

    async def test_health(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestRouteRegistration:
    """Verify that all expected routes are registered in the app."""

    def test_all_routes_registered(self):
        """Check that all API route prefixes are present."""
        route_paths = [route.path for route in app.routes if hasattr(route, "path")]

        expected_prefixes = [
            "/v1/spaces",
            "/v1/entities",
            "/v1/relations",
            "/v1/memories",
            "/v1/search",
            "/v1/conversations",
            "/v1/documents",
            "/profile",
            "/v1/graph",
            "/v1/tasks",
        ]

        for prefix in expected_prefixes:
            matching = [p for p in route_paths if p.startswith(prefix)]
            assert len(matching) > 0, f"Route prefix {prefix} not found in registered routes. Routes: {route_paths}"

    def test_graph_neighborhood_route(self):
        """Check that the graph neighborhood route is registered."""
        route_paths = [route.path for route in app.routes if hasattr(route, "path")]
        assert any("/v1/graph/neighborhood" in p for p in route_paths)

    def test_tasks_routes(self):
        """Check that tasks routes are registered."""
        route_paths = [route.path for route in app.routes if hasattr(route, "path")]
        assert any("/v1/tasks/" in p for p in route_paths)
        assert any("/v1/tasks/{task_id}" in p for p in route_paths)


class TestWorkerIntegration:
    """Verify that the Worker is wired into the app lifecycle."""

    def test_worker_instance_exists(self):
        """The app should have a Worker instance."""
        from app.main import worker
        assert worker is not None

    def test_worker_registered_in_tasks_api(self):
        """The tasks API should have access to the worker."""
        from app.api.tasks import get_worker
        w = get_worker()
        assert w is not None


class TestAPIValidation:
    """Test API request validation for key endpoints."""

    async def test_search_without_query(self, client):
        """POST /v1/search/ without required 'q' should return 422."""
        response = await client.post("/v1/search/", json={})
        assert response.status_code == 422

    async def test_documents_list_without_space_id(self, client):
        """GET /v1/documents/ without space_id should return 422."""
        response = await client.get("/v1/documents/")
        assert response.status_code == 422

    async def test_graph_neighborhood_without_entity_id(self, client):
        """GET /v1/graph/neighborhood without entity_id should return 422."""
        response = await client.get("/v1/graph/neighborhood")
        assert response.status_code == 422