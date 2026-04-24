"""API integration tests for /v1/tasks/ — background task status endpoints."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.worker import Worker


@pytest.fixture
async def worker_client():
    """Set up a Worker and wire it into the app, then provide an HTTP client."""
    worker = Worker(max_concurrent=2)
    worker.start()

    # Wire worker into the app
    from app.api.tasks import set_worker
    set_worker(worker)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, worker

    # Cleanup
    await worker.stop()
    set_worker(None)  # Reset global


class TestTasksAPI:
    """Tests for the Tasks API endpoints."""

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, worker_client):
        client, worker = worker_client
        response = await client.get("/v1/tasks/")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_submit_and_get_task(self, worker_client):
        client, worker = worker_client

        async def handler(payload):
            return {"result": payload.get("input", "default")}

        worker.register("test_task", handler)

        task_id = await worker.submit("test_task", {"input": "hello"})

        # Wait for completion
        for _ in range(50):
            task = worker.get_task(task_id)
            if task and task.status.value in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        response = await client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == task_id
        assert data["task_type"] == "test_task"
        assert data["status"] == "completed"
        assert data["result"] == {"result": "hello"}
        assert data["started_at"] is not None
        assert data["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, worker_client):
        client, worker = worker_client
        response = await client.get("/v1/tasks/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_tasks_with_filter(self, worker_client):
        client, worker = worker_client

        async def handler_a(payload):
            return {}

        async def handler_b(payload):
            return {}

        worker.register("type_a", handler_a)
        worker.register("type_b", handler_b)

        await worker.submit("type_a", {})
        await worker.submit("type_b", {})
        await worker.submit("type_a", {})

        # Wait for tasks to complete
        await asyncio.sleep(0.3)

        response = await client.get("/v1/tasks/", params={"task_type": "type_a"})
        assert response.status_code == 200
        data = response.json()
        assert all(item["task_type"] == "type_a" for item in data["items"])
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_tasks_with_limit(self, worker_client):
        client, worker = worker_client

        async def handler(payload):
            return {}

        worker.register("limit_test", handler)

        for i in range(5):
            await worker.submit("limit_test", {"i": i})

        await asyncio.sleep(0.3)

        response = await client.get("/v1/tasks/", params={"limit": 2})
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 2

    @pytest.mark.asyncio
    async def test_task_error_response(self, worker_client):
        client, worker = worker_client

        async def failing_handler(payload):
            raise ValueError("Processing error!")

        worker.register("failing", failing_handler)

        task_id = await worker.submit("failing", {})

        for _ in range(50):
            task = worker.get_task(task_id)
            if task and task.status.value in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        response = await client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "Processing error!" in data["error"]
        # Failed tasks should NOT have a result key
        assert "result" not in data

    @pytest.mark.asyncio
    async def test_worker_not_initialized_returns_503(self):
        """When worker is not set, the API should return 503."""
        from app.api.tasks import set_worker
        set_worker(None)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/tasks/")
        assert response.status_code == 503