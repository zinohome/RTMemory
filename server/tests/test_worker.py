"""Tests for the async Worker — task submission, concurrency, and status tracking."""

import asyncio

import pytest

from app.worker import Task, TaskStatus, Worker


class TestTask:
    """Tests for the Task dataclass."""

    def test_task_defaults(self):
        task = Task(id="t1", task_type="test", payload={"key": "val"})
        assert task.status == TaskStatus.queued
        assert task.result is None
        assert task.error is None
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_status_values(self):
        assert TaskStatus.queued == "queued"
        assert TaskStatus.running == "running"
        assert TaskStatus.completed == "completed"
        assert TaskStatus.failed == "failed"


class TestWorker:
    """Tests for the async Worker."""

    def test_register_handler(self):
        worker = Worker()
        async def handler(payload): return {}
        worker.register("test_type", handler)
        assert "test_type" in worker._handlers

    def test_register_overwrite_handler(self):
        worker = Worker()
        async def handler1(payload): return {"v": 1}
        async def handler2(payload): return {"v": 2}
        worker.register("test", handler1)
        worker.register("test", handler2)
        assert worker._handlers["test"] is handler2

    def test_submit_unknown_type_raises(self):
        worker = Worker()
        with pytest.raises(ValueError, match="Unknown task type"):
            asyncio.get_event_loop().run_until_complete(
                worker.submit("nonexistent", {})
            )

    @pytest.mark.asyncio
    async def test_submit_and_complete_task(self):
        worker = Worker(max_concurrent=2)
        worker.start()

        async def handler(payload):
            return {"processed": payload.get("data", "")}

        worker.register("process", handler)

        task_id = await worker.submit("process", {"data": "hello"})
        assert task_id is not None

        # Wait for task to complete
        for _ in range(50):
            task = worker.get_task(task_id)
            if task.status in (TaskStatus.completed, TaskStatus.failed):
                break
            await asyncio.sleep(0.05)

        task = worker.get_task(task_id)
        assert task.status == TaskStatus.completed
        assert task.result == {"processed": "hello"}
        assert task.started_at is not None
        assert task.completed_at is not None

        await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_failing_task(self):
        worker = Worker(max_concurrent=2)
        worker.start()

        async def handler(payload):
            raise RuntimeError("Task failed!")

        worker.register("fail_type", handler)

        task_id = await worker.submit("fail_type", {})
        for _ in range(50):
            task = worker.get_task(task_id)
            if task.status in (TaskStatus.completed, TaskStatus.failed):
                break
            await asyncio.sleep(0.05)

        task = worker.get_task(task_id)
        assert task.status == TaskStatus.failed
        assert "Task failed!" in task.error
        assert task.completed_at is not None

        await worker.stop()

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Multiple tasks should execute concurrently up to max_concurrent."""
        worker = Worker(max_concurrent=4)
        worker.start()

        execution_order = []

        async def handler(payload):
            execution_order.append(payload["name"])
            await asyncio.sleep(0.05)
            return {"name": payload["name"]}

        worker.register("concurrent", handler)

        task_ids = []
        for i in range(4):
            tid = await worker.submit("concurrent", {"name": f"task_{i}"})
            task_ids.append(tid)

        # Wait for all tasks
        for _ in range(100):
            all_done = all(
                worker.get_task(tid).status in (TaskStatus.completed, TaskStatus.failed)
                for tid in task_ids
            )
            if all_done:
                break
            await asyncio.sleep(0.05)

        # All 4 tasks should have completed
        for tid in task_ids:
            task = worker.get_task(tid)
            assert task.status == TaskStatus.completed

        await worker.stop()

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Tasks beyond max_concurrent should wait for a slot."""
        worker = Worker(max_concurrent=1)
        worker.start()

        running_count = 0
        max_concurrent_seen = 0

        async def handler(payload):
            nonlocal running_count, max_concurrent_seen
            running_count += 1
            max_concurrent_seen = max(max_concurrent_seen, running_count)
            await asyncio.sleep(0.1)
            running_count -= 1
            return {}

        worker.register("limited", handler)

        task_ids = []
        for i in range(3):
            tid = await worker.submit("limited", {"i": i})
            task_ids.append(tid)

        for _ in range(200):
            all_done = all(
                worker.get_task(tid).status in (TaskStatus.completed, TaskStatus.failed)
                for tid in task_ids
            )
            if all_done:
                break
            await asyncio.sleep(0.05)

        # With max_concurrent=1, only one task should run at a time
        assert max_concurrent_seen <= 1

        await worker.stop()

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        worker = Worker(max_concurrent=4)
        worker.start()

        async def handler(payload):
            return {}

        worker.register("list_test", handler)

        tid1 = await worker.submit("list_test", {"a": 1})
        tid2 = await worker.submit("list_test", {"b": 2})

        # Wait for tasks to complete
        for _ in range(50):
            t1 = worker.get_task(tid1)
            t2 = worker.get_task(tid2)
            if (t1.status in (TaskStatus.completed, TaskStatus.failed) and
                t2.status in (TaskStatus.completed, TaskStatus.failed)):
                break
            await asyncio.sleep(0.05)

        tasks = worker.list_tasks()
        assert len(tasks) >= 2
        assert tasks[0].created_at >= tasks[1].created_at  # Most recent first

        await worker.stop()

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_type(self):
        worker = Worker(max_concurrent=4)
        worker.start()

        async def handler_a(payload):
            await asyncio.sleep(0.05)
            return {}

        async def handler_b(payload):
            await asyncio.sleep(0.05)
            return {}

        worker.register("type_a", handler_a)
        worker.register("type_b", handler_b)

        await worker.submit("type_a", {})
        await worker.submit("type_b", {})
        await worker.submit("type_a", {})

        # Wait for tasks to complete
        await asyncio.sleep(0.3)

        type_a_tasks = worker.list_tasks(task_type="type_a")
        assert all(t.task_type == "type_a" for t in type_a_tasks)
        assert len(type_a_tasks) == 2

        type_b_tasks = worker.list_tasks(task_type="type_b")
        assert all(t.task_type == "type_b" for t in type_b_tasks)
        assert len(type_b_tasks) == 1

        await worker.stop()

    @pytest.mark.asyncio
    async def test_get_task_not_found(self):
        worker = Worker()
        assert worker.get_task("nonexistent_id") is None

    @pytest.mark.asyncio
    async def test_stop_waits_for_running_tasks(self):
        worker = Worker(max_concurrent=1)
        worker.start()

        async def handler(payload):
            await asyncio.sleep(0.3)
            return {"done": True}

        worker.register("slow", handler)
        task_id = await worker.submit("slow", {})

        # Give the task a moment to start
        await asyncio.sleep(0.05)

        # Stop should wait for the task to finish
        await worker.stop()

        # After stop, the task should have completed
        task = worker.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.completed
        assert task.result == {"done": True}