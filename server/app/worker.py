"""Background task worker for RTMemory — async task processing.

Processes document pipeline, deep scans, and profile invalidation
using a simple in-memory task queue with asyncio.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine
import logging

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class Task:
    id: str
    task_type: str
    payload: dict[str, Any]
    status: TaskStatus = TaskStatus.queued
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None


# Type for task handlers
TaskHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class Worker:
    """In-memory async task worker.

    Submits tasks to a background asyncio loop. Integrates with
    FastAPI lifecycle via start()/stop() methods.

    Completed tasks are automatically evicted after _TASK_RETENTION_SECONDS
    to prevent unbounded memory growth. A maximum of _MAX_COMPLETED_TASKS
    completed/failed tasks are retained.
    """

    # Retention config for completed tasks
    _MAX_COMPLETED_TASKS = 1000
    _TASK_RETENTION_SECONDS = 3600  # 1 hour

    def __init__(self, max_concurrent: int = 4) -> None:
        self._tasks: dict[str, Task] = {}
        self._handlers: dict[str, TaskHandler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC

    def register(self, task_type: str, handler: TaskHandler) -> None:
        """Register a handler for a task type."""
        self._handlers[task_type] = handler

    def start(self) -> None:
        """Start the worker (call from FastAPI lifespan)."""
        self._running = True
        logger.info("Worker started with handlers: %s", list(self._handlers.keys()))

    async def stop(self) -> None:
        """Stop the worker, waiting for running tasks to complete."""
        self._running = False
        # Wait a moment for running tasks
        running = [t for t in self._tasks.values() if t.status == TaskStatus.running]
        if running:
            logger.info("Worker stopping, waiting for %d running tasks...", len(running))
            for _ in range(50):  # Wait up to 5 seconds
                if all(t.status != TaskStatus.running for t in running):
                    break
                await asyncio.sleep(0.1)
        logger.info("Worker stopped")

    async def submit(self, task_type: str, payload: dict[str, Any]) -> str:
        """Submit a task and return its ID."""
        if task_type not in self._handlers:
            raise ValueError(f"Unknown task type: {task_type}")

        task_id = str(uuid.uuid4())
        task = Task(id=task_id, task_type=task_type, payload=payload)
        self._tasks[task_id] = task

        # Create background task and store reference to prevent GC
        bg_task = asyncio.create_task(self._run_task(task))
        self._background_tasks.add(bg_task)
        bg_task.add_done_callback(self._background_tasks.discard)

        return task_id

    async def _run_task(self, task: Task) -> None:
        """Run a single task with semaphore-controlled concurrency."""
        async with self._semaphore:
            task.status = TaskStatus.running
            task.started_at = datetime.now(timezone.utc)
            try:
                handler = self._handlers[task.task_type]
                task.result = await handler(task.payload)
                task.status = TaskStatus.completed
            except Exception as e:
                task.status = TaskStatus.failed
                task.error = str(e)
                logger.error("Task %s (%s) failed: %s", task.id, task.task_type, e)
            finally:
                task.completed_at = datetime.now(timezone.utc)
                self._evict_old_tasks()

    def _evict_old_tasks(self) -> None:
        """Remove completed/failed tasks beyond retention limits."""
        now = datetime.now(timezone.utc)
        # Remove tasks older than retention period
        expired_ids = [
            tid for tid, t in self._tasks.items()
            if t.status in (TaskStatus.completed, TaskStatus.failed)
            and t.completed_at is not None
            and (now - t.completed_at).total_seconds() > self._TASK_RETENTION_SECONDS
        ]
        for tid in expired_ids:
            del self._tasks[tid]

        # Enforce max completed tasks
        completed = [
            (tid, t) for tid, t in self._tasks.items()
            if t.status in (TaskStatus.completed, TaskStatus.failed)
        ]
        if len(completed) > self._MAX_COMPLETED_TASKS:
            completed.sort(key=lambda x: x[1].completed_at or datetime.min.replace(tzinfo=timezone.utc))
            to_remove = completed[: len(completed) - self._MAX_COMPLETED_TASKS]
            for tid, _ in to_remove:
                del self._tasks[tid]

    def get_task(self, task_id: str) -> Task | None:
        """Get task status by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self, task_type: str | None = None, limit: int = 50) -> list[Task]:
        """List tasks, optionally filtered by type."""
        tasks = list(self._tasks.values())
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]