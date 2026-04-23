"""API routes for /v1/tasks/ — background task status tracking.

GET  /v1/tasks/       — List tasks (optionally filtered by type)
GET  /v1/tasks/:id     — Get task status and result
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.worker import TaskStatus, Worker

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

# Global worker instance — set during app startup
_worker: Worker | None = None


def set_worker(worker: Worker) -> None:
    """Set the global Worker instance (called during app startup)."""
    global _worker
    _worker = worker


def get_worker() -> Worker:
    """FastAPI dependency — returns the configured Worker."""
    if _worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    return _worker


@router.get("/")
async def list_tasks(
    task_type: str | None = Query(default=None, description="Filter by task type"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List background tasks, optionally filtered by type."""
    worker = get_worker()
    tasks = worker.list_tasks(task_type=task_type, limit=limit)
    return {
        "items": [
            {
                "id": t.id,
                "task_type": t.task_type,
                "status": t.status.value,
                "created_at": t.created_at.isoformat(),
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "error": t.error,
            }
            for t in tasks
        ],
        "total": len(tasks),
    }


@router.get("/{task_id}")
async def get_task(task_id: str):
    """Get task status and result by ID."""
    worker = get_worker()
    task = worker.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    result = {
        "id": task.id,
        "task_type": task.task_type,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "error": task.error,
    }
    if task.status == TaskStatus.completed and task.result is not None:
        result["result"] = task.result
    return result