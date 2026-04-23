"""API routes for /v1/memories/."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    GraphTraversalOut,
    GraphTraversalParams,
    MemoryCreate,
    MemoryForget,
    MemoryListOut,
    MemoryOut,
    MemorySourceOut,
    MemoryUpdate,
    MemoryVersionChainOut,
)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


@router.post("/", response_model=MemoryOut, status_code=201)
async def create_memory(
    data: MemoryCreate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Create a new memory."""
    result = await engine.create_memory(data)
    await engine.session.commit()
    return result


@router.get("/", response_model=MemoryListOut)
async def list_memories(
    org_id: uuid.UUID = Query(...),
    space_id: Optional[uuid.UUID] = Query(default=None),
    memory_type: Optional[str] = Query(default=None),
    entity_id: Optional[uuid.UUID] = Query(default=None),
    include_forgotten: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: GraphEngine = Depends(_get_engine),
):
    """List memories with optional filters and pagination."""
    result = await engine.list_memories(
        org_id=org_id,
        space_id=space_id,
        memory_type=memory_type,
        entity_id=entity_id,
        include_forgotten=include_forgotten,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/{memory_id}", response_model=MemoryOut)
async def get_memory(
    memory_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get a memory by ID."""
    result = await engine.get_memory(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@router.get("/{memory_id}/versions", response_model=MemoryVersionChainOut)
async def get_memory_versions(
    memory_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get the full version chain for a memory."""
    try:
        result = await engine.get_memory_version_chain(memory_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{memory_id}/sources", response_model=list[MemorySourceOut])
async def get_memory_sources(
    memory_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get source documents for a memory."""
    return await engine.get_memory_sources(memory_id)


@router.patch("/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: uuid.UUID,
    data: MemoryUpdate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Update a memory (creates a new version in the chain)."""
    try:
        result = await engine.update_memory(memory_id, data)
        await engine.session.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{memory_id}", response_model=MemoryOut)
async def forget_memory(
    memory_id: uuid.UUID,
    data: MemoryForget = Body(default=MemoryForget()),
    engine: GraphEngine = Depends(_get_engine),
):
    """Soft-delete (forget) a memory."""
    try:
        result = await engine.forget_memory(memory_id, data)
        await engine.session.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Graph Traversal ────────────────────────────────────────────

@router.post("/traverse", response_model=GraphTraversalOut)
async def traverse_graph(
    data: GraphTraversalParams,
    engine: GraphEngine = Depends(_get_engine),
):
    """Traverse the knowledge graph from a starting entity.

    Uses recursive CTE for multi-hop traversal.
    Requires PostgreSQL backend.
    """
    return await engine.traverse_graph(data)