"""API routes for /v1/entities/."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    EntityCreate,
    EntityListOut,
    EntityOut,
    EntityUpdate,
)

router = APIRouter(prefix="/v1/entities", tags=["entities"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


@router.post("/", response_model=EntityOut, status_code=201)
async def create_entity(
    data: EntityCreate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Create a new entity."""
    result = await engine.create_entity(data)
    return result


@router.get("/", response_model=EntityListOut)
async def list_entities(
    org_id: uuid.UUID = Query(...),
    space_id: Optional[uuid.UUID] = Query(default=None),
    entity_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: GraphEngine = Depends(_get_engine),
):
    """List entities with optional filters and pagination."""
    result = await engine.list_entities(
        org_id=org_id,
        space_id=space_id,
        entity_type=entity_type,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/{entity_id}", response_model=EntityOut)
async def get_entity(
    entity_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get an entity by ID."""
    result = await engine.get_entity(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return result


@router.patch("/{entity_id}", response_model=EntityOut)
async def update_entity(
    entity_id: uuid.UUID,
    data: EntityUpdate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Update an entity."""
    try:
        result = await engine.update_entity(entity_id, data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{entity_id}", status_code=204)
async def delete_entity(
    entity_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Delete an entity."""
    try:
        await engine.delete_entity(entity_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))