"""API routes for /v1/relations/."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    RelationCreate,
    RelationListOut,
    RelationOut,
    RelationUpdate,
)

router = APIRouter(prefix="/v1/relations", tags=["relations"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


@router.post("/", response_model=RelationOut, status_code=201)
async def create_relation(
    data: RelationCreate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Create a new relation. Handles contradiction automatically."""
    result = await engine.create_relation(data)
    return result


@router.get("/", response_model=RelationListOut)
async def list_relations(
    org_id: uuid.UUID = Query(...),
    space_id: Optional[uuid.UUID] = Query(default=None),
    source_entity_id: Optional[uuid.UUID] = Query(default=None),
    target_entity_id: Optional[uuid.UUID] = Query(default=None),
    relation_type: Optional[str] = Query(default=None),
    is_current: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: GraphEngine = Depends(_get_engine),
):
    """List relations with optional filters and pagination."""
    result = await engine.list_relations(
        org_id=org_id,
        space_id=space_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        is_current=is_current,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/{relation_id}", response_model=RelationOut)
async def get_relation(
    relation_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Get a relation by ID."""
    result = await engine.get_relation(relation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Relation not found")
    return result


@router.patch("/{relation_id}", response_model=RelationOut)
async def update_relation(
    relation_id: uuid.UUID,
    data: RelationUpdate,
    engine: GraphEngine = Depends(_get_engine),
):
    """Update a relation."""
    try:
        result = await engine.update_relation(relation_id, data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{relation_id}", status_code=204)
async def delete_relation(
    relation_id: uuid.UUID,
    engine: GraphEngine = Depends(_get_engine),
):
    """Delete a relation."""
    try:
        await engine.delete_relation(relation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))