"""Graph visualization API — GET /v1/graph/neighborhood.

Returns a sub-graph around a given entity for visualization,
using the GraphEngine's traverse method.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.graph_engine import GraphEngine
from app.schemas.graph import (
    EntityOut,
    GraphNeighborhoodOut,
    RelationOut,
)

router = APIRouter(prefix="/v1/graph", tags=["graph"])

# Global engine — set during app startup
_graph_engine: GraphEngine | None = None


def set_graph_engine(engine: GraphEngine) -> None:
    """Set the global GraphEngine instance."""
    global _graph_engine
    _graph_engine = engine


def get_graph_engine() -> GraphEngine:
    """FastAPI dependency — returns the configured GraphEngine."""
    if _graph_engine is None:
        raise HTTPException(status_code=503, detail="GraphEngine not initialized")
    return _graph_engine


@router.get("/neighborhood", response_model=GraphNeighborhoodOut)
async def get_neighborhood(
    entity_id: uuid.UUID = Query(..., description="Center entity ID"),
    max_hops: int = Query(default=2, ge=1, le=5, description="Max traversal depth"),
    engine: GraphEngine = Depends(get_graph_engine),
):
    """Get the graph neighborhood around an entity.

    Returns the center entity plus all entities and relations reachable
    within `max_hops` steps. Useful for graph visualization.
    """
    result = await engine.traverse(
        entity_id=entity_id,
        max_hops=max_hops,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    center_entity = result["center"]
    entities = result.get("entities", [])
    relations = result.get("relations", [])

    return GraphNeighborhoodOut(
        center=EntityOut(
            id=center_entity.id,
            name=center_entity.name,
            entity_type=center_entity.entity_type,
            description=center_entity.description,
            confidence=center_entity.confidence,
            space_id=center_entity.space_id,
            created_at=center_entity.created_at,
            updated_at=center_entity.updated_at,
        ),
        entities=[
            EntityOut(
                id=e.id,
                name=e.name,
                entity_type=e.entity_type,
                description=e.description,
                confidence=e.confidence,
                space_id=e.space_id,
                created_at=e.created_at,
                updated_at=e.updated_at,
            )
            for e in entities
        ],
        relations=[
            RelationOut(
                id=r.id,
                source_entity_id=r.source_entity_id,
                target_entity_id=r.target_entity_id,
                relation_type=r.relation_type,
                value=r.value,
                valid_from=r.valid_from,
                valid_to=r.valid_to,
                confidence=r.confidence,
                is_current=r.is_current,
                source_count=r.source_count,
                space_id=r.space_id,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in relations
        ],
        depth=max_hops,
    )