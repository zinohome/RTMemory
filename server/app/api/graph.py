"""Graph visualization API route — GET /v1/graph/neighborhood.

Provides a D3.js/Cytoscape-compatible graph neighborhood endpoint that
wraps GraphEngine.traverse_graph() and transforms the output into a
format suitable for front-end graph rendering.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.graph_engine import GraphEngine
from app.db.session import get_session
from app.schemas.graph import (
    EntityOut,
    GraphTraversalOut,
    GraphTraversalParams,
    RelationOut,
    TraversedRelationOut,
)

router = APIRouter(prefix="/v1/graph", tags=["graph"])


async def _get_engine(session: AsyncSession = Depends(get_session)) -> GraphEngine:
    return GraphEngine(session)


# ── Response schemas for D3/Cytoscape ───────────────────────────────


class GraphNode(BaseModel):
    """Node suitable for D3.js / Cytoscape rendering."""
    id: str
    label: str
    entityType: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 1.0


class GraphEdge(BaseModel):
    """Edge suitable for D3.js / Cytoscape rendering."""
    id: str
    source: str
    target: str
    label: str
    value: Optional[str] = None
    confidence: float = 1.0
    validFrom: Optional[str] = None
    validTo: Optional[str] = None
    isCurrent: bool = True


class GraphNeighborhoodResponse(BaseModel):
    """Response format for graph neighborhood visualization.

    This format is directly consumable by D3.js force-directed layouts
    and Cytoscape.js:
      - nodes[].id, nodes[].label for node rendering
      - edges[].source, edges[].target for link rendering
    """
    center: str
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    maxHops: int = 3


# ── Endpoint ────────────────────────────────────────────────────────


@router.get("/neighborhood", response_model=GraphNeighborhoodResponse)
async def get_graph_neighborhood(
    entity_id: uuid.UUID = Query(..., description="Center entity ID"),
    space_id: Optional[uuid.UUID] = Query(default=None, description="Filter by space"),
    max_hops: int = Query(default=3, ge=1, le=10, description="Max traversal hops"),
    relation_types: Optional[str] = Query(
        default=None, description="Comma-separated relation types to follow"
    ),
    direction: str = Query(
        default="both", description="Traversal direction: outgoing, incoming, or both"
    ),
    engine: GraphEngine = Depends(_get_engine),
) -> GraphNeighborhoodResponse:
    """Get the neighborhood of an entity for graph visualization.

    Traverses the knowledge graph from the given entity using recursive CTE
    and returns nodes/edges in a D3.js/Cytoscape-compatible format.

    This is a visualization-friendly wrapper around the core
    ``GraphEngine.traverse_graph()`` method.
    """
    # Validate direction
    if direction not in ("outgoing", "incoming", "both"):
        raise HTTPException(
            status_code=400,
            detail="direction must be one of: outgoing, incoming, both",
        )

    # Parse comma-separated relation types
    parsed_relation_types: Optional[list[str]] = None
    if relation_types:
        parsed_relation_types = [rt.strip() for rt in relation_types.split(",") if rt.strip()]
        # Validate relation types — only alphanumeric + underscore
        import re
        _valid_rt = re.compile(r"^[a-zA-Z0-9_]+$")
        for rt in parsed_relation_types:
            if not _valid_rt.match(rt):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid relation type: {rt!r}. Only alphanumeric characters and underscores are allowed.",
                )

    params = GraphTraversalParams(
        entity_id=entity_id,
        max_hops=max_hops,
        relation_types=parsed_relation_types,
        direction=direction,
    )

    try:
        traversal: GraphTraversalOut = await engine.traverse_graph(params)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Convert engine output to D3/Cytoscape format
    nodes: list[GraphNode] = []
    for entity in traversal.entities:
        nodes.append(
            GraphNode(
                id=str(entity.id),
                label=entity.name,
                entityType=entity.entity_type.value
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type),
                description=entity.description,
                confidence=entity.confidence,
            )
        )

    edges: list[GraphEdge] = []
    for traversed in traversal.relations:
        rel: RelationOut = traversed.relation
        edges.append(
            GraphEdge(
                id=str(rel.id),
                source=str(rel.source_entity_id),
                target=str(rel.target_entity_id),
                label=rel.relation_type,
                value=rel.value or None,
                confidence=rel.confidence,
                validFrom=rel.valid_from.isoformat() if rel.valid_from else None,
                validTo=rel.valid_to.isoformat() if rel.valid_to else None,
                isCurrent=rel.is_current,
            )
        )

    return GraphNeighborhoodResponse(
        center=str(entity_id),
        nodes=nodes,
        edges=edges,
        maxHops=max_hops,
    )
