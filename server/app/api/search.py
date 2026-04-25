"""POST /v1/search/ — unified hybrid search endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.search_engine import SearchEngine
from app.api.deps import db_session
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/v1/search", tags=["search"])


def _get_embedding_service():
    """Dependency injection for embedding service.

    Uses app.state.embedding_service if set (initialized at startup),
    otherwise creates one from config settings.
    """
    from app.config import get_settings
    from app.core.embedding import create_embedding_service
    settings = get_settings()
    return create_embedding_service(settings.embedding)


def _get_llm_adapter():
    """Dependency injection for LLM adapter (optional).

    Returns None if no LLM is configured.
    """
    try:
        from app.config import get_settings
        from app.core.llm import create_llm_adapter
        settings = get_settings()
        return create_llm_adapter(settings.llm)
    except Exception:
        return None


@router.post("/", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(db_session),
    embedding_service=Depends(_get_embedding_service),
    llm_adapter=Depends(_get_llm_adapter),
) -> SearchResponse:
    """Execute a hybrid search across memories, entities, and documents.

    Pipeline: QueryProcessor -> [Vector, Graph, Keyword] -> RRF -> Profile Boost -> Results

    The search combines three channels:
    - **Vector**: pgvector cosine similarity across memories, chunks, entities
    - **Graph**: Recursive CTE from identified entities (up to 3 hops)
    - **Keyword**: PostgreSQL tsvector full-text search (simple config for Chinese)

    Results are fused via Reciprocal Rank Fusion (k=60) and optionally
    boosted by user profile (entity match x1.5, preference match x1.2).
    """
    org_id = await _resolve_org_id(db, request.space_id)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Space {request.space_id} not found",
        )

    engine = SearchEngine(
        db_session=db,
        embedding_service=embedding_service,
        llm_adapter=llm_adapter,
    )

    response = await engine.search(request, org_id=org_id)
    return response


async def _resolve_org_id(db: AsyncSession, space_id: uuid.UUID) -> uuid.UUID | None:
    """Look up the org_id for a given space."""
    from sqlalchemy import text
    result = await db.execute(
        text("SELECT org_id FROM spaces WHERE id = :space_id"),
        {"space_id": str(space_id)},
    )
    row = result.fetchone()
    return row.org_id if row else None