"""Three search channel implementations: vector, graph, keyword."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# ── Shared result type ──────────────────────────────────────────


@dataclass
class ChannelResult:
    """Result container for a single search channel."""

    items: list[dict[str, Any]] = field(default_factory=list)
    channel: str = ""
    timing_ms: float = 0.0


# ── Vector Search Channel ──────────────────────────────────────


class VectorSearchChannel:
    """Search across memories, chunks, and entities using pgvector cosine similarity.

    SQL pattern:
        1 - (embedding <=> $query_vec) AS similarity
    """

    def __init__(self, db_session: AsyncSession):
        self._session = db_session

    async def search(
        self,
        query_vec: list[float],
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        limit: int = 20,
        chunk_threshold: float = 0.0,
        document_threshold: float = 0.0,
    ) -> ChannelResult:
        """Run vector similarity search across memories, chunks, and entities."""
        t0 = time.perf_counter()
        items: list[dict[str, Any]] = []

        # ── Vector search across memories ───────────────────────
        mem_sql = text("""
            SELECT
                m.id,
                'memory' AS type,
                m.content,
                1 - (m.embedding <=> :query_vec) AS similarity,
                m.entity_id,
                e.name AS entity_name,
                e.entity_type AS entity_type,
                m.metadata AS metadata_,
                m.created_at
            FROM memories m
            LEFT JOIN entities e ON m.entity_id = e.id
            WHERE m.space_id = :space_id
              AND m.org_id = :org_id
              AND m.is_forgotten = false
              AND m.embedding IS NOT NULL
            ORDER BY m.embedding <=> :query_vec
            LIMIT :limit
        """)
        mem_result = await self._session.execute(
            mem_sql,
            {
                "query_vec": str(query_vec),
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in mem_result.fetchall():
            score = float(row.similarity) if row.similarity is not None else 0.0
            if score >= chunk_threshold:
                items.append({
                    "id": row.id,
                    "type": row.type,
                    "content": row.content,
                    "score": score,
                    "entity_id": row.entity_id,
                    "entity_name": row.entity_name,
                    "entity_type": row.entity_type,
                    "metadata": row.metadata_,
                    "created_at": row.created_at,
                })

        # ── Vector search across chunks ─────────────────────────
        chunk_sql = text("""
            SELECT
                c.id,
                'document_chunk' AS type,
                c.content,
                1 - (c.embedding <=> :query_vec) AS similarity,
                c.document_id,
                d.title AS doc_title,
                d.url AS doc_url,
                d.id AS doc_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.space_id = :space_id
              AND d.org_id = :org_id
              AND d.status = 'done'
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> :query_vec
            LIMIT :limit
        """)
        chunk_result = await self._session.execute(
            chunk_sql,
            {
                "query_vec": str(query_vec),
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in chunk_result.fetchall():
            score = float(row.similarity) if row.similarity is not None else 0.0
            if score >= chunk_threshold:
                items.append({
                    "id": row.id,
                    "type": row.type,
                    "content": row.content,
                    "score": score,
                    "document_id": row.document_id,
                    "document": {"id": row.doc_id, "title": row.doc_title, "url": row.doc_url},
                })

        # ── Vector search across entities ────────────────────────
        ent_sql = text("""
            SELECT
                e.id,
                'entity' AS type,
                e.name AS content,
                1 - (e.embedding <=> :query_vec) AS similarity,
                e.entity_type
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND e.embedding IS NOT NULL
            ORDER BY e.embedding <=> :query_vec
            LIMIT :limit
        """)
        ent_result = await self._session.execute(
            ent_sql,
            {
                "query_vec": str(query_vec),
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in ent_result.fetchall():
            score = float(row.similarity) if row.similarity is not None else 0.0
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": score,
                "entity_name": row.content,
                "entity_type": row.entity_type,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChannelResult(items=items, channel="vector", timing_ms=elapsed_ms)

    def _build_metadata_filter_clause(self, filters: dict | None) -> tuple[str, dict]:
        """Build SQL WHERE clause from metadata filter dict.

        Supports AND/OR nested structure:
        {"AND": [{"key": "source", "value": "slack"}]}
        {"OR": [{"key": "source", "value": "slack"}, {"key": "source", "value": "email"}]}

        Returns (sql_fragment, params_dict).
        """
        if not filters:
            return "", {}

        conditions = []
        params = {}
        op = "AND"

        if "AND" in filters:
            items = filters["AND"]
            op = "AND"
        elif "OR" in filters:
            items = filters["OR"]
            op = "OR"
        else:
            items = [filters]
            op = "AND"

        for i, f in enumerate(items):
            key = f.get("key", "")
            value = f.get("value", "")
            param_key = f"mf_{key}_{i}"
            conditions.append(f"metadata->>:{param_key} = :{param_key}_val")
            params[param_key] = key
            params[f"{param_key}_val"] = str(value)

        if not conditions:
            return "", {}

        clause = f" {op} ".join(conditions)
        return f"AND ({clause})", params


# ── Graph Traversal Channel ────────────────────────────────────


class GraphSearchChannel:
    """Search by traversing the knowledge graph from identified seed entities.

    Uses a recursive CTE to walk up to 3 hops from seed entities.
    Closer hops get higher scores: score = base_score / (depth + 1)
    """

    DEPTH_SCORES = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.3}

    def __init__(self, db_session: AsyncSession):
        self._session = db_session

    async def search(
        self,
        seed_entity_ids: list[uuid.UUID],
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        max_depth: int = 3,
        limit: int = 20,
    ) -> ChannelResult:
        """Traverse graph from seed entities using recursive CTE."""
        t0 = time.perf_counter()

        if not seed_entity_ids:
            return ChannelResult(items=[], channel="graph", timing_ms=0.0)

        items: list[dict[str, Any]] = []

        cte_sql = text("""
            WITH RECURSIVE graph_traverse AS (
                SELECT
                    e.id,
                    e.name,
                    e.entity_type,
                    e.description,
                    0 AS depth,
                    NULL::text AS relation_type,
                    e.confidence
                FROM entities e
                WHERE e.id = ANY(:seed_ids)
                  AND e.space_id = :space_id
                  AND e.org_id = :org_id

                UNION ALL

                SELECT
                    target.id,
                    target.name,
                    target.entity_type,
                    target.description,
                    gt.depth + 1,
                    r.relation_type,
                    r.confidence
                FROM graph_traverse gt
                JOIN relations r ON r.source_entity_id = gt.id
                JOIN entities target ON r.target_entity_id = target.id
                WHERE r.is_current = true
                  AND r.space_id = :space_id
                  AND r.org_id = :org_id
                  AND gt.depth < :max_depth
            )
            SELECT DISTINCT ON (id)
                id,
                name,
                entity_type,
                description,
                depth,
                relation_type,
                confidence
            FROM graph_traverse
            ORDER BY id, depth ASC
            LIMIT :limit
        """)

        result = await self._session.execute(
            cte_sql,
            {
                "seed_ids": [str(eid) for eid in seed_entity_ids],
                "space_id": str(space_id),
                "org_id": str(org_id),
                "max_depth": max_depth,
                "limit": limit,
            },
        )

        for row in result.fetchall():
            depth = int(row.depth)
            base_score = self.DEPTH_SCORES.get(depth, 0.1)
            conf = float(row.confidence) if row.confidence is not None else 0.5
            score = base_score * conf

            items.append({
                "id": row.id,
                "type": "entity",
                "content": row.description or row.name,
                "score": score,
                "depth": depth,
                "relation_type": row.relation_type,
                "entity_name": row.name,
                "entity_type": row.entity_type,
            })

        # Also fetch memories attached to traversed entities
        if items:
            entity_ids = [item["id"] for item in items]
            mem_sql = text("""
                SELECT
                    m.id,
                    m.content,
                    m.confidence,
                    m.entity_id,
                    e.name AS entity_name,
                    e.entity_type AS entity_type,
                    m.metadata AS metadata_,
                    m.created_at
                FROM memories m
                LEFT JOIN entities e ON m.entity_id = e.id
                WHERE m.entity_id = ANY(:entity_ids)
                  AND m.space_id = :space_id
                  AND m.org_id = :org_id
                  AND m.is_forgotten = false
                ORDER BY m.confidence DESC
                LIMIT :limit
            """)
            mem_result = await self._session.execute(
                mem_sql,
                {
                    "entity_ids": [str(eid) for eid in entity_ids],
                    "space_id": str(space_id),
                    "org_id": str(org_id),
                    "limit": limit,
                },
            )
            for row in mem_result.fetchall():
                items.append({
                    "id": row.id,
                    "type": "memory",
                    "content": row.content,
                    "score": float(row.confidence) * 0.7,
                    "depth": None,
                    "entity_id": row.entity_id,
                    "entity_name": row.entity_name,
                    "entity_type": row.entity_type,
                    "metadata": row.metadata_,
                    "created_at": row.created_at,
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChannelResult(items=items, channel="graph", timing_ms=elapsed_ms)


# ── Keyword Full-Text Search Channel ────────────────────────────


class KeywordSearchChannel:
    """Full-text search using PostgreSQL tsvector with 'simple' config for Chinese.

    Uses to_tsvector('simple', ...) + to_tsquery('simple', ...) for Chinese support
    since 'simple' config does not stem and treats each token as-is.
    """

    def __init__(self, db_session: AsyncSession):
        self._session = db_session

    async def search(
        self,
        query_text: str,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        limit: int = 20,
    ) -> ChannelResult:
        """Run full-text search across memories and document chunks."""
        t0 = time.perf_counter()
        items: list[dict[str, Any]] = []

        # Convert query text to tsquery: split on whitespace, join with &
        tokens = query_text.strip().split()
        tsquery_str = " & ".join(tokens)

        # ── FTS on memories ────────────────────────────────────
        mem_sql = text("""
            SELECT
                m.id,
                'memory' AS type,
                m.content,
                ts_rank(
                    to_tsvector('simple', m.content),
                    to_tsquery('simple', :tsquery)
                ) AS rank,
                m.entity_id,
                e.name AS entity_name,
                e.entity_type AS entity_type,
                m.metadata AS metadata_,
                m.created_at
            FROM memories m
            LEFT JOIN entities e ON m.entity_id = e.id
            WHERE m.space_id = :space_id
              AND m.org_id = :org_id
              AND m.is_forgotten = false
              AND to_tsvector('simple', m.content) @@ to_tsquery('simple', :tsquery)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        mem_result = await self._session.execute(
            mem_sql,
            {
                "tsquery": tsquery_str,
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in mem_result.fetchall():
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": float(row.rank) if row.rank is not None else 0.0,
                "entity_id": row.entity_id,
                "entity_name": row.entity_name,
                "entity_type": row.entity_type,
                "metadata": row.metadata_,
                "created_at": row.created_at,
            })

        # ── FTS on chunks ───────────────────────────────────────
        chunk_sql = text("""
            SELECT
                c.id,
                'document_chunk' AS type,
                c.content,
                ts_rank(
                    to_tsvector('simple', c.content),
                    to_tsquery('simple', :tsquery)
                ) AS rank,
                c.document_id,
                d.title AS doc_title,
                d.url AS doc_url,
                d.id AS doc_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.space_id = :space_id
              AND d.org_id = :org_id
              AND d.status = 'done'
              AND to_tsvector('simple', c.content) @@ to_tsquery('simple', :tsquery)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        chunk_result = await self._session.execute(
            chunk_sql,
            {
                "tsquery": tsquery_str,
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in chunk_result.fetchall():
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": float(row.rank) if row.rank is not None else 0.0,
                "document_id": row.document_id,
                "document": {"id": row.doc_id, "title": row.doc_title, "url": row.doc_url},
            })

        # ── FTS on entities (name + description) ────────────────
        ent_sql = text("""
            SELECT
                e.id,
                'entity' AS type,
                COALESCE(e.description, e.name) AS content,
                ts_rank(
                    to_tsvector('simple', COALESCE(e.description, '') || ' ' || e.name),
                    to_tsquery('simple', :tsquery)
                ) AS rank,
                e.name AS entity_name,
                e.entity_type
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND (
                to_tsvector('simple', e.name) @@ to_tsquery('simple', :tsquery)
                OR to_tsvector('simple', COALESCE(e.description, '')) @@ to_tsquery('simple', :tsquery)
              )
            ORDER BY rank DESC
            LIMIT :limit
        """)
        ent_result = await self._session.execute(
            ent_sql,
            {
                "tsquery": tsquery_str,
                "space_id": str(space_id),
                "org_id": str(org_id),
                "limit": limit,
            },
        )
        for row in ent_result.fetchall():
            items.append({
                "id": row.id,
                "type": row.type,
                "content": row.content,
                "score": float(row.rank) if row.rank is not None else 0.0,
                "entity_name": row.entity_name,
                "entity_type": row.entity_type,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChannelResult(items=items, channel="keyword", timing_ms=elapsed_ms)