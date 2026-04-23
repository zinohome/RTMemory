"""SearchEngine -- hybrid search orchestration.

Combines vector search, graph traversal, and keyword search channels,
fuses them with RRF, and applies profile boosting.

Architecture:
    Query -> QueryProcessor -> [Vector, Graph, Keyword] -> RRF -> Profile Boost -> Results
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.search_channels import (
    ChannelResult,
    VectorSearchChannel,
    GraphSearchChannel,
    KeywordSearchChannel,
)
from app.core.search_fusion import (
    FusedResult,
    reciprocal_rank_fusion,
    apply_profile_boost,
)
from app.core.query_processor import QueryProcessor
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTiming,
    SearchProfile,
    SearchMode,
    SearchChannel,
    ResultType,
    EntityBrief,
    DocumentBrief,
)


class SearchEngine:
    """Hybrid search engine combining vector, graph, and keyword channels.

    Usage:
        engine = SearchEngine(db_session=session, embedding_service=embed_svc, llm_adapter=llm)
        response = await engine.search(request, org_id=org_id)
    """

    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: Any,
        llm_adapter: Any | None = None,
    ):
        self._session = db_session
        self._embedding_service = embedding_service
        self._llm_adapter = llm_adapter

        self._vector_channel = VectorSearchChannel(db_session)
        self._graph_channel = GraphSearchChannel(db_session)
        self._keyword_channel = KeywordSearchChannel(db_session)
        self._query_processor = QueryProcessor(db_session, llm_adapter)

    async def search(
        self,
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> SearchResponse:
        """Execute a hybrid search request.

        Pipeline:
        1. Process query (entity recognition + optional rewrite)
        2. Run search channels in parallel
        3. Fuse results with RRF
        4. Apply profile boost if user_id provided
        5. Build response with timing
        """
        t0 = time.perf_counter()

        # ── Step 1: Query processing ────────────────────────────
        rewrite_t0 = time.perf_counter()
        processed = await self._query_processor.process(
            query=request.q,
            space_id=request.space_id,
            org_id=org_id,
            rewrite_query=request.rewrite_query,
        )
        rewrite_ms = (time.perf_counter() - rewrite_t0) * 1000

        # ── Step 2: Determine active channels ───────────────────
        active_channels = self._resolve_channels(request)

        # ── Step 3: Get query embedding for vector search ───────
        query_vec: list[float] | None = None
        if SearchChannel.vector in active_channels:
            try:
                embed_result = await self._embedding_service.embed([processed.effective_query])
                query_vec = embed_result[0]
            except Exception:
                # If embedding fails, skip vector channel
                active_channels.discard(SearchChannel.vector)

        # ── Step 4: Run channels in parallel ────────────────────
        channel_tasks: dict[str, asyncio.Task] = {}

        if SearchChannel.vector in active_channels and query_vec is not None:
            channel_tasks["vector"] = asyncio.create_task(
                self._run_vector_search(query_vec, request, org_id)
            )

        if SearchChannel.graph in active_channels:
            channel_tasks["graph"] = asyncio.create_task(
                self._run_graph_search(processed.entity_ids, request, org_id)
            )

        if SearchChannel.keyword in active_channels:
            channel_tasks["keyword"] = asyncio.create_task(
                self._run_keyword_search(processed.effective_query, request, org_id)
            )

        # Gather all channel results
        channel_results: dict[str, ChannelResult] = {}
        if channel_tasks:
            done = await asyncio.gather(*channel_tasks.values(), return_exceptions=True)
            for ch_name, result in zip(channel_tasks.keys(), done):
                if isinstance(result, Exception):
                    channel_results[ch_name] = ChannelResult(items=[], channel=ch_name, timing_ms=0.0)
                else:
                    channel_results[ch_name] = result

        # ── Step 5: RRF fusion ──────────────────────────────────
        fusion_t0 = time.perf_counter()
        channel_item_map: dict[str, list[dict[str, Any]]] = {}
        for ch_name, ch_result in channel_results.items():
            channel_item_map[ch_name] = ch_result.items

        channel_item_map = self._filter_by_mode(channel_item_map, request.mode)
        fused = reciprocal_rank_fusion(channel_item_map, k=60)
        fusion_ms = (time.perf_counter() - fusion_t0) * 1000

        # ── Step 6: Profile Boost ───────────────────────────────
        profile_t0 = time.perf_counter()
        profile: SearchProfile | None = None
        if request.user_id is not None:
            user_entity_id = await self._get_user_entity_id(request.user_id, request.space_id, org_id)
            pref_ids = await self._get_user_preference_ids(user_entity_id, request.space_id, org_id) if user_entity_id else []
            fused = apply_profile_boost(fused, user_entity_id=user_entity_id, user_preference_entity_ids=pref_ids)

            if request.include_profile and user_entity_id:
                profile = await self._build_profile_snapshot(user_entity_id, request.space_id, org_id)
        profile_ms = (time.perf_counter() - profile_t0) * 1000

        # ── Step 7: Build response ──────────────────────────────
        results = self._build_result_items(fused, request)

        total_ms = (time.perf_counter() - t0) * 1000
        timing = SearchTiming(
            total_ms=round(total_ms, 2),
            vector_ms=round(channel_results.get("vector", ChannelResult()).timing_ms, 2) if "vector" in channel_results else None,
            graph_ms=round(channel_results.get("graph", ChannelResult()).timing_ms, 2) if "graph" in channel_results else None,
            keyword_ms=round(channel_results.get("keyword", ChannelResult()).timing_ms, 2) if "keyword" in channel_results else None,
            fusion_ms=round(fusion_ms, 2),
            profile_ms=round(profile_ms, 2),
            rewrite_ms=round(rewrite_ms, 2) if request.rewrite_query else None,
        )

        return SearchResponse(
            results=results[:request.limit],
            profile=profile,
            timing=timing,
            query=processed.effective_query,
        )

    # ── Channel runners ─────────────────────────────────────────

    async def _run_vector_search(
        self,
        query_vec: list[float],
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> ChannelResult:
        """Run vector similarity search."""
        return await self._vector_channel.search(
            query_vec=query_vec,
            space_id=request.space_id,
            org_id=org_id,
            limit=request.limit,
            chunk_threshold=request.chunk_threshold,
            document_threshold=request.document_threshold,
        )

    async def _run_graph_search(
        self,
        seed_entity_ids: list[uuid.UUID],
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> ChannelResult:
        """Run graph traversal search."""
        return await self._graph_channel.search(
            seed_entity_ids=seed_entity_ids,
            space_id=request.space_id,
            org_id=org_id,
            max_depth=3,
            limit=request.limit,
        )

    async def _run_keyword_search(
        self,
        query_text: str,
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> ChannelResult:
        """Run keyword full-text search."""
        return await self._keyword_channel.search(
            query_text=query_text,
            space_id=request.space_id,
            org_id=org_id,
            limit=request.limit,
        )

    # ── Channel resolution ──────────────────────────────────────

    def _resolve_channels(self, request: SearchRequest) -> set[SearchChannel]:
        """Determine which channels to run based on request parameters."""
        if request.channels is not None:
            return set(request.channels)

        if request.mode == SearchMode.hybrid:
            return {SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword}
        elif request.mode == SearchMode.memory_only:
            return {SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword}
        elif request.mode == SearchMode.documents_only:
            return {SearchChannel.vector, SearchChannel.keyword}
        return {SearchChannel.vector, SearchChannel.graph, SearchChannel.keyword}

    def _filter_by_mode(
        self,
        channel_item_map: dict[str, list[dict[str, Any]]],
        mode: SearchMode,
    ) -> dict[str, list[dict[str, Any]]]:
        """Filter channel results by search mode (memory_only vs documents_only)."""
        if mode == SearchMode.memory_only:
            filtered: dict[str, list[dict[str, Any]]] = {}
            for ch, items in channel_item_map.items():
                filtered[ch] = [i for i in items if i.get("type") in ("memory", "entity")]
            return filtered
        elif mode == SearchMode.documents_only:
            filtered = {}
            for ch, items in channel_item_map.items():
                filtered[ch] = [i for i in items if i.get("type") in ("document_chunk", "document")]
            return filtered
        return channel_item_map

    # ── Profile helpers ─────────────────────────────────────────

    async def _get_user_entity_id(
        self,
        user_id: uuid.UUID,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> uuid.UUID | None:
        """Look up the entity associated with a user_id.

        Convention: entity metadata contains user_id field.
        """
        sql = text("""
            SELECT e.id
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND e.metadata->>'user_id' = :user_id
            LIMIT 1
        """)
        result = await self._session.execute(sql, {
            "space_id": str(space_id),
            "org_id": str(org_id),
            "user_id": str(user_id),
        })
        row = result.fetchone()
        return row.id if row else None

    async def _get_user_preference_ids(
        self,
        user_entity_id: uuid.UUID,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        """Get entity IDs that the user entity has 'prefers' relations to."""
        sql = text("""
            SELECT r.target_entity_id
            FROM relations r
            WHERE r.source_entity_id = :entity_id
              AND r.relation_type = 'prefers'
              AND r.is_current = true
              AND r.space_id = :space_id
              AND r.org_id = :org_id
        """)
        result = await self._session.execute(sql, {
            "entity_id": str(user_entity_id),
            "space_id": str(space_id),
            "org_id": str(org_id),
        })
        return [row.target_entity_id for row in result.fetchall()]

    async def _build_profile_snapshot(
        self,
        user_entity_id: uuid.UUID,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> SearchProfile:
        """Build a lightweight profile snapshot for the search response.

        Simplified version -- full profile computation is in ProfileEngine.
        """
        identity_sql = text("""
            SELECT r.relation_type, t.name AS target_name, t.entity_type
            FROM relations r
            JOIN entities t ON r.target_entity_id = t.id
            WHERE r.source_entity_id = :entity_id
              AND r.is_current = true
              AND r.space_id = :space_id
              AND r.org_id = :org_id
        """)
        result = await self._session.execute(identity_sql, {
            "entity_id": str(user_entity_id),
            "space_id": str(space_id),
            "org_id": str(org_id),
        })

        identity: dict[str, Any] = {}
        preferences: dict[str, Any] = {}
        current_status: dict[str, Any] = {}

        for row in result.fetchall():
            rel_type = row.relation_type
            target_name = row.target_name
            if rel_type in ("lives_in", "works_at", "role"):
                identity[rel_type] = target_name
            elif rel_type == "prefers":
                etype = row.entity_type or "other"
                preferences.setdefault(etype, []).append(target_name)
            else:
                current_status[rel_type] = target_name

        return SearchProfile(
            identity=identity if identity else None,
            preferences=preferences if preferences else None,
            current_status=current_status if current_status else None,
        )

    # ── Result builder ─────────────────────────────────────────

    def _build_result_items(
        self,
        fused: list[FusedResult],
        request: SearchRequest,
    ) -> list[SearchResultItem]:
        """Convert FusedResult list to API SearchResultItem list."""
        items: list[SearchResultItem] = []
        for fr in fused:
            entity_brief = None
            if fr.entity_name:
                entity_brief = EntityBrief(name=fr.entity_name, type=fr.entity_type or "")

            doc_brief = None
            if fr.document:
                doc_brief = DocumentBrief(
                    id=fr.document.get("id", fr.document_id or uuid.UUID(int=0)),
                    title=fr.document.get("title", ""),
                    url=fr.document.get("url"),
                )

            try:
                result_type = ResultType(fr.type)
            except ValueError:
                result_type = ResultType.memory

            items.append(SearchResultItem(
                type=result_type,
                id=fr.id,
                content=fr.content,
                score=round(fr.boosted_score, 6),
                source="+".join(fr.source_channels),
                entity=entity_brief,
                document=doc_brief,
                metadata=fr.metadata,
                created_at=fr.created_at,
            ))
        return items

    # ── Document assembly ────────────────────────────────────────

    async def _assemble_document_results(
        self,
        chunk_items: list[dict[str, Any]],
        request: SearchRequest,
        org_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """Assemble chunks into document-level results when include_full_docs=True.

        If only_matching_chunks=True, return chunks as-is.
        If include_full_docs=True, fetch full documents and attach their chunks.
        """
        if request.only_matching_chunks or not request.include_full_docs:
            return chunk_items

        doc_ids = list({item["document_id"] for item in chunk_items if "document_id" in item})
        if not doc_ids:
            return chunk_items

        sql = text("""
            SELECT d.id, d.title, d.content, d.url, d.summary, d.metadata
            FROM documents d
            WHERE d.id = ANY(:doc_ids)
              AND d.space_id = :space_id
              AND d.org_id = :org_id
        """)
        result = await self._session.execute(sql, {
            "doc_ids": [str(did) for did in doc_ids],
            "space_id": str(request.space_id),
            "org_id": str(org_id),
        })

        doc_map: dict[uuid.UUID, dict] = {}
        for row in result.fetchall():
            doc_map[row.id] = {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "url": row.url,
                "summary": row.summary if request.include_summary else None,
                "metadata": row.metadata,
            }

        assembled = []
        for item in chunk_items:
            doc_id = item.get("document_id")
            if doc_id and doc_id in doc_map:
                item["document"] = doc_map[doc_id]
            assembled.append(item)

        return assembled