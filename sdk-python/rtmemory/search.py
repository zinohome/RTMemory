"""SearchNamespace — async methods for the /v1/search/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import SearchMode, SearchRequest, SearchResponse, SearchFilterGroup


class SearchNamespace:
    """Namespace for hybrid search operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def __call__(
        self,
        q: str,
        space_id: str | None = None,
        user_id: str | None = None,
        mode: SearchMode = SearchMode.hybrid,
        channels: list[str] | None = None,
        limit: int = 10,
        include_profile: bool = False,
        chunk_threshold: float = 0.0,
        document_threshold: float = 0.0,
        only_matching_chunks: bool = False,
        include_full_docs: bool = False,
        include_summary: bool = False,
        filters: SearchFilterGroup | dict[str, Any] | None = None,
        rewrite_query: bool = False,
        rerank: bool = False,
    ) -> SearchResponse:
        """Execute a hybrid search across memories, documents, and graph."""
        filters_dump = None
        if filters is not None:
            if isinstance(filters, SearchFilterGroup):
                filters_dump = filters.model_dump(exclude_none=True)
            else:
                filters_dump = filters

        body = SearchRequest(
            q=q,
            space_id=space_id,
            user_id=user_id,
            mode=mode,
            channels=channels,
            limit=limit,
            include_profile=include_profile,
            chunk_threshold=chunk_threshold,
            document_threshold=document_threshold,
            only_matching_chunks=only_matching_chunks,
            include_full_docs=include_full_docs,
            include_summary=include_summary,
            filters=filters_dump,
            rewrite_query=rewrite_query,
            rerank=rerank,
        )
        resp = await self._http.post("/v1/search/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())