"""MemoriesNamespace — async methods for the /v1/memories/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import (
    Memory,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryForgetRequest,
    MemoryListResponse,
    MemoryUpdateRequest,
)


class MemoriesNamespace:
    """Namespace for memory CRUD operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def add(
        self,
        content: str,
        space_id: str,
        user_id: str | None = None,
        custom_id: str | None = None,
        entity_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryAddResponse:
        """Add a new memory (triggers extraction pipeline)."""
        body = MemoryAddRequest(
            content=content,
            space_id=space_id,
            user_id=user_id,
            custom_id=custom_id,
            entity_context=entity_context,
            metadata=metadata,
        )
        resp = await self._http.post("/v1/memories/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return MemoryAddResponse.model_validate(resp.json())

    async def list(
        self,
        space_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> MemoryListResponse:
        """List memories with pagination and optional filtering."""
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if space_id is not None:
            params["space_id"] = space_id
        resp = await self._http.get("/v1/memories/", params=params)
        resp.raise_for_status()
        return MemoryListResponse.model_validate(resp.json())

    async def get(self, id: str) -> Memory:
        """Get a single memory by ID (includes version chain)."""
        resp = await self._http.get(f"/v1/memories/{id}")
        resp.raise_for_status()
        return Memory.model_validate(resp.json())

    async def update(
        self,
        id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Update a memory's content and/or metadata."""
        body = MemoryUpdateRequest(content=content, metadata=metadata)
        resp = await self._http.patch(f"/v1/memories/{id}", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return Memory.model_validate(resp.json())

    async def forget(
        self,
        memory_id: str | None = None,
        content_match: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Forget a memory by ID or content match (soft delete)."""
        body = MemoryForgetRequest(memory_id=memory_id, content_match=content_match, reason=reason)
        resp = await self._http.post("/v1/memories/forget", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return resp.json()