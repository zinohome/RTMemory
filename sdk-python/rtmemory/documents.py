"""DocumentsNamespace — async methods for the /v1/documents/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import (
    Document,
    DocumentAddRequest,
    DocumentListResponse,
)


class DocumentsNamespace:
    """Namespace for document management operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def add(
        self,
        content: str,
        space_id: str,
        title: str | None = None,
    ) -> Document:
        """Add a document by content (text or URL)."""
        body = DocumentAddRequest(content=content, space_id=space_id, title=title)
        resp = await self._http.post("/v1/documents/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return Document.model_validate(resp.json())

    async def upload(self, file: str, space_id: str) -> Document:
        """Upload a file (multipart) as a document."""
        with open(file, "rb") as f:
            files = {"file": (file, f)}
            data = {"space_id": (None, space_id)}
            resp = await self._http.post("/v1/documents/upload", files=files, data=data)
        resp.raise_for_status()
        return Document.model_validate(resp.json())

    async def list(
        self,
        space_id: str | None = None,
        status: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
        offset: int = 0,
        limit: int = 20,
    ) -> DocumentListResponse:
        """List documents with optional status filter and sorting."""
        params: dict[str, Any] = {"sort": sort, "order": order, "offset": offset, "limit": limit}
        if space_id is not None:
            params["space_id"] = space_id
        if status is not None:
            params["status"] = status
        resp = await self._http.get("/v1/documents/", params=params)
        resp.raise_for_status()
        return DocumentListResponse.model_validate(resp.json())

    async def get(self, id: str) -> Document:
        """Get a single document with associated memories."""
        resp = await self._http.get(f"/v1/documents/{id}")
        resp.raise_for_status()
        return Document.model_validate(resp.json())

    async def delete(self, id: str) -> dict[str, Any]:
        """Delete a document."""
        resp = await self._http.delete(f"/v1/documents/{id}")
        resp.raise_for_status()
        return resp.json()