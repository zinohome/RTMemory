"""SpacesNamespace — async methods for the /v1/spaces/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import Space, SpaceCreateRequest, SpaceListResponse


class SpacesNamespace:
    """Namespace for space management operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def create(
        self,
        name: str,
        description: str | None = None,
    ) -> Space:
        """Create a new space."""
        body = SpaceCreateRequest(name=name, description=description)
        resp = await self._http.post("/v1/spaces/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return Space.model_validate(resp.json())

    async def list(self) -> SpaceListResponse:
        """List all spaces."""
        resp = await self._http.get("/v1/spaces/")
        resp.raise_for_status()
        data = resp.json()
        # Handle case where API returns a list directly
        if isinstance(data, list):
            return SpaceListResponse(items=data, total=len(data))
        return SpaceListResponse.model_validate(data)

    async def get(self, id: str) -> Space:
        """Get space details."""
        resp = await self._http.get(f"/v1/spaces/{id}")
        resp.raise_for_status()
        return Space.model_validate(resp.json())

    async def delete(self, id: str) -> dict[str, Any]:
        """Delete a space."""
        resp = await self._http.delete(f"/v1/spaces/{id}")
        resp.raise_for_status()
        return resp.json()