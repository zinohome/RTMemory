"""ProfileNamespace — async methods for the /v1/profile/ API."""

from __future__ import annotations

import httpx

from rtmemory.types import ProfileRequest, ProfileResponse


class ProfileNamespace:
    """Namespace for user profile operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def get(
        self,
        entity_id: str,
        space_id: str,
        q: str | None = None,
        fresh: bool = False,
    ) -> ProfileResponse:
        """Get (or compute) a user profile from the knowledge graph."""
        body = ProfileRequest(entity_id=entity_id, space_id=space_id, q=q, fresh=fresh)
        resp = await self._http.post("/v1/profile/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return ProfileResponse.model_validate(resp.json())