"""GraphNamespace — async methods for the /v1/graph/ API."""

from __future__ import annotations

import httpx

from rtmemory.types import GraphNeighborhood


class GraphNamespace:
    """Namespace for graph visualization and traversal."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def get_neighborhood(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> GraphNeighborhood:
        """Get the neighborhood subgraph around an entity."""
        resp = await self._http.get(f"/v1/graph/{entity_id}", params={"depth": depth})
        resp.raise_for_status()
        return GraphNeighborhood.model_validate(resp.json())