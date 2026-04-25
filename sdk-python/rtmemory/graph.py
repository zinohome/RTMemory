"""GraphNamespace — async methods for the /v1/graph/ API."""

from __future__ import annotations

from typing import Optional

import httpx

from rtmemory.types import GraphNeighborhood


class GraphNamespace:
    """Namespace for graph visualization and traversal."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def neighborhood(
        self,
        entity_id: str,
        space_id: Optional[str] = None,
        max_hops: int = 3,
        relation_types: Optional[list[str]] = None,
        direction: str = "both",
    ) -> GraphNeighborhood:
        """Get the neighborhood subgraph around an entity.

        Calls GET /v1/graph/neighborhood with the specified parameters.
        """
        params: dict[str, str | int] = {
            "entity_id": entity_id,
            "max_hops": max_hops,
            "direction": direction,
        }
        if space_id is not None:
            params["space_id"] = space_id
        if relation_types is not None:
            params["relation_types"] = ",".join(relation_types)
        resp = await self._http.get("/v1/graph/neighborhood", params=params)
        resp.raise_for_status()
        return GraphNeighborhood.model_validate(resp.json())