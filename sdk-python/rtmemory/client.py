"""RTMemoryClient — async-first Python SDK entry point."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.memories import MemoriesNamespace
from rtmemory.search import SearchNamespace
from rtmemory.profile import ProfileNamespace
from rtmemory.documents import DocumentsNamespace
from rtmemory.conversations import ConversationsNamespace
from rtmemory.graph import GraphNamespace
from rtmemory.spaces import SpacesNamespace


class RTMemoryClient:
    """Main entry point for the RTMemory Python SDK.

    Usage::

        async with RTMemoryClient(base_url="http://localhost:8000", api_key="sk-...") as client:
            result = await client.memories.add(content="...", space_id="sp_001")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._external_client = http_client is not None
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        # Attach namespaces
        self.memories = MemoriesNamespace(self._http)
        self.search = SearchNamespace(self._http)
        self.profile = ProfileNamespace(self._http)
        self.documents = DocumentsNamespace(self._http)
        self.conversations = ConversationsNamespace(self._http)
        self.graph = GraphNamespace(self._http)
        self.spaces = SpacesNamespace(self._http)

    async def __aenter__(self) -> RTMemoryClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client (only if we own it)."""
        if not self._external_client:
            await self._http.aclose()