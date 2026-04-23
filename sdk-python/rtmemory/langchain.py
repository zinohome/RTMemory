"""LangChain integration — RTMemory as a LangChain Tool and VectorStore.

Provides:
- RTMemoryTool: A LangChain BaseTool that wraps search/add/get_profile operations.
- RTMemoryVectorStore: A LangChain VectorStore that uses RTMemory hybrid search.

Usage::

    from rtmemory.langchain import RTMemoryTool, RTMemoryVectorStore

    # As a Tool
    tool = RTMemoryTool(base_url="http://localhost:8000", space_id="sp_001")

    # As a VectorStore
    vs = RTMemoryVectorStore(base_url="http://localhost:8000", space_id="sp_001")
    results = vs.similarity_search("user preferences", k=5)
"""
from __future__ import annotations

from typing import Any

try:
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
except ImportError as exc:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install rtmemory-server[langchain]"
    ) from exc

from rtmemory.client import RTMemoryClient


class RTMemoryTool(BaseTool):
    """LangChain tool wrapping RTMemory search and memory operations.

    The tool performs a hybrid search query against the RTMemory server
    and returns formatted results as a string.
    """

    name: str = "rtmemory_search"
    description: str = (
        "Search the user's memories and knowledge base. "
        "Use this when you need to recall information about the user, "
        "their preferences, past conversations, or documents in the knowledge base."
    )
    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    space_id: str = "default"
    user_id: str | None = None
    search_mode: str = "hybrid"
    limit: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Synchronous run — delegates to async via httpx sync client."""
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "q": query,
            "space_id": self.space_id,
            "mode": self.search_mode,
            "limit": self.limit,
            "include_profile": True,
        }
        if self.user_id:
            body["user_id"] = self.user_id

        with httpx.Client(base_url=self.base_url, headers=headers, timeout=30.0) as client:
            resp = client.post("/v1/search/", json=body)
            resp.raise_for_status()
            data = resp.json()

        # Format results
        lines: list[str] = []
        for r in data.get("results", []):
            source = r.get("source", "unknown")
            content = r.get("content", "")
            score = r.get("score", 0.0)
            lines.append(f"[{source}] (score: {score:.3f}) {content}")

        profile = data.get("profile")
        if profile:
            lines.append("\n--- User Profile ---")
            for layer in ["identity", "preferences", "currentStatus", "relationships"]:
                layer_data = profile.get(layer, {})
                if layer_data:
                    lines.append(f"{layer}: {layer_data}")

        return "\n".join(lines) if lines else "No results found."

    async def _arun(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Async run — uses the async RTMemoryClient."""
        async with RTMemoryClient(base_url=self.base_url, api_key=self.api_key) as client:
            result = await client.search(
                q=query,
                space_id=self.space_id,
                user_id=self.user_id,
                mode=self.search_mode,
                limit=self.limit,
                include_profile=True,
            )

        lines: list[str] = []
        for r in result.results:
            lines.append(f"[{r.source}] (score: {r.score:.3f}) {r.content}")

        if result.profile:
            lines.append("\n--- User Profile ---")
            for layer_name in ["identity", "preferences", "current_status", "relationships"]:
                layer = getattr(result.profile, layer_name, None)
                if layer:
                    lines.append(f"{layer_name}: {layer}")

        return "\n".join(lines) if lines else "No results found."


class RTMemoryVectorStore(VectorStore):
    """LangChain VectorStore backed by RTMemory hybrid search.

    Supports similarity_search and add_texts operations via the RTMemory API.
    """

    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    space_id: str = "default"
    user_id: str | None = None

    class Config:
        arbitrary_types_allowed = True

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for documents similar to the query."""
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {
            "q": query,
            "space_id": self.space_id,
            "mode": "hybrid",
            "limit": k,
        }
        if self.user_id:
            body["user_id"] = self.user_id

        with httpx.Client(base_url=self.base_url, headers=headers, timeout=30.0) as client:
            resp = client.post("/v1/search/", json=body)
            resp.raise_for_status()
            data = resp.json()

        documents: list[Document] = []
        for r in data.get("results", []):
            metadata = r.get("metadata", {})
            metadata["source"] = r.get("source", "")
            metadata["score"] = r.get("score", 0.0)
            if r.get("entity"):
                metadata["entity"] = r["entity"]
            if r.get("document"):
                metadata["document"] = r["document"]
            documents.append(Document(page_content=r.get("content", ""), metadata=metadata))

        return documents

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Async search for documents similar to the query."""
        async with RTMemoryClient(base_url=self.base_url, api_key=self.api_key) as client:
            result = await client.search(
                q=query,
                space_id=self.space_id,
                user_id=self.user_id,
                mode="hybrid",
                limit=k,
            )

        documents: list[Document] = []
        for r in result.results:
            metadata: dict[str, Any] = {}
            metadata["source"] = r.source
            metadata["score"] = r.score
            if r.entity:
                metadata["entity"] = {"name": r.entity.name, "type": r.entity.type}
            if r.document:
                metadata["document"] = {"title": r.document.title, "url": r.document.url}
            documents.append(Document(page_content=r.content, metadata=metadata))

        return documents

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts as memories to RTMemory."""
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        ids: list[str] = []
        with httpx.Client(base_url=self.base_url, headers=headers, timeout=30.0) as client:
            for i, text in enumerate(texts):
                body: dict[str, Any] = {
                    "content": text,
                    "space_id": self.space_id,
                }
                if self.user_id:
                    body["user_id"] = self.user_id
                if metadatas and i < len(metadatas):
                    body["metadata"] = metadatas[i]

                resp = client.post("/v1/memories/", json=body)
                resp.raise_for_status()
                data = resp.json()
                ids.append(data.get("id", ""))

        return ids

    async def aadd_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async add texts as memories to RTMemory."""
        async with RTMemoryClient(base_url=self.base_url, api_key=self.api_key) as client:
            ids: list[str] = []
            for i, text in enumerate(texts):
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                resp = await client.memories.add(
                    content=text,
                    space_id=self.space_id,
                    user_id=self.user_id,
                    metadata=meta,
                )
                ids.append(resp.id)
            return ids

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any = None,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> "RTMemoryVectorStore":
        """Create an RTMemoryVectorStore and add texts to it.

        Note: The `embedding` parameter is ignored — RTMemory uses its own
        embedding service on the server side.
        """
        instance = cls(**kwargs)
        instance.add_texts(texts, metadatas=metadatas)
        return instance