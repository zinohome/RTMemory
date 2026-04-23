"""OpenAI embedding service.

Provides text embeddings via the OpenAI embeddings API.
Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.
"""

from __future__ import annotations

import httpx

from app.core.embedding.service import EmbeddingService


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using the OpenAI embeddings API.

    Args:
        api_key: OpenAI API key.
        model: Embedding model name. Defaults to "text-embedding-3-small".
        base_url: API base URL. Defaults to "https://api.openai.com/v1".
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    # Known dimensions for OpenAI embedding models
    _MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings using the OpenAI embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors.

        Uses a lookup table for known OpenAI models.
        Returns 1536 (the most common dimension) for unknown models.

        Returns:
            Integer dimension of the embedding vectors.
        """
        return self._MODEL_DIMENSIONS.get(self._model, 1536)