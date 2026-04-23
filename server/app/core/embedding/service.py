"""Embedding Service abstract base class.

Defines the unified interface for all embedding providers.
Each provider (local sentence-transformers, OpenAI) implements this protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Abstract base class for text embedding services.

    All providers must implement:
    - embed: batch embedding of text strings
    - get_dimension: return the embedding vector dimension
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into vector representations.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.
            Each vector is a list of floats with dimension get_dimension().
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors produced by this service.

        Returns:
            Integer dimension of the embedding vectors.
        """
        ...