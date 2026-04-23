"""Local embedding service using sentence-transformers.

Provides local text embedding by lazy-loading a sentence-transformers model.
No API calls are made -- everything runs on the local machine.
"""

from __future__ import annotations

import numpy as np

from app.core.embedding.service import EmbeddingService

# Module-level import alias for testability — mock this in tests
from sentence_transformers import SentenceTransformer as _SentenceTransformer


class LocalEmbeddingService(EmbeddingService):
    """Embedding service using a locally-loaded sentence-transformers model.

    The model is lazy-loaded on the first call to embed() or get_dimension(),
    avoiding the overhead of loading the model at import time.

    Args:
        model_name: HuggingFace model name or path.
            Defaults to "BAAI/bge-base-zh-v1.5" (Chinese-optimized embedding model).
    """

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            self._model = _SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings using the local model.

        The model is lazy-loaded on the first call.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.
        """
        if not texts:
            return []

        self._load_model()
        embeddings: np.ndarray = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors.

        The model is lazy-loaded if not already loaded.

        Returns:
            Integer dimension of the embedding vectors.
        """
        self._load_model()
        return self._dimension