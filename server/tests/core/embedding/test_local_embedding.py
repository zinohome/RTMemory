"""Tests for LocalEmbeddingService using sentence-transformers."""

import unittest.mock

import numpy as np
import pytest

from app.core.embedding.local_embedding import LocalEmbeddingService


class TestLocalEmbeddingService:
    """Tests for LocalEmbeddingService."""

    def test_get_dimension_before_embed(self):
        """get_dimension() lazy-loads the model and returns the dimension."""
        mock_model = unittest.mock.MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="BAAI/bge-base-zh-v1.5")
            dim = service.get_dimension()
            assert dim == 768

    async def test_embed_single_text(self):
        """embed() returns embeddings for a single text."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            result = await service.embed(["hello"])
            assert result == [[0.1, 0.2, 0.3]]
            mock_model.encode.assert_called_once_with(["hello"], normalize_embeddings=True)

    async def test_embed_multiple_texts(self):
        """embed() returns embeddings for multiple texts."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            result = await service.embed(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

    async def test_embed_lazy_loads_model(self):
        """embed() lazy-loads the model on first call."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_sentence_embedding_dimension.return_value = 2

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls:
            service = LocalEmbeddingService(model_name="test-model")
            # Model should not be loaded yet
            mock_cls.assert_not_called()

            # Model should be loaded on first embed call
            await service.embed(["hello"])
            mock_cls.assert_called_once_with("test-model")

            # Second call should not reload
            await service.embed(["world"])
            assert mock_cls.call_count == 1

    async def test_embed_empty_list(self):
        """embed() returns empty list for empty input."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([])
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            result = await service.embed([])
            assert result == []

    def test_default_model_name(self):
        """LocalEmbeddingService defaults to bge-base-zh-v1.5."""
        service = LocalEmbeddingService()
        assert service._model_name == "BAAI/bge-base-zh-v1.5"

    async def test_get_dimension_matches_embed_dimension(self):
        """get_dimension() returns the same dimension as the embeddings."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 6

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            dim = service.get_dimension()
            result = await service.embed(["hello"])
            assert dim == len(result[0])