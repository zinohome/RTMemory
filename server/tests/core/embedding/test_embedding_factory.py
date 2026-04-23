"""Tests for create_embedding_service factory function."""

import unittest.mock

import pytest

from app.config import EmbeddingConfig
from app.core.embedding import create_embedding_service
from app.core.embedding.service import EmbeddingService
from app.core.embedding.local_embedding import LocalEmbeddingService
from app.core.embedding.openai_embedding import OpenAIEmbeddingService


class TestCreateEmbeddingService:
    """Tests for the create_embedding_service factory function."""

    def test_creates_local_embedding_service(self):
        """create_embedding_service creates LocalEmbeddingService for provider='local'."""
        config = EmbeddingConfig(
            provider="local",
            model="BAAI/bge-base-zh-v1.5",
        )
        service = create_embedding_service(config)
        assert isinstance(service, LocalEmbeddingService)
        assert isinstance(service, EmbeddingService)
        assert service._model_name == "BAAI/bge-base-zh-v1.5"

    def test_creates_openai_embedding_service(self):
        """create_embedding_service creates OpenAIEmbeddingService for provider='openai'."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )
        service = create_embedding_service(config)
        assert isinstance(service, OpenAIEmbeddingService)
        assert isinstance(service, EmbeddingService)
        assert service._api_key == "sk-test-key"
        assert service._model == "text-embedding-3-small"

    def test_creates_openai_with_custom_base_url(self):
        """create_embedding_service uses custom base_url for OpenAI."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
            base_url="https://proxy.example.com/v1",
        )
        service = create_embedding_service(config)
        assert isinstance(service, OpenAIEmbeddingService)
        assert service._base_url == "https://proxy.example.com/v1"

    def test_openai_default_base_url(self):
        """create_embedding_service defaults to OpenAI's base URL."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )
        service = create_embedding_service(config)
        assert service._base_url == "https://api.openai.com/v1"

    def test_raises_for_unknown_provider(self):
        """create_embedding_service raises ValueError for unknown provider."""
        config = EmbeddingConfig(
            provider="cohere",
            model="embed-english-v3",
            api_key="test-key",
        )
        with pytest.raises(ValueError, match="Unknown embedding provider: cohere"):
            create_embedding_service(config)

    def test_case_insensitive_provider(self):
        """create_embedding_service is case-insensitive for provider name."""
        config = EmbeddingConfig(
            provider="Local",
            model="BAAI/bge-base-zh-v1.5",
        )
        service = create_embedding_service(config)
        assert isinstance(service, LocalEmbeddingService)

    async def test_created_local_service_can_embed(self):
        """Factory-created local embedding service has working embed method."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = __import__("numpy").array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with unittest.mock.patch(
            "app.core.embedding.local_embedding._SentenceTransformer",
            return_value=mock_model,
        ):
            config = EmbeddingConfig(provider="local", model="test-model")
            service = create_embedding_service(config)
            result = await service.embed(["hello"])
            assert result == [[0.1, 0.2, 0.3]]

    async def test_created_openai_service_can_embed(self):
        """Factory-created OpenAI embedding service has working embed method."""
        import httpx
        from tests.conftest import make_httpx_response

        mock_client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
        )
        service = create_embedding_service(config, client=mock_client)
        result = await service.embed(["hello"])
        assert result == [[0.1, 0.2]]