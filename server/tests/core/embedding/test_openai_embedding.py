"""Tests for OpenAIEmbeddingService."""

import unittest.mock

import httpx
import pytest

from app.core.embedding.openai_embedding import OpenAIEmbeddingService
from tests.conftest import make_httpx_response


class TestOpenAIEmbeddingService:
    """Tests for OpenAIEmbeddingService."""

    async def test_embed_single_text(self, mock_client):
        """embed() returns embeddings for a single text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
            client=mock_client,
        )
        result = await service.embed(["hello"])

        assert result == [[0.1, 0.2, 0.3]]

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["model"] == "text-embedding-3-small"
        assert call_args.kwargs["json"]["input"] == ["hello"]

    async def test_embed_multiple_texts(self, mock_client):
        """embed() returns embeddings for multiple texts."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
            client=mock_client,
        )
        result = await service.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    async def test_embed_sends_authorization_header(self, mock_client):
        """embed() includes Bearer token in Authorization header."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-my-secret",
            model="text-embedding-3-small",
            client=mock_client,
        )
        await service.embed(["test"])

        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer sk-my-secret"

    async def test_embed_api_error(self, mock_client):
        """embed() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
            response=httpx.Response(401, json={"error": {"message": "Invalid API key"}}),
        )

        service = OpenAIEmbeddingService(
            api_key="sk-invalid",
            model="text-embedding-3-small",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.embed(["test"])

    def test_get_dimension_text_embedding_3_small(self):
        """get_dimension() returns 1536 for text-embedding-3-small."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
        )
        assert service.get_dimension() == 1536

    def test_get_dimension_text_embedding_3_large(self):
        """get_dimension() returns 3072 for text-embedding-3-large."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-large",
        )
        assert service.get_dimension() == 3072

    def test_get_dimension_text_embedding_ada_002(self):
        """get_dimension() returns 1536 for text-embedding-ada-002."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-ada-002",
        )
        assert service.get_dimension() == 1536

    def test_get_dimension_unknown_model_defaults_to_1536(self):
        """get_dimension() returns 1536 for unknown models."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="my-custom-model",
        )
        assert service.get_dimension() == 1536

    async def test_embed_uses_custom_base_url(self, mock_client):
        """embed() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
            base_url="https://proxy.example.com/v1",
            client=mock_client,
        )
        await service.embed(["test"])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://proxy.example.com/v1/embeddings"