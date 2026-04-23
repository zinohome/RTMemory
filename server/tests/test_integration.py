"""Integration tests for LLM adapter and embedding service configuration.

These tests verify that the config.yaml can be loaded and used to create
adapters and services through the factory functions. All HTTP calls are mocked.
"""

import unittest.mock
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml

from app.config import AppConfig, load_config
from app.core.llm import create_llm_adapter
from app.core.llm.ollama_adapter import OllamaAdapter
from app.core.llm.openai_adapter import OpenAIAdapter
from app.core.llm.anthropic_adapter import AnthropicAdapter
from app.core.embedding import create_embedding_service
from app.core.embedding.local_embedding import LocalEmbeddingService
from app.core.embedding.openai_embedding import OpenAIEmbeddingService
from tests.conftest import make_httpx_response


class TestConfigIntegration:
    """Tests for loading config.yaml and creating adapters."""

    def test_load_default_config(self):
        """Load the project's config.yaml and verify structure."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            pytest.skip("config.yaml not found at project root")

        config = load_config(config_path)
        assert config.llm.provider in ("openai", "anthropic", "ollama")
        assert config.embedding.provider in ("local", "openai")
        assert config.llm.model
        assert config.embedding.model

    def test_create_ollama_adapter_from_config(self):
        """Create OllamaAdapter from an Ollama config."""
        config_yaml = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)
        adapter = create_llm_adapter(config.llm)
        assert isinstance(adapter, OllamaAdapter)
        assert adapter._model == "qwen2.5:7b"

    def test_create_openai_adapter_from_config(self):
        """Create OpenAIAdapter from an OpenAI config."""
        config_yaml = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config = AppConfig(**config_yaml)
        adapter = create_llm_adapter(config.llm)
        assert isinstance(adapter, OpenAIAdapter)

    def test_create_anthropic_adapter_from_config(self):
        """Create AnthropicAdapter from an Anthropic config."""
        config_yaml = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key": "sk-ant-test-key",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)
        adapter = create_llm_adapter(config.llm)
        assert isinstance(adapter, AnthropicAdapter)

    def test_create_local_embedding_from_config(self):
        """Create LocalEmbeddingService from a local config."""
        config_yaml = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)
        service = create_embedding_service(config.embedding)
        assert isinstance(service, LocalEmbeddingService)

    def test_create_openai_embedding_from_config(self):
        """Create OpenAIEmbeddingService from an OpenAI config."""
        config_yaml = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config = AppConfig(**config_yaml)
        service = create_embedding_service(config.embedding)
        assert isinstance(service, OpenAIEmbeddingService)


class TestEndToEndWithMocking:
    """End-to-end tests with mocked HTTP calls."""

    async def test_ollama_adapter_full_flow(self):
        """Full flow: config -> factory -> adapter -> complete + complete_structured."""
        config_yaml = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock complete response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Ollama!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

        adapter = create_llm_adapter(config.llm, client=mock_client)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == "Hello from Ollama!"

    async def test_openai_adapter_then_embedding_flow(self):
        """Full flow: LLM complete -> embedding with mocked calls."""
        config_yaml = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config = AppConfig(**config_yaml)

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock LLM response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Extracted: Alice likes Python"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

        adapter = create_llm_adapter(config.llm, client=mock_client)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "What does Alice like?"}],
        )
        assert "Alice" in result

        # Mock embedding response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1] * 1536},
                ],
                "model": "text-embedding-3-small",
            },
        )

        embedding_service = create_embedding_service(config.embedding, client=mock_client)
        embeddings = await embedding_service.embed(["Alice likes Python"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert embedding_service.get_dimension() == 1536

    async def test_anthropic_adapter_structured_output_flow(self):
        """Full flow: Anthropic complete_structured with mocked response."""
        config_yaml = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key": "sk-ant-test-key",
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock structured response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"entities": [{"name": "Bob", "type": "person"}], "relations": [], "memories": []}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
            },
        )

        adapter = create_llm_adapter(config.llm, client=mock_client)
        schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
                "relations": {"type": "array"},
                "memories": {"type": "array"},
            },
        }
        result = await adapter.complete_structured(
            messages=[{"role": "user", "content": "Extract info from: Bob lives in NYC"}],
            schema=schema,
        )
        assert result["entities"][0]["name"] == "Bob"
        assert result["entities"][0]["type"] == "person"