"""Tests for create_llm_adapter factory function."""

import pytest

from app.config import LLMConfig
from app.core.llm import create_llm_adapter
from app.core.llm.adapter import LLMAdapter
from app.core.llm.openai_adapter import OpenAIAdapter
from app.core.llm.anthropic_adapter import AnthropicAdapter
from app.core.llm.ollama_adapter import OllamaAdapter


class TestCreateLLMAdapter:
    """Tests for the create_llm_adapter factory function."""

    def test_creates_openai_adapter(self):
        """create_llm_adapter creates OpenAIAdapter for provider='openai'."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test-key",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._api_key == "sk-test-key"
        assert adapter._model == "gpt-4o"
        assert adapter._base_url == "https://api.openai.com/v1"

    def test_creates_openai_adapter_with_custom_base_url(self):
        """create_llm_adapter respects custom base_url for OpenAI."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test-key",
            base_url="https://proxy.example.com/v1",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter._base_url == "https://proxy.example.com/v1"

    def test_creates_anthropic_adapter(self):
        """create_llm_adapter creates AnthropicAdapter for provider='anthropic'."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test-key",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._api_key == "sk-ant-test-key"
        assert adapter._model == "claude-sonnet-4-20250514"
        assert adapter._base_url == "https://api.anthropic.com"

    def test_creates_anthropic_adapter_with_custom_base_url(self):
        """create_llm_adapter respects custom base_url for Anthropic."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test-key",
            base_url="https://proxy.anthropic.example.com",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._base_url == "https://proxy.anthropic.example.com"

    def test_creates_ollama_adapter(self):
        """create_llm_adapter creates OllamaAdapter for provider='ollama'."""
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OllamaAdapter)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._model == "qwen2.5:7b"
        assert adapter._base_url == "http://localhost:11434"

    def test_creates_ollama_adapter_with_default_base_url(self):
        """create_llm_adapter uses default Ollama base_url when not specified."""
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OllamaAdapter)
        assert adapter._base_url == "http://localhost:11434"

    def test_raises_for_unknown_provider(self):
        """create_llm_adapter raises ValueError for unknown provider."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-pro",
            api_key="test-key",
        )
        with pytest.raises(ValueError, match="Unknown LLM provider: gemini"):
            create_llm_adapter(config)

    def test_case_insensitive_provider(self):
        """create_llm_adapter is case-insensitive for provider name."""
        config = LLMConfig(
            provider="OpenAI",
            model="gpt-4o",
            api_key="sk-test-key",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)

    async def test_created_openai_adapter_can_complete(self):
        """Factory-created OpenAI adapter has working complete method."""
        import unittest.mock
        import httpx
        from tests.conftest import make_httpx_response

        mock_client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Factory works!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

        config = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        adapter = create_llm_adapter(config, client=mock_client)
        result = await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        assert result == "Factory works!"