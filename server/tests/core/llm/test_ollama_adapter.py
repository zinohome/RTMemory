"""Tests for Ollama LLM adapter."""

import json
import unittest.mock

import httpx
import pytest

from app.core.llm.ollama_adapter import OllamaAdapter
from tests.conftest import make_httpx_response


class TestOllamaAdapterComplete:
    """Tests for OllamaAdapter.complete method."""

    async def test_complete_basic(self, mock_client):
        """complete() sends messages to Ollama and returns assistant text."""
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
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello from Ollama!"

    async def test_complete_uses_openai_compatible_endpoint(self, mock_client):
        """complete() uses /v1/chat/completions endpoint (OpenAI-compatible)."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://localhost:11434/v1/chat/completions"

    async def test_complete_sends_model_and_stream_false(self, mock_client):
        """complete() sends model, stream=false, and other parameters."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=512,
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 512

    async def test_complete_with_response_format(self, mock_client):
        """complete() passes response_format to the Ollama API."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"key": "value"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    async def test_complete_no_api_key(self, mock_client):
        """complete() does not send Authorization header (Ollama is local)."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        headers = call_args.kwargs["headers"]
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    async def test_complete_api_error(self, mock_client):
        """complete() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="500 Internal Server Error",
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
            response=httpx.Response(500, json={"error": "model not found"}),
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

    async def test_complete_custom_base_url(self, mock_client):
        """complete() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Remote"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            base_url="http://remote-ollama.example.com",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://remote-ollama.example.com/v1/chat/completions"


class TestOllamaAdapterCompleteStructured:
    """Tests for OllamaAdapter.complete_structured method."""

    async def test_complete_structured_basic(self, mock_client):
        """complete_structured() returns parsed JSON matching the schema."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"name": "Carol", "age": 35}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await adapter.complete_structured(
            messages=[{"role": "user", "content": "Extract person info"}],
            schema=schema,
        )

        assert result == {"name": "Carol", "age": 35}

        # Verify response_format={"type": "json_object"} was sent
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}

        # Verify schema instruction was added to last user message
        last_msg = payload["messages"][-1]
        assert "Output valid JSON" in last_msg["content"]
        assert "schema" in last_msg["content"].lower() or "properties" in last_msg["content"]

    async def test_complete_structured_appends_to_user_message(self, mock_client):
        """complete_structured() appends instruction to the last user message."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"items": []}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[
                {"role": "system", "content": "You are a data extractor."},
                {"role": "user", "content": "Extract items from: apples, oranges"},
            ],
            schema={"type": "object", "properties": {"items": {"type": "array"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        # System message should be unchanged
        assert payload["messages"][0] == {"role": "system", "content": "You are a data extractor."}
        # User message should have schema instruction appended
        user_msg = payload["messages"][1]
        assert user_msg["content"].startswith("Extract items from: apples, oranges")
        assert "Output valid JSON" in user_msg["content"]

    async def test_complete_structured_adds_user_message_if_none(self, mock_client):
        """complete_structured() adds a user message if none exists."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"result": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[{"role": "system", "content": "You are helpful."}],
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        # Should have original system message plus a new user message
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert "Output valid JSON" in payload["messages"][1]["content"]

    async def test_complete_structured_invalid_json_raises(self, mock_client):
        """complete_structured() raises ValueError when Ollama returns invalid JSON."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I cannot produce JSON."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        with pytest.raises(ValueError, match="Ollama did not return valid JSON"):
            await adapter.complete_structured(
                messages=[{"role": "user", "content": "Test"}],
                schema={"type": "object"},
            )