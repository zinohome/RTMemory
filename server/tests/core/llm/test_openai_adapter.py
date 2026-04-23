"""Tests for OpenAI LLM adapter."""

import json
import unittest.mock

import httpx
import pytest

from app.core.llm.openai_adapter import OpenAIAdapter
from tests.conftest import make_httpx_response


class TestOpenAIAdapterComplete:
    """Tests for OpenAIAdapter.complete method."""

    async def test_complete_basic(self, mock_client):
        """complete() sends messages to OpenAI and returns assistant text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello!"
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["model"] == "gpt-4o"
        assert call_args.kwargs["json"]["messages"] == [{"role": "user", "content": "Hi"}]
        assert call_args.kwargs["json"]["temperature"] == 0.7
        assert call_args.kwargs["json"]["max_tokens"] == 1024

    async def test_complete_with_system_message(self, mock_client):
        """complete() sends system + user messages to OpenAI."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I can help."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Help me"},
            ],
        )

        assert result == "I can help."
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me"},
        ]

    async def test_complete_custom_parameters(self, mock_client):
        """complete() uses custom temperature and max_tokens."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Precise answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Be precise"}],
            temperature=0.1,
            max_tokens=2048,
        )

        assert result == "Precise answer"
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["temperature"] == 0.1
        assert call_args.kwargs["json"]["max_tokens"] == 2048

    async def test_complete_with_response_format(self, mock_client):
        """complete() passes response_format to OpenAI API."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
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

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    async def test_complete_api_error(self, mock_client):
        """complete() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            response=httpx.Response(401, json={"error": {"message": "Invalid API key"}}),
        )

        adapter = OpenAIAdapter(
            api_key="sk-invalid",
            model="gpt-4o",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

    async def test_complete_uses_authorization_header(self, mock_client):
        """complete() includes Bearer token in Authorization header."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
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

        adapter = OpenAIAdapter(
            api_key="sk-my-secret-key",
            model="gpt-4o",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer sk-my-secret-key"
        assert call_args.kwargs["headers"]["Content-Type"] == "application/json"

    async def test_complete_custom_base_url(self, mock_client):
        """complete() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Proxy response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://proxy.example.com/v1",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://proxy.example.com/v1/chat/completions"


class TestOpenAIAdapterCompleteStructured:
    """Tests for OpenAIAdapter.complete_structured method."""

    async def test_complete_structured_basic(self, mock_client):
        """complete_structured() returns parsed JSON matching the schema."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "Alice", "age": 30}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
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

        assert result == {"name": "Alice", "age": 30}

        # Verify json_schema response_format is passed
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "response"
        assert payload["response_format"]["json_schema"]["schema"] == schema

    async def test_complete_structured_uses_low_temperature(self, mock_client):
        """complete_structured() defaults to temperature=0.1."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"result": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[{"role": "user", "content": "Test"}],
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["temperature"] == 0.1

    async def test_complete_structured_invalid_json_raises(self, mock_client):
        """complete_structured() raises ValueError when LLM returns invalid JSON."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Not JSON at all"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        with pytest.raises(ValueError, match="OpenAI did not return valid JSON"):
            await adapter.complete_structured(
                messages=[{"role": "user", "content": "Test"}],
                schema={"type": "object"},
            )