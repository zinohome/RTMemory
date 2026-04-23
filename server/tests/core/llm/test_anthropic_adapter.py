"""Tests for Anthropic LLM adapter."""

import json
import unittest.mock

import httpx
import pytest

from app.core.llm.anthropic_adapter import AnthropicAdapter
from tests.conftest import make_httpx_response


class TestAnthropicAdapterComplete:
    """Tests for AnthropicAdapter.complete method."""

    async def test_complete_basic(self, mock_client):
        """complete() sends messages to Anthropic and returns assistant text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello from Claude!"

    async def test_complete_extracts_system_message(self, mock_client):
        """complete() sends system message as a top-level parameter."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I am helpful."}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 4},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        # System message should be in top-level "system" field, not in messages
        assert payload["system"] == "You are helpful."
        assert all(msg["role"] != "system" for msg in payload["messages"])

    async def test_complete_sends_anthropic_headers(self, mock_client):
        """complete() includes x-api-key and anthropic-version headers."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "OK"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["x-api-key"] == "sk-ant-test-key"
        assert call_args.kwargs["headers"]["anthropic-version"] == "2023-06-01"

    async def test_complete_custom_parameters(self, mock_client):
        """complete() uses custom temperature and max_tokens."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Precise"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete(
            messages=[{"role": "user", "content": "Be precise"}],
            temperature=0.1,
            max_tokens=2048,
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["temperature"] == 0.1
        assert call_args.kwargs["json"]["max_tokens"] == 2048

    async def test_complete_api_error(self, mock_client):
        """complete() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            response=httpx.Response(401, json={"error": {"message": "Invalid API key"}}),
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-invalid",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

    async def test_complete_custom_base_url(self, mock_client):
        """complete() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Proxy"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            base_url="https://proxy.anthropic.example.com",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://proxy.anthropic.example.com/v1/messages"


class TestAnthropicAdapterCompleteStructured:
    """Tests for AnthropicAdapter.complete_structured method."""

    async def test_complete_structured_basic(self, mock_client):
        """complete_structured() returns parsed JSON matching the schema."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"name": "Bob", "age": 25}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
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

        assert result == {"name": "Bob", "age": 25}

        # Verify schema instruction was added to system message
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert "system" in payload
        assert "valid JSON matching this schema" in payload["system"]

    async def test_complete_structured_adds_system_if_missing(self, mock_client):
        """complete_structured() adds a system message if none exists."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"result": true}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[{"role": "user", "content": "Test"}],
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert "system" in payload
        assert "valid JSON" in payload["system"]

    async def test_complete_structured_appends_to_existing_system(self, mock_client):
        """complete_structured() appends schema instruction to existing system message."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"items": []}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[
                {"role": "system", "content": "You are a data extractor."},
                {"role": "user", "content": "Extract items"},
            ],
            schema={"type": "object", "properties": {"items": {"type": "array"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["system"].startswith("You are a data extractor.")
        assert "valid JSON" in payload["system"]

    async def test_complete_structured_invalid_json_raises(self, mock_client):
        """complete_structured() raises ValueError when Anthropic returns invalid JSON."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I cannot produce JSON."}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        with pytest.raises(ValueError, match="Anthropic did not return valid JSON"):
            await adapter.complete_structured(
                messages=[{"role": "user", "content": "Test"}],
                schema={"type": "object"},
            )