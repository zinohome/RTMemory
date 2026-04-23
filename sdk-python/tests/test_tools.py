"""Tests for rtmemory.tools — generic agent tool definitions."""

import pytest
import respx
import httpx

from rtmemory import RTMemoryClient
from rtmemory.tools import get_memory_tools


BASE = "http://localhost:8000"


class TestGetMemoryTools:
    @pytest.mark.asyncio
    async def test_returns_five_tools(self):
        async with RTMemoryClient(base_url=BASE) as client:
            tools = get_memory_tools(client, space_id="sp_001")
        assert len(tools) == 5
        names = [t["name"] for t in tools]
        assert "search_memories" in names
        assert "add_memory" in names
        assert "get_profile" in names
        assert "forget_memory" in names
        assert "add_document" in names

    @pytest.mark.asyncio
    async def test_each_tool_has_required_keys(self):
        async with RTMemoryClient(base_url=BASE) as client:
            tools = get_memory_tools(client, space_id="sp_001")
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "function" in tool
            assert tool["parameters"]["type"] == "object"

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_memories_tool_calls_client(self):
        respx.post(f"{BASE}/v1/search/").mock(
            return_value=httpx.Response(
                200,
                json={"results": [], "timing_ms": 10},
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            tools = get_memory_tools(client, space_id="sp_001", user_id="user_001")
            fn = next(t["function"] for t in tools if t["name"] == "search_memories")
            result = await fn(q="What does Zhang Jun use?")
        assert "results" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_add_memory_tool_calls_client(self):
        respx.post(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(
                200,
                json={"id": "mem_001", "content": "test", "confidence": 1.0},
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            tools = get_memory_tools(client, space_id="sp_001")
            fn = next(t["function"] for t in tools if t["name"] == "add_memory")
            result = await fn(content="I prefer dark mode")
        assert result["id"] == "mem_001"