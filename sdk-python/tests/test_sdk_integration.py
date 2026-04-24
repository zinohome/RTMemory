"""SDK integration tests — verify request/response shape for each namespace."""

import uuid
from datetime import datetime, timezone

import httpx
import pytest
import respx

from rtmemory import RTMemoryClient
from rtmemory.types import (
    ConversationAddResponse,
    Document,
    DocumentListResponse,
    GraphNeighborhood,
    Memory,
    MemoryAddResponse,
    MemoryListResponse,
    ProfileResponse,
    SearchResponse,
    Space,
    SpaceListResponse,
)

BASE = "http://localhost:8000"


class TestMemoryNamespaceIntegration:
    """Integration tests for MemoriesNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_add_memory(self):
        respx.post(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(200, json={
                "id": "mem_001",
                "content": "User likes dark mode",
                "confidence": 1.0,
                "entity_id": None,
                "relation_ids": [],
                "memory_type": "fact",
                "custom_id": None,
                "created_at": "2026-01-01T00:00:00+00:00",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.memories.add(
                content="User likes dark mode",
                space_id="sp_001",
            )
        assert result.id == "mem_001"
        assert result.content == "User likes dark mode"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_memories(self):
        respx.get(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(200, json={
                "items": [],
                "total": 0,
                "offset": 0,
                "limit": 20,
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.memories.list(space_id="sp_001")
        assert result.total == 0
        assert result.items == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_forget_memory(self):
        respx.post(f"{BASE}/v1/memories/forget").mock(
            return_value=httpx.Response(200, json={
                "id": "mem_001",
                "content": "User likes dark mode",
                "is_forgotten": True,
                "forget_reason": "outdated",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.memories.forget(
                memory_id="mem_001",
                reason="outdated",
            )
        assert result["is_forgotten"] is True


class TestSearchNamespaceIntegration:
    """Integration tests for SearchNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        respx.post(f"{BASE}/v1/search/").mock(
            return_value=httpx.Response(200, json={
                "results": [
                    {
                        "type": "memory",
                        "content": "User prefers dark theme",
                        "score": 0.95,
                        "source": "vector",
                        "entity": None,
                        "document": None,
                        "metadata": {},
                    }
                ],
                "timing": {"total_ms": 50.0},
                "query": "theme preference",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.search(
                q="theme preference",
                space_id="sp_001",
                mode="hybrid",
            )
        assert len(result.results) == 1
        assert result.results[0].content == "User prefers dark theme"
        assert result.results[0].score == 0.95


class TestProfileNamespaceIntegration:
    """Integration tests for ProfileNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_profile(self):
        respx.post(f"{BASE}/v1/profile").mock(
            return_value=httpx.Response(200, json={
                "profile": {
                    "identity": {"name": "Alice"},
                    "preferences": {"theme": "dark"},
                    "current_status": {},
                    "relationships": {},
                    "dynamic_memories": [],
                },
                "confidence": {"identity": 0.9, "preferences": 0.8},
                "computed_at": "2026-01-01T00:00:00+00:00",
                "timing_ms": 25.0,
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.profile.get(
                entity_id="ent_001",
                space_id="sp_001",
            )
        assert result.profile.identity == {"name": "Alice"}
        assert result.confidence["identity"] == 0.9


class TestDocumentsNamespaceIntegration:
    """Integration tests for DocumentsNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_add_document(self):
        respx.post(f"{BASE}/v1/documents/").mock(
            return_value=httpx.Response(200, json={
                "id": "doc_001",
                "title": "Test Document",
                "doc_type": "text",
                "status": "queued",
                "space_id": "sp_001",
                "created_at": "2026-01-01T00:00:00+00:00",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.documents.add(
                content="Test content",
                space_id="sp_001",
                title="Test Document",
            )
        assert result.id == "doc_001"
        assert result.status == "queued"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_documents(self):
        respx.get(f"{BASE}/v1/documents/").mock(
            return_value=httpx.Response(200, json={
                "items": [],
                "total": 0,
                "offset": 0,
                "limit": 20,
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.documents.list(space_id="sp_001")
        assert result.total == 0


class TestConversationsNamespaceIntegration:
    """Integration tests for ConversationsNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_add_conversation(self):
        respx.post(f"{BASE}/v1/conversations/").mock(
            return_value=httpx.Response(200, json={
                "id": "conv_001",
                "memory_ids": ["mem_001", "mem_002"],
                "entity_ids": ["ent_001"],
                "created_at": "2026-01-01T00:00:00+00:00",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.conversations.add(
                messages=[
                    {"role": "user", "content": "I prefer dark mode"},
                ],
                space_id="sp_001",
            )
        assert result.id == "conv_001"
        assert len(result.memory_ids) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_end_conversation(self):
        respx.post(f"{BASE}/v1/conversations/end").mock(
            return_value=httpx.Response(200, json={
                "id": "conv_001",
                "memory_ids": [],
                "entity_ids": [],
                "created_at": "2026-01-01T00:00:00+00:00",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.conversations.end("conv_001")
        assert result.id == "conv_001"


class TestSpacesNamespaceIntegration:
    """Integration tests for SpacesNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_create_space(self):
        respx.post(f"{BASE}/v1/spaces/").mock(
            return_value=httpx.Response(201, json={
                "id": "sp_002",
                "name": "New Space",
                "description": "A new space",
                "org_id": "org_001",
                "is_default": False,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.spaces.create(
                name="New Space",
                description="A new space",
                org_id="org_001",
            )
        assert result.id == "sp_002"
        assert result.name == "New Space"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_spaces(self):
        respx.get(f"{BASE}/v1/spaces/").mock(
            return_value=httpx.Response(200, json={
                "items": [
                    {
                        "id": "sp_001",
                        "name": "Default",
                        "description": None,
                        "org_id": "org_001",
                        "is_default": True,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
                "total": 1,
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.spaces.list()
        assert result.total == 1
        assert result.items[0].name == "Default"


class TestGraphNamespaceIntegration:
    """Integration tests for GraphNamespace."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_neighborhood(self):
        entity_id = str(uuid.uuid4())
        respx.post(f"{BASE}/v1/memories/traverse").mock(
            return_value=httpx.Response(200, json={
                "center": {
                    "id": entity_id,
                    "name": "Alice",
                    "entity_type": "person",
                    "description": "",
                    "confidence": 1.0,
                },
                "entities": [
                    {"id": entity_id, "name": "Alice", "entity_type": "person", "description": "", "confidence": 1.0},
                    {"id": str(uuid.uuid4()), "name": "Bob", "entity_type": "person", "description": "", "confidence": 1.0},
                ],
                "relations": [
                    {
                        "id": str(uuid.uuid4()),
                        "source_entity_id": entity_id,
                        "target_entity_id": str(uuid.uuid4()),
                        "relation_type": "knows",
                        "value": "",
                        "valid_from": "2026-01-01T00:00:00+00:00",
                        "valid_to": None,
                        "confidence": 1.0,
                        "is_current": True,
                    }
                ],
                "depth": 3,
            })
        )

        async with RTMemoryClient(base_url=BASE) as client:
            result = await client.graph.neighborhood(entity_id=entity_id, max_hops=3)
        assert result.center.name == "Alice"
        assert len(result.entities) == 2
        assert len(result.relations) == 1