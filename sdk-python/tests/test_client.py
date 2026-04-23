"""Tests for RTMemoryClient and all namespace methods using respx mock transport."""

import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from rtmemory import RTMemoryClient
from rtmemory.types import (
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


# ── Client construction ──────────────────────────────────────────────────


class TestClientConstruction:
    def test_client_defaults(self):
        client = RTMemoryClient(base_url=BASE)
        assert client.base_url == BASE
        assert client.api_key is None
        # Need to close properly for async client
        import asyncio
        asyncio.run(client.close())

    def test_client_with_api_key(self):
        client = RTMemoryClient(base_url=BASE, api_key="sk-test")
        assert client.api_key == "sk-test"
        import asyncio
        asyncio.run(client.close())

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with RTMemoryClient(base_url=BASE) as c:
            assert c.base_url == BASE


# ── Memories namespace ───────────────────────────────────────────────────


class TestMemoriesNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_add(self):
        respx.post(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "mem_001",
                    "content": "I moved to Beijing",
                    "custom_id": "ext_123",
                    "confidence": 0.9,
                    "entity_id": "ent_001",
                    "relation_ids": ["rel_001"],
                    "memory_type": "fact",
                    "created_at": "2026-04-23T10:00:00Z",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.memories.add(
                content="I moved to Beijing",
                space_id="sp_001",
                custom_id="ext_123",
                entity_context="Zhang Jun's knowledge base",
                metadata={"source": "slack"},
            )
        assert resp.id == "mem_001"
        assert resp.content == "I moved to Beijing"
        assert resp.confidence == 0.9

    @respx.mock
    @pytest.mark.asyncio
    async def test_list(self):
        respx.get(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "mem_001",
                            "content": "I moved to Beijing",
                            "confidence": 0.9,
                            "space_id": "sp_001",
                        }
                    ],
                    "total": 1,
                    "offset": 0,
                    "limit": 20,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.memories.list(space_id="sp_001")
        assert resp.total == 1
        assert resp.items[0].id == "mem_001"

    @respx.mock
    @pytest.mark.asyncio
    async def test_get(self):
        respx.get(f"{BASE}/v1/memories/mem_001").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "mem_001",
                    "content": "I moved to Beijing",
                    "confidence": 0.9,
                    "space_id": "sp_001",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.memories.get("mem_001")
        assert resp.id == "mem_001"

    @respx.mock
    @pytest.mark.asyncio
    async def test_update(self):
        respx.patch(f"{BASE}/v1/memories/mem_001").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "mem_001",
                    "content": "Updated content",
                    "confidence": 0.9,
                    "space_id": "sp_001",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.memories.update(
                "mem_001", content="Updated content", metadata={"edited": True}
            )
        assert resp.content == "Updated content"

    @respx.mock
    @pytest.mark.asyncio
    async def test_forget_by_id(self):
        respx.post(f"{BASE}/v1/memories/forget").mock(
            return_value=httpx.Response(200, json={"forgotten": True, "memory_id": "mem_001"})
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.memories.forget(memory_id="mem_001", reason="user requested")
        assert resp["forgotten"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_forget_by_content_match(self):
        respx.post(f"{BASE}/v1/memories/forget").mock(
            return_value=httpx.Response(200, json={"forgotten": True, "count": 2})
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.memories.forget(content_match="lives in Shanghai", reason="outdated")
        assert resp["forgotten"] is True


# ── Search namespace ─────────────────────────────────────────────────────


class TestSearchNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self):
        respx.post(f"{BASE}/v1/search/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "type": "memory",
                            "content": "Zhang Jun uses Next.js",
                            "score": 0.92,
                            "source": "vector+graph",
                            "entity": {"name": "Zhang Jun", "type": "person"},
                            "metadata": {},
                        }
                    ],
                    "profile": {
                        "identity": {"name": "Zhang Jun"},
                        "preferences": {},
                        "current_status": {},
                        "relationships": {},
                        "dynamic_memories": [],
                    },
                    "timing_ms": 45,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.search(
                q="What framework does Zhang Jun use?",
                space_id="sp_001",
                user_id="user_001",
                mode="hybrid",
                include_profile=True,
                chunk_threshold=0.5,
                document_threshold=0.0,
                only_matching_chunks=False,
                include_full_docs=True,
                include_summary=True,
                filters=None,
                rewrite_query=False,
                rerank=False,
            )
        assert len(resp.results) == 1
        assert resp.results[0].score == 0.92
        assert resp.profile is not None


# ── Profile namespace ────────────────────────────────────────────────────


class TestProfileNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get(self):
        respx.post(f"{BASE}/v1/profile/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "profile": {
                        "identity": {"name": "Zhang Jun", "location": "Beijing"},
                        "preferences": {"stack": ["Python", "TypeScript"]},
                        "current_status": {"focus": "knowledge graph"},
                        "relationships": {},
                        "dynamic_memories": ["Working on RTMemory"],
                    },
                    "confidence": {"location": 0.95, "stack": 0.85},
                    "search_results": [],
                    "computed_at": "2026-04-23T10:00:00Z",
                    "timing_ms": 48,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.profile.get(
                entity_id="ent_001", space_id="sp_001", q="frontend frameworks", fresh=False
            )
        assert resp.profile.identity["name"] == "Zhang Jun"
        assert resp.confidence["location"] == 0.95


# ── Documents namespace ──────────────────────────────────────────────────


class TestDocumentsNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_add(self):
        respx.post(f"{BASE}/v1/documents/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "doc_001",
                    "title": "Tech Guide",
                    "content": "https://example.com/guide",
                    "status": "queued",
                    "space_id": "sp_001",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.documents.add(
                content="https://example.com/guide", space_id="sp_001", title="Tech Guide"
            )
        assert resp.id == "doc_001"
        assert resp.status.value == "queued"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list(self):
        respx.get(f"{BASE}/v1/documents/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "doc_001",
                            "title": "Tech Guide",
                            "status": "done",
                            "space_id": "sp_001",
                        }
                    ],
                    "total": 1,
                    "offset": 0,
                    "limit": 20,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.documents.list(space_id="sp_001", status="done", sort="created_at", order="desc")
        assert resp.total == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_get(self):
        respx.get(f"{BASE}/v1/documents/doc_001").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "doc_001",
                    "title": "Tech Guide",
                    "status": "done",
                    "space_id": "sp_001",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.documents.get("doc_001")
        assert resp.id == "doc_001"

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete(self):
        respx.delete(f"{BASE}/v1/documents/doc_001").mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.documents.delete("doc_001")
        assert resp["deleted"] is True


# ── Conversations namespace ──────────────────────────────────────────────


class TestConversationsNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_add(self):
        respx.post(f"{BASE}/v1/conversations/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "conv_001",
                    "memory_ids": ["mem_010"],
                    "entity_ids": ["ent_002"],
                    "created_at": "2026-04-23T11:00:00Z",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.conversations.add(
                messages=[{"role": "user", "content": "I just started learning Rust"}],
                space_id="sp_001",
                user_id="user_001",
            )
        assert resp.id == "conv_001"

    @respx.mock
    @pytest.mark.asyncio
    async def test_end(self):
        respx.post(f"{BASE}/v1/conversations/end").mock(
            return_value=httpx.Response(200, json={"deep_scan_triggered": True})
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.conversations.end(conversation_id="conv_001", space_id="sp_001")
        assert resp["deep_scan_triggered"] is True


# ── Graph namespace ──────────────────────────────────────────────────────


class TestGraphNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_neighborhood(self):
        respx.get(f"{BASE}/v1/graph/ent_001").mock(
            return_value=httpx.Response(
                200,
                json={
                    "center": {
                        "id": "ent_001",
                        "name": "Zhang Jun",
                        "entity_type": "person",
                        "description": "Software engineer",
                        "confidence": 0.95,
                    },
                    "entities": [
                        {
                            "id": "ent_002",
                            "name": "Beijing",
                            "entity_type": "location",
                            "description": "City",
                            "confidence": 0.9,
                        }
                    ],
                    "relations": [
                        {
                            "id": "rel_001",
                            "source_entity_id": "ent_001",
                            "target_entity_id": "ent_002",
                            "relation_type": "lives_in",
                            "confidence": 0.95,
                            "is_current": True,
                        }
                    ],
                    "depth": 2,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.graph.get_neighborhood(entity_id="ent_001", depth=2)
        assert resp.center.name == "Zhang Jun"
        assert len(resp.relations) == 1


# ── Spaces namespace ─────────────────────────────────────────────────────


class TestSpacesNamespace:
    @respx.mock
    @pytest.mark.asyncio
    async def test_create(self):
        respx.post(f"{BASE}/v1/spaces/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "sp_002",
                    "name": "Project KB",
                    "description": "Knowledge base for the project",
                    "is_default": False,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.spaces.create(name="Project KB", description="Knowledge base for the project")
        assert resp.id == "sp_002"
        assert resp.name == "Project KB"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list(self):
        respx.get(f"{BASE}/v1/spaces/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {"id": "sp_001", "name": "Default", "is_default": True},
                    ],
                    "total": 1,
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.spaces.list()
        assert resp.total == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_get(self):
        respx.get(f"{BASE}/v1/spaces/sp_001").mock(
            return_value=httpx.Response(
                200,
                json={"id": "sp_001", "name": "Default", "is_default": True},
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.spaces.get("sp_001")
        assert resp.name == "Default"

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete(self):
        respx.delete(f"{BASE}/v1/spaces/sp_001").mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.spaces.delete("sp_001")
        assert resp["deleted"] is True