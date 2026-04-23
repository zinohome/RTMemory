# RTMemory SDK & Integrations — 客户端与集成层

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Python SDK, JavaScript SDK, MCP Server, LangChain integration, Claude Code integration, and generic agent tools.

**Architecture:** Python SDK is async-first using httpx. JS SDK uses native fetch. MCP Server exposes 6 tools via stdio. LangChain integration wraps SDK methods as Tool objects. Claude adapter maps file operations to memory operations.

**Tech Stack:** Python 3.12 (httpx, pydantic, mcp), TypeScript (fetch, zod)

---

## Phase 1: Python SDK — Types & Client Shell

### Step 1.1 — Create Python SDK pyproject.toml and package skeleton

- [ ] Create `sdk-python/pyproject.toml` and `sdk-python/rtmemory/__init__.py`

**File: `sdk-python/pyproject.toml`**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rtmemory"
version = "0.1.0"
description = "Async Python SDK for RTMemory — temporal knowledge-graph driven AI memory"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
dependencies = [
    "httpx>=0.27,<1",
    "pydantic>=2.7,<3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "respx>=0.22",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**File: `sdk-python/rtmemory/__init__.py`**
```python
"""RTMemory Python SDK — async-first client for the RTMemory server."""

from rtmemory.client import RTMemoryClient
from rtmemory.types import (
    Memory,
    MemoryAddResponse,
    SearchResponse,
    SearchResult,
    ProfileResponse,
    Document,
    DocumentListResponse,
    Space,
    GraphNeighborhood,
    ConversationAddResponse,
)

__all__ = [
    "RTMemoryClient",
    "Memory",
    "MemoryAddResponse",
    "SearchResponse",
    "SearchResult",
    "ProfileResponse",
    "Document",
    "DocumentListResponse",
    "Space",
    "GraphNeighborhood",
    "ConversationAddResponse",
]

__version__ = "0.1.0"
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
pip install -e ".[dev]"
# Expected: Successfully installed rtmemory-0.1.0 ...
```

**Commit:** `feat(sdk-python): add pyproject.toml and package init`

---

### Step 1.2 — Define all Pydantic types for SDK request/response models

- [ ] Create `sdk-python/rtmemory/types.py`

**File: `sdk-python/rtmemory/types.py`**
```python
"""Pydantic models for RTMemory API request and response objects."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class SearchMode(str, Enum):
    hybrid = "hybrid"
    memory_only = "memory_only"
    documents_only = "documents_only"


class DocumentStatus(str, Enum):
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    done = "done"
    failed = "failed"


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


# ── Memories ─────────────────────────────────────────────────────────────

class Memory(BaseModel):
    id: str
    content: str
    custom_id: str | None = None
    memory_type: MemoryType | None = None
    entity_id: str | None = None
    relation_id: str | None = None
    confidence: float = 1.0
    decay_rate: float = 0.01
    is_forgotten: bool = False
    forget_reason: str | None = None
    version: int = 1
    parent_id: str | None = None
    root_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    space_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryAddRequest(BaseModel):
    content: str
    space_id: str
    user_id: str | None = None
    custom_id: str | None = None
    entity_context: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryAddResponse(BaseModel):
    id: str
    content: str
    custom_id: str | None = None
    confidence: float = 1.0
    entity_id: str | None = None
    relation_ids: list[str] = Field(default_factory=list)
    memory_type: MemoryType | None = None
    created_at: datetime | None = None


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryForgetRequest(BaseModel):
    memory_id: str | None = None
    content_match: str | None = None
    reason: str | None = None


class MemoryListResponse(BaseModel):
    items: list[Memory] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 20


# ── Search ───────────────────────────────────────────────────────────────

class SearchFilter(BaseModel):
    key: str
    value: Any
    operator: str = "eq"


class SearchFilterGroup(BaseModel):
    AND: list[SearchFilter | SearchFilterGroup] | None = None
    OR: list[SearchFilter | SearchFilterGroup] | None = None


class SearchRequest(BaseModel):
    q: str
    space_id: str | None = None
    user_id: str | None = None
    mode: SearchMode = SearchMode.hybrid
    channels: list[str] | None = None
    limit: int = 10
    include_profile: bool = False
    chunk_threshold: float = 0.0
    document_threshold: float = 0.0
    only_matching_chunks: bool = False
    include_full_docs: bool = False
    include_summary: bool = False
    filters: SearchFilterGroup | None = None
    rewrite_query: bool = False
    rerank: bool = False


class SearchResultEntity(BaseModel):
    name: str
    type: str | None = None


class SearchResultDocument(BaseModel):
    title: str | None = None
    url: str | None = None


class SearchResult(BaseModel):
    type: str  # "memory" | "document_chunk" | "entity"
    content: str
    score: float
    source: str  # e.g. "vector+graph"
    entity: SearchResultEntity | None = None
    document: SearchResultDocument | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfileData(BaseModel):
    identity: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    current_status: dict[str, Any] = Field(default_factory=dict)
    relationships: dict[str, Any] = Field(default_factory=dict)
    dynamic_memories: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    profile: ProfileData | None = None
    timing_ms: int = 0


# ── Profile ──────────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    entity_id: str
    space_id: str
    q: str | None = None
    fresh: bool = False


class ProfileConfidence(BaseModel):
    """Confidence values keyed by profile attribute."""
    pass


class ProfileResponse(BaseModel):
    profile: ProfileData
    confidence: dict[str, float] = Field(default_factory=dict)
    search_results: list[SearchResult] = Field(default_factory=list)
    computed_at: datetime | None = None
    timing_ms: int = 0


# ── Documents ────────────────────────────────────────────────────────────

class Document(BaseModel):
    id: str
    title: str | None = None
    content: str | None = None
    doc_type: str | None = None
    url: str | None = None
    status: DocumentStatus = DocumentStatus.queued
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    space_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentAddRequest(BaseModel):
    content: str
    space_id: str
    title: str | None = None


class DocumentListResponse(BaseModel):
    items: list[Document] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 20


# ── Conversations ────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ConversationAddRequest(BaseModel):
    messages: list[ConversationMessage]
    space_id: str
    user_id: str | None = None


class ConversationAddResponse(BaseModel):
    id: str
    memory_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class ConversationEndRequest(BaseModel):
    conversation_id: str
    space_id: str


# ── Spaces ───────────────────────────────────────────────────────────────

class Space(BaseModel):
    id: str
    name: str
    description: str | None = None
    org_id: str | None = None
    owner_id: str | None = None
    container_tag: str | None = None
    is_default: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SpaceCreateRequest(BaseModel):
    name: str
    description: str | None = None


class SpaceListResponse(BaseModel):
    items: list[Space] = Field(default_factory=list)
    total: int = 0


# ── Graph ────────────────────────────────────────────────────────────────

class GraphEntity(BaseModel):
    id: str
    name: str
    entity_type: str | None = None
    description: str | None = None
    confidence: float = 1.0


class GraphRelation(BaseModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    value: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 1.0
    is_current: bool = True


class GraphNeighborhood(BaseModel):
    center: GraphEntity
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    depth: int = 1
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.types import Memory, SearchResponse, ProfileResponse, Document; print('Types OK')"
# Expected: Types OK
```

**Commit:** `feat(sdk-python): add all Pydantic request/response types`

---

### Step 1.3 — Write TDD tests for RTMemoryClient and namespaces

- [ ] Create `sdk-python/tests/__init__.py` and `sdk-python/tests/test_client.py`

**File: `sdk-python/tests/__init__.py`**
```python
```

**File: `sdk-python/tests/test_client.py`**
```python
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
        client.close()

    def test_client_with_api_key(self):
        client = RTMemoryClient(base_url=BASE, api_key="sk-test")
        assert client.api_key == "sk-test"
        client.close()

    def test_async_context_manager(self):
        async def _go():
            async with RTMemoryClient(base_url=BASE) as c:
                assert c.base_url == BASE

        import asyncio
        asyncio.run(_go())


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
    async def test_upload(self):
        respx.post(f"{BASE}/v1/documents/upload").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "doc_002",
                    "title": "kb.pdf",
                    "status": "queued",
                    "space_id": "sp_001",
                },
            )
        )
        async with RTMemoryClient(base_url=BASE) as client:
            resp = await client.documents.upload(file="tests/fixtures/dummy.pdf", space_id="sp_001")
        assert resp.id == "doc_002"

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
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
pip install respx pytest-asyncio
# Tests will fail until client is implemented — that's TDD red phase
```

**Commit:** `test(sdk-python): add TDD tests for all client namespaces`

---

### Step 1.4 — Implement RTMemoryClient with HTTP transport layer

- [ ] Create `sdk-python/rtmemory/client.py`

**File: `sdk-python/rtmemory/client.py`**
```python
"""RTMemoryClient — async-first Python SDK entry point."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.memories import MemoriesNamespace
from rtmemory.search import SearchNamespace
from rtmemory.profile import ProfileNamespace
from rtmemory.documents import DocumentsNamespace
from rtmemory.conversations import ConversationsNamespace
from rtmemory.graph import GraphNamespace
from rtmemory.spaces import SpacesNamespace


class RTMemoryClient:
    """Main entry point for the RTMemory Python SDK.

    Usage::

        async with RTMemoryClient(base_url="http://localhost:8000", api_key="sk-...") as client:
            result = await client.memories.add(content="...", space_id="sp_001")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._external_client = http_client is not None
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        # Attach namespaces
        self.memories = MemoriesNamespace(self._http)
        self.search = SearchNamespace(self._http)
        self.profile = ProfileNamespace(self._http)
        self.documents = DocumentsNamespace(self._http)
        self.conversations = ConversationsNamespace(self._http)
        self.graph = GraphNamespace(self._http)
        self.spaces = SpacesNamespace(self._http)

    async def __aenter__(self) -> RTMemoryClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client (only if we own it)."""
        if not self._external_client:
            # Synchronous close for non-async contexts
            try:
                import asyncio
                asyncio.get_event_loop().create_task(self._http.aclose())
            except RuntimeError:
                pass
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.client import RTMemoryClient; print('Client import OK')"
# Expected: Client import OK  (namespace modules not yet created, will fail here)
```

**Commit:** `feat(sdk-python): add RTMemoryClient entry point`

---

### Step 1.5 — Implement MemoriesNamespace

- [ ] Create `sdk-python/rtmemory/memories.py`

**File: `sdk-python/rtmemory/memories.py`**
```python
"""MemoriesNamespace — async methods for the /v1/memories/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import (
    Memory,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryForgetRequest,
    MemoryListResponse,
    MemoryUpdateRequest,
)


class MemoriesNamespace:
    """Namespace for memory CRUD operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def add(
        self,
        content: str,
        space_id: str,
        user_id: str | None = None,
        custom_id: str | None = None,
        entity_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryAddResponse:
        """Add a new memory (triggers extraction pipeline)."""
        body = MemoryAddRequest(
            content=content,
            space_id=space_id,
            user_id=user_id,
            custom_id=custom_id,
            entity_context=entity_context,
            metadata=metadata,
        )
        resp = await self._http.post("/v1/memories/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return MemoryAddResponse.model_validate(resp.json())

    async def list(
        self,
        space_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> MemoryListResponse:
        """List memories with pagination and optional filtering."""
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if space_id is not None:
            params["space_id"] = space_id
        resp = await self._http.get("/v1/memories/", params=params)
        resp.raise_for_status()
        return MemoryListResponse.model_validate(resp.json())

    async def get(self, id: str) -> Memory:
        """Get a single memory by ID (includes version chain)."""
        resp = await self._http.get(f"/v1/memories/{id}")
        resp.raise_for_status()
        return Memory.model_validate(resp.json())

    async def update(
        self,
        id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Update a memory's content and/or metadata."""
        body = MemoryUpdateRequest(content=content, metadata=metadata)
        resp = await self._http.patch(f"/v1/memories/{id}", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return Memory.model_validate(resp.json())

    async def forget(
        self,
        memory_id: str | None = None,
        content_match: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Forget a memory by ID or content match (soft delete)."""
        body = MemoryForgetRequest(memory_id=memory_id, content_match=content_match, reason=reason)
        resp = await self._http.post("/v1/memories/forget", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return resp.json()
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.memories import MemoriesNamespace; print('MemoriesNamespace OK')"
# Expected: MemoriesNamespace OK
```

**Commit:** `feat(sdk-python): implement MemoriesNamespace`

---

### Step 1.6 — Implement SearchNamespace

- [ ] Create `sdk-python/rtmemory/search.py`

**File: `sdk-python/rtmemory/search.py`**
```python
"""SearchNamespace — async methods for the /v1/search/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import SearchMode, SearchRequest, SearchResponse, SearchFilterGroup


class SearchNamespace:
    """Namespace for hybrid search operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def __call__(
        self,
        q: str,
        space_id: str | None = None,
        user_id: str | None = None,
        mode: SearchMode = SearchMode.hybrid,
        channels: list[str] | None = None,
        limit: int = 10,
        include_profile: bool = False,
        chunk_threshold: float = 0.0,
        document_threshold: float = 0.0,
        only_matching_chunks: bool = False,
        include_full_docs: bool = False,
        include_summary: bool = False,
        filters: SearchFilterGroup | dict[str, Any] | None = None,
        rewrite_query: bool = False,
        rerank: bool = False,
    ) -> SearchResponse:
        """Execute a hybrid search across memories, documents, and graph."""
        filters_dump = None
        if filters is not None:
            if isinstance(filters, SearchFilterGroup):
                filters_dump = filters.model_dump(exclude_none=True)
            else:
                filters_dump = filters

        body = SearchRequest(
            q=q,
            space_id=space_id,
            user_id=user_id,
            mode=mode,
            channels=channels,
            limit=limit,
            include_profile=include_profile,
            chunk_threshold=chunk_threshold,
            document_threshold=document_threshold,
            only_matching_chunks=only_matching_chunks,
            include_full_docs=include_full_docs,
            include_summary=include_summary,
            filters=filters_dump,
            rewrite_query=rewrite_query,
            rerank=rerank,
        )
        resp = await self._http.post("/v1/search/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.search import SearchNamespace; print('SearchNamespace OK')"
# Expected: SearchNamespace OK
```

**Commit:** `feat(sdk-python): implement SearchNamespace`

---

### Step 1.7 — Implement ProfileNamespace

- [ ] Create `sdk-python/rtmemory/profile.py`

**File: `sdk-python/rtmemory/profile.py`**
```python
"""ProfileNamespace — async methods for the /v1/profile/ API."""

from __future__ import annotations

import httpx

from rtmemory.types import ProfileRequest, ProfileResponse


class ProfileNamespace:
    """Namespace for user profile operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def get(
        self,
        entity_id: str,
        space_id: str,
        q: str | None = None,
        fresh: bool = False,
    ) -> ProfileResponse:
        """Get (or compute) a user profile from the knowledge graph."""
        body = ProfileRequest(entity_id=entity_id, space_id=space_id, q=q, fresh=fresh)
        resp = await self._http.post("/v1/profile/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return ProfileResponse.model_validate(resp.json())
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.profile import ProfileNamespace; print('ProfileNamespace OK')"
# Expected: ProfileNamespace OK
```

**Commit:** `feat(sdk-python): implement ProfileNamespace`

---

### Step 1.8 — Implement DocumentsNamespace

- [ ] Create `sdk-python/rtmemory/documents.py`

**File: `sdk-python/rtmemory/documents.py`**
```python
"""DocumentsNamespace — async methods for the /v1/documents/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import (
    Document,
    DocumentAddRequest,
    DocumentListResponse,
)


class DocumentsNamespace:
    """Namespace for document management operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def add(
        self,
        content: str,
        space_id: str,
        title: str | None = None,
    ) -> Document:
        """Add a document by content (text or URL)."""
        body = DocumentAddRequest(content=content, space_id=space_id, title=title)
        resp = await self._http.post("/v1/documents/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return Document.model_validate(resp.json())

    async def upload(self, file: str, space_id: str) -> Document:
        """Upload a file (multipart) as a document."""
        with open(file, "rb") as f:
            files = {"file": (file, f)}
            data = {"space_id": (None, space_id)}
            resp = await self._http.post("/v1/documents/upload", files=files, data=data)
        resp.raise_for_status()
        return Document.model_validate(resp.json())

    async def list(
        self,
        space_id: str | None = None,
        status: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
        offset: int = 0,
        limit: int = 20,
    ) -> DocumentListResponse:
        """List documents with optional status filter and sorting."""
        params: dict[str, Any] = {"sort": sort, "order": order, "offset": offset, "limit": limit}
        if space_id is not None:
            params["space_id"] = space_id
        if status is not None:
            params["status"] = status
        resp = await self._http.get("/v1/documents/", params=params)
        resp.raise_for_status()
        return DocumentListResponse.model_validate(resp.json())

    async def get(self, id: str) -> Document:
        """Get a single document with associated memories."""
        resp = await self._http.get(f"/v1/documents/{id}")
        resp.raise_for_status()
        return Document.model_validate(resp.json())

    async def delete(self, id: str) -> dict[str, Any]:
        """Delete a document."""
        resp = await self._http.delete(f"/v1/documents/{id}")
        resp.raise_for_status()
        return resp.json()
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.documents import DocumentsNamespace; print('DocumentsNamespace OK')"
# Expected: DocumentsNamespace OK
```

**Commit:** `feat(sdk-python): implement DocumentsNamespace`

---

### Step 1.9 — Implement ConversationsNamespace

- [ ] Create `sdk-python/rtmemory/conversations.py`

**File: `sdk-python/rtmemory/conversations.py`**
```python
"""ConversationsNamespace — async methods for the /v1/conversations/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import (
    ConversationAddRequest,
    ConversationAddResponse,
    ConversationMessage,
)


class ConversationsNamespace:
    """Namespace for conversation memory operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def add(
        self,
        messages: list[dict[str, str] | ConversationMessage],
        space_id: str,
        user_id: str | None = None,
    ) -> ConversationAddResponse:
        """Submit a conversation fragment (triggers extraction)."""
        parsed_messages = []
        for m in messages:
            if isinstance(m, ConversationMessage):
                parsed_messages.append(m)
            else:
                parsed_messages.append(ConversationMessage(**m))
        body = ConversationAddRequest(messages=parsed_messages, space_id=space_id, user_id=user_id)
        resp = await self._http.post("/v1/conversations/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return ConversationAddResponse.model_validate(resp.json())

    async def end(
        self,
        conversation_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        """End a conversation (triggers deep scan)."""
        body = {"conversation_id": conversation_id, "space_id": space_id}
        resp = await self._http.post("/v1/conversations/end", json=body)
        resp.raise_for_status()
        return resp.json()
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.conversations import ConversationsNamespace; print('ConversationsNamespace OK')"
# Expected: ConversationsNamespace OK
```

**Commit:** `feat(sdk-python): implement ConversationsNamespace`

---

### Step 1.10 — Implement GraphNamespace

- [ ] Create `sdk-python/rtmemory/graph.py`

**File: `sdk-python/rtmemory/graph.py`**
```python
"""GraphNamespace — async methods for the /v1/graph/ API."""

from __future__ import annotations

import httpx

from rtmemory.types import GraphNeighborhood


class GraphNamespace:
    """Namespace for graph visualization and traversal."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def get_neighborhood(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> GraphNeighborhood:
        """Get the neighborhood subgraph around an entity."""
        resp = await self._http.get(f"/v1/graph/{entity_id}", params={"depth": depth})
        resp.raise_for_status()
        return GraphNeighborhood.model_validate(resp.json())
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.graph import GraphNamespace; print('GraphNamespace OK')"
# Expected: GraphNamespace OK
```

**Commit:** `feat(sdk-python): implement GraphNamespace`

---

### Step 1.11 — Implement SpacesNamespace

- [ ] Create `sdk-python/rtmemory/spaces.py`

**File: `sdk-python/rtmemory/spaces.py`**
```python
"""SpacesNamespace — async methods for the /v1/spaces/ API."""

from __future__ import annotations

from typing import Any

import httpx

from rtmemory.types import Space, SpaceCreateRequest, SpaceListResponse


class SpacesNamespace:
    """Namespace for space management operations."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def create(
        self,
        name: str,
        description: str | None = None,
    ) -> Space:
        """Create a new space."""
        body = SpaceCreateRequest(name=name, description=description)
        resp = await self._http.post("/v1/spaces/", json=body.model_dump(exclude_none=True))
        resp.raise_for_status()
        return Space.model_validate(resp.json())

    async def list(self) -> SpaceListResponse:
        """List all spaces."""
        resp = await self._http.get("/v1/spaces/")
        resp.raise_for_status()
        return SpaceListResponse.model_validate(resp.json())

    async def get(self, id: str) -> Space:
        """Get space details."""
        resp = await self._http.get(f"/v1/spaces/{id}")
        resp.raise_for_status()
        return Space.model_validate(resp.json())

    async def delete(self, id: str) -> dict[str, Any]:
        """Delete a space."""
        resp = await self._http.delete(f"/v1/spaces/{id}")
        resp.raise_for_status()
        return resp.json()
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.spaces import SpacesNamespace; print('SpacesNamespace OK')"
# Expected: SpacesNamespace OK
```

**Commit:** `feat(sdk-python): implement SpacesNamespace`

---

### Step 1.12 — Run all Python SDK tests against mock server

- [ ] Verify all tests pass with respx mock

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -m pytest tests/ -v
# Expected: all Test* classes pass, ~15 tests
```

**Commit:** `test(sdk-python): all namespace tests pass against mock server`

---

## Phase 2: Generic Agent Tools (Python SDK)

### Step 2.1 — Implement get_memory_tools for any LLM agent

- [ ] Create `sdk-python/rtmemory/tools.py`

**File: `sdk-python/rtmemory/tools.py`**
```python
"""Generic LLM Agent tool definitions wrapping RTMemoryClient methods.

Usage::

    from rtmemory import RTMemoryClient
    from rtmemory.tools import get_memory_tools

    client = RTMemoryClient(base_url="http://localhost:8000", api_key="sk-...")
    tools = get_memory_tools(client, space_id="sp_001", user_id="user_001")
    # tools is a list of dicts with "name", "description", "parameters", "function"
"""

from __future__ import annotations

from typing import Any, Callable

from rtmemory.client import RTMemoryClient


def get_memory_tools(
    client: RTMemoryClient,
    space_id: str,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return tool definitions suitable for any LLM agent framework.

    Each tool dict contains:
      - name: tool identifier
      - description: human-readable description for the LLM
      - parameters: JSON Schema for the tool's input
      - function: async callable that executes the tool

    Returns a list of 5 tools: search_memories, add_memory, get_profile,
    forget_memory, add_document.
    """

    async def _search_memories(q: str, mode: str = "hybrid", limit: int = 5) -> dict[str, Any]:
        """Search memories and knowledge base."""
        resp = await client.search(
            q=q,
            space_id=space_id,
            user_id=user_id,
            mode=mode,
            limit=limit,
        )
        return resp.model_dump()

    async def _add_memory(content: str, entity_context: str | None = None) -> dict[str, Any]:
        """Add a new memory."""
        resp = await client.memories.add(
            content=content,
            space_id=space_id,
            user_id=user_id,
            entity_context=entity_context,
        )
        return resp.model_dump()

    async def _get_profile(entity_id: str, q: str | None = None) -> dict[str, Any]:
        """Get a user's profile from the knowledge graph."""
        resp = await client.profile.get(
            entity_id=entity_id,
            space_id=space_id,
            q=q,
        )
        return resp.model_dump()

    async def _forget_memory(memory_id: str | None = None, content_match: str | None = None, reason: str | None = None) -> dict[str, Any]:
        """Forget (soft-delete) a memory by ID or content match."""
        resp = await client.memories.forget(
            memory_id=memory_id,
            content_match=content_match,
            reason=reason,
        )
        return resp  # already a dict

    async def _add_document(content: str, title: str | None = None) -> dict[str, Any]:
        """Add a document to the knowledge base."""
        resp = await client.documents.add(
            content=content,
            space_id=space_id,
            title=title,
        )
        return resp.model_dump()

    return [
        {
            "name": "search_memories",
            "description": "Search the user's memories and knowledge base. Use this when you need to recall information about the user, their preferences, past conversations, or documents in the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "mode": {"type": "string", "enum": ["hybrid", "memory_only", "documents_only"], "description": "Search mode", "default": "hybrid"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["q"],
            },
            "function": _search_memories,
        },
        {
            "name": "add_memory",
            "description": "Add a new memory for the user. Use this when the user shares new facts, preferences, or status changes that should be remembered for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store"},
                    "entity_context": {"type": "string", "description": "Optional context to guide entity extraction"},
                },
                "required": ["content"],
            },
            "function": _add_memory,
        },
        {
            "name": "get_profile",
            "description": "Get the user's profile from the knowledge graph. Use this to understand the user's identity, preferences, current status, and relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "The entity ID of the user"},
                    "q": {"type": "string", "description": "Optional query to include relevant search results with the profile"},
                },
                "required": ["entity_id"],
            },
            "function": _get_profile,
        },
        {
            "name": "forget_memory",
            "description": "Forget (soft-delete) a memory. Use this when the user asks to remove or correct outdated information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID to forget"},
                    "content_match": {"type": "string", "description": "Fuzzy content match to find and forget memories"},
                    "reason": {"type": "string", "description": "Reason for forgetting"},
                },
            },
            "function": _forget_memory,
        },
        {
            "name": "add_document",
            "description": "Add a document (text or URL) to the knowledge base. Use this when the user wants to ingest a document for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Document content (text or URL)"},
                    "title": {"type": "string", "description": "Document title"},
                },
                "required": ["content"],
            },
            "function": _add_document,
        },
    ]
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -c "from rtmemory.tools import get_memory_tools; print('tools OK')"
# Expected: tools OK
```

**Commit:** `feat(sdk-python): implement get_memory_tools for generic LLM agents`

---

### Step 2.2 — Write tests for get_memory_tools

- [ ] Create `sdk-python/tests/test_tools.py`

**File: `sdk-python/tests/test_tools.py`**
```python
"""Tests for rtmemory.tools — generic agent tool definitions."""

import pytest
import respx
import httpx

from rtmemory import RTMemoryClient
from rtmemory.tools import get_memory_tools


BASE = "http://localhost:8000"


class TestGetMemoryTools:
    def test_returns_five_tools(self):
        async def _go():
            async with RTMemoryClient(base_url=BASE) as client:
                tools = get_memory_tools(client, space_id="sp_001")
                return tools
        import asyncio
        tools = asyncio.run(_go())
        assert len(tools) == 5
        names = [t["name"] for t in tools]
        assert "search_memories" in names
        assert "add_memory" in names
        assert "get_profile" in names
        assert "forget_memory" in names
        assert "add_document" in names

    def test_each_tool_has_required_keys(self):
        async def _go():
            async with RTMemoryClient(base_url=BASE) as client:
                return get_memory_tools(client, space_id="sp_001")
        import asyncio
        tools = asyncio.run(_go())
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
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -m pytest tests/test_tools.py -v
# Expected: 4 tests pass
```

**Commit:** `test(sdk-python): add tests for get_memory_tools`

---

## Phase 3: JavaScript SDK

### Step 3.1 — Create JS SDK package.json and TypeScript config

- [ ] Create `sdk-js/package.json` and `sdk-js/tsconfig.json`

**File: `sdk-js/package.json`**
```json
{
  "name": "rtmemory",
  "version": "0.1.0",
  "description": "TypeScript SDK for RTMemory — temporal knowledge-graph driven AI memory",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "type": "module",
  "scripts": {
    "build": "tsc",
    "test": "node --experimental-vm-modules node_modules/.bin/vitest run"
  },
  "dependencies": {},
  "peerDependencies": {},
  "devDependencies": {
    "typescript": "^5.5",
    "vitest": "^2.0",
    "zod": "^3.23"
  },
  "license": "MIT"
}
```

**File: `sdk-js/tsconfig.json`**
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "dist",
    "rootDir": "src",
    "declaration": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npm install
# Expected: added typescript, vitest, zod
```

**Commit:** `feat(sdk-js): add package.json and tsconfig.json`

---

### Step 3.2 — Define all Zod schemas and TypeScript types

- [ ] Create `sdk-js/src/types.ts`

**File: `sdk-js/src/types.ts`**
```typescript
/**
 * Zod schemas and inferred TypeScript types for RTMemory API objects.
 */

import { z } from "zod";

// ── Enums ──────────────────────────────────────────────────────────────

export const MemoryTypeSchema = z.enum(["fact", "preference", "status", "inference"]);
export type MemoryType = z.infer<typeof MemoryTypeSchema>;

export const SearchModeSchema = z.enum(["hybrid", "memory_only", "documents_only"]);
export type SearchMode = z.infer<typeof SearchModeSchema>;

export const DocumentStatusSchema = z.enum([
  "queued", "extracting", "chunking", "embedding", "done", "failed",
]);
export type DocumentStatus = z.infer<typeof DocumentStatusSchema>;

// ── Memories ───────────────────────────────────────────────────────────

export const MemorySchema = z.object({
  id: z.string(),
  content: z.string(),
  customId: z.string().nullable().optional(),
  memoryType: MemoryTypeSchema.nullable().optional(),
  entityId: z.string().nullable().optional(),
  relationId: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
  decayRate: z.number().default(0.01),
  isForgotten: z.boolean().default(false),
  forgetReason: z.string().nullable().optional(),
  version: z.number().default(1),
  parentId: z.string().nullable().optional(),
  rootId: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
  spaceId: z.string().nullable().optional(),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
});
export type Memory = z.infer<typeof MemorySchema>;

export const MemoryAddRequestSchema = z.object({
  content: z.string(),
  spaceId: z.string(),
  userId: z.string().nullable().optional(),
  customId: z.string().nullable().optional(),
  entityContext: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).nullable().optional(),
});
export type MemoryAddRequest = z.infer<typeof MemoryAddRequestSchema>;

export const MemoryAddResponseSchema = z.object({
  id: z.string(),
  content: z.string(),
  customId: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
  entityId: z.string().nullable().optional(),
  relationIds: z.array(z.string()).default([]),
  memoryType: MemoryTypeSchema.nullable().optional(),
  createdAt: z.string().nullable().optional(),
});
export type MemoryAddResponse = z.infer<typeof MemoryAddResponseSchema>;

export const MemoryForgetRequestSchema = z.object({
  memoryId: z.string().nullable().optional(),
  contentMatch: z.string().nullable().optional(),
  reason: z.string().nullable().optional(),
});
export type MemoryForgetRequest = z.infer<typeof MemoryForgetRequestSchema>;

export const MemoryListResponseSchema = z.object({
  items: z.array(MemorySchema).default([]),
  total: z.number().default(0),
  offset: z.number().default(0),
  limit: z.number().default(20),
});
export type MemoryListResponse = z.infer<typeof MemoryListResponseSchema>;

// ── Search ─────────────────────────────────────────────────────────────

export const SearchFilterSchema = z.object({
  key: z.string(),
  value: z.unknown(),
  operator: z.string().default("eq"),
});

export const SearchFilterGroupSchema: z.ZodType<SearchFilterGroup> = z.object({
  AND: z.lazy(() => z.array(z.union([SearchFilterSchema, SearchFilterGroupSchema]))).nullable().optional(),
  OR: z.lazy(() => z.array(z.union([SearchFilterSchema, SearchFilterGroupSchema]))).nullable().optional(),
});
export type SearchFilterGroup = {
  AND?: (z.infer<typeof SearchFilterSchema> | SearchFilterGroup)[] | null;
  OR?: (z.infer<typeof SearchFilterSchema> | SearchFilterGroup)[] | null;
};

export const SearchRequestSchema = z.object({
  q: z.string(),
  spaceId: z.string().nullable().optional(),
  userId: z.string().nullable().optional(),
  mode: SearchModeSchema.default("hybrid"),
  channels: z.array(z.string()).nullable().optional(),
  limit: z.number().default(10),
  includeProfile: z.boolean().default(false),
  chunkThreshold: z.number().default(0.0),
  documentThreshold: z.number().default(0.0),
  onlyMatchingChunks: z.boolean().default(false),
  includeFullDocs: z.boolean().default(false),
  includeSummary: z.boolean().default(false),
  filters: SearchFilterGroupSchema.nullable().optional(),
  rewriteQuery: z.boolean().default(false),
  rerank: z.boolean().default(false),
});
export type SearchRequest = z.infer<typeof SearchRequestSchema>;

export const SearchResultEntitySchema = z.object({
  name: z.string(),
  type: z.string().nullable().optional(),
});

export const SearchResultDocumentSchema = z.object({
  title: z.string().nullable().optional(),
  url: z.string().nullable().optional(),
});

export const SearchResultSchema = z.object({
  type: z.string(),
  content: z.string(),
  score: z.number(),
  source: z.string(),
  entity: SearchResultEntitySchema.nullable().optional(),
  document: SearchResultDocumentSchema.nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
});
export type SearchResult = z.infer<typeof SearchResultSchema>;

export const ProfileDataSchema = z.object({
  identity: z.record(z.unknown()).default({}),
  preferences: z.record(z.unknown()).default({}),
  currentStatus: z.record(z.unknown()).default({}),
  relationships: z.record(z.unknown()).default({}),
  dynamicMemories: z.array(z.string()).default([]),
});
export type ProfileData = z.infer<typeof ProfileDataSchema>;

export const SearchResponseSchema = z.object({
  results: z.array(SearchResultSchema).default([]),
  profile: ProfileDataSchema.nullable().optional(),
  timingMs: z.number().default(0),
});
export type SearchResponse = z.infer<typeof SearchResponseSchema>;

// ── Profile ────────────────────────────────────────────────────────────

export const ProfileRequestSchema = z.object({
  entityId: z.string(),
  spaceId: z.string(),
  q: z.string().nullable().optional(),
  fresh: z.boolean().default(false),
});
export type ProfileRequest = z.infer<typeof ProfileRequestSchema>;

export const ProfileResponseSchema = z.object({
  profile: ProfileDataSchema,
  confidence: z.record(z.number()).default({}),
  searchResults: z.array(SearchResultSchema).default([]),
  computedAt: z.string().nullable().optional(),
  timingMs: z.number().default(0),
});
export type ProfileResponse = z.infer<typeof ProfileResponseSchema>;

// ── Documents ───────────────────────────────────────────────────────────

export const DocumentSchema = z.object({
  id: z.string(),
  title: z.string().nullable().optional(),
  content: z.string().nullable().optional(),
  docType: z.string().nullable().optional(),
  url: z.string().nullable().optional(),
  status: DocumentStatusSchema.default("queued"),
  summary: z.string().nullable().optional(),
  metadata: z.record(z.unknown()).default({}),
  spaceId: z.string().nullable().optional(),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
});
export type Document = z.infer<typeof DocumentSchema>;

export const DocumentAddRequestSchema = z.object({
  content: z.string(),
  spaceId: z.string(),
  title: z.string().nullable().optional(),
});
export type DocumentAddRequest = z.infer<typeof DocumentAddRequestSchema>;

export const DocumentListResponseSchema = z.object({
  items: z.array(DocumentSchema).default([]),
  total: z.number().default(0),
  offset: z.number().default(0),
  limit: z.number().default(20),
});
export type DocumentListResponse = z.infer<typeof DocumentListResponseSchema>;

// ── Conversations ──────────────────────────────────────────────────────

export const ConversationMessageSchema = z.object({
  role: z.string(),
  content: z.string(),
});
export type ConversationMessage = z.infer<typeof ConversationMessageSchema>;

export const ConversationAddRequestSchema = z.object({
  messages: z.array(ConversationMessageSchema),
  spaceId: z.string(),
  userId: z.string().nullable().optional(),
});
export type ConversationAddRequest = z.infer<typeof ConversationAddRequestSchema>;

export const ConversationAddResponseSchema = z.object({
  id: z.string(),
  memoryIds: z.array(z.string()).default([]),
  entityIds: z.array(z.string()).default([]),
  createdAt: z.string().nullable().optional(),
});
export type ConversationAddResponse = z.infer<typeof ConversationAddResponseSchema>;

// ── Spaces ──────────────────────────────────────────────────────────────

export const SpaceSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().nullable().optional(),
  orgId: z.string().nullable().optional(),
  ownerId: z.string().nullable().optional(),
  containerTag: z.string().nullable().optional(),
  isDefault: z.boolean().default(false),
  createdAt: z.string().nullable().optional(),
  updatedAt: z.string().nullable().optional(),
});
export type Space = z.infer<typeof SpaceSchema>;

export const SpaceCreateRequestSchema = z.object({
  name: z.string(),
  description: z.string().nullable().optional(),
});
export type SpaceCreateRequest = z.infer<typeof SpaceCreateRequestSchema>;

export const SpaceListResponseSchema = z.object({
  items: z.array(SpaceSchema).default([]),
  total: z.number().default(0),
});
export type SpaceListResponse = z.infer<typeof SpaceListResponseSchema>;

// ── Graph ───────────────────────────────────────────────────────────────

export const GraphEntitySchema = z.object({
  id: z.string(),
  name: z.string(),
  entityType: z.string().nullable().optional(),
  description: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
});

export const GraphRelationSchema = z.object({
  id: z.string(),
  sourceEntityId: z.string(),
  targetEntityId: z.string(),
  relationType: z.string(),
  value: z.string().nullable().optional(),
  validFrom: z.string().nullable().optional(),
  validTo: z.string().nullable().optional(),
  confidence: z.number().default(1.0),
  isCurrent: z.boolean().default(true),
});

export const GraphNeighborhoodSchema = z.object({
  center: GraphEntitySchema,
  entities: z.array(GraphEntitySchema).default([]),
  relations: z.array(GraphRelationSchema).default([]),
  depth: z.number().default(1),
});
export type GraphNeighborhood = z.infer<typeof GraphNeighborhoodSchema>;
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npx tsc --noEmit src/types.ts
# Expected: no errors
```

**Commit:** `feat(sdk-js): add Zod schemas and TypeScript types`

---

### Step 3.3 — Implement RTMemoryClient class (JS)

- [ ] Create `sdk-js/src/client.ts`

**File: `sdk-js/src/client.ts`**
```typescript
/**
 * RTMemoryClient — main entry point for the JavaScript/TypeScript SDK.
 *
 * Usage:
 *   const client = new RTMemoryClient({ baseUrl: "http://localhost:8000", apiKey: "sk-..." });
 *   await client.memories.add({ content: "I moved to Beijing", spaceId: "sp_001" });
 */

import type {
  Memory,
  MemoryAddRequest,
  MemoryAddResponse,
  MemoryListResponse,
  MemoryForgetRequest,
  SearchRequest,
  SearchResponse,
  ProfileRequest,
  ProfileResponse,
  Document,
  DocumentAddRequest,
  DocumentListResponse,
  ConversationAddRequest,
  ConversationAddResponse,
  Space,
  SpaceCreateRequest,
  SpaceListResponse,
  GraphNeighborhood,
} from "./types.js";

import {
  MemoryAddResponseSchema,
  MemoryListResponseSchema,
  MemorySchema,
  SearchResponseSchema,
  ProfileResponseSchema,
  DocumentSchema,
  DocumentListResponseSchema,
  ConversationAddResponseSchema,
  SpaceSchema,
  SpaceListResponseSchema,
  GraphNeighborhoodSchema,
} from "./types.js";

// ── Helpers ──────────────────────────────────────────────────────────────

/** Convert camelCase object keys to snake_case for the API. */
function toSnakeCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const snake = key.replace(/[A-Z]/g, (ch) => "_" + ch.toLowerCase());
    if (value !== undefined && value !== null) {
      result[snake] = value;
    }
  }
  return result;
}

/** Convert snake_case object keys to camelCase for the client. */
function toCamelCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const camel = key.replace(/_([a-z])/g, (_, ch: string) => ch.toUpperCase());
    result[camel] = value;
  }
  return result;
}

// ── Memories Namespace ───────────────────────────────────────────────────

export class MemoriesNamespace {
  constructor(private http: HttpClient) {}

  async add(params: MemoryAddRequest): Promise<MemoryAddResponse> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.post("/v1/memories/", body);
    return MemoryAddResponseSchema.parse(toCamelCase(json));
  }

  async list(params?: { spaceId?: string; offset?: number; limit?: number }): Promise<MemoryListResponse> {
    const query: Record<string, string> = {};
    if (params?.spaceId) query.space_id = params.spaceId;
    if (params?.offset !== undefined) query.offset = String(params.offset);
    if (params?.limit !== undefined) query.limit = String(params.limit);
    const json = await this.http.get("/v1/memories/", query);
    return MemoryListResponseSchema.parse(toCamelCase(json));
  }

  async get(id: string): Promise<Memory> {
    const json = await this.http.get(`/v1/memories/${id}`);
    return MemorySchema.parse(toCamelCase(json));
  }

  async update(id: string, params: { content?: string; metadata?: Record<string, unknown> }): Promise<Memory> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.patch(`/v1/memories/${id}`, body);
    return MemorySchema.parse(toCamelCase(json));
  }

  async forget(params: MemoryForgetRequest): Promise<Record<string, unknown>> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    return await this.http.post("/v1/memories/forget", body);
  }
}

// ── Search Namespace ────────────────────────────────────────────────────

export class SearchNamespace {
  constructor(private http: HttpClient) {}

  async search(params: SearchRequest): Promise<SearchResponse> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.post("/v1/search/", body);
    return SearchResponseSchema.parse(toCamelCase(json));
  }
}

// ── Profile Namespace ───────────────────────────────────────────────────

export class ProfileNamespace {
  constructor(private http: HttpClient) {}

  async get(params: ProfileRequest): Promise<ProfileResponse> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.post("/v1/profile/", body);
    return ProfileResponseSchema.parse(toCamelCase(json));
  }
}

// ── Documents Namespace ──────────────────────────────────────────────────

export class DocumentsNamespace {
  constructor(private http: HttpClient) {}

  async add(params: DocumentAddRequest): Promise<Document> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.post("/v1/documents/", body);
    return DocumentSchema.parse(toCamelCase(json));
  }

  async upload(file: File, spaceId: string): Promise<Document> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("space_id", spaceId);
    const json = await this.http.postMultipart("/v1/documents/upload", formData);
    return DocumentSchema.parse(toCamelCase(json));
  }

  async list(params: {
    spaceId?: string;
    status?: string;
    sort?: string;
    order?: string;
    offset?: number;
    limit?: number;
  }): Promise<DocumentListResponse> {
    const query: Record<string, string> = {};
    if (params.spaceId) query.space_id = params.spaceId;
    if (params.status) query.status = params.status;
    if (params.sort) query.sort = params.sort;
    if (params.order) query.order = params.order;
    if (params.offset !== undefined) query.offset = String(params.offset);
    if (params.limit !== undefined) query.limit = String(params.limit);
    const json = await this.http.get("/v1/documents/", query);
    return DocumentListResponseSchema.parse(toCamelCase(json));
  }

  async get(id: string): Promise<Document> {
    const json = await this.http.get(`/v1/documents/${id}`);
    return DocumentSchema.parse(toCamelCase(json));
  }

  async delete(id: string): Promise<Record<string, unknown>> {
    return await this.http.delete(`/v1/documents/${id}`);
  }
}

// ── Conversations Namespace ─────────────────────────────────────────────

export class ConversationsNamespace {
  constructor(private http: HttpClient) {}

  async add(params: ConversationAddRequest): Promise<ConversationAddResponse> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.post("/v1/conversations/", body);
    return ConversationAddResponseSchema.parse(toCamelCase(json));
  }

  async end(params: { conversationId: string; spaceId: string }): Promise<Record<string, unknown>> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    return await this.http.post("/v1/conversations/end", body);
  }
}

// ── Graph Namespace ─────────────────────────────────────────────────────

export class GraphNamespace {
  constructor(private http: HttpClient) {}

  async getNeighborhood(params: { entityId: string; depth?: number }): Promise<GraphNeighborhood> {
    const query: Record<string, string> = {};
    if (params.depth !== undefined) query.depth = String(params.depth);
    const json = await this.http.get(`/v1/graph/${params.entityId}`, query);
    return GraphNeighborhoodSchema.parse(toCamelCase(json));
  }
}

// ── Spaces Namespace ─────────────────────────────────────────────────────

export class SpacesNamespace {
  constructor(private http: HttpClient) {}

  async create(params: SpaceCreateRequest): Promise<Space> {
    const body = toSnakeCase(params as unknown as Record<string, unknown>);
    const json = await this.http.post("/v1/spaces/", body);
    return SpaceSchema.parse(toCamelCase(json));
  }

  async list(): Promise<SpaceListResponse> {
    const json = await this.http.get("/v1/spaces/");
    return SpaceListResponseSchema.parse(toCamelCase(json));
  }

  async get(id: string): Promise<Space> {
    const json = await this.http.get(`/v1/spaces/${id}`);
    return SpaceSchema.parse(toCamelCase(json));
  }

  async delete(id: string): Promise<Record<string, unknown>> {
    return await this.http.delete(`/v1/spaces/${id}`);
  }
}

// ── HTTP Client (internal) ───────────────────────────────────────────────

class HttpClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, apiKey?: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = {
      "Content-Type": "application/json",
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
    };
  }

  private async request(method: string, path: string, opts?: { body?: unknown; query?: Record<string, string> }): Promise<unknown> {
    let url = `${this.baseUrl}${path}`;
    if (opts?.query && Object.keys(opts.query).length > 0) {
      const params = new URLSearchParams(opts.query);
      url += `?${params.toString()}`;
    }
    const init: RequestInit = {
      method,
      headers: this.headers,
    };
    if (opts?.body !== undefined) {
      init.body = JSON.stringify(opts.body);
    }
    const resp = await fetch(url, init);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`RTMemory API error ${resp.status}: ${text}`);
    }
    return await resp.json();
  }

  async get(path: string, query?: Record<string, string>): Promise<Record<string, unknown>> {
    return (await this.request("GET", path, { query })) as Record<string, unknown>;
  }

  async post(path: string, body: unknown): Promise<Record<string, unknown>> {
    return (await this.request("POST", path, { body })) as Record<string, unknown>;
  }

  async patch(path: string, body: unknown): Promise<Record<string, unknown>> {
    return (await this.request("PATCH", path, { body })) as Record<string, unknown>;
  }

  async delete(path: string): Promise<Record<string, unknown>> {
    return (await this.request("DELETE", path)) as Record<string, unknown>;
  }

  async postMultipart(path: string, formData: FormData): Promise<Record<string, unknown>> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {};
    if (this.headers["Authorization"]) {
      headers["Authorization"] = this.headers["Authorization"];
    }
    const resp = await fetch(url, {
      method: "POST",
      headers,
      body: formData,
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`RTMemory API error ${resp.status}: ${text}`);
    }
    return await resp.json();
  }
}

// ── RTMemoryClient ───────────────────────────────────────────────────────

export interface RTMemoryClientOptions {
  baseUrl: string;
  apiKey?: string;
}

export class RTMemoryClient {
  public memories: MemoriesNamespace;
  public search: SearchNamespace;
  public profile: ProfileNamespace;
  public documents: DocumentsNamespace;
  public conversations: ConversationsNamespace;
  public graph: GraphNamespace;
  public spaces: SpacesNamespace;

  private http: HttpClient;

  constructor(options: RTMemoryClientOptions) {
    this.http = new HttpClient(options.baseUrl, options.apiKey);
    this.memories = new MemoriesNamespace(this.http);
    this.search = new SearchNamespace(this.http);
    this.profile = new ProfileNamespace(this.http);
    this.documents = new DocumentsNamespace(this.http);
    this.conversations = new ConversationsNamespace(this.http);
    this.graph = new GraphNamespace(this.http);
    this.spaces = new SpacesNamespace(this.http);
  }
}
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npx tsc --noEmit src/client.ts
# Expected: no errors
```

**Commit:** `feat(sdk-js): implement RTMemoryClient with all namespaces`

---

### Step 3.4 — Create JS SDK index.ts barrel export

- [ ] Create `sdk-js/src/index.ts`

**File: `sdk-js/src/index.ts`**
```typescript
/**
 * RTMemory JavaScript/TypeScript SDK — barrel export.
 */

export { RTMemoryClient, type RTMemoryClientOptions } from "./client.js";
export {
  MemoriesNamespace,
  SearchNamespace,
  ProfileNamespace,
  DocumentsNamespace,
  ConversationsNamespace,
  GraphNamespace,
  SpacesNamespace,
} from "./client.js";

export type {
  Memory,
  MemoryType,
  MemoryAddRequest,
  MemoryAddResponse,
  MemoryForgetRequest,
  MemoryListResponse,
  SearchMode,
  SearchRequest,
  SearchResponse,
  SearchResult,
  ProfileRequest,
  ProfileResponse,
  ProfileData,
  Document,
  DocumentStatus,
  DocumentAddRequest,
  DocumentListResponse,
  ConversationMessage,
  ConversationAddRequest,
  ConversationAddResponse,
  Space,
  SpaceCreateRequest,
  SpaceListResponse,
  GraphNeighborhood,
} from "./types.js";
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npx tsc --noEmit
# Expected: no errors
```

**Commit:** `feat(sdk-js): add barrel export index.ts`

---

### Step 3.5 — Write JS SDK tests with vitest

- [ ] Create `sdk-js/src/__tests__/client.test.ts` and `sdk-js/vitest.config.ts`

**File: `sdk-js/vitest.config.ts`**
```typescript
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
  },
});
```

**File: `sdk-js/src/__tests__/client.test.ts`**
```typescript
/**
 * Tests for the JS RTMemoryClient using vitest + mocked fetch.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RTMemoryClient } from "../client.js";

const BASE = "http://localhost:8000";

function mockFetch(response: unknown, status = 200): void {
  (globalThis as any).fetch = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => response,
    text: async () => JSON.stringify(response),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("RTMemoryClient construction", () => {
  it("creates client with baseUrl", () => {
    const client = new RTMemoryClient({ baseUrl: BASE });
    expect(client.memories).toBeDefined();
    expect(client.search).toBeDefined();
    expect(client.profile).toBeDefined();
    expect(client.documents).toBeDefined();
    expect(client.conversations).toBeDefined();
    expect(client.graph).toBeDefined();
    expect(client.spaces).toBeDefined();
  });

  it("creates client with apiKey", () => {
    const client = new RTMemoryClient({ baseUrl: BASE, apiKey: "sk-test" });
    expect(client).toBeDefined();
  });
});

describe("MemoriesNamespace", () => {
  it("add sends POST /v1/memories/", async () => {
    mockFetch({
      id: "mem_001",
      content: "I moved to Beijing",
      confidence: 0.9,
      entity_id: "ent_001",
      relation_ids: ["rel_001"],
      memory_type: "fact",
      created_at: "2026-04-23T10:00:00Z",
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.memories.add({
      content: "I moved to Beijing",
      spaceId: "sp_001",
      customId: "ext_123",
      entityContext: "Zhang Jun's knowledge base",
    });
    expect(resp.id).toBe("mem_001");
    expect(resp.confidence).toBe(0.9);
  });

  it("list sends GET /v1/memories/", async () => {
    mockFetch({
      items: [{ id: "mem_001", content: "test", confidence: 1.0 }],
      total: 1,
      offset: 0,
      limit: 20,
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.memories.list({ spaceId: "sp_001" });
    expect(resp.total).toBe(1);
  });

  it("forget sends POST /v1/memories/forget", async () => {
    mockFetch({ forgotten: true, memory_id: "mem_001" });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.memories.forget({ memoryId: "mem_001", reason: "user requested" });
    expect(resp.forgotten).toBe(true);
  });
});

describe("SearchNamespace", () => {
  it("search sends POST /v1/search/", async () => {
    mockFetch({
      results: [
        {
          type: "memory",
          content: "Zhang Jun uses Next.js",
          score: 0.92,
          source: "vector+graph",
          entity: { name: "Zhang Jun", type: "person" },
          metadata: {},
        },
      ],
      profile: {
        identity: { name: "Zhang Jun" },
        preferences: {},
        current_status: {},
        relationships: {},
        dynamic_memories: [],
      },
      timing_ms: 45,
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.search.search({
      q: "What framework does Zhang Jun use?",
      spaceId: "sp_001",
      mode: "hybrid",
      includeProfile: true,
    });
    expect(resp.results.length).toBe(1);
    expect(resp.results[0].score).toBe(0.92);
    expect(resp.profile).toBeDefined();
  });
});

describe("ProfileNamespace", () => {
  it("get sends POST /v1/profile/", async () => {
    mockFetch({
      profile: {
        identity: { name: "Zhang Jun", location: "Beijing" },
        preferences: {},
        current_status: {},
        relationships: {},
        dynamic_memories: [],
      },
      confidence: { location: 0.95 },
      search_results: [],
      computed_at: "2026-04-23T10:00:00Z",
      timing_ms: 48,
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.profile.get({
      entityId: "ent_001",
      spaceId: "sp_001",
    });
    expect(resp.profile.identity.name).toBe("Zhang Jun");
    expect(resp.confidence.location).toBe(0.95);
  });
});

describe("DocumentsNamespace", () => {
  it("add sends POST /v1/documents/", async () => {
    mockFetch({
      id: "doc_001",
      title: "Tech Guide",
      status: "queued",
      space_id: "sp_001",
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.documents.add({
      content: "https://example.com/guide",
      spaceId: "sp_001",
      title: "Tech Guide",
    });
    expect(resp.id).toBe("doc_001");
  });

  it("list sends GET /v1/documents/", async () => {
    mockFetch({
      items: [{ id: "doc_001", title: "Guide", status: "done" }],
      total: 1,
      offset: 0,
      limit: 20,
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.documents.list({ spaceId: "sp_001", status: "done" });
    expect(resp.total).toBe(1);
  });

  it("delete sends DELETE /v1/documents/:id", async () => {
    mockFetch({ deleted: true });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.documents.delete("doc_001");
    expect(resp.deleted).toBe(true);
  });
});

describe("ConversationsNamespace", () => {
  it("add sends POST /v1/conversations/", async () => {
    mockFetch({
      id: "conv_001",
      memory_ids: ["mem_010"],
      entity_ids: ["ent_002"],
      created_at: "2026-04-23T11:00:00Z",
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.conversations.add({
      messages: [{ role: "user", content: "I just started learning Rust" }],
      spaceId: "sp_001",
      userId: "user_001",
    });
    expect(resp.id).toBe("conv_001");
  });

  it("end sends POST /v1/conversations/end", async () => {
    mockFetch({ deep_scan_triggered: true });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.conversations.end({ conversationId: "conv_001", spaceId: "sp_001" });
    expect(resp.deep_scan_triggered).toBe(true);
  });
});

describe("GraphNamespace", () => {
  it("getNeighborhood sends GET /v1/graph/:entityId", async () => {
    mockFetch({
      center: { id: "ent_001", name: "Zhang Jun", entity_type: "person", confidence: 0.95 },
      entities: [{ id: "ent_002", name: "Beijing", entity_type: "location", confidence: 0.9 }],
      relations: [
        {
          id: "rel_001",
          source_entity_id: "ent_001",
          target_entity_id: "ent_002",
          relation_type: "lives_in",
          confidence: 0.95,
          is_current: true,
        },
      ],
      depth: 2,
    });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.graph.getNeighborhood({ entityId: "ent_001", depth: 2 });
    expect(resp.center.name).toBe("Zhang Jun");
    expect(resp.relations.length).toBe(1);
  });
});

describe("SpacesNamespace", () => {
  it("create sends POST /v1/spaces/", async () => {
    mockFetch({ id: "sp_002", name: "Project KB", is_default: false });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.spaces.create({ name: "Project KB" });
    expect(resp.id).toBe("sp_002");
  });

  it("list sends GET /v1/spaces/", async () => {
    mockFetch({ items: [{ id: "sp_001", name: "Default", is_default: true }], total: 1 });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.spaces.list();
    expect(resp.total).toBe(1);
  });

  it("delete sends DELETE /v1/spaces/:id", async () => {
    mockFetch({ deleted: true });
    const client = new RTMemoryClient({ baseUrl: BASE });
    const resp = await client.spaces.delete("sp_001");
    expect(resp.deleted).toBe(true);
  });
});
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npm test
# Expected: all tests pass (~15 tests)
```

**Commit:** `test(sdk-js): add vitest tests for all client namespaces`

---

### Step 3.6 — Build the JS SDK for distribution

- [ ] Run the TypeScript compiler and verify output

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npx tsc
ls dist/
# Expected: index.js, index.d.ts, client.js, client.d.ts, types.js, types.d.ts
```

**Commit:** `build(sdk-js): compile TypeScript to dist/`

---

## Phase 4: MCP Server

### Step 4.1 — Create MCP Server pyproject.toml

- [ ] Create `integrations/mcp-server/pyproject.toml` and `integrations/mcp-server/__init__.py`

**File: `integrations/mcp-server/pyproject.toml`**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rtmemory-mcp-server"
version = "0.1.0"
description = "MCP Server for RTMemory — exposes memory tools via stdio"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
dependencies = [
    "mcp>=1.0",
    "rtmemory",
]

[project.scripts]
rtmemory-mcp = "rtmemory_mcp_server.main:main"
```

**File: `integrations/mcp-server/__init__.py`**
```python
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/mcp-server
pip install -e "."
# Expected: Successfully installed rtmemory-mcp-server-0.1.0
```

**Commit:** `feat(mcp-server): add pyproject.toml and package skeleton`

---

### Step 4.2 — Implement MCP Server with 6 tools

- [ ] Create `integrations/mcp-server/rtmemory_mcp_server/__init__.py` and `integrations/mcp-server/rtmemory_mcp_server/main.py`

**File: `integrations/mcp-server/rtmemory_mcp_server/__init__.py`**
```python
"""RTMemory MCP Server — exposes memory tools via Model Context Protocol."""
```

**File: `integrations/mcp-server/rtmemory_mcp_server/main.py`**
```python
"""RTMemory MCP Server — 6 tools exposed via stdio transport.

Tools:
  - rtmemory_search         : Search memories and knowledge base
  - rtmemory_add            : Add a memory
  - rtmemory_profile        : Get user profile
  - rtmemory_forget         : Forget a memory
  - rtmemory_add_document   : Add a document to the knowledge base
  - rtmemory_list_documents : List documents in the knowledge base
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from rtmemory import RTMemoryClient


def create_server() -> Server:
    """Create and configure the MCP server with all RTMemory tools."""
    server = Server("rtmemory-mcp-server")

    base_url = os.environ.get("RTMEMORY_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("RTMEMORY_API_KEY")
    default_space_id = os.environ.get("RTMEMORY_SPACE_ID", "")
    default_user_id = os.environ.get("RTMEMORY_USER_ID", "")

    client = RTMemoryClient(base_url=base_url, api_key=api_key)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="rtmemory_search",
                description="Search the user's memories and knowledge base. Use this to recall information about the user, their preferences, past conversations, or documents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Search query"},
                        "space_id": {"type": "string", "description": "Space ID (defaults to RTMEMORY_SPACE_ID)"},
                        "mode": {"type": "string", "enum": ["hybrid", "memory_only", "documents_only"], "description": "Search mode", "default": "hybrid"},
                        "limit": {"type": "integer", "description": "Max number of results", "default": 5},
                        "include_profile": {"type": "boolean", "description": "Include user profile with results", "default": False},
                    },
                    "required": ["q"],
                },
            ),
            Tool(
                name="rtmemory_add",
                description="Add a new memory for the user. Use this when the user shares new facts, preferences, or status changes that should be remembered.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The memory content to store"},
                        "space_id": {"type": "string", "description": "Space ID"},
                        "entity_context": {"type": "string", "description": "Context to guide entity extraction"},
                        "metadata": {"type": "object", "description": "Optional metadata dict"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="rtmemory_profile",
                description="Get a user's profile from the knowledge graph. Use this to understand the user's identity, preferences, current status, and relationships.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string", "description": "Entity ID of the user"},
                        "space_id": {"type": "string", "description": "Space ID"},
                        "q": {"type": "string", "description": "Optional query to include relevant search results"},
                        "fresh": {"type": "boolean", "description": "Force fresh recomputation", "default": False},
                    },
                    "required": ["entity_id"],
                },
            ),
            Tool(
                name="rtmemory_forget",
                description="Forget (soft-delete) a memory. Use this when the user asks to remove or correct outdated information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID to forget"},
                        "content_match": {"type": "string", "description": "Fuzzy content match to find and forget memories"},
                        "reason": {"type": "string", "description": "Reason for forgetting"},
                    },
                },
            ),
            Tool(
                name="rtmemory_add_document",
                description="Add a document (text or URL) to the knowledge base. Use this when the user wants to ingest a document for future reference.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Document content (text or URL)"},
                        "title": {"type": "string", "description": "Document title"},
                        "space_id": {"type": "string", "description": "Space ID"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="rtmemory_list_documents",
                description="List documents in the knowledge base. Use this to see what documents are available or check processing status.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "space_id": {"type": "string", "description": "Space ID"},
                        "status": {"type": "string", "description": "Filter by status (queued, extracting, chunking, embedding, done, failed)"},
                        "sort": {"type": "string", "description": "Sort field", "default": "created_at"},
                        "order": {"type": "string", "description": "Sort order (asc/desc)", "default": "desc"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = await _dispatch(name, arguments)
        except Exception as exc:
            result = {"error": str(exc)}

        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _dispatch(name: str, args: dict[str, Any]) -> Any:
        if name == "rtmemory_search":
            space_id = args.get("space_id") or default_space_id
            user_id = args.get("user_id") or default_user_id
            resp = await client.search(
                q=args["q"],
                space_id=space_id,
                user_id=user_id,
                mode=args.get("mode", "hybrid"),
                limit=args.get("limit", 5),
                include_profile=args.get("include_profile", False),
            )
            return resp.model_dump()

        elif name == "rtmemory_add":
            space_id = args.get("space_id") or default_space_id
            resp = await client.memories.add(
                content=args["content"],
                space_id=space_id,
                entity_context=args.get("entity_context"),
                metadata=args.get("metadata"),
            )
            return resp.model_dump()

        elif name == "rtmemory_profile":
            space_id = args.get("space_id") or default_space_id
            resp = await client.profile.get(
                entity_id=args["entity_id"],
                space_id=space_id,
                q=args.get("q"),
                fresh=args.get("fresh", False),
            )
            return resp.model_dump()

        elif name == "rtmemory_forget":
            resp = await client.memories.forget(
                memory_id=args.get("memory_id"),
                content_match=args.get("content_match"),
                reason=args.get("reason"),
            )
            return resp

        elif name == "rtmemory_add_document":
            space_id = args.get("space_id") or default_space_id
            resp = await client.documents.add(
                content=args["content"],
                space_id=space_id,
                title=args.get("title"),
            )
            return resp.model_dump()

        elif name == "rtmemory_list_documents":
            space_id = args.get("space_id") or default_space_id
            resp = await client.documents.list(
                space_id=space_id,
                status=args.get("status"),
                sort=args.get("sort", "created_at"),
                order=args.get("order", "desc"),
                limit=args.get("limit", 10),
            )
            return resp.model_dump()

        else:
            return {"error": f"Unknown tool: {name}"}

    return server


async def run() -> None:
    """Run the MCP server via stdio transport."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for the rtmemory-mcp console script."""
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/mcp-server
python -c "from rtmemory_mcp_server.main import create_server; print('MCP server import OK')"
# Expected: MCP server import OK
```

**Commit:** `feat(mcp-server): implement MCP Server with 6 tools via stdio`

---

### Step 4.3 — Write MCP Server tests

- [ ] Create `integrations/mcp-server/tests/__init__.py` and `integrations/mcp-server/tests/test_main.py`

**File: `integrations/mcp-server/tests/__init__.py`**
```python
```

**File: `integrations/mcp-server/tests/test_main.py`**
```python
"""Tests for the RTMemory MCP Server tool definitions."""

import pytest

from rtmemory_mcp_server.main import create_server


@pytest.mark.asyncio
async def test_list_tools_returns_six():
    server = create_server()
    # The server.list_tools handler is registered via decorator.
    # We need to call the internal handler directly.
    # Access the registered handler through the server's request handlers.
    handler = server.request_handlers.get("tools/list")
    assert handler is not None, "tools/list handler not registered"
    result = await handler(None)
    tools = result.root if hasattr(result, "root") else result
    names = [t.name for t in tools]
    assert len(tools) == 6
    assert "rtmemory_search" in names
    assert "rtmemory_add" in names
    assert "rtmemory_profile" in names
    assert "rtmemory_forget" in names
    assert "rtmemory_add_document" in names
    assert "rtmemory_list_documents" in names


@pytest.mark.asyncio
async def test_tool_schemas_have_required_fields():
    server = create_server()
    handler = server.request_handlers.get("tools/list")
    result = await handler(None)
    tools = result.root if hasattr(result, "root") else result
    for tool in tools:
        assert tool.name, f"tool missing name"
        assert tool.description, f"tool {tool.name} missing description"
        assert tool.inputSchema, f"tool {tool.name} missing inputSchema"
        assert tool.inputSchema["type"] == "object", f"tool {tool.name} inputSchema not object"
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/mcp-server
pip install pytest pytest-asyncio
python -m pytest tests/ -v
# Expected: 2 tests pass (tool list and schema validation)
```

**Commit:** `test(mcp-server): add tests for tool list and schema validation`

---

## Phase 5: LangChain Integration

### Step 5.1 — Implement RTMemoryTools for LangChain

- [ ] Create `integrations/langchain/rtmemory_langchain/__init__.py` and `integrations/langchain/rtmemory_langchain/tools.py`

**File: `integrations/langchain/pyproject.toml`**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rtmemory-langchain"
version = "0.1.0"
description = "LangChain integration for RTMemory"
requires-python = ">=3.12"
license = "MIT"
dependencies = [
    "langchain-core>=0.2",
    "rtmemory",
]
```

**File: `integrations/langchain/rtmemory_langchain/__init__.py`**
```python
"""RTMemory LangChain integration."""

from rtmemory_langchain.tools import RTMemoryTools

__all__ = ["RTMemoryTools"]
```

**File: `integrations/langchain/rtmemory_langchain/tools.py`**
```python
"""RTMemoryTools — LangChain Tool objects wrapping RTMemoryClient methods.

Usage::

    from rtmemory_langchain import RTMemoryTools

    tools = RTMemoryTools(
        base_url="http://localhost:8000",
        api_key="sk-...",
        space_id="sp_001",
        user_id="user_001",
    )
    langchain_tools = tools.get_tools()
    # Returns: [search_memories, add_memory, get_profile, forget_memory, add_document, list_documents]
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from rtmemory import RTMemoryClient


# ── Input schemas ────────────────────────────────────────────────────────

class SearchMemoriesInput(BaseModel):
    q: str = Field(description="Search query")
    mode: str = Field(default="hybrid", description="Search mode: hybrid, memory_only, documents_only")
    limit: int = Field(default=5, description="Maximum number of results")


class AddMemoryInput(BaseModel):
    content: str = Field(description="The memory content to store")
    entity_context: str | None = Field(default=None, description="Context to guide entity extraction")


class GetProfileInput(BaseModel):
    entity_id: str = Field(description="Entity ID of the user")
    q: str | None = Field(default=None, description="Optional query to include relevant search results")


class ForgetMemoryInput(BaseModel):
    memory_id: str | None = Field(default=None, description="Memory ID to forget")
    content_match: str | None = Field(default=None, description="Fuzzy content match to find and forget memories")
    reason: str | None = Field(default=None, description="Reason for forgetting")


class AddDocumentInput(BaseModel):
    content: str = Field(description="Document content (text or URL)")
    title: str | None = Field(default=None, description="Document title")


class ListDocumentsInput(BaseModel):
    status: str | None = Field(default=None, description="Filter by status")
    sort: str = Field(default="created_at", description="Sort field")
    order: str = Field(default="desc", description="Sort order")
    limit: int = Field(default=10, description="Max results")


# ── RTMemoryTools ─────────────────────────────────────────────────────────

class RTMemoryTools:
    """Generates LangChain StructuredTool objects wrapping RTMemoryClient."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        space_id: str = "",
        user_id: str | None = None,
    ) -> None:
        self.client = RTMemoryClient(base_url=base_url, api_key=api_key)
        self.space_id = space_id
        self.user_id = user_id

    def get_tools(self) -> list[StructuredTool]:
        """Return a list of LangChain StructuredTool objects."""
        return [
            self._search_memories_tool(),
            self._add_memory_tool(),
            self._get_profile_tool(),
            self._forget_memory_tool(),
            self._add_document_tool(),
            self._list_documents_tool(),
        ]

    def _search_memories_tool(self) -> StructuredTool:
        async def _func(q: str, mode: str = "hybrid", limit: int = 5) -> str:
            import json
            resp = await self.client.search(
                q=q,
                space_id=self.space_id,
                user_id=self.user_id,
                mode=mode,
                limit=limit,
            )
            return json.dumps(resp.model_dump(), default=str)

        return StructuredTool.from_function(
            coroutine=_func,
            name="search_memories",
            description="Search the user's memories and knowledge base. Use this to recall information about the user, their preferences, past conversations, or documents.",
            args_schema=SearchMemoriesInput,
        )

    def _add_memory_tool(self) -> StructuredTool:
        async def _func(content: str, entity_context: str | None = None) -> str:
            import json
            resp = await self.client.memories.add(
                content=content,
                space_id=self.space_id,
                user_id=self.user_id,
                entity_context=entity_context,
            )
            return json.dumps(resp.model_dump(), default=str)

        return StructuredTool.from_function(
            coroutine=_func,
            name="add_memory",
            description="Add a new memory for the user. Use this when the user shares new facts, preferences, or status changes that should be remembered.",
            args_schema=AddMemoryInput,
        )

    def _get_profile_tool(self) -> StructuredTool:
        async def _func(entity_id: str, q: str | None = None) -> str:
            import json
            resp = await self.client.profile.get(
                entity_id=entity_id,
                space_id=self.space_id,
                q=q,
            )
            return json.dumps(resp.model_dump(), default=str)

        return StructuredTool.from_function(
            coroutine=_func,
            name="get_profile",
            description="Get the user's profile from the knowledge graph. Use this to understand the user's identity, preferences, and current status.",
            args_schema=GetProfileInput,
        )

    def _forget_memory_tool(self) -> StructuredTool:
        async def _func(
            memory_id: str | None = None,
            content_match: str | None = None,
            reason: str | None = None,
        ) -> str:
            import json
            resp = await self.client.memories.forget(
                memory_id=memory_id,
                content_match=content_match,
                reason=reason,
            )
            return json.dumps(resp, default=str)

        return StructuredTool.from_function(
            coroutine=_func,
            name="forget_memory",
            description="Forget (soft-delete) a memory. Use this when the user asks to remove or correct outdated information.",
            args_schema=ForgetMemoryInput,
        )

    def _add_document_tool(self) -> StructuredTool:
        async def _func(content: str, title: str | None = None) -> str:
            import json
            resp = await self.client.documents.add(
                content=content,
                space_id=self.space_id,
                title=title,
            )
            return json.dumps(resp.model_dump(), default=str)

        return StructuredTool.from_function(
            coroutine=_func,
            name="add_document",
            description="Add a document (text or URL) to the knowledge base.",
            args_schema=AddDocumentInput,
        )

    def _list_documents_tool(self) -> StructuredTool:
        async def _func(
            status: str | None = None,
            sort: str = "created_at",
            order: str = "desc",
            limit: int = 10,
        ) -> str:
            import json
            resp = await self.client.documents.list(
                space_id=self.space_id,
                status=status,
                sort=sort,
                order=order,
                limit=limit,
            )
            return json.dumps(resp.model_dump(), default=str)

        return StructuredTool.from_function(
            coroutine=_func,
            name="list_documents",
            description="List documents in the knowledge base. Use this to see what documents are available or check processing status.",
            args_schema=ListDocumentsInput,
        )
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/langchain
pip install -e .
python -c "from rtmemory_langchain import RTMemoryTools; print('LangChain integration OK')"
# Expected: LangChain integration OK
```

**Commit:** `feat(langchain): implement RTMemoryTools with 6 LangChain Tool objects`

---

### Step 5.2 — Write LangChain integration tests

- [ ] Create `integrations/langchain/tests/test_tools.py`

**File: `integrations/langchain/tests/__init__.py`**
```python
```

**File: `integrations/langchain/tests/test_tools.py`**
```python
"""Tests for RTMemoryTools LangChain integration."""

import pytest

from rtmemory_langchain.tools import RTMemoryTools


BASE = "http://localhost:8000"


class TestRTMemoryTools:
    def test_get_tools_returns_six(self):
        tools_obj = RTMemoryTools(base_url=BASE, space_id="sp_001")
        tools = tools_obj.get_tools()
        assert len(tools) == 6
        names = [t.name for t in tools]
        assert "search_memories" in names
        assert "add_memory" in names
        assert "get_profile" in names
        assert "forget_memory" in names
        assert "add_document" in names
        assert "list_documents" in names

    def test_each_tool_is_coroutine(self):
        tools_obj = RTMemoryTools(base_url=BASE, space_id="sp_001")
        tools = tools_obj.get_tools()
        for tool in tools:
            assert tool.coroutine is not None, f"tool {tool.name} missing async coroutine"

    def test_each_tool_has_args_schema(self):
        tools_obj = RTMemoryTools(base_url=BASE, space_id="sp_001")
        tools = tools_obj.get_tools()
        for tool in tools:
            assert tool.args_schema is not None, f"tool {tool.name} missing args_schema"

    def test_search_memories_input_schema(self):
        from rtmemory_langchain.tools import SearchMemoriesInput
        schema = SearchMemoriesInput.model_json_schema()
        assert "q" in schema["properties"]
        assert schema["properties"]["q"]["type"] == "string"

    def test_add_memory_input_schema(self):
        from rtmemory_langchain.tools import AddMemoryInput
        schema = AddMemoryInput.model_json_schema()
        assert "content" in schema["properties"]
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/langchain
pip install pytest
python -m pytest tests/ -v
# Expected: 5 tests pass
```

**Commit:** `test(langchain): add tests for RTMemoryTools`

---

## Phase 6: Claude Code Integration

### Step 6.1 — Implement ClaudeMemoryAdapter

- [ ] Create `integrations/claude/rtmemory_claude/__init__.py` and `integrations/claude/rtmemory_claude/memory_adapter.py`

**File: `integrations/claude/pyproject.toml`**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rtmemory-claude"
version = "0.1.0"
description = "Claude Code integration for RTMemory"
requires-python = ">=3.12"
license = "MIT"
dependencies = [
    "rtmemory",
]
```

**File: `integrations/claude/rtmemory_claude/__init__.py`**
```python
"""RTMemory Claude Code integration."""

from rtmemory_claude.memory_adapter import ClaudeMemoryAdapter

__all__ = ["ClaudeMemoryAdapter"]
```

**File: `integrations/claude/rtmemory_claude/memory_adapter.py`**
```python
"""ClaudeMemoryAdapter — maps file-system-style memory operations to RTMemory.

Claude Code uses a file-system-like interface for memory (view, create,
str_replace, insert, delete, rename). This adapter maps those operations
to RTMemory's search, add, update, and forget methods.

Usage::

    from rtmemory_claude import ClaudeMemoryAdapter

    adapter = ClaudeMemoryAdapter(
        base_url="http://localhost:8000",
        api_key="sk-...",
        space_id="sp_001",
        user_id="user_001",
    )
    # adapter.view("user preferences")
    # adapter.create("user prefers dark mode for IDE")
    # adapter.str_replace("mem_001", "user prefers light mode for IDE")
    # adapter.delete("mem_001")
"""

from __future__ import annotations

import json
from typing import Any

from rtmemory import RTMemoryClient


class ClaudeMemoryAdapter:
    """Adapter that maps Claude Code's file-system-style memory operations
    to RTMemory's search, add, update, and forget methods.

    The adapter translates:
      - view(path)     -> search(q=path) returns concatenated results
      - create(path)   -> add(content, space_id)
      - str_replace()  -> update(memory_id, content)
      - insert()       -> add(content, space_id) (append as new memory)
      - delete(path)   -> forget(memory_id)
      - rename(old,new) -> add(new content) + forget(old memory)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        space_id: str = "",
        user_id: str | None = None,
    ) -> None:
        self.client = RTMemoryClient(base_url=base_url, api_key=api_key)
        self.space_id = space_id
        self.user_id = user_id

    async def view(self, path: str) -> str:
        """View memory content by searching for it.

        Maps Claude's file view to a search query. Returns formatted
        text results similar to viewing a file.
        """
        resp = await self.client.search(
            q=path,
            space_id=self.space_id,
            user_id=self.user_id,
            mode="hybrid",
            limit=5,
        )
        if not resp.results:
            return f"# No memories found for: {path}"
        lines = [f"# Memory results for: {path}", ""]
        for i, r in enumerate(resp.results, 1):
            lines.append(f"## Result {i} (score: {r.score:.2f}, source: {r.source})")
            lines.append(r.content)
            if r.entity:
                lines.append(f"  Entity: {r.entity.name} ({r.entity.type})")
            lines.append("")
        return "\n".join(lines)

    async def create(self, path: str, content: str) -> str:
        """Create a new memory.

        Maps Claude's file create to adding a memory.
        The path is used as entity_context to guide extraction.
        """
        resp = await self.client.memories.add(
            content=content,
            space_id=self.space_id,
            user_id=self.user_id,
            entity_context=path,
        )
        return json.dumps({"id": resp.id, "content": resp.content, "confidence": resp.confidence})

    async def str_replace(self, memory_id: str, old_content: str, new_content: str) -> str:
        """Replace content in a memory by updating it.

        Maps Claude's str_replace to a memory update. The old_content
        parameter is accepted for interface compatibility but the update
        replaces the full memory content.
        """
        resp = await self.client.memories.update(
            id=memory_id,
            content=new_content,
        )
        return json.dumps({"id": resp.id, "content": resp.content})

    async def insert(self, path: str, content: str) -> str:
        """Insert (append) content as a new memory.

        Maps Claude's insert to adding a new memory rather than
        modifying an existing one, since memories are immutable
        in the temporal graph — updates create new versions.
        """
        resp = await self.client.memories.add(
            content=content,
            space_id=self.space_id,
            user_id=self.user_id,
            entity_context=path,
        )
        return json.dumps({"id": resp.id, "content": resp.content, "confidence": resp.confidence})

    async def delete(self, memory_id: str) -> str:
        """Delete (forget) a memory by ID.

        Maps Claude's file delete to soft-forgetting a memory.
        """
        resp = await self.client.memories.forget(
            memory_id=memory_id,
            reason="Claude Code delete operation",
        )
        return json.dumps(resp)

    async def rename(self, old_memory_id: str, new_path: str) -> str:
        """Rename a memory by creating a new one and forgetting the old.

        Maps Claude's rename to add(forwarding content) + forget(old).
        The new memory references the old path.
        """
        # First, get the old memory content
        old_memory = await self.client.memories.get(old_memory_id)
        # Create new memory with updated entity_context
        new_resp = await self.client.memories.add(
            content=old_memory.content,
            space_id=self.space_id,
            user_id=self.user_id,
            entity_context=new_path,
        )
        # Forget the old memory
        await self.client.memories.forget(
            memory_id=old_memory_id,
            reason=f"Renamed to {new_resp.id}",
        )
        return json.dumps({
            "old_id": old_memory_id,
            "new_id": new_resp.id,
            "new_content": new_resp.content,
        })
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/claude
pip install -e .
python -c "from rtmemory_claude import ClaudeMemoryAdapter; print('ClaudeMemoryAdapter OK')"
# Expected: ClaudeMemoryAdapter OK
```

**Commit:** `feat(claude): implement ClaudeMemoryAdapter for file-system-style memory ops`

---

### Step 6.2 — Write ClaudeMemoryAdapter tests

- [ ] Create `integrations/claude/tests/test_memory_adapter.py`

**File: `integrations/claude/tests/__init__.py`**
```python
```

**File: `integrations/claude/tests/test_memory_adapter.py`**
```python
"""Tests for ClaudeMemoryAdapter."""

import json

import httpx
import pytest
import respx

from rtmemory_claude.memory_adapter import ClaudeMemoryAdapter


BASE = "http://localhost:8000"


class TestClaudeMemoryAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_view_returns_formatted_results(self):
        respx.post(f"{BASE}/v1/search/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "type": "memory",
                            "content": "User prefers dark mode",
                            "score": 0.95,
                            "source": "vector",
                            "entity": {"name": "User", "type": "person"},
                            "metadata": {},
                        }
                    ],
                    "timing_ms": 10,
                },
            )
        )
        adapter = ClaudeMemoryAdapter(base_url=BASE, space_id="sp_001")
        result = await adapter.view("user preferences")
        assert "dark mode" in result
        assert "0.95" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_view_no_results(self):
        respx.post(f"{BASE}/v1/search/").mock(
            return_value=httpx.Response(200, json={"results": [], "timing_ms": 5})
        )
        adapter = ClaudeMemoryAdapter(base_url=BASE, space_id="sp_001")
        result = await adapter.view("nonexistent topic")
        assert "No memories found" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_create_adds_memory(self):
        respx.post(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(
                200,
                json={"id": "mem_new", "content": "I prefer dark mode", "confidence": 0.9},
            )
        )
        adapter = ClaudeMemoryAdapter(base_url=BASE, space_id="sp_001")
        result = await adapter.create("user preferences", "I prefer dark mode")
        parsed = json.loads(result)
        assert parsed["id"] == "mem_new"

    @respx.mock
    @pytest.mark.asyncio
    async def test_str_replace_updates_memory(self):
        respx.patch(f"{BASE}/v1/memories/mem_001").mock(
            return_value=httpx.Response(
                200,
                json={"id": "mem_001", "content": "I prefer light mode", "confidence": 0.9},
            )
        )
        adapter = ClaudeMemoryAdapter(base_url=BASE, space_id="sp_001")
        result = await adapter.str_replace("mem_001", "dark mode", "light mode")
        parsed = json.loads(result)
        assert parsed["content"] == "I prefer light mode"

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_forgets_memory(self):
        respx.post(f"{BASE}/v1/memories/forget").mock(
            return_value=httpx.Response(200, json={"forgotten": True, "memory_id": "mem_001"})
        )
        adapter = ClaudeMemoryAdapter(base_url=BASE, space_id="sp_001")
        result = await adapter.delete("mem_001")
        parsed = json.loads(result)
        assert parsed["forgotten"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_insert_creates_new_memory(self):
        respx.post(f"{BASE}/v1/memories/").mock(
            return_value=httpx.Response(
                200,
                json={"id": "mem_ins", "content": "Also likes vim", "confidence": 0.8},
            )
        )
        adapter = ClaudeMemoryAdapter(base_url=BASE, space_id="sp_001")
        result = await adapter.insert("editor preferences", "Also likes vim")
        parsed = json.loads(result)
        assert parsed["id"] == "mem_ins"
```

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/claude
pip install pytest pytest-asyncio respx
python -m pytest tests/ -v
# Expected: 6 tests pass
```

**Commit:** `test(claude): add tests for ClaudeMemoryAdapter`

---

## Phase 7: Integration Verification and Polish

### Step 7.1 — Run all Python SDK tests end-to-end

- [ ] Run the full Python SDK test suite to verify everything still works

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
python -m pytest tests/ -v --tb=short
# Expected: all ~19 tests pass (15 client + 4 tools)
```

**Commit:** `test(sdk-python): verify all SDK tests pass end-to-end`

---

### Step 7.2 — Run all JS SDK tests

- [ ] Run vitest to verify the JS SDK

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npm test
# Expected: all ~15 tests pass
```

**Commit:** `test(sdk-js): verify all JS SDK tests pass`

---

### Step 7.3 — Run MCP server, LangChain, and Claude integration tests

- [ ] Run all integration test suites

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/mcp-server && python -m pytest tests/ -v
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/langchain && python -m pytest tests/ -v
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/claude && python -m pytest tests/ -v
# Expected: 2 + 5 + 6 = 13 tests pass across all integrations
```

**Commit:** `test(integrations): verify all integration tests pass`

---

### Step 7.4 — Verify Python SDK package installs cleanly

- [ ] Fresh install test

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-python
pip install -e ".[dev]" --force-reinstall --no-deps
python -c "from rtmemory import RTMemoryClient; print('SDK import OK')"
# Expected: SDK import OK
```

**Commit:** `chore(sdk-python): verify clean install`

---

### Step 7.5 — Build and verify JS SDK distribution

- [ ] Build the JS SDK and verify output

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/sdk-js
npx tsc
ls -la dist/
# Expected: index.js, index.d.ts, client.js, client.d.ts, types.js, types.d.ts
```

**Commit:** `build(sdk-js): verify distribution build`

---

### Step 7.6 — Verify MCP Server can start

- [ ] Test MCP server entry point loads without error

**Run:**
```bash
cd /home/ubuntu/ReToneProjects/RTMemory/integrations/mcp-server
pip install -e .
timeout 3 rtmemory-mcp || true
# Expected: server starts, then times out (no stdin) — no import errors
```

**Commit:** `chore(mcp-server): verify entry point loads`

---

## Summary of All Files Created

| File | Purpose |
|------|---------|
| `sdk-python/pyproject.toml` | Python SDK build config |
| `sdk-python/rtmemory/__init__.py` | Python SDK package init with exports |
| `sdk-python/rtmemory/types.py` | All Pydantic request/response models |
| `sdk-python/rtmemory/client.py` | RTMemoryClient main entry point |
| `sdk-python/rtmemory/memories.py` | MemoriesNamespace |
| `sdk-python/rtmemory/search.py` | SearchNamespace |
| `sdk-python/rtmemory/profile.py` | ProfileNamespace |
| `sdk-python/rtmemory/documents.py` | DocumentsNamespace |
| `sdk-python/rtmemory/conversations.py` | ConversationsNamespace |
| `sdk-python/rtmemory/graph.py` | GraphNamespace |
| `sdk-python/rtmemory/spaces.py` | SpacesNamespace |
| `sdk-python/rtmemory/tools.py` | get_memory_tools() for generic LLM agents |
| `sdk-python/tests/__init__.py` | Test package init |
| `sdk-python/tests/test_client.py` | TDD tests for all client namespaces |
| `sdk-python/tests/test_tools.py` | Tests for get_memory_tools |
| `sdk-js/package.json` | JS SDK package config |
| `sdk-js/tsconfig.json` | TypeScript compiler config |
| `sdk-js/vitest.config.ts` | Vitest test config |
| `sdk-js/src/types.ts` | All Zod schemas + TypeScript types |
| `sdk-js/src/client.ts` | RTMemoryClient + all namespaces |
| `sdk-js/src/index.ts` | Barrel export |
| `sdk-js/src/__tests__/client.test.ts` | Vitest tests for JS SDK |
| `integrations/mcp-server/pyproject.toml` | MCP Server build config |
| `integrations/mcp-server/rtmemory_mcp_server/__init__.py` | MCP Server package init |
| `integrations/mcp-server/rtmemory_mcp_server/main.py` | MCP Server with 6 tools |
| `integrations/mcp-server/tests/__init__.py` | Test package init |
| `integrations/mcp-server/tests/test_main.py` | MCP Server tests |
| `integrations/langchain/pyproject.toml` | LangChain integration build config |
| `integrations/langchain/rtmemory_langchain/__init__.py` | LangChain package init |
| `integrations/langchain/rtmemory_langchain/tools.py` | RTMemoryTools class |
| `integrations/langchain/tests/__init__.py` | Test package init |
| `integrations/langchain/tests/test_tools.py` | LangChain integration tests |
| `integrations/claude/pyproject.toml` | Claude integration build config |
| `integrations/claude/rtmemory_claude/__init__.py` | Claude package init |
| `integrations/claude/rtmemory_claude/memory_adapter.py` | ClaudeMemoryAdapter |
| `integrations/claude/tests/__init__.py` | Test package init |
| `integrations/claude/tests/test_memory_adapter.py` | Claude adapter tests |

**Total: 37 files, ~47 tests, 28 implementation steps**