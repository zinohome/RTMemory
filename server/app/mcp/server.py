"""RTMemory MCP Server — exposes RTMemory operations as MCP tools.

Tools provided:
  - add_memory: Add a new memory to a space
  - search_memory: Search memories and documents
  - get_profile: Get a user profile
  - forget_memory: Soft-delete a memory
  - add_document: Upload a document
  - list_spaces: List available spaces

Run with: python -m app.mcp.server

Uses stdio transport for integration with Claude Desktop, Cursor, etc.
"""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any

# MCP SDK — graceful import check
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    print(
        "MCP SDK not installed. Run: pip install mcp",
        file=sys.stderr,
    )
    raise

# Create the MCP server
app = Server("rtmemory")

# ── Tool definitions ─────────────────────────────────────────────

TOOLS = [
    Tool(
        name="add_memory",
        description="Add a new memory to an RTMemory space. Extracts entities and relations automatically.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content text"},
                "space_id": {"type": "string", "description": "UUID of the space"},
                "org_id": {"type": "string", "description": "UUID of the org"},
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "status", "inference"],
                    "default": "fact",
                },
                "custom_id": {"type": "string", "description": "Optional external ID"},
            },
            "required": ["content", "space_id", "org_id"],
        },
    ),
    Tool(
        name="search_memory",
        description="Search memories, entities, and documents in RTMemory. Returns ranked results.",
        inputSchema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query"},
                "space_id": {"type": "string", "description": "UUID of the space"},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["q", "space_id"],
        },
    ),
    Tool(
        name="get_profile",
        description="Get a computed user profile from the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "UUID of the user entity"},
                "space_id": {"type": "string", "description": "UUID of the space"},
            },
            "required": ["entity_id", "space_id"],
        },
    ),
    Tool(
        name="forget_memory",
        description="Soft-delete (forget) a memory by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID of the memory to forget"},
                "reason": {"type": "string", "default": "", "description": "Reason for forgetting"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="add_document",
        description="Upload a document to RTMemory for extraction and indexing.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "content": {"type": "string", "description": "Document content text"},
                "space_id": {"type": "string", "description": "UUID of the space"},
                "org_id": {"type": "string", "description": "UUID of the org"},
            },
            "required": ["title", "content", "space_id", "org_id"],
        },
    ),
    Tool(
        name="list_spaces",
        description="List available RTMemory spaces.",
        inputSchema={
            "type": "object",
            "properties": {
                "org_id": {"type": "string", "description": "UUID of the org to filter by"},
            },
            "required": ["org_id"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return available MCP tools."""
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle a tool invocation by forwarding to the RTMemory API."""
    # Lazy import to avoid circular deps
    from app.config import get_settings
    from app.db.session import get_session_factory
    from app.core.graph_engine import GraphEngine

    result_data: dict[str, Any] = {}

    try:
        if name == "add_memory":
            factory = get_session_factory()
            async with factory() as session:
                engine = GraphEngine(session)
                from app.schemas.graph import MemoryCreate, MemoryType
                data = MemoryCreate(
                    content=arguments["content"],
                    space_id=uuid.UUID(arguments["space_id"]),
                    org_id=uuid.UUID(arguments["org_id"]),
                    memory_type=MemoryType(arguments.get("memory_type", "fact")),
                    custom_id=arguments.get("custom_id"),
                )
                memory = await engine.create_memory(data)
                await session.commit()
                result_data = {"id": str(memory.id), "content": memory.content, "version": memory.version}

        elif name == "search_memory":
            from app.core.search_engine import SearchEngine
            from app.core.embedding import create_embedding_service
            from app.core.llm import create_llm_adapter
            settings = get_settings()
            factory = get_session_factory()
            async with factory() as session:
                embedding_svc = create_embedding_service(settings.embedding)
                llm_adapter = create_llm_adapter(settings.llm)
                search_engine = SearchEngine(session, embedding_svc, llm_adapter)
                from app.schemas.search import SearchRequest
                req = SearchRequest(
                    q=arguments["q"],
                    space_id=uuid.UUID(arguments["space_id"]),
                    limit=arguments.get("limit", 10),
                )
                resp = await search_engine.search(req)
                result_data = {
                    "results": [
                        {"type": r.type, "content": r.content, "score": r.score}
                        for r in resp.results
                    ],
                    "timing_ms": resp.timing_ms,
                }

        elif name == "get_profile":
            from app.core.profile_engine import ProfileEngine
            factory = get_session_factory()
            async with factory() as session:
                engine = ProfileEngine(graph_engine=GraphEngine(session))
                profile = await engine.compute(
                    entity_id=arguments["entity_id"],
                    space_id=arguments["space_id"],
                )
                result_data = profile.model_dump(mode="json")

        elif name == "forget_memory":
            factory = get_session_factory()
            async with factory() as session:
                engine = GraphEngine(session)
                from app.schemas.graph import MemoryForget
                result = await engine.forget_memory(
                    uuid.UUID(arguments["memory_id"]),
                    MemoryForget(forget_reason=arguments.get("reason", "")),
                )
                await session.commit()
                result_data = {"id": str(result.id), "is_forgotten": result.is_forgotten}

        elif name == "add_document":
            factory = get_session_factory()
            async with factory() as session:
                engine = GraphEngine(session)
                from app.db.models import Document
                now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
                doc = Document(
                    id=uuid.uuid4(),
                    title=arguments["title"],
                    content=arguments["content"],
                    doc_type="text",
                    status="queued",
                    org_id=uuid.UUID(arguments["org_id"]),
                    space_id=uuid.UUID(arguments["space_id"]),
                    created_at=now,
                    updated_at=now,
                )
                session.add(doc)
                await session.commit()
                result_data = {"id": str(doc.id), "status": doc.status}

        elif name == "list_spaces":
            factory = get_session_factory()
            async with factory() as session:
                from sqlalchemy import select, func
                from app.db.models import Space
                stmt = select(Space).where(Space.org_id == uuid.UUID(arguments["org_id"]))
                result = await session.execute(stmt)
                spaces = result.scalars().all()
                result_data = {
                    "spaces": [
                        {"id": str(s.id), "name": s.name, "description": s.description}
                        for s in spaces
                    ]
                }

        else:
            result_data = {"error": f"Unknown tool: {name}"}

    except Exception as e:
        result_data = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(result_data, default=str))]


async def main():
    """Run the MCP server via stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())