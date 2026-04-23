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