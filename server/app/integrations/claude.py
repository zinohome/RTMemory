"""Claude Code adapter — integrates RTMemory with Claude Code via MCP protocol.

Provides an MCP server that exposes RTMemory operations as tools
that Claude Code can invoke during conversations.

Usage from Claude Code settings::

    {
      "mcpServers": {
        "rtmemory": {
          "command": "python",
          "args": ["-m", "app.integrations.claude"],
          "env": {
            "RTMEMORY_BASE_URL": "http://localhost:8000",
            "RTMEMORY_API_KEY": "sk-...",
            "RTMEMORY_DEFAULT_SPACE": "sp_001"
          }
        }
      }
    }
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx


# ── Configuration ────────────────────────────────────────────────────────

BASE_URL = os.environ.get("RTMEMORY_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("RTMEMORY_API_KEY", "")
DEFAULT_SPACE = os.environ.get("RTMEMORY_DEFAULT_SPACE", "default")


# ── HTTP helper ──────────────────────────────────────────────────────────

async def _request(method: str, path: str, body: dict | None = None, params: dict | None = None) -> Any:
    """Make an HTTP request to the RTMemory server."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=30.0) as client:
        if method == "GET":
            resp = await client.get(path, params=params)
        elif method == "POST":
            resp = await client.post(path, json=body)
        elif method == "DELETE":
            resp = await client.delete(path, json=body)
        else:
            raise ValueError(f"Unsupported method: {method}")
        resp.raise_for_status()
        return resp.json()


# ── Tool implementations ─────────────────────────────────────────────────

async def tool_add_memory(arguments: dict[str, Any]) -> str:
    """Add a memory to RTMemory."""
    content = arguments["content"]
    space_id = arguments.get("space_id", DEFAULT_SPACE)
    user_id = arguments.get("user_id")
    entity_context = arguments.get("entity_context")

    body: dict[str, Any] = {"content": content, "space_id": space_id}
    if user_id:
        body["user_id"] = user_id
    if entity_context:
        body["entity_context"] = entity_context

    result = await _request("POST", "/v1/memories/", body)
    return json.dumps(result, indent=2)


async def tool_search_memory(arguments: dict[str, Any]) -> str:
    """Search memories and knowledge base."""
    q = arguments["q"]
    space_id = arguments.get("space_id", DEFAULT_SPACE)
    mode = arguments.get("mode", "hybrid")
    limit = arguments.get("limit", 5)
    include_profile = arguments.get("include_profile", False)

    body = {
        "q": q,
        "space_id": space_id,
        "mode": mode,
        "limit": limit,
        "include_profile": include_profile,
    }
    result = await _request("POST", "/v1/search/", body)
    return json.dumps(result, indent=2)


async def tool_get_profile(arguments: dict[str, Any]) -> str:
    """Get a user's profile from the knowledge graph."""
    entity_id = arguments["entity_id"]
    space_id = arguments.get("space_id", DEFAULT_SPACE)
    q = arguments.get("q")
    fresh = arguments.get("fresh", False)

    body = {"entity_id": entity_id, "space_id": space_id, "fresh": fresh}
    if q:
        body["q"] = q

    result = await _request("POST", "/v1/profile", body)
    return json.dumps(result, indent=2)


async def tool_forget_memory(arguments: dict[str, Any]) -> str:
    """Forget (soft-delete) a memory."""
    memory_id = arguments.get("memory_id")
    content_match = arguments.get("content_match")
    reason = arguments.get("reason")

    body: dict[str, Any] = {}
    if memory_id:
        body["memory_id"] = memory_id
    if content_match:
        body["content_match"] = content_match
    if reason:
        body["reason"] = reason

    result = await _request("POST", "/v1/memories/forget", body)
    return json.dumps(result, indent=2)


async def tool_add_document(arguments: dict[str, Any]) -> str:
    """Add a document to the knowledge base."""
    content = arguments["content"]
    space_id = arguments.get("space_id", DEFAULT_SPACE)
    title = arguments.get("title")

    body: dict[str, Any] = {"content": content, "space_id": space_id}
    if title:
        body["title"] = title

    result = await _request("POST", "/v1/documents/", body)
    return json.dumps(result, indent=2)


async def tool_list_spaces(arguments: dict[str, Any]) -> str:
    """List available memory spaces."""
    result = await _request("GET", "/v1/spaces/")
    return json.dumps(result, indent=2)


# ── MCP Protocol Implementation (stdio transport) ────────────────────────

TOOLS = [
    {
        "name": "add_memory",
        "description": "Add a new memory for the user. Use this when the user shares new facts, preferences, or status changes that should be remembered for future conversations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "space_id": {"type": "string", "description": "Space ID (optional, uses default)"},
                "user_id": {"type": "string", "description": "User ID (optional)"},
                "entity_context": {"type": "string", "description": "Context to guide entity extraction (optional)"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memory",
        "description": "Search the user's memories and knowledge base. Use this when you need to recall information about the user, their preferences, past conversations, or documents in the knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query"},
                "space_id": {"type": "string", "description": "Space ID (optional, uses default)"},
                "mode": {"type": "string", "enum": ["hybrid", "memory_only", "documents_only"], "description": "Search mode", "default": "hybrid"},
                "limit": {"type": "integer", "description": "Max results", "default": 5},
                "include_profile": {"type": "boolean", "description": "Include user profile", "default": false},
            },
            "required": ["q"],
        },
    },
    {
        "name": "get_profile",
        "description": "Get the user's profile from the knowledge graph. Use this to understand the user's identity, preferences, current status, and relationships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "The entity ID of the user"},
                "space_id": {"type": "string", "description": "Space ID (optional, uses default)"},
                "q": {"type": "string", "description": "Optional query to include search results (optional)"},
                "fresh": {"type": "boolean", "description": "Force fresh computation", "default": false},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "forget_memory",
        "description": "Forget (soft-delete) a memory. Use this when the user asks to remove or correct outdated information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to forget"},
                "content_match": {"type": "string", "description": "Fuzzy content match to find and forget"},
                "reason": {"type": "string", "description": "Reason for forgetting"},
            },
        },
    },
    {
        "name": "add_document",
        "description": "Add a document (text or URL) to the knowledge base. Use this when the user wants to ingest a document for future reference.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Document content (text or URL)"},
                "title": {"type": "string", "description": "Document title (optional)"},
                "space_id": {"type": "string", "description": "Space ID (optional, uses default)"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "list_spaces",
        "description": "List available memory spaces.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

TOOL_HANDLERS = {
    "add_memory": tool_add_memory,
    "search_memory": tool_search_memory,
    "get_profile": tool_get_profile,
    "forget_memory": tool_forget_memory,
    "add_document": tool_add_document,
    "list_spaces": tool_list_spaces,
}


def _send(message: dict[str, Any]) -> None:
    """Send a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


async def handle_request(request: dict[str, Any]) -> None:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    request_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        _send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "rtmemory", "version": "0.1.0"},
            },
        })
    elif method == "notifications/initialized":
        pass  # No response needed
    elif method == "tools/list":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            _send({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            })
        else:
            try:
                result_text = await handler(arguments)
                _send({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                    },
                })
            except Exception as e:
                _send({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                })
    else:
        _send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })


async def run_stdio() -> None:
    """Run the MCP server on stdio transport."""
    import asyncio

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_event_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            await handle_request(request)
        except json.JSONDecodeError:
            _send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_stdio())