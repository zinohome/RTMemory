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