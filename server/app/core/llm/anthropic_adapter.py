"""Anthropic LLM adapter.

Implements LLMAdapter for the Anthropic messages API.
Uses prompt-based JSON instructions for structured output
since Anthropic does not natively support response_format.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class AnthropicAdapter:
    """LLM adapter for the Anthropic messages API.

    Handles the Anthropic-specific API format:
    - System messages are sent as a top-level "system" parameter
    - Messages list contains only user/assistant messages
    - Structured output uses prompt engineering with JSON schema instructions

    Args:
        api_key: Anthropic API key.
        model: Model identifier, e.g. "claude-sonnet-4-20250514".
        base_url: API base URL. Defaults to "https://api.anthropic.com".
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    def _prepare_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Extract system message from messages list.

        Anthropic requires the system message as a separate top-level parameter.
        This method extracts it and returns (system, filtered_messages).

        Args:
            messages: List of message dicts that may include a system message.

        Returns:
            Tuple of (system_content_or_None, filtered_messages_without_system).
        """
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        return system, filtered

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to Anthropic and return the assistant text response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                System messages are extracted and sent as a top-level parameter.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens in the response.
            response_format: Ignored for Anthropic. Anthropic does not natively
                support response_format; use complete_structured for JSON output.

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        system, filtered = self._prepare_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": filtered,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            payload["system"] = system

        response = await self._client.post(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        # Anthropic returns content as a list of content blocks
        return data["content"][0]["text"]

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to Anthropic and return a parsed JSON response.

        Uses prompt engineering: adds the JSON schema instruction to the system
        message (or creates one) since Anthropic does not natively support
        structured output via response_format.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        schema_instruction = (
            "You must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )

        modified = []
        has_system = False
        for msg in messages:
            if msg["role"] == "system":
                modified.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + schema_instruction,
                })
                has_system = True
            else:
                modified.append(msg)

        if not has_system:
            modified.insert(0, {"role": "system", "content": schema_instruction})

        text = await self.complete(
            messages=modified,
            temperature=temperature,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Anthropic did not return valid JSON: {text}") from e