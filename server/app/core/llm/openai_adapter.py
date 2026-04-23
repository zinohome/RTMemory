"""OpenAI LLM adapter.

Implements LLMAdapter for the OpenAI chat completions API.
Supports GPT-4o, GPT-4o-mini, and other OpenAI models.
Uses native json_schema response_format for structured output.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAIAdapter:
    """LLM adapter for OpenAI chat completions API.

    Args:
        api_key: OpenAI API key (starts with "sk-").
        model: Model identifier, e.g. "gpt-4o", "gpt-4o-mini".
        base_url: API base URL. Defaults to "https://api.openai.com/v1".
            Can be set to a proxy or Azure OpenAI endpoint.
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to OpenAI and return the assistant text response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format spec, e.g. {"type": "json_object"}.

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to OpenAI and return a parsed JSON response.

        Uses OpenAI's native json_schema response_format for structured output.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
            },
        }
        text = await self.complete(
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI did not return valid JSON: {text}") from e