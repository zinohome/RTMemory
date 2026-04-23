"""Ollama LLM adapter.

Implements LLMAdapter for Ollama using the OpenAI-compatible API endpoint.
Ollama provides an OpenAI-compatible API at /v1/chat/completions, which
allows us to use the same request format as the OpenAI adapter.

For structured output, uses response_format={"type": "json_object"} and
adds "Output valid JSON" instruction to the prompt.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaAdapter:
    """LLM adapter for Ollama using the OpenAI-compatible API.

    Uses Ollama's /v1/chat/completions endpoint which is compatible with
    the OpenAI chat completions format. No API key is required since
    Ollama runs locally.

    Args:
        model: Ollama model identifier, e.g. "qwen2.5:7b", "llama3.1:8b".
        base_url: Ollama server URL. Defaults to "http://localhost:11434".
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
    ):
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
        """Send messages to Ollama and return the assistant text response.

        Uses the OpenAI-compatible /v1/chat/completions endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
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
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
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
        """Send messages to Ollama and return a parsed JSON response.

        Uses Ollama's response_format={"type": "json_object"} and adds
        "Output valid JSON" instruction to the prompt along with the schema.

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
            "\n\nOutput valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        # Add the instruction to the last user message
        modified = [msg.copy() for msg in messages]
        if modified and modified[-1]["role"] == "user":
            modified[-1]["content"] = modified[-1]["content"] + schema_instruction
        else:
            modified.append({"role": "user", "content": "Output valid JSON." + schema_instruction})

        text = await self.complete(
            messages=modified,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama did not return valid JSON: {text}") from e