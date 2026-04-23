"""LLM Adapter abstract base class.

Defines the unified interface for all LLM providers.
Each provider (OpenAI, Anthropic, Ollama) implements this protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    """Abstract base class for LLM chat completion adapters.

    All providers must implement:
    - complete: standard chat completion returning raw text
    - complete_structured: chat completion returning parsed JSON
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to the LLM and return the text response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Example: [{"role": "user", "content": "Hello"}]
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format specification for the response.
                For OpenAI: {"type": "json_object"} or {"type": "json_schema", ...}
                For Ollama: {"type": "json_object"}
                Anthropic: Not natively supported; ignored.

        Returns:
            The assistant's response text.
        """
        ...

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to the LLM and return a parsed JSON response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for more deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        ...