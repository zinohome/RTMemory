from .adapter import LLMAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .ollama_adapter import OllamaAdapter

__all__ = [
    "LLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
]


def create_llm_adapter(config, client=None):
    """Factory: create an LLM adapter from config.

    Args:
        config: LLMConfig instance with provider, model, etc.
        client: Optional httpx.AsyncClient for testing.

    Returns:
        LLMAdapter instance.
    """
    provider = config.provider.lower()
    if provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            client=client,
        )
    elif provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://api.anthropic.com",
            client=client,
        )
    elif provider == "ollama":
        from .ollama_adapter import OllamaAdapter

        return OllamaAdapter(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434",
            client=client,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")