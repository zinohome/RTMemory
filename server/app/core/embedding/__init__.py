from .service import EmbeddingService
from .local_embedding import LocalEmbeddingService
from .openai_embedding import OpenAIEmbeddingService

__all__ = [
    "EmbeddingService",
    "LocalEmbeddingService",
    "OpenAIEmbeddingService",
]


def create_embedding_service(config, client=None):
    """Factory: create an embedding service from config.

    Args:
        config: EmbeddingConfig instance with provider, model, etc.
        client: Optional httpx.AsyncClient for testing.

    Returns:
        EmbeddingService instance.
    """
    provider = config.provider.lower()
    if provider == "local":
        from .local_embedding import LocalEmbeddingService

        return LocalEmbeddingService(model_name=config.model)
    elif provider == "openai":
        from .openai_embedding import OpenAIEmbeddingService

        return OpenAIEmbeddingService(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            client=client,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")