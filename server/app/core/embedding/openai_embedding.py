"""OpenAI embedding service - stub."""


class OpenAIEmbeddingService:
    """Stub — will be implemented in Phase 10."""

    def __init__(self, api_key="", model="text-embedding-3-small", base_url="https://api.openai.com/v1", client=None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url