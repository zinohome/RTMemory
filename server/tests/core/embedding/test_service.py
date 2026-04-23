"""Tests for EmbeddingService abstract base class."""

import pytest

from app.core.embedding.service import EmbeddingService


def test_embedding_service_cannot_be_instantiated_directly():
    """EmbeddingService is abstract and cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract method"):
        EmbeddingService()


def test_incomplete_subclass_cannot_be_instantiated():
    """Subclass without all abstract methods cannot be instantiated."""

    class IncompleteService(EmbeddingService):
        async def embed(self, texts):
            return [[0.1]]

    with pytest.raises(TypeError, match="abstract method"):
        IncompleteService()


def test_complete_subclass_can_be_instantiated():
    """Subclass implementing all abstract methods can be instantiated."""

    class CompleteService(EmbeddingService):
        async def embed(self, texts):
            return [[0.1, 0.2, 0.3]]

        def get_dimension(self):
            return 3

    service = CompleteService()
    assert isinstance(service, EmbeddingService)


def test_embedding_service_has_required_methods():
    """EmbeddingService defines embed and get_dimension abstract methods."""
    abstract_methods = EmbeddingService.__abstractmethods__
    assert "embed" in abstract_methods
    assert "get_dimension" in abstract_methods