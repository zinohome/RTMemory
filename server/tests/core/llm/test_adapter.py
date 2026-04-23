"""Tests for LLMAdapter abstract base class."""

import pytest

from app.core.llm.adapter import LLMAdapter


def test_llm_adapter_cannot_be_instantiated_directly():
    """LLMAdapter is abstract and cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract method"):
        LLMAdapter()


def test_incomplete_subclass_cannot_be_instantiated():
    """Subclass without all abstract methods cannot be instantiated."""

    class IncompleteAdapter(LLMAdapter):
        async def complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
            return "hello"

    with pytest.raises(TypeError, match="abstract method"):
        IncompleteAdapter()


def test_complete_subclass_can_be_instantiated():
    """Subclass implementing all abstract methods can be instantiated."""

    class CompleteAdapter(LLMAdapter):
        async def complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
            return "test response"

        async def complete_structured(self, messages, schema, temperature=0.1):
            return {"result": True}

    adapter = CompleteAdapter()
    assert isinstance(adapter, LLMAdapter)


def test_llm_adapter_has_required_methods():
    """LLMAdapter defines complete and complete_structured abstract methods."""
    abstract_methods = LLMAdapter.__abstractmethods__
    assert "complete" in abstract_methods
    assert "complete_structured" in abstract_methods