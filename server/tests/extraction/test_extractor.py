"""Tests for Extractor — Layer 2 LLM structured extraction."""

import pytest

from app.extraction.extractor import Extractor
from app.schemas.extraction import ExtractionResult, ExtractedEntity, ExtractedRelation, ExtractedMemory, ExtractedContradiction


# ── Mock LLM adapter ─────────────────────────────────────────────

class MockLLMAdapter:
    """Mock LLM adapter that returns predefined structured responses."""

    def __init__(self, response: dict | None = None):
        self._response = response or self._default_response()
        self.call_count = 0
        self.last_messages = None
        self.last_schema = None

    def _default_response(self) -> dict:
        return {
            "entities": [
                {"name": "张军", "type": "person", "description": "软件工程师", "confidence": 0.95},
                {"name": "北京", "type": "location", "description": "中国首都", "confidence": 0.9},
            ],
            "relations": [
                {
                    "source": "张军",
                    "target": "北京",
                    "relation": "lives_in",
                    "value": "",
                    "valid_from": "2024-01",
                    "valid_to": None,
                    "confidence": 0.95,
                },
            ],
            "memories": [
                {
                    "content": "张军搬到了北京",
                    "type": "fact",
                    "confidence": 0.9,
                    "entity_name": "张军",
                },
            ],
            "contradictions": [],
        }

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict,
        temperature: float = 0.1,
    ) -> dict:
        self.call_count += 1
        self.last_messages = messages
        self.last_schema = schema
        return self._response


class MockLLMAdapterEmpty(MockLLMAdapter):
    """Returns empty extraction result."""

    def __init__(self):
        super().__init__(response={
            "entities": [],
            "relations": [],
            "memories": [],
            "contradictions": [],
        })


# ── Tests ────────────────────────────────────────────────────────

class TestExtractorBasic:
    """Basic extraction tests using mock LLM adapter."""

    @pytest.fixture
    def extractor(self):
        return Extractor(llm_adapter=MockLLMAdapter())

    @pytest.fixture
    def extractor_empty(self):
        return Extractor(llm_adapter=MockLLMAdapterEmpty())

    async def test_extract_returns_extraction_result(self, extractor):
        result = await extractor.extract("我搬到北京了")
        assert isinstance(result, ExtractionResult)

    async def test_extract_returns_entities(self, extractor):
        result = await extractor.extract("我搬到北京了")
        assert len(result.entities) == 2
        assert result.entities[0].name == "张军"
        assert result.entities[1].name == "北京"

    async def test_extract_returns_relations(self, extractor):
        result = await extractor.extract("我搬到北京了")
        assert len(result.relations) == 1
        assert result.relations[0].relation == "lives_in"
        assert result.relations[0].source == "张军"
        assert result.relations[0].target == "北京"

    async def test_extract_returns_memories(self, extractor):
        result = await extractor.extract("我搬到北京了")
        assert len(result.memories) == 1
        assert "北京" in result.memories[0].content

    async def test_extract_returns_no_contradictions(self, extractor):
        result = await extractor.extract("我搬到北京了")
        assert len(result.contradictions) == 0

    async def test_extract_empty_result(self, extractor_empty):
        result = await extractor_empty.extract("你好")
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert len(result.memories) == 0

    async def test_extract_calls_llm_once(self, extractor):
        await extractor.extract("我搬到北京了")
        assert extractor.llm_adapter.call_count == 1

    async def test_extract_passes_messages_to_llm(self, extractor):
        await extractor.extract("我搬到北京了")
        messages = extractor.llm_adapter.last_messages
        assert len(messages) >= 1
        # System message should contain extraction instructions
        assert any("system" in m.get("role", "") for m in messages)


class TestExtractorWithContext:
    """Tests for extraction with entity_context hint."""

    async def test_extract_with_entity_context(self):
        mock = MockLLMAdapter()
        extractor = Extractor(llm_adapter=mock)
        await extractor.extract("最近在用Next.js", entity_context="这是关于张军的知识库")
        messages = mock.last_messages
        # Entity context should appear in the prompt
        prompt_text = " ".join(m.get("content", "") for m in messages)
        assert "张军" in prompt_text


class TestExtractorContradictions:
    """Tests for contradiction detection in extraction."""

    async def test_extract_with_contradiction(self):
        mock_response = {
            "entities": [
                {"name": "张军", "type": "person", "description": "软件工程师", "confidence": 0.95},
                {"name": "北京", "type": "location", "description": "", "confidence": 0.9},
            ],
            "relations": [
                {
                    "source": "张军",
                    "target": "北京",
                    "relation": "lives_in",
                    "value": "",
                    "valid_from": "2024-06",
                    "valid_to": None,
                    "confidence": 0.95,
                },
            ],
            "memories": [
                {
                    "content": "张军搬到了北京",
                    "type": "fact",
                    "confidence": 0.9,
                    "entity_name": "张军",
                },
            ],
            "contradictions": [
                {
                    "new": "lives_in(北京)",
                    "old": "lives_in(上海)",
                    "resolution": "update",
                },
            ],
        }
        mock = MockLLMAdapter(response=mock_response)
        extractor = Extractor(llm_adapter=mock)
        result = await extractor.extract("我搬到北京了，之前在上海")
        assert len(result.contradictions) == 1
        assert result.contradictions[0].resolution == "update"
        assert result.contradictions[0].new == "lives_in(北京)"


class TestExtractorConversation:
    """Tests for multi-message conversation extraction."""

    async def test_extract_conversation(self):
        mock = MockLLMAdapter()
        extractor = Extractor(llm_adapter=mock)
        messages = [
            {"role": "user", "content": "我最近在学Rust"},
            {"role": "assistant", "content": "Rust是一门系统编程语言"},
            {"role": "user", "content": "我决定用Rust重写我的后端服务"},
        ]
        result = await extractor.extract_conversation(messages)
        assert isinstance(result, ExtractionResult)

    async def test_extract_conversation_calls_llm_once(self):
        mock = MockLLMAdapter()
        extractor = Extractor(llm_adapter=mock)
        messages = [
            {"role": "user", "content": "我用Python"},
            {"role": "assistant", "content": "Python很好"},
        ]
        await extractor.extract_conversation(messages)
        assert mock.call_count == 1