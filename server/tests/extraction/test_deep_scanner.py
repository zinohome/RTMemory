"""Tests for DeepScanner — Layer 3 deep scan at conversation end."""

import pytest

from app.extraction.deep_scanner import DeepScanner
from app.schemas.extraction import DeepScanResult


class MockLLMAdapter:
    """Mock LLM adapter for deep scan testing."""

    def __init__(self, response: dict | None = None):
        self._response = response or self._default_response()
        self.call_count = 0
        self.last_messages = None

    def _default_response(self) -> dict:
        return {
            "entities": [
                {"name": "张军", "type": "person", "description": "全栈工程师", "confidence": 0.95},
                {"name": "RTMemory", "type": "project", "description": "时序知识图谱项目", "confidence": 0.9},
            ],
            "relations": [
                {
                    "source": "张军",
                    "target": "RTMemory",
                    "relation": "works_on",
                    "value": "",
                    "valid_from": "2026-04",
                    "valid_to": None,
                    "confidence": 0.85,
                },
            ],
            "memories": [
                {"content": "张军正在开发RTMemory项目", "type": "status", "confidence": 0.8, "entity_name": "张军"},
                {"content": "张军偏好Python和TypeScript", "type": "preference", "confidence": 0.75, "entity_name": "张军"},
            ],
            "contradictions": [],
            "confidence_adjustments": [
                {
                    "target_type": "relation",
                    "target_id": "00000000-0000-0000-0000-000000000001",
                    "old_confidence": 0.7,
                    "new_confidence": 0.9,
                    "reason": "多次提及确认",
                },
            ],
        }

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict,
        temperature: float = 0.1,
    ) -> dict:
        self.call_count += 1
        self.last_messages = messages
        return self._response


class TestDeepScannerBasic:
    """Basic deep scanner tests."""

    @pytest.fixture
    def scanner(self):
        return DeepScanner(llm_adapter=MockLLMAdapter())

    async def test_deep_scan_returns_result(self, scanner):
        messages = [
            {"role": "user", "content": "我最近在做一个知识图谱项目"},
            {"role": "assistant", "content": "听起来很有意思"},
            {"role": "user", "content": "叫RTMemory，用Python和TypeScript写的"},
            {"role": "user", "content": "我偏好简洁的代码风格"},
        ]
        result = await scanner.deep_scan(messages)
        assert isinstance(result, DeepScanResult)

    async def test_deep_scan_returns_entities(self, scanner):
        messages = [
            {"role": "user", "content": "我在做RTMemory项目"},
        ]
        result = await scanner.deep_scan(messages)
        assert len(result.entities) >= 1

    async def test_deep_scan_returns_memories(self, scanner):
        messages = [
            {"role": "user", "content": "我偏好Python"},
        ]
        result = await scanner.deep_scan(messages)
        assert len(result.memories) >= 1

    async def test_deep_scan_returns_confidence_adjustments(self, scanner):
        messages = [
            {"role": "user", "content": "我在做RTMemory项目"},
        ]
        result = await scanner.deep_scan(messages)
        assert len(result.confidence_adjustments) >= 1
        adj = result.confidence_adjustments[0]
        assert adj.new_confidence > adj.old_confidence

    async def test_deep_scan_calls_llm_once(self, scanner):
        messages = [
            {"role": "user", "content": "我用Python"},
        ]
        await scanner.deep_scan(messages)
        assert scanner.llm_adapter.call_count == 1


class TestDeepScannerEmpty:
    """Deep scanner with empty result."""

    async def test_deep_scan_empty_messages(self):
        mock = MockLLMAdapter(response={
            "entities": [],
            "relations": [],
            "memories": [],
            "contradictions": [],
            "confidence_adjustments": [],
        })
        scanner = DeepScanner(llm_adapter=mock)
        result = await scanner.deep_scan([])
        assert len(result.entities) == 0
        assert len(result.memories) == 0


class TestDeepScannerWithContext:
    """Deep scanner with entity context."""

    async def test_deep_scan_with_entity_context(self):
        mock = MockLLMAdapter()
        scanner = DeepScanner(llm_adapter=mock)
        messages = [
            {"role": "user", "content": "我最近在研究知识图谱"},
        ]
        await scanner.deep_scan(messages, entity_context="这是张军的知识库")
        messages_sent = mock.last_messages
        prompt_text = " ".join(m.get("content", "") for m in messages_sent)
        assert "张军" in prompt_text


class TestDeepScannerThreshold:
    """Deep scanner respects message count threshold."""

    async def test_deep_scan_skip_below_threshold(self):
        mock = MockLLMAdapter()
        scanner = DeepScanner(llm_adapter=mock, min_messages=3)
        # Only 2 messages — below threshold
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = await scanner.deep_scan(messages)
        assert mock.call_count == 0
        # Should still return a valid (empty) result
        assert isinstance(result, DeepScanResult)

    async def test_deep_scan_triggers_at_threshold(self):
        mock = MockLLMAdapter()
        scanner = DeepScanner(llm_adapter=mock, min_messages=3)
        messages = [
            {"role": "user", "content": "我搬到北京了"},
            {"role": "assistant", "content": "北京不错"},
            {"role": "user", "content": "我用Python"},
        ]
        result = await scanner.deep_scan(messages)
        assert mock.call_count == 1