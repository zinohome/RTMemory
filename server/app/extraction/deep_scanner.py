"""DeepScanner — Layer 3: batch deep scan at conversation end.

Triggered when a conversation ends or accumulates N messages.
Captures implicit preferences, cross-message correlations,
status changes, and confidence adjustments that individual
message extraction might miss.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.schemas.extraction import (
    ConfidenceAdjustment,
    ContradictionResolution,
    DeepScanResult,
    EntityType,
    ExtractedContradiction,
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelation,
    MemoryType,
)

# JSON Schema for deep scan LLM output (extends extraction schema with confidence_adjustments)
DEEP_SCAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": [e.value for e in EntityType]},
                    "description": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "value": {"type": "string"},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_to": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["source", "target", "relation"],
            },
        },
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": [m.value for m in MemoryType]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "entity_name": {"type": ["string", "null"]},
                },
                "required": ["content", "type"],
            },
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "new": {"type": "string"},
                    "old": {"type": "string"},
                    "resolution": {
                        "type": "string",
                        "enum": [r.value for r in ContradictionResolution],
                    },
                },
                "required": ["new", "old", "resolution"],
            },
        },
        "confidence_adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_type": {"type": "string"},
                    "target_id": {"type": "string", "format": "uuid"},
                    "old_confidence": {"type": "number"},
                    "new_confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["target_type", "target_id", "old_confidence", "new_confidence"],
            },
        },
    },
    "required": ["entities", "relations", "memories", "contradictions", "confidence_adjustments"],
}

DEEP_SCAN_SYSTEM_PROMPT = """\
你是一个深度知识扫描器。你现在拿到了一段完整对话，请进行深度分析。

在单条消息提取的基础上，额外关注：
1. 隐含偏好：用户没有直接说"我喜欢X"，但行为模式暗示了偏好
2. 跨消息关联：不同消息之间的因果或关联关系
3. 状态变化：用户的状态在对话中发生了什么变化
4. 置信度调整：多次提及的事实应提升置信度

输出 JSON 包含：
- entities: 提取的实体
- relations: 提取的关系
- memories: 提取的记忆（特别是隐含的偏好和状态变化）
- contradictions: 发现的矛盾
- confidence_adjustments: 需要调整置信度的已有记忆/关系

置信度调整格式：
{"target_type": "memory" 或 "relation", "target_id": "UUID", "old_confidence": 0.7, "new_confidence": 0.9, "reason": "原因"}

规则同即时提取：实体类型只能是 person/org/location/concept/project/technology，
记忆类型只能是 fact/preference/status/inference，置信度 [0,1]。
"""


class DeepScanner:
    """Layer 3: deep scan triggered at conversation end.

    Scans the full conversation to capture patterns that only emerge
    across multiple messages — implicit preferences, state changes,
    cross-message correlations, and confidence adjustments.
    """

    def __init__(
        self,
        llm_adapter: Any,
        min_messages: int = 1,
    ) -> None:
        """Initialize the deep scanner.

        Args:
            llm_adapter: LLM adapter with async complete_structured method.
            min_messages: Minimum number of messages to trigger deep scan.
                If conversation has fewer messages, return empty result
                without calling the LLM.
        """
        self.llm_adapter = llm_adapter
        self.min_messages = min_messages

    def _build_messages(
        self,
        conversation: list[dict[str, str]],
        entity_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build LLM messages for deep scan.

        Args:
            conversation: List of message dicts with role/content.
            entity_context: Optional context hint.

        Returns:
            LLM message list.
        """
        conv_text = "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}"
            for m in conversation
        )

        user_content = f"请对以下完整对话进行深度扫描提取：\n\n{conv_text}"
        if entity_context:
            user_content = f"上下文提示：{entity_context}\n\n{user_content}"

        return [
            {"role": "system", "content": DEEP_SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_result(self, raw: dict) -> DeepScanResult:
        """Parse raw LLM output into DeepScanResult.

        Args:
            raw: Dict from the LLM adapter.

        Returns:
            DeepScanResult with all extracted items and confidence adjustments.
        """
        entities = [
            ExtractedEntity(
                name=e.get("name", ""),
                type=EntityType(e.get("type", "person")),
                description=e.get("description", ""),
                confidence=e.get("confidence", 0.8),
            )
            for e in raw.get("entities", [])
        ]

        relations = [
            ExtractedRelation(
                source=r.get("source", ""),
                target=r.get("target", ""),
                relation=r.get("relation", ""),
                value=r.get("value", ""),
                valid_from=r.get("valid_from"),
                valid_to=r.get("valid_to"),
                confidence=r.get("confidence", 0.8),
            )
            for r in raw.get("relations", [])
        ]

        memories = [
            ExtractedMemory(
                content=m.get("content", ""),
                type=MemoryType(m.get("type", "fact")),
                confidence=m.get("confidence", 0.8),
                entity_name=m.get("entity_name"),
            )
            for m in raw.get("memories", [])
        ]

        contradictions = [
            ExtractedContradiction(
                new=c.get("new", ""),
                old=c.get("old", ""),
                resolution=ContradictionResolution(c.get("resolution", "update")),
            )
            for c in raw.get("contradictions", [])
        ]

        confidence_adjustments = [
            ConfidenceAdjustment(
                target_type=a.get("target_type", "memory"),
                target_id=uuid.UUID(a.get("target_id", "00000000-0000-0000-0000-000000000000")),
                old_confidence=float(a.get("old_confidence", 0.5)),
                new_confidence=float(a.get("new_confidence", 0.5)),
                reason=a.get("reason", ""),
            )
            for a in raw.get("confidence_adjustments", [])
        ]

        return DeepScanResult(
            entities=entities,
            relations=relations,
            memories=memories,
            contradictions=contradictions,
            confidence_adjustments=confidence_adjustments,
        )

    async def deep_scan(
        self,
        messages: list[dict[str, str]],
        entity_context: str | None = None,
    ) -> DeepScanResult:
        """Perform deep scan on a complete conversation.

        Args:
            messages: Full conversation messages.
            entity_context: Optional context hint.

        Returns:
            DeepScanResult with all extracted items and confidence adjustments.
            Returns empty result if messages count is below min_messages threshold.
        """
        if len(messages) < self.min_messages:
            return DeepScanResult()

        llm_messages = self._build_messages(messages, entity_context=entity_context)
        raw = await self.llm_adapter.complete_structured(
            messages=llm_messages,
            schema=DEEP_SCAN_OUTPUT_SCHEMA,
            temperature=0.1,
        )
        return self._parse_result(raw)