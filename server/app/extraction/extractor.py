"""Extractor — Layer 2: LLM structured extraction.

Extracts entities, relations, memories, and contradictions from
conversation messages using the LLM adapter's complete_structured method.
"""

from __future__ import annotations

from typing import Any

from app.schemas.extraction import (
    ContradictionResolution,
    ExtractionResult,
    ExtractedContradiction,
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelation,
    EntityType,
    MemoryType,
)

# JSON Schema for structured LLM output
EXTRACTION_OUTPUT_SCHEMA: dict[str, Any] = {
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
    },
    "required": ["entities", "relations", "memories", "contradictions"],
}

SYSTEM_PROMPT = """\
你是一个知识图谱提取器。从对话中提取实体、关系、记忆和矛盾。

规则：
1. 实体类型只能是：person, org, location, concept, project, technology
2. 记忆类型只能是：fact, preference, status, inference
3. 矛盾解决方式只能是：update（替换旧值）, extend（时间区间扩展）, ignore（忽略新值）
4. 置信度范围 [0, 1]
5. 关系的 valid_from 使用 ISO 日期或年月格式（如 2024-01）
6. 只提取明确陈述的事实，不要推测
7. 如果对话是英文，用英文输出；如果中文，用中文输出
"""

SYSTEM_PROMPT_EN = """\
You are a knowledge graph extractor. Extract entities, relations, memories, and contradictions from conversations.

Rules:
1. Entity types must be one of: person, org, location, concept, project, technology
2. Memory types must be one of: fact, preference, status, inference
3. Contradiction resolutions must be one of: update (replace old), extend (time interval extension), ignore (keep old)
4. Confidence range: [0, 1]
5. Relation valid_from uses ISO date or year-month format (e.g. 2024-01)
6. Only extract explicitly stated facts, do not speculate
7. Output in the same language as the input
"""


class Extractor:
    """Layer 2: LLM-based structured extraction.

    Takes a single message or conversation and calls the LLM adapter's
    complete_structured method to produce an ExtractionResult.
    """

    def __init__(self, llm_adapter: Any) -> None:
        """Initialize with an LLM adapter that implements complete_structured.

        Args:
            llm_adapter: Any object with an async complete_structured method
                that accepts (messages, schema, temperature) and returns a dict.
        """
        self.llm_adapter = llm_adapter

    def _build_messages(
        self,
        text: str,
        entity_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the LLM message list for extraction.

        Args:
            text: The text to extract from.
            entity_context: Optional context hint (e.g. "这是张军的知识库").

        Returns:
            List of message dicts for the LLM.
        """
        user_content = f"请从以下内容中提取实体、关系、记忆和矛盾：\n\n{text}"
        if entity_context:
            user_content = f"上下文提示：{entity_context}\n\n{user_content}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_conversation_messages(
        self,
        messages: list[dict[str, str]],
        entity_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the LLM message list for multi-message conversation extraction.

        Args:
            messages: Conversation messages (list of dicts with role/content).
            entity_context: Optional context hint.

        Returns:
            List of message dicts for the LLM.
        """
        # Format the conversation as readable text
        conv_text = "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}"
            for m in messages
        )

        user_content = f"请从以下对话中提取实体、关系、记忆和矛盾：\n\n{conv_text}"
        if entity_context:
            user_content = f"上下文提示：{entity_context}\n\n{user_content}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_result(self, raw: dict) -> ExtractionResult:
        """Parse the raw LLM JSON output into an ExtractionResult.

        Args:
            raw: The dict returned by the LLM adapter's complete_structured.

        Returns:
            ExtractionResult with typed entities, relations, memories, contradictions.
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

        return ExtractionResult(
            entities=entities,
            relations=relations,
            memories=memories,
            contradictions=contradictions,
        )

    async def extract(
        self,
        text: str,
        entity_context: str | None = None,
    ) -> ExtractionResult:
        """Extract entities, relations, memories, and contradictions from a single text.

        Args:
            text: The text to extract from.
            entity_context: Optional context hint to guide extraction.

        Returns:
            ExtractionResult with all extracted items.
        """
        messages = self._build_messages(text, entity_context=entity_context)
        raw = await self.llm_adapter.complete_structured(
            messages=messages,
            schema=EXTRACTION_OUTPUT_SCHEMA,
            temperature=0.1,
        )
        return self._parse_result(raw)

    async def extract_conversation(
        self,
        messages: list[dict[str, str]],
        entity_context: str | None = None,
    ) -> ExtractionResult:
        """Extract from a multi-message conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            entity_context: Optional context hint.

        Returns:
            ExtractionResult with all extracted items.
        """
        llm_messages = self._build_conversation_messages(
            messages, entity_context=entity_context,
        )
        raw = await self.llm_adapter.complete_structured(
            messages=llm_messages,
            schema=EXTRACTION_OUTPUT_SCHEMA,
            temperature=0.1,
        )
        return self._parse_result(raw)