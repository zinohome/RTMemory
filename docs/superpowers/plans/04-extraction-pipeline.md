# RTMemory Extraction Pipeline — 三层提取流水线

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-layer extraction pipeline and document processing pipeline for ingesting text/PDF/webpages.

**Architecture:** Layer 1 (regex rules, zero cost) → Layer 2 (LLM structured extraction) → Layer 3 (deep scan at conversation end). Document: extract → chunk → embed → graph-extract → index. Async via FastAPI BackgroundTasks.

**Tech Stack:** Python 3.12, FastAPI, PyMuPDF, trafilatura

**Depends on:** Plan 02 (LLM adapter, embedding service), Plan 03 (graph engine)

---

## File Map

```
server/
  app/
    extraction/
      __init__.py
      fact_detector.py        # Layer 1: regex + rule-based filtering
      extractor.py            # Layer 2: LLM structured extraction
      deep_scanner.py          # Layer 3: batch deep scan at conversation end
      document_processor.py   # Document: extract → chunk → embed → graph-extract → index
    api/
      conversations.py        # /v1/conversations/ routes
      documents.py             # /v1/documents/ routes
    schemas/
      extraction.py            # Pydantic schemas for extraction pipeline
  tests/
    extraction/
      __init__.py
      test_fact_detector.py
      test_extractor.py
      test_deep_scanner.py
      test_document_processor.py
      test_api_conversations.py
      test_api_documents.py
```

---

## Phase 1: FactDetector — Layer 1 Regex Filtering

### Step 1: Create extraction package skeleton

- [ ] **Create directories and init files**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
mkdir -p server/app/extraction
mkdir -p server/tests/extraction
touch server/app/extraction/__init__.py
touch server/tests/__init__.py
touch server/tests/extraction/__init__.py
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && mkdir -p server/app/extraction server/tests/extraction && touch server/app/extraction/__init__.py server/tests/__init__.py server/tests/extraction/__init__.py
```

**Expected:** Directories and `__init__.py` files created.

---

### Step 2: Write FactDetector test (TDD — failing first)

- [ ] **Write** `server/tests/extraction/test_fact_detector.py`

```python
"""Tests for FactDetector — Layer 1 regex-based fact detection."""

import pytest

from app.extraction.fact_detector import FactDetector


@pytest.fixture
def detector():
    return FactDetector()


# ── Chinese patterns ────────────────────────────────────────────

class TestChinesePatterns:
    """Chinese regex patterns for fact detection."""

    def test_chinese_identity_is(self, detector):
        """'我是' pattern — identity statement."""
        assert detector.should_extract("我是张军") is True

    def test_chinese_identity_at(self, detector):
        """'我在' pattern — location/status."""
        assert detector.should_extract("我在北京工作") is True

    def test_chinese_have(self, detector):
        """'我有' pattern — possession."""
        assert detector.should_extract("我有一只猫") is True

    def test_chinese_use(self, detector):
        """'我用' pattern — tool/preference."""
        assert detector.should_extract("我用Python写代码") is True

    def test_chinese_like(self, detector):
        """'我喜欢' pattern — preference."""
        assert detector.should_extract("我喜欢TypeScript") is True

    def test_chinese_prefer(self, detector):
        """'我偏好' pattern — preference."""
        assert detector.should_extract("我偏好简洁的设计") is True

    def test_chinese_moved(self, detector):
        """'我搬到' pattern — location change."""
        assert detector.should_extract("我搬到北京了") is True

    def test_chinese_switched(self, detector):
        """'我换' pattern — change."""
        assert detector.should_extract("我换了一个新手机") is True

    def test_chinese_changed(self, detector):
        """'我改' pattern — change."""
        assert detector.should_extract("我改用VS Code了") is True

    def test_chinese_we_use(self, detector):
        """'我们用' pattern — group decision."""
        assert detector.should_extract("我们用FastAPI做后端") is True

    def test_chinese_we_chose(self, detector):
        """'我们选' pattern — group decision."""
        assert detector.should_extract("我们选了React作为前端框架") is True

    def test_chinese_we_decided(self, detector):
        """'我们决定' pattern — group decision."""
        assert detector.should_extract("我们决定迁移到Kubernetes") is True

    def test_chinese_we_planned(self, detector):
        """'我们计划' pattern — plan."""
        assert detector.should_extract("我们计划下周发布新版本") is True

    def test_chinese_recommend_keyword(self, detector):
        """'推荐' keyword."""
        assert detector.should_extract("有没有推荐的IDE？") is True

    def test_chinese_suggest_keyword(self, detector):
        """'建议' keyword."""
        assert detector.should_extract("我建议用Docker部署") is True

    def test_chinese_preference_keyword(self, detector):
        """'偏好' keyword."""
        assert detector.should_extract("用户的偏好是暗色主题") is True

    def test_chinese_habit_keyword(self, detector):
        """'习惯' keyword."""
        assert detector.should_extract("我的习惯是早上写代码") is True


# ── English patterns ────────────────────────────────────────────

class TestEnglishPatterns:
    """English regex patterns for fact detection."""

    def test_english_i_am(self, detector):
        assert detector.should_extract("I am a software engineer") is True

    def test_english_i_work(self, detector):
        assert detector.should_extract("I work at Google") is True

    def test_english_i_live(self, detector):
        assert detector.should_extract("I live in Tokyo") is True

    def test_english_i_use(self, detector):
        assert detector.should_extract("I use VS Code for development") is True

    def test_english_i_like(self, detector):
        assert detector.should_extract("I like Python") is True

    def test_english_i_prefer(self, detector):
        assert detector.should_extract("I prefer dark mode") is True

    def test_english_i_moved(self, detector):
        assert detector.should_extract("I moved to Berlin") is True

    def test_english_i_switched(self, detector):
        assert detector.should_extract("I switched to Vim") is True

    def test_english_i_changed(self, detector):
        assert detector.should_extract("I changed my editor") is True

    def test_english_we_use(self, detector):
        assert detector.should_extract("We use Kubernetes") is True

    def test_english_we_chose(self, detector):
        assert detector.should_extract("We chose PostgreSQL") is True

    def test_english_we_decided(self, detector):
        assert detector.should_extract("We decided to migrate") is True

    def test_english_we_plan(self, detector):
        assert detector.should_extract("We plan to launch next week") is True

    def test_english_recommend_keyword(self, detector):
        assert detector.should_extract("Any recommendations for a good IDE?") is True

    def test_english_suggest_keyword(self, detector):
        assert detector.should_extract("I suggest using Docker") is True

    def test_english_preference_keyword(self, detector):
        assert detector.should_extract("My preference is light theme") is True

    def test_english_habit_keyword(self, detector):
        assert detector.should_extract("My habit is to code at night") is True

    def test_english_i_have(self, detector):
        assert detector.should_extract("I have a cat named Whiskers") is True

    def test_english_my_name_is(self, detector):
        assert detector.should_extract("My name is Alice") is True

    def test_english_my_job_is(self, detector):
        assert detector.should_extract("My job is frontend development") is True


# ── Negative cases — casual chat should be filtered ─────────────

class TestNegativeCases:
    """Messages that should NOT trigger extraction."""

    def test_simple_greeting(self, detector):
        assert detector.should_extract("你好") is False

    def test_english_greeting(self, detector):
        assert detector.should_extract("Hello") is False

    def test_casual_thanks(self, detector):
        assert detector.should_extract("谢谢") is False

    def test_english_thanks(self, detector):
        assert detector.should_extract("Thanks!") is False

    def test_acknowledgment(self, detector):
        assert detector.should_extract("好的，知道了") is False

    def test_english_ok(self, detector):
        assert detector.should_extract("OK, got it") is False

    def test_question_without_fact(self, detector):
        assert detector.should_extract("今天天气怎么样？") is False

    def test_english_weather_question(self, detector):
        assert detector.should_extract("How's the weather?") is False

    def test_empty_string(self, detector):
        assert detector.should_extract("") is False

    def test_whitespace_only(self, detector):
        assert detector.should_extract("   ") is False

    def test_emoji_only(self, detector):
        assert detector.should_extract("👍") is False


# ── Context-aware boosting ───────────────────────────────────────

class TestContextBoost:
    """Context list can boost detection when message is ambiguous."""

    def test_no_context_pure_question(self, detector):
        assert detector.should_extract("这个怎么用？") is False

    def test_context_boosts_ambiguous(self, detector):
        """When recent context contains fact-like statements,
        even ambiguous follow-ups should be extracted."""
        ctx = ["我最近在学Rust", "Rust的借用检查器挺难的"]
        assert detector.should_extract("这个怎么用？", context=ctx) is True

    def test_context_empty_list_no_boost(self, detector):
        assert detector.should_extract("这个怎么用？", context=[]) is False

    def test_context_non_fact_no_boost(self, detector):
        ctx = ["你好", "谢谢"]
        assert detector.should_extract("这个怎么用？", context=[]) is False


# ── Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_mixed_chinese_english(self, detector):
        """Mixed language message should still match."""
        assert detector.should_extract("我用Next.js做前端") is True

    def test_long_message_with_fact(self, detector):
        """Long message containing a fact pattern."""
        msg = "今天开了三个会，不过我还是决定用Go重写那个服务"
        assert detector.should_extract(msg) is True

    def test_fact_at_end_of_message(self, detector):
        assert detector.should_extract("天气不错，我搬到了深圳") is True
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_fact_detector.py -v 2>&1 | tail -5
```

**Expected:** All tests FAIL (module not found).

---

### Step 3: Implement FactDetector

- [ ] **Write** `server/app/extraction/fact_detector.py`

```python
"""FactDetector — Layer 1: lightweight regex + rule-based filtering.

Zero-cost pre-filter that catches 70-80% of factual statements without
calling an LLM. Supports Chinese and English patterns.
"""

from __future__ import annotations

import re


class FactDetector:
    """Determines whether a message likely contains new facts, preferences,
    or status changes worth extracting into the knowledge graph.

    Uses compiled regex rules for both Chinese and English. Optional
    context-aware boosting when recent conversation history contains
    fact-like patterns.
    """

    # ── Chinese patterns ──────────────────────────────────────
    _ZH_PATTERNS: list[str] = [
        r"我(?:是|在|有|用|喜欢|偏好|搬到|换了?|改)",
        r"我们(?:用|选|决定|计划|换|改)",
        r"(?:推荐|建议|偏好|习惯)",
    ]

    # ── English patterns ─────────────────────────────────────
    _EN_PATTERNS: list[str] = [
        r"(?:I|i)\s+(?:am|work|live|use|like|prefer|moved|switched|changed|have)",
        r"(?:[Ww]e)\s+(?:use|chose|decided|plan|switched|changed)",
        r"(?:[Mm]y)\s+(?:name|job|role|team|company|location)\s+is",
        r"(?:recommend|suggest|preference|habit)",
    ]

    def __init__(self) -> None:
        # Compile all patterns for performance
        self._zh_rules: list[re.Pattern[str]] = [
            re.compile(p) for p in self._ZH_PATTERNS
        ]
        self._en_rules: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self._EN_PATTERNS
        ]

        # Pattern to detect fact-like statements in context
        self._context_fact_re: re.Pattern[str] = re.compile(
            r"我(?:是|在|有|用|喜欢|偏好|搬到|换了?|改)"
            r"|(?:I|i)\s+(?:am|work|live|use|like|prefer|moved|switched|have)"
            r"|(?:推荐|建议|偏好|习惯)"
            r"|(?:recommend|suggest|preference|habit)"
        )

    def _match_rules(self, message: str) -> bool:
        """Check message against all compiled regex rules."""
        for rule in self._zh_rules:
            if rule.search(message):
                return True
        for rule in self._en_rules:
            if rule.search(message):
                return True
        return False

    def should_extract(
        self,
        message: str,
        context: list[str] | None = None,
    ) -> bool:
        """Determine if the message should be sent to Layer 2 extraction.

        Args:
            message: The user message to evaluate.
            context: Optional list of recent conversation messages.
                When provided and the message is ambiguous (e.g. a follow-up
                question), if any context message matches a fact pattern,
                this message is boosted to be extracted too.

        Returns:
            True if the message likely contains extractable facts.
        """
        if not message or not message.strip():
            return False

        # Direct rule match — always extract
        if self._match_rules(message):
            return True

        # Context-aware boost: if recent messages contain facts,
        # follow-up questions may refer to them
        if context:
            for ctx_msg in context:
                if self._context_fact_re.search(ctx_msg):
                    return True

        return False
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_fact_detector.py -v 2>&1 | tail -20
```

**Expected:**

```
tests/extraction/test_fact_detector.py::TestChinesePatterns::test_chinese_identity_is PASSED
...
tests/extraction/test_fact_detector.py::TestNegativeCases::test_emoji_only PASSED
...
42 passed
```

---

### Step 4: Commit FactDetector

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/extraction/__init__.py server/app/extraction/fact_detector.py server/tests/extraction/__init__.py server/tests/extraction/test_fact_detector.py
git commit -m "feat(extraction): add FactDetector — Layer 1 regex+rule fact filtering (zh+en)"
```

---

## Phase 2: Extraction Schemas

### Step 5: Create Pydantic schemas for extraction pipeline

- [ ] **Write** `server/app/schemas/extraction.py`

```python
"""Pydantic schemas for the extraction pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Extraction result types ──────────────────────────────────────

class EntityType(str, Enum):
    person = "person"
    org = "org"
    location = "location"
    concept = "concept"
    project = "project"
    technology = "technology"


class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class ContradictionResolution(str, Enum):
    update = "update"
    extend = "extend"
    ignore = "ignore"


# ── Layer 2 extraction output ────────────────────────────────────

class ExtractedEntity(BaseModel):
    """A single entity extracted from conversation."""
    name: str = Field(..., min_length=1, max_length=500)
    type: EntityType = Field(default=EntityType.person)
    description: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    """A single relation extracted from conversation."""
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    value: str = Field(default="")
    valid_from: Optional[str] = Field(default=None, description="ISO date or partial like 2024-01")
    valid_to: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedMemory(BaseModel):
    """A single memory extracted from conversation."""
    content: str = Field(..., min_length=1)
    type: MemoryType = Field(default=MemoryType.fact)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    entity_name: Optional[str] = Field(default=None, description="Name of the primary entity this memory is about")


class ExtractedContradiction(BaseModel):
    """A detected contradiction between new and existing knowledge."""
    new: str = Field(..., min_length=1, description="New relation or fact, e.g. lives_in(Beijing)")
    old: str = Field(..., min_length=1, description="Old relation or fact, e.g. lives_in(Shanghai)")
    resolution: ContradictionResolution = Field(default=ContradictionResolution.update)


class ExtractionResult(BaseModel):
    """Complete structured output from Layer 2 extraction."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    memories: list[ExtractedMemory] = Field(default_factory=list)
    contradictions: list[ExtractedContradiction] = Field(default_factory=list)


# ── Deep scan result (Layer 3) ──────────────────────────────────

class DeepScanResult(BaseModel):
    """Result from Layer 3 deep scan — richer than single extraction."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    memories: list[ExtractedMemory] = Field(default_factory=list)
    contradictions: list[ExtractedContradiction] = Field(default_factory=list)
    confidence_adjustments: list[ConfidenceAdjustment] = Field(default_factory=list)


class ConfidenceAdjustment(BaseModel):
    """An adjustment to confidence of an existing memory/relation."""
    target_type: str = Field(..., description="memory or relation")
    target_id: uuid.UUID = Field(...)
    old_confidence: float = Field(...)
    new_confidence: float = Field(...)
    reason: str = Field(default="")


# ── Conversation API schemas ────────────────────────────────────

class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class ConversationSubmitRequest(BaseModel):
    """Request body for POST /v1/conversations/."""
    messages: list[ConversationMessage] = Field(..., min_length=1)
    space_id: uuid.UUID
    user_id: Optional[str] = Field(default=None)
    entity_context: Optional[str] = Field(default=None, description="Context hint to guide extraction")
    metadata: Optional[dict] = Field(default=None)


class ConversationSubmitResponse(BaseModel):
    """Response for conversation submission."""
    conversation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    extracted: ExtractionResult = Field(default_factory=ExtractionResult)
    skipped: bool = Field(default=False, description="True if FactDetector filtered all messages")
    message_count: int = Field(...)


class ConversationEndRequest(BaseModel):
    """Request body for POST /v1/conversations/end."""
    conversation_id: uuid.UUID
    space_id: uuid.UUID
    user_id: Optional[str] = Field(default=None)


class ConversationEndResponse(BaseModel):
    """Response for conversation end (deep scan)."""
    conversation_id: uuid.UUID
    deep_scan_result: DeepScanResult = Field(default_factory=DeepScanResult)
    message_count: int = Field(default=0)


# ── Document API schemas ────────────────────────────────────────

class DocumentType(str, Enum):
    text = "text"
    pdf = "pdf"
    webpage = "webpage"


class DocumentStatus(str, Enum):
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    done = "done"
    failed = "failed"


class DocumentCreateRequest(BaseModel):
    """Request body for POST /v1/documents/."""
    title: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None, description="Raw text content (for doc_type=text)")
    url: Optional[str] = Field(default=None, description="URL to fetch (for doc_type=webpage)")
    doc_type: DocumentType = Field(default=DocumentType.text)
    space_id: uuid.UUID
    metadata: Optional[dict] = Field(default=None)


class DocumentUploadResponse(BaseModel):
    """Response for document upload/creation."""
    id: uuid.UUID
    title: Optional[str] = None
    doc_type: DocumentType
    status: DocumentStatus = Field(default=DocumentStatus.queued)
    space_id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentOut(BaseModel):
    """Full document output."""
    id: uuid.UUID
    title: Optional[str] = None
    doc_type: DocumentType
    url: Optional[str] = None
    status: DocumentStatus
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    space_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated document list."""
    items: list[DocumentOut] = Field(default_factory=list)
    total: int = Field(default=0)
    offset: int = Field(default=0)
    limit: int = Field(default=20)
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from app.schemas.extraction import ExtractionResult, DeepScanResult, ConversationSubmitRequest, DocumentCreateRequest; print('schemas OK')"
```

**Expected:** `schemas OK`

---

### Step 6: Commit extraction schemas

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/schemas/extraction.py
git commit -m "feat(extraction): add Pydantic schemas for extraction pipeline, conversations, and documents"
```

---

## Phase 3: Extractor — Layer 2 LLM Structured Extraction

### Step 7: Write Extractor test (TDD — failing first)

- [ ] **Write** `server/tests/extraction/test_extractor.py`

```python
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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_extractor.py -v 2>&1 | tail -5
```

**Expected:** All tests FAIL (module not found).

---

### Step 8: Implement Extractor

- [ ] **Write** `server/app/extraction/extractor.py`

```python
"""Extractor — Layer 2: LLM structured extraction.

Extracts entities, relations, memories, and contradictions from
conversation messages using the LLM adapter's complete_structured method.
"""

from __future__ import annotations

import json
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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_extractor.py -v 2>&1 | tail -20
```

**Expected:**

```
tests/extraction/test_extractor.py::TestExtractorBasic::test_extract_returns_extraction_result PASSED
...
tests/extraction/test_extractor.py::TestExtractorConversation::test_extract_conversation_calls_llm_once PASSED
11 passed
```

---

### Step 9: Commit Extractor

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/extraction/extractor.py server/tests/extraction/test_extractor.py
git commit -m "feat(extraction): add Extractor — Layer 2 LLM structured extraction with mock tests"
```

---

## Phase 4: DeepScanner — Layer 3 Batch Deep Scan

### Step 10: Write DeepScanner test (TDD — failing first)

- [ ] **Write** `server/tests/extraction/test_deep_scanner.py`

```python
"""Tests for DeepScanner — Layer 3 deep scan at conversation end."""

import pytest

from app.extraction.deep_scanner import DeepScanner
from app.schemas.extraction import DeepScanResult


class MockLLMAdapter:
    """Mock LLM adapter for deep scan testing."""

    def __init__(self, response: dict | None = None):
        self._response = response or self._default_response()
        self.call_count = 0

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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_deep_scanner.py -v 2>&1 | tail -5
```

**Expected:** All tests FAIL (module not found).

---

### Step 11: Implement DeepScanner

- [ ] **Write** `server/app/extraction/deep_scanner.py`

```python
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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_deep_scanner.py -v 2>&1 | tail -20
```

**Expected:**

```
tests/extraction/test_deep_scanner.py::TestDeepScannerBasic::test_deep_scan_returns_result PASSED
...
tests/extraction/test_deep_scanner.py::TestDeepScannerThreshold::test_deep_scan_triggers_at_threshold PASSED
9 passed
```

---

### Step 12: Commit DeepScanner

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/extraction/deep_scanner.py server/tests/extraction/test_deep_scanner.py
git commit -m "feat(extraction): add DeepScanner — Layer 3 batch deep scan with confidence adjustments"
```

---

## Phase 5: DocumentProcessor — Document Extraction Pipeline

### Step 13: Write DocumentProcessor test (TDD — failing first)

- [ ] **Write** `server/tests/extraction/test_document_processor.py`

```python
"""Tests for DocumentProcessor — text/PDF/webpage → chunk → embed → graph-extract → index."""

import pytest

from app.extraction.document_processor import DocumentProcessor, Chunker
from app.schemas.extraction import DocumentType


# ── Mock LLM adapter ─────────────────────────────────────────────

class MockLLMAdapter:
    """Mock LLM adapter for document extraction."""

    def __init__(self):
        self.call_count = 0

    async def complete_structured(self, messages, schema, temperature=0.1):
        self.call_count += 1
        return {
            "entities": [
                {"name": "FastAPI", "type": "technology", "description": "Web framework", "confidence": 0.9},
            ],
            "relations": [],
            "memories": [
                {"content": "FastAPI is a modern web framework", "type": "fact", "confidence": 0.8, "entity_name": "FastAPI"},
            ],
            "contradictions": [],
        }


# ── Mock embedding service ───────────────────────────────────────

class MockEmbeddingService:
    """Mock embedding service that returns fixed-dimension vectors."""

    def __init__(self, dim: int = 1536):
        self._dim = dim
        self.call_count = 0

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        return [0.01] * self._dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [[0.01] * self._dim for _ in texts]


# ── Chunker tests ────────────────────────────────────────────────

class TestChunker:
    """Tests for text chunking by paragraphs."""

    @pytest.fixture
    def chunker(self):
        return Chunker(max_chunk_size=200, overlap_size=50)

    def test_chunk_single_paragraph(self, chunker):
        text = "Hello world."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].position == 0

    def test_chunk_multiple_paragraphs(self, chunker):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        assert len(chunks) == 3
        assert chunks[0].position == 0
        assert chunks[1].position == 1
        assert chunks[2].position == 2

    def test_chunk_long_paragraph_split(self, chunker):
        """A paragraph exceeding max_chunk_size should be split."""
        long_para = " ".join(["word"] * 300)  # ~1800 chars
        chunks = chunker.chunk(long_para)
        assert len(chunks) >= 2
        # Each chunk should respect max_chunk_size (with overlap tolerance)
        for c in chunks:
            assert len(c.content) <= chunker.max_chunk_size + chunker.overlap_size

    def test_chunk_empty_text(self, chunker):
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_chunk_whitespace_only(self, chunker):
        chunks = chunker.chunk("   \n\n  \n  ")
        assert len(chunks) == 0

    def test_chunk_preserves_positions(self, chunker):
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
        chunks = chunker.chunk(text)
        for i, c in enumerate(chunks):
            assert c.position == i

    def test_chunk_single_line_breaks_not_split(self, chunker):
        """Single newlines should not cause a split — only double newlines (paragraphs)."""
        text = "Line one\nLine two\nLine three"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1


# ── DocumentProcessor tests ────────────────────────────────────

class TestDocumentProcessorBasic:
    """Basic document processor tests."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(
            llm_adapter=MockLLMAdapter(),
            embedding_service=MockEmbeddingService(),
        )

    async def test_process_text_document(self, processor):
        result = await processor.process(
            content="FastAPI is a modern web framework for Python.",
            doc_type=DocumentType.text,
            title="FastAPI Intro",
        )
        assert result.status == "done"
        assert result.title == "FastAPI Intro"
        assert len(result.chunks) >= 1
        assert len(result.entities) >= 1

    async def test_process_text_document_has_chunks(self, processor):
        result = await processor.process(
            content="First para.\n\nSecond para.\n\nThird para.",
            doc_type=DocumentType.text,
        )
        assert len(result.chunks) == 3

    async def test_process_text_document_has_embeddings(self, processor):
        result = await processor.process(
            content="Some content here.",
            doc_type=DocumentType.text,
        )
        # Each chunk should have an embedding
        for chunk in result.chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1536

    async def test_process_text_document_summary_embedding(self, processor):
        result = await processor.process(
            content="FastAPI is great.",
            doc_type=DocumentType.text,
        )
        assert result.summary_embedding is not None
        assert len(result.summary_embedding) == 1536


class TestDocumentProcessorPDF:
    """PDF processing tests (mocked)."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(
            llm_adapter=MockLLMAdapter(),
            embedding_service=MockEmbeddingService(),
        )

    async def test_process_pdf_from_bytes(self, processor):
        """Test PDF processing with mock PDF bytes."""
        # We mock the _extract_pdf method for testing
        result = await processor.process(
            content="PDF text extracted: FastAPI is great.",
            doc_type=DocumentType.pdf,
            title="Test PDF",
        )
        assert result.status == "done"


class TestDocumentProcessorWebpage:
    """Webpage processing tests (mocked)."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(
            llm_adapter=MockLLMAdapter(),
            embedding_service=MockEmbeddingService(),
        )

    async def test_process_webpage_from_text(self, processor):
        """Test webpage processing with pre-fetched text."""
        result = await processor.process(
            content="Webpage content about Python programming.",
            doc_type=DocumentType.webpage,
            title="Python Docs",
        )
        assert result.status == "done"


class TestDocumentProcessorStatus:
    """Document processing status tracking."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(
            llm_adapter=MockLLMAdapter(),
            embedding_service=MockEmbeddingService(),
        )

    async def test_status_progresses(self, processor):
        result = await processor.process(
            content="Some text content.",
            doc_type=DocumentType.text,
        )
        # Final status should be "done"
        assert result.status == "done"

    async def test_failed_status_on_error(self):
        """If extraction fails, status should be 'failed'."""

        class FailLLM:
            async def complete_structured(self, messages, schema, temperature=0.1):
                raise RuntimeError("LLM unavailable")

        processor = DocumentProcessor(
            llm_adapter=FailLLM(),
            embedding_service=MockEmbeddingService(),
        )
        result = await processor.process(
            content="Some text.",
            doc_type=DocumentType.text,
        )
        assert result.status == "failed"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_document_processor.py -v 2>&1 | tail -5
```

**Expected:** All tests FAIL (module not found).

---

### Step 14: Implement DocumentProcessor

- [ ] **Write** `server/app/extraction/document_processor.py`

```python
"""DocumentProcessor — text/PDF/webpage → chunk → embed → graph-extract → index.

Pipeline:
1. Extract text from source (PDF via PyMuPDF, webpage via trafilatura, text passthrough)
2. Chunk text by paragraphs with overlap
3. Generate embeddings for each chunk + document summary
4. Extract entities and relations from document content via LLM
5. Return structured result for indexing into DB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas.extraction import (
    ContradictionResolution,
    DocumentType,
    EntityType,
    ExtractedContradiction,
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelation,
    ExtractionResult,
    MemoryType,
)

logger = logging.getLogger(__name__)

# Reuse the same extraction schema from Extractor
from app.extraction.extractor import EXTRACTION_OUTPUT_SCHEMA


@dataclass
class ChunkResult:
    """A single text chunk with its embedding."""
    content: str
    position: int
    embedding: list[float] | None = None


@dataclass
class DocumentProcessResult:
    """Complete result from document processing pipeline."""
    title: str | None = None
    doc_type: DocumentType = DocumentType.text
    status: str = "queued"  # queued → extracting → chunking → embedding → done/failed
    chunks: list[ChunkResult] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    memories: list[ExtractedMemory] = field(default_factory=list)
    summary: str = ""
    summary_embedding: list[float] | None = None
    raw_text: str = ""
    error: str | None = None


class Chunker:
    """Splits text into chunks by paragraphs with optional overlap.

    Paragraph boundaries are detected by double newlines (\\n\\n).
    Single newlines are treated as line breaks within a paragraph.
    Paragraphs exceeding max_chunk_size are further split by sentences
    or by character boundary with overlap.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        overlap_size: int = 200,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Split a paragraph that exceeds max_chunk_size.

        Strategy: split by sentence boundaries (. ! ?), then by character
        boundary if sentences are too long.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        # Split by sentence-ending punctuation
        import re
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current) + len(sentence) + 1 <= self.max_chunk_size:
                current = f"{current} {sentence}".strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                # If a single sentence is too long, split by character boundary
                if len(sentence) > self.max_chunk_size:
                    start = 0
                    while start < len(sentence):
                        end = min(start + self.max_chunk_size, len(sentence))
                        chunks.append(sentence[start:end])
                        start = end - self.overlap_size
                        if start >= len(sentence) - self.overlap_size:
                            break
                else:
                    current = sentence
        if current:
            chunks.append(current)

        return chunks

    def chunk(self, text: str) -> list[ChunkResult]:
        """Split text into chunks by paragraph boundaries.

        Args:
            text: The raw text to chunk.

        Returns:
            List of ChunkResult with content and position.
        """
        if not text or not text.strip():
            return []

        # Split by double newline (paragraph boundaries)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        results: list[ChunkResult] = []
        position = 0

        for para in paragraphs:
            sub_chunks = self._split_long_paragraph(para)
            for sub in sub_chunks:
                results.append(ChunkResult(content=sub, position=position))
                position += 1

        return results


class DocumentProcessor:
    """Full document processing pipeline.

    Supports text (passthrough), PDF (PyMuPDF), and webpage (trafilatura).
    Processes through: extract → chunk → embed → graph-extract → result.
    """

    def __init__(
        self,
        llm_adapter: Any,
        embedding_service: Any,
        chunker: Chunker | None = None,
    ) -> None:
        """Initialize the document processor.

        Args:
            llm_adapter: LLM adapter with async complete_structured method.
            embedding_service: Embedding service with async embed(text) and
                embed_batch(texts) methods returning list[float].
            chunker: Optional Chunker instance. Defaults to Chunker().
        """
        self.llm_adapter = llm_adapter
        self.embedding_service = embedding_service
        self.chunker = chunker or Chunker()

    async def _extract_text(
        self,
        content: str | bytes | None,
        doc_type: DocumentType,
        url: str | None = None,
    ) -> str:
        """Extract raw text from the document source.

        Args:
            content: Raw text content (for text type), bytes (for PDF), or None.
            doc_type: Type of document.
            url: URL for webpage type.

        Returns:
            Extracted plain text.
        """
        if doc_type == DocumentType.text:
            return content or ""

        elif doc_type == DocumentType.pdf:
            return self._extract_pdf(content)

        elif doc_type == DocumentType.webpage:
            if content:
                return content
            if url:
                return self._extract_webpage(url)
            return ""

        return content or ""

    def _extract_pdf(self, content: str | bytes) -> str:
        """Extract text from PDF using PyMuPDF.

        Args:
            content: PDF bytes or already-extracted text (for testing).

        Returns:
            Extracted text from all pages.
        """
        # If content is a string, it's pre-extracted text (useful for testing)
        if isinstance(content, str):
            return content

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=content, filetype="pdf")
            text_parts: list[str] = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed, returning raw content")
            return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    def _extract_webpage(self, url: str) -> str:
        """Extract text from a webpage URL using trafilatura.

        Args:
            url: The URL to fetch and extract text from.

        Returns:
            Extracted plain text from the webpage.
        """
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError(f"Failed to fetch URL: {url}")
            result = trafilatura.extract(downloaded)
            if not result:
                raise ValueError(f"Failed to extract text from URL: {url}")
            return result
        except ImportError:
            logger.warning("trafilatura not installed, cannot extract webpage")
            raise ValueError("trafilatura is required for webpage extraction")
        except Exception as e:
            logger.error(f"Webpage extraction failed for {url}: {e}")
            raise

    async def _embed_chunks(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
        """Generate embeddings for all chunks.

        Args:
            chunks: List of ChunkResult without embeddings.

        Returns:
            Same chunks with embeddings populated.
        """
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_service.embed_batch(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    async def _generate_summary_embedding(self, summary: str) -> list[float] | None:
        """Generate embedding for the document summary.

        Args:
            summary: Document summary text.

        Returns:
            Embedding vector or None if summary is empty.
        """
        if not summary:
            return None
        return await self.embedding_service.embed(summary)

    async def _extract_graph(self, text: str) -> ExtractionResult:
        """Extract entities and relations from document text via LLM.

        Args:
            text: The full document text.

        Returns:
            ExtractionResult with entities, relations, memories, contradictions.
        """
        from app.extraction.extractor import Extractor

        extractor = Extractor(llm_adapter=self.llm_adapter)
        return await extractor.extract(text)

    def _generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a simple extractive summary (first N chars).

        For production, this would use LLM summarization. For now,
        use the first max_length characters as a summary.

        Args:
            text: Full document text.
            max_length: Maximum summary length in characters.

        Returns:
            Summary string.
        """
        if len(text) <= max_length:
            return text
        # Truncate at the last sentence boundary within max_length
        truncated = text[:max_length]
        last_period = max(
            truncated.rfind("."),
            truncated.rfind("。"),
            truncated.rfind("!"),
            truncated.rfind("?"),
        )
        if last_period > max_length // 2:
            return truncated[: last_period + 1]
        return truncated

    async def process(
        self,
        content: str | bytes | None = None,
        doc_type: DocumentType = DocumentType.text,
        title: str | None = None,
        url: str | None = None,
    ) -> DocumentProcessResult:
        """Process a document through the full pipeline.

        Pipeline: extract → chunk → embed → graph-extract → result.
        Status progresses: queued → extracting → chunking → embedding → done.
        On any failure, status is set to "failed" with error message.

        Args:
            content: Document content (text, PDF bytes, or pre-fetched text).
            doc_type: Type of document.
            title: Optional document title.
            url: URL for webpage documents.

        Returns:
            DocumentProcessResult with chunks, entities, embeddings, and status.
        """
        result = DocumentProcessResult(
            title=title,
            doc_type=doc_type,
        )

        try:
            # Step 1: Extract text
            result.status = "extracting"
            raw_text = await self._extract_text(content, doc_type, url=url)
            result.raw_text = raw_text

            if not raw_text.strip():
                result.status = "failed"
                result.error = "No text content extracted from document"
                return result

            # Step 2: Chunk
            result.status = "chunking"
            chunks = self.chunker.chunk(raw_text)
            result.chunks = chunks

            # Step 3: Embed chunks
            result.status = "embedding"
            result.chunks = await self._embed_chunks(chunks)

            # Step 4: Generate summary + summary embedding
            result.summary = self._generate_summary(raw_text)
            result.summary_embedding = await self._generate_summary_embedding(result.summary)

            # Step 5: Graph extraction from document
            extraction = await self._extract_graph(raw_text)
            result.entities = extraction.entities
            result.relations = extraction.relations
            result.memories = extraction.memories

            result.status = "done"
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            result.status = "failed"
            result.error = str(e)

        return result
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_document_processor.py -v 2>&1 | tail -20
```

**Expected:**

```
tests/extraction/test_document_processor.py::TestChunker::test_chunk_single_paragraph PASSED
...
tests/extraction/test_document_processor.py::TestDocumentProcessorStatus::test_failed_status_on_error PASSED
16 passed
```

---

### Step 15: Commit DocumentProcessor

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/extraction/document_processor.py server/tests/extraction/test_document_processor.py
git commit -m "feat(extraction): add DocumentProcessor — text/PDF/webpage → chunk → embed → graph-extract pipeline"
```

---

## Phase 6: Conversation API Routes

### Step 16: Write conversation API test (TDD — failing first)

- [ ] **Write** `server/tests/extraction/test_api_conversations.py`

```python
"""Tests for /v1/conversations/ API routes."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Mock extraction pipeline ────────────────────────────────────

class MockLLMAdapter:
    async def complete_structured(self, messages, schema, temperature=0.1):
        return {
            "entities": [
                {"name": "张军", "type": "person", "description": "工程师", "confidence": 0.9},
            ],
            "relations": [],
            "memories": [
                {"content": "张军搬到北京", "type": "fact", "confidence": 0.9, "entity_name": "张军"},
            ],
            "contradictions": [],
        }


class MockEmbeddingService:
    async def embed(self, text):
        return [0.01] * 1536

    async def embed_batch(self, texts):
        return [[0.01] * 1536 for _ in texts]


# ── Conversation submit ─────────────────────────────────────────

class TestConversationSubmit:
    """POST /v1/conversations/ — submit conversation messages."""

    def test_submit_conversation_returns_200(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "extracted" in data
        assert "skipped" in data
        assert "message_count" in data

    def test_submit_conversation_count(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message_count"] == 3

    def test_submit_empty_messages_returns_422(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [],
                "space_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert resp.status_code == 422

    def test_submit_skipped_when_no_facts(self, client):
        """Messages with no fact patterns should be skipped."""
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ],
                "space_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is True

    def test_submit_with_entity_context(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": "00000000-0000-0000-0000-000000000001",
                "entity_context": "这是张军的知识库",
            },
        )
        assert resp.status_code == 200


# ── Conversation end (deep scan) ────────────────────────────────

class TestConversationEnd:
    """POST /v1/conversations/end — trigger deep scan."""

    def test_conversation_end_returns_200(self, client):
        resp = client.post(
            "/v1/conversations/end",
            json={
                "conversation_id": "00000000-0000-0000-0000-000000000099",
                "space_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "deep_scan_result" in data
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_api_conversations.py -v 2>&1 | tail -5
```

**Expected:** All tests FAIL (module not found or 404).

---

### Step 17: Implement conversation API routes

- [ ] **Write** `server/app/api/conversations.py`

```python
"""API routes for /v1/conversations/ — conversation memory extraction.

POST /v1/conversations/     — Submit conversation messages, trigger Layer 1+2 extraction
POST /v1/conversations/end  — End conversation, trigger Layer 3 deep scan
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends

from app.extraction.fact_detector import FactDetector
from app.extraction.extractor import Extractor
from app.extraction.deep_scanner import DeepScanner
from app.schemas.extraction import (
    ConversationEndRequest,
    ConversationEndResponse,
    ConversationSubmitRequest,
    ConversationSubmitResponse,
    DeepScanResult,
    ExtractionResult,
)

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

# ── In-memory conversation store (to be replaced with DB in integration) ──

_conversations: dict[uuid.UUID, list[dict[str, str]]] = {}


def _get_fact_detector() -> FactDetector:
    """Dependency: get or create FactDetector singleton."""
    return FactDetector()


def _get_extractor() -> Extractor:
    """Dependency: get or create Extractor.

    In production, this would use the real LLM adapter from the app state.
    For now, create a placeholder that raises if not overridden.
    """
    from app.core.llm import create_llm_adapter
    from app.config import get_config
    try:
        config = get_config()
        llm_adapter = create_llm_adapter(config.llm)
        return Extractor(llm_adapter=llm_adapter)
    except Exception:
        # Fallback: will be injected via app.state in production
        raise RuntimeError(
            "LLM adapter not configured. Set LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL env vars."
        )


def _get_deep_scanner() -> DeepScanner:
    """Dependency: get or create DeepScanner."""
    from app.core.llm import create_llm_adapter
    from app.config import get_config
    try:
        config = get_config()
        llm_adapter = create_llm_adapter(config.llm)
        return DeepScanner(llm_adapter=llm_adapter, min_messages=3)
    except Exception:
        raise RuntimeError(
            "LLM adapter not configured. Set LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL env vars."
        )


@router.post("/", response_model=ConversationSubmitResponse)
async def submit_conversation(
    request: ConversationSubmitRequest,
    background_tasks: BackgroundTasks,
    detector: FactDetector = Depends(_get_fact_detector),
    extractor: Extractor = Depends(_get_extractor),
):
    """Submit conversation messages and trigger extraction pipeline.

    Layer 1 (FactDetector) filters messages that contain factual content.
    Layer 2 (Extractor) performs structured extraction on messages that pass.
    If no messages pass Layer 1, the response has skipped=True.
    """
    conversation_id = uuid.uuid4()

    # Store conversation messages for later deep scan
    message_dicts = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]
    _conversations[conversation_id] = message_dicts

    # Layer 1: Filter messages that contain facts
    context_messages = [m.content for m in request.messages]
    fact_messages = [
        m for m in request.messages
        if detector.should_extract(m.content, context=context_messages)
    ]

    # If no fact-like messages, skip extraction
    if not fact_messages:
        return ConversationSubmitResponse(
            conversation_id=conversation_id,
            extracted=ExtractionResult(),
            skipped=True,
            message_count=len(request.messages),
        )

    # Layer 2: Extract from fact-like messages
    fact_dicts = [{"role": m.role, "content": m.content} for m in fact_messages]
    extracted = await extractor.extract_conversation(
        fact_dicts,
        entity_context=request.entity_context,
    )

    return ConversationSubmitResponse(
        conversation_id=conversation_id,
        extracted=extracted,
        skipped=False,
        message_count=len(request.messages),
    )


@router.post("/end", response_model=ConversationEndResponse)
async def end_conversation(
    request: ConversationEndRequest,
    background_tasks: BackgroundTasks,
    deep_scanner: DeepScanner = Depends(_get_deep_scanner),
):
    """End a conversation and trigger Layer 3 deep scan.

    The deep scan processes the full conversation to capture implicit
    preferences, cross-message correlations, and confidence adjustments.
    """
    messages = _conversations.pop(request.conversation_id, [])

    if not messages:
        return ConversationEndResponse(
            conversation_id=request.conversation_id,
            deep_scan_result=DeepScanResult(),
            message_count=0,
        )

    # Layer 3: Deep scan the full conversation
    deep_scan_result = await deep_scanner.deep_scan(
        messages,
        entity_context=None,  # Could be stored from submit
    )

    return ConversationEndResponse(
        conversation_id=request.conversation_id,
        deep_scan_result=deep_scan_result,
        message_count=len(messages),
    )
```

Now we need to register the router in the FastAPI app. Let us also create a minimal `main.py` that can be tested.

- [ ] **Write** `server/app/main.py`

```python
"""RTMemory Server — FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router

app = FastAPI(
    title="RTMemory",
    version="0.1.0",
    description="Temporal Knowledge Graph-Driven AI Memory System",
)

# Register API routers
app.include_router(conversations_router)
app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Write** `server/app/api/__init__.py`

```python
# RTMemory API
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_api_conversations.py -v 2>&1 | tail -15
```

**Expected:** Some tests pass (basic structure), some may fail if LLM adapter is not configured. We need to inject mocks for testing.

The test file uses `TestClient` which requires the app dependencies to work. Let us update the test to use dependency overrides.

- [ ] **Update** `server/tests/extraction/test_api_conversations.py` — replace with version that uses dependency overrides

```python
"""Tests for /v1/conversations/ API routes."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.extraction.fact_detector import FactDetector
from app.extraction.extractor import Extractor
from app.extraction.deep_scanner import DeepScanner
from app.api.conversations import _get_fact_detector, _get_extractor, _get_deep_scanner


# ── Mock LLM adapter ─────────────────────────────────────────────

class MockLLMAdapter:
    async def complete_structured(self, messages, schema, temperature=0.1):
        return {
            "entities": [
                {"name": "张军", "type": "person", "description": "工程师", "confidence": 0.9},
            ],
            "relations": [],
            "memories": [
                {"content": "张军搬到北京", "type": "fact", "confidence": 0.9, "entity_name": "张军"},
            ],
            "contradictions": [],
        }


class MockDeepLLMAdapter:
    async def complete_structured(self, messages, schema, temperature=0.1):
        return {
            "entities": [],
            "relations": [],
            "memories": [],
            "contradictions": [],
            "confidence_adjustments": [],
        }


@pytest.fixture
def client():
    # Override dependencies with mocks
    app.dependency_overrides[_get_fact_detector] = lambda: FactDetector()
    app.dependency_overrides[_get_extractor] = lambda: Extractor(llm_adapter=MockLLMAdapter())
    app.dependency_overrides[_get_deep_scanner] = lambda: DeepScanner(
        llm_adapter=MockDeepLLMAdapter(), min_messages=1,
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


SAMPLE_SPACE_ID = "00000000-0000-0000-0000-000000000001"


# ── Conversation submit ─────────────────────────────────────────

class TestConversationSubmit:
    """POST /v1/conversations/ — submit conversation messages."""

    def test_submit_conversation_returns_200(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "extracted" in data
        assert "skipped" in data
        assert "message_count" in data

    def test_submit_conversation_count(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message_count"] == 3

    def test_submit_empty_messages_returns_422(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 422

    def test_submit_skipped_when_no_facts(self, client):
        """Messages with no fact patterns should be skipped."""
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is True

    def test_submit_with_entity_context(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": SAMPLE_SPACE_ID,
                "entity_context": "这是张军的知识库",
            },
        )
        assert resp.status_code == 200

    def test_submit_fact_message_not_skipped(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我用Python写代码"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is False

    def test_submit_extraction_has_entities(self, client):
        resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        data = resp.json()
        assert len(data["extracted"]["entities"]) >= 1


# ── Conversation end (deep scan) ────────────────────────────────

class TestConversationEnd:
    """POST /v1/conversations/end — trigger deep scan."""

    def test_conversation_end_returns_200(self, client):
        conv_id = str(uuid.uuid4())
        resp = client.post(
            "/v1/conversations/end",
            json={
                "conversation_id": conv_id,
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "deep_scan_result" in data

    def test_conversation_end_unknown_id_returns_empty_result(self, client):
        """Unknown conversation ID should return empty result, not error."""
        resp = client.post(
            "/v1/conversations/end",
            json={
                "conversation_id": str(uuid.uuid4()),
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message_count"] == 0

    def test_conversation_submit_then_end(self, client):
        """Submit a conversation, then end it — should find the stored messages."""
        # Submit first
        submit_resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        conv_id = submit_resp.json()["conversation_id"]

        # End it
        end_resp = client.post(
            "/v1/conversations/end",
            json={
                "conversation_id": conv_id,
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert end_resp.status_code == 200
        data = end_resp.json()
        assert data["message_count"] == 1
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_api_conversations.py -v 2>&1 | tail -20
```

**Expected:**

```
tests/extraction/test_api_conversations.py::TestConversationSubmit::test_submit_conversation_returns_200 PASSED
...
tests/extraction/test_api_conversations.py::TestConversationEnd::test_conversation_submit_then_end PASSED
10 passed
```

---

### Step 18: Commit conversation API

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/api/__init__.py server/app/api/conversations.py server/app/main.py server/tests/extraction/test_api_conversations.py
git commit -m "feat(api): add /v1/conversations/ routes with dependency-overridden mock tests"
```

---

## Phase 7: Document API Routes

### Step 19: Write document API test (TDD — failing first)

- [ ] **Write** `server/tests/extraction/test_api_documents.py`

```python
"""Tests for /v1/documents/ API routes."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.extraction.document_processor import DocumentProcessor
from app.api.documents import _get_document_processor


# ── Mock services ───────────────────────────────────────────────

class MockLLMAdapter:
    async def complete_structured(self, messages, schema, temperature=0.1):
        return {
            "entities": [
                {"name": "FastAPI", "type": "technology", "description": "Web framework", "confidence": 0.9},
            ],
            "relations": [],
            "memories": [
                {"content": "FastAPI is a web framework", "type": "fact", "confidence": 0.8, "entity_name": "FastAPI"},
            ],
            "contradictions": [],
        }


class MockEmbeddingService:
    async def embed(self, text):
        return [0.01] * 1536

    async def embed_batch(self, texts):
        return [[0.01] * 1536 for _ in texts]


@pytest.fixture
def client():
    processor = DocumentProcessor(
        llm_adapter=MockLLMAdapter(),
        embedding_service=MockEmbeddingService(),
    )
    app.dependency_overrides[_get_document_processor] = lambda: processor
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


SAMPLE_SPACE_ID = "00000000-0000-0000-0000-000000000001"


# ── Document create ────────────────────────────────────────────

class TestDocumentCreate:
    """POST /v1/documents/ — create a text document."""

    def test_create_text_document_returns_200(self, client):
        resp = client.post(
            "/v1/documents/",
            json={
                "content": "FastAPI is a modern web framework for Python.",
                "doc_type": "text",
                "space_id": SAMPLE_SPACE_ID,
                "title": "FastAPI Intro",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["title"] == "FastAPI Intro"
        assert "id" in data

    def test_create_text_document_default_type(self, client):
        resp = client.post(
            "/v1/documents/",
            json={
                "content": "Some text content.",
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_type"] == "text"

    def test_create_webpage_document(self, client):
        resp = client.post(
            "/v1/documents/",
            json={
                "content": "Webpage content about Python.",
                "doc_type": "webpage",
                "space_id": SAMPLE_SPACE_ID,
                "title": "Python Docs",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_type"] == "webpage"

    def test_create_document_missing_space_id_returns_422(self, client):
        resp = client.post(
            "/v1/documents/",
            json={
                "content": "Some content",
            },
        )
        assert resp.status_code == 422


# ── Document status tracking ──────────────────────────────────

class TestDocumentStatus:
    """Document status: queued → extracting → chunking → embedding → done/failed."""

    def test_successful_document_status_done(self, client):
        resp = client.post(
            "/v1/documents/",
            json={
                "content": "FastAPI is great.",
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        data = resp.json()
        assert data["status"] == "done"

    def test_failed_document_status(self):
        """When LLM fails, document status should be 'failed'."""

        class FailLLM:
            async def complete_structured(self, messages, schema, temperature=0.1):
                raise RuntimeError("LLM unavailable")

        fail_processor = DocumentProcessor(
            llm_adapter=FailLLM(),
            embedding_service=MockEmbeddingService(),
        )
        app.dependency_overrides[_get_document_processor] = lambda: fail_processor
        with TestClient(app) as c:
            resp = c.post(
                "/v1/documents/",
                json={
                    "content": "Some content.",
                    "space_id": SAMPLE_SPACE_ID,
                },
            )
            data = resp.json()
            assert data["status"] == "failed"
        app.dependency_overrides.clear()


# ── Document list ──────────────────────────────────────────────

class TestDocumentList:
    """GET /v1/documents/ — list documents."""

    def test_list_documents_returns_200(self, client):
        resp = client.get(
            "/v1/documents/",
            params={"space_id": SAMPLE_SPACE_ID},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    def test_list_documents_empty(self, client):
        resp = client.get(
            "/v1/documents/",
            params={"space_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0


# ── Document delete ────────────────────────────────────────────

class TestDocumentDelete:
    """DELETE /v1/documents/:id — delete a document."""

    def test_delete_document_returns_200(self, client):
        # Create first
        create_resp = client.post(
            "/v1/documents/",
            json={
                "content": "Test document.",
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        doc_id = create_resp.json()["id"]

        # Delete
        resp = client.delete(f"/v1/documents/{doc_id}")
        assert resp.status_code == 200

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete(f"/v1/documents/{uuid.uuid4()}")
        assert resp.status_code == 404


# ── Document get ───────────────────────────────────────────────

class TestDocumentGet:
    """GET /v1/documents/:id — get document details."""

    def test_get_document_returns_200(self, client):
        create_resp = client.post(
            "/v1/documents/",
            json={
                "content": "FastAPI docs content.",
                "space_id": SAMPLE_SPACE_ID,
                "title": "FastAPI",
            },
        )
        doc_id = create_resp.json()["id"]

        resp = client.get(f"/v1/documents/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "FastAPI"

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get(f"/v1/documents/{uuid.uuid4()}")
        assert resp.status_code == 404
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_api_documents.py -v 2>&1 | tail -5
```

**Expected:** All tests FAIL (module not found or 404).

---

### Step 20: Implement document API routes

- [ ] **Write** `server/app/api/documents.py`

```python
"""API routes for /v1/documents/ — document management and processing.

POST   /v1/documents/      — Create and process a document (text/pdf/webpage)
GET    /v1/documents/       — List documents (paginated, filterable by status)
GET    /v1/documents/:id    — Get document details
DELETE /v1/documents/:id    — Delete a document

Processing is done asynchronously via FastAPI BackgroundTasks.
Document status: queued → extracting → chunking → embedding → done/failed
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.extraction.document_processor import DocumentProcessor
from app.schemas.extraction import (
    DocumentCreateRequest,
    DocumentListResponse,
    DocumentOut,
    DocumentStatus,
    DocumentType,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/v1/documents", tags=["documents"])

# ── In-memory document store (to be replaced with DB in integration) ──────

_documents: dict[uuid.UUID, dict] = {}


def _get_document_processor() -> DocumentProcessor:
    """Dependency: get or create DocumentProcessor.

    In production, this would use real LLM adapter and embedding service
    from the app state.
    """
    from app.core.llm import create_llm_adapter
    from app.core.embedding import create_embedding_service
    from app.config import get_config
    try:
        config = get_config()
        llm_adapter = create_llm_adapter(config.llm)
        embedding_service = create_embedding_service(config.embedding)
        return DocumentProcessor(
            llm_adapter=llm_adapter,
            embedding_service=embedding_service,
        )
    except Exception:
        raise RuntimeError(
            "LLM/Embedding not configured. Set environment variables."
        )


async def _process_document_background(
    doc_id: uuid.UUID,
    processor: DocumentProcessor,
    content: str | None,
    doc_type: DocumentType,
    url: str | None,
) -> None:
    """Background task for document processing.

    Updates the in-memory document store with processing results.
    Status progresses: queued → extracting → chunking → embedding → done/failed.

    Args:
        doc_id: Document UUID.
        processor: DocumentProcessor instance.
        content: Document content.
        doc_type: Type of document.
        url: URL for webpage documents.
    """
    result = await processor.process(
        content=content,
        doc_type=doc_type,
        url=url,
    )

    # Update stored document with results
    if doc_id in _documents:
        doc = _documents[doc_id]
        doc["status"] = result.status
        doc["summary"] = result.summary
        if result.error:
            doc["metadata"] = {**(doc.get("metadata") or {}), "error": result.error}
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/", response_model=DocumentUploadResponse)
async def create_document(
    request: DocumentCreateRequest,
    background_tasks: BackgroundTasks,
    processor: DocumentProcessor = Depends(_get_document_processor),
):
    """Create a new document and start async processing.

    The document is created with status='queued' and processing happens
    in the background via FastAPI BackgroundTasks.
    """
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # Store document metadata
    _documents[doc_id] = {
        "id": str(doc_id),
        "title": request.title,
        "doc_type": request.doc_type.value,
        "url": request.url,
        "status": DocumentStatus.queued.value,
        "summary": None,
        "metadata": request.metadata,
        "space_id": str(request.space_id),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    # Schedule background processing
    background_tasks.add_task(
        _process_document_background,
        doc_id=doc_id,
        processor=processor,
        content=request.content,
        doc_type=request.doc_type,
        url=request.url,
    )

    return DocumentUploadResponse(
        id=doc_id,
        title=request.title,
        doc_type=request.doc_type,
        status=DocumentStatus.queued,
        space_id=request.space_id,
        created_at=now,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    space_id: uuid.UUID = Query(...),
    status: Optional[str] = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
):
    """List documents with optional status filter and pagination."""
    items = [
        doc for doc in _documents.values()
        if doc.get("space_id") == str(space_id)
        and (status is None or doc.get("status") == status)
    ]
    # Sort by created_at descending
    items.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    total = len(items)
    paginated = items[offset : offset + limit]

    return DocumentListResponse(
        items=[
            DocumentOut(
                id=uuid.UUID(d["id"]),
                title=d.get("title"),
                doc_type=DocumentType(d["doc_type"]),
                url=d.get("url"),
                status=DocumentStatus(d["status"]),
                summary=d.get("summary"),
                metadata=d.get("metadata"),
                space_id=uuid.UUID(d["space_id"]),
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
            )
            for d in paginated
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: uuid.UUID):
    """Get document details by ID."""
    doc = _documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentOut(
        id=uuid.UUID(doc["id"]),
        title=doc.get("title"),
        doc_type=DocumentType(doc["doc_type"]),
        url=doc.get("url"),
        status=DocumentStatus(doc["status"]),
        summary=doc.get("summary"),
        metadata=doc.get("metadata"),
        space_id=uuid.UUID(doc["space_id"]),
        created_at=datetime.fromisoformat(doc["created_at"]),
        updated_at=datetime.fromisoformat(doc["updated_at"]),
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: uuid.UUID):
    """Delete a document by ID."""
    if doc_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")
    del _documents[doc_id]
    return {"deleted": True, "id": str(doc_id)}
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_api_documents.py -v 2>&1 | tail -20
```

**Expected:**

```
tests/extraction/test_api_documents.py::TestDocumentCreate::test_create_text_document_returns_200 PASSED
...
tests/extraction/test_api_documents.py::TestDocumentGet::test_get_nonexistent_returns_404 PASSED
12 passed
```

---

### Step 21: Commit document API

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/api/documents.py server/tests/extraction/test_api_documents.py
git commit -m "feat(api): add /v1/documents/ routes with async BackgroundTasks processing"
```

---

## Phase 8: Wire Up the Extraction Package

### Step 22: Update extraction __init__.py with public API

- [ ] **Update** `server/app/extraction/__init__.py`

```python
"""RTMemory Extraction Pipeline — three-layer extraction + document processing.

Layer 1: FactDetector — regex + rule-based filtering (zero cost)
Layer 2: Extractor — LLM structured extraction
Layer 3: DeepScanner — batch deep scan at conversation end
Document: DocumentProcessor — text/PDF/webpage → chunk → embed → graph-extract → index
"""

from app.extraction.fact_detector import FactDetector
from app.extraction.extractor import Extractor
from app.extraction.deep_scanner import DeepScanner
from app.extraction.document_processor import DocumentProcessor, Chunker

__all__ = [
    "FactDetector",
    "Extractor",
    "DeepScanner",
    "DocumentProcessor",
    "Chunker",
]
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from app.extraction import FactDetector, Extractor, DeepScanner, DocumentProcessor, Chunker; print('extraction package OK')"
```

**Expected:** `extraction package OK`

---

### Step 23: Run all extraction tests together

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/ -v 2>&1 | tail -30
```

**Expected:** All tests pass — approximately 42+ tests across all extraction test files.

```
tests/extraction/test_fact_detector.py ......
tests/extraction/test_extractor.py ...........
tests/extraction/test_deep_scanner.py .........
tests/extraction/test_document_processor.py ................
tests/extraction/test_api_conversations.py ..........
tests/extraction/test_api_documents.py ............
============================== XX passed ==============================
```

---

### Step 24: Commit extraction package wiring

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/extraction/__init__.py
git commit -m "feat(extraction): wire up extraction package with public API exports"
```

---

## Phase 9: Integration — Full Pipeline End-to-End Test

### Step 25: Write end-to-end integration test

- [ ] **Write** `server/tests/extraction/test_integration.py`

```python
"""End-to-end integration test for the full extraction pipeline.

Tests the complete flow:
- Conversation submit → Layer 1 filter → Layer 2 extract
- Conversation end → Layer 3 deep scan
- Document create → extract → chunk → embed → graph-extract
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.extraction.fact_detector import FactDetector
from app.extraction.extractor import Extractor
from app.extraction.deep_scanner import DeepScanner
from app.extraction.document_processor import DocumentProcessor
from app.api.conversations import _get_fact_detector, _get_extractor, _get_deep_scanner
from app.api.documents import _get_document_processor


# ── Mock adapters ───────────────────────────────────────────────

class MockLLMAdapter:
    """Returns realistic extraction results."""

    async def complete_structured(self, messages, schema, temperature=0.1):
        # Inspect the messages to return contextually appropriate results
        prompt = " ".join(m.get("content", "") for m in messages)

        if "深度" in prompt or "deep" in prompt.lower():
            # Deep scan response
            return {
                "entities": [
                    {"name": "张军", "type": "person", "description": "全栈工程师", "confidence": 0.95},
                    {"name": "RTMemory", "type": "project", "description": "知识图谱项目", "confidence": 0.9},
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
                    {"content": "张军正在开发RTMemory", "type": "status", "confidence": 0.85, "entity_name": "张军"},
                    {"content": "张军偏好Python和TypeScript", "type": "preference", "confidence": 0.8, "entity_name": "张军"},
                ],
                "contradictions": [],
                "confidence_adjustments": [],
            }

        # Document extraction or conversation extraction
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
                {"content": "张军搬到了北京", "type": "fact", "confidence": 0.9, "entity_name": "张军"},
                {"content": "张军偏好Python", "type": "preference", "confidence": 0.85, "entity_name": "张军"},
            ],
            "contradictions": [],
        }


class MockEmbeddingService:
    async def embed(self, text):
        return [0.01] * 1536

    async def embed_batch(self, texts):
        return [[0.01] * 1536 for _ in texts]


@pytest.fixture
def client():
    app.dependency_overrides[_get_fact_detector] = lambda: FactDetector()
    app.dependency_overrides[_get_extractor] = lambda: Extractor(llm_adapter=MockLLMAdapter())
    app.dependency_overrides[_get_deep_scanner] = lambda: DeepScanner(
        llm_adapter=MockLLMAdapter(), min_messages=1,
    )
    app.dependency_overrides[_get_document_processor] = lambda: DocumentProcessor(
        llm_adapter=MockLLMAdapter(),
        embedding_service=MockEmbeddingService(),
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


SAMPLE_SPACE_ID = "00000000-0000-0000-0000-000000000001"


class TestFullPipeline:
    """End-to-end test of the complete extraction pipeline."""

    def test_conversation_submit_and_end(self, client):
        """Submit a conversation, then end it — full 3-layer pipeline."""

        # Step 1: Submit conversation (Layer 1 + Layer 2)
        submit_resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我搬到北京了"},
                    {"role": "assistant", "content": "北京是个好地方！"},
                    {"role": "user", "content": "我用Python写代码"},
                ],
                "space_id": SAMPLE_SPACE_ID,
                "entity_context": "张军的知识库",
            },
        )
        assert submit_resp.status_code == 200
        submit_data = submit_resp.json()
        assert submit_data["skipped"] is False
        assert len(submit_data["extracted"]["entities"]) >= 1
        assert len(submit_data["extracted"]["memories"]) >= 1
        conv_id = submit_data["conversation_id"]

        # Step 2: End conversation (Layer 3 deep scan)
        end_resp = client.post(
            "/v1/conversations/end",
            json={
                "conversation_id": conv_id,
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert end_resp.status_code == 200
        end_data = end_resp.json()
        assert end_data["message_count"] == 3
        assert len(end_data["deep_scan_result"]["entities"]) >= 1

    def test_conversation_skip_then_submit_fact(self, client):
        """Non-fact messages are skipped, but fact messages trigger extraction."""

        # First: casual chat
        resp1 = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp1.json()["skipped"] is True

        # Second: fact message
        resp2 = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我是软件工程师"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert resp2.json()["skipped"] is False

    def test_document_create_and_list(self, client):
        """Create a document, then list it — full document pipeline."""

        # Create
        create_resp = client.post(
            "/v1/documents/",
            json={
                "content": "FastAPI is a modern web framework for building APIs with Python.",
                "doc_type": "text",
                "space_id": SAMPLE_SPACE_ID,
                "title": "FastAPI Guide",
            },
        )
        assert create_resp.status_code == 200
        doc_id = create_resp.json()["id"]

        # List
        list_resp = client.get(
            "/v1/documents/",
            params={"space_id": SAMPLE_SPACE_ID},
        )
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] >= 1

        # Get
        get_resp = client.get(f"/v1/documents/{doc_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["title"] == "FastAPI Guide"

    def test_document_and_conversation_independent(self, client):
        """Document processing and conversation extraction are independent."""

        # Create a document
        client.post(
            "/v1/documents/",
            json={
                "content": "Some document content.",
                "space_id": SAMPLE_SPACE_ID,
            },
        )

        # Submit a conversation
        conv_resp = client.post(
            "/v1/conversations/",
            json={
                "messages": [
                    {"role": "user", "content": "我喜欢Python"},
                ],
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        assert conv_resp.status_code == 200
        assert conv_resp.json()["skipped"] is False
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/test_integration.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/extraction/test_integration.py::TestFullPipeline::test_conversation_submit_and_end PASSED
tests/extraction/test_integration.py::TestFullPipeline::test_conversation_skip_then_submit_fact PASSED
tests/extraction/test_integration.py::TestFullPipeline::test_document_create_and_list PASSED
tests/extraction/test_integration.py::TestFullPipeline::test_document_and_conversation_independent PASSED
4 passed
```

---

### Step 26: Commit integration test

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/tests/extraction/test_integration.py
git commit -m "test(extraction): add end-to-end integration test for full 3-layer + document pipeline"
```

---

## Phase 10: Final Validation

### Step 27: Run the complete test suite

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/extraction/ -v --tb=short 2>&1
```

**Expected:** All tests pass — FactDetector (42), Extractor (11), DeepScanner (9), DocumentProcessor (16), Conversations API (10), Documents API (12), Integration (4). Total ~104 tests.

---

### Step 28: Verify all files exist

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && find server/app/extraction server/app/api server/app/schemas/extraction.py server/tests/extraction -type f | sort
```

**Expected:**

```
server/app/api/__init__.py
server/app/api/conversations.py
server/app/api/documents.py
server/app/extraction/__init__.py
server/app/extraction/deep_scanner.py
server/app/extraction/document_processor.py
server/app/extraction/extractor.py
server/app/extraction/fact_detector.py
server/app/schemas/extraction.py
server/tests/extraction/__init__.py
server/tests/extraction/test_api_conversations.py
server/tests/extraction/test_api_documents.py
server/tests/extraction/test_deep_scanner.py
server/tests/extraction/test_document_processor.py
server/tests/extraction/test_extractor.py
server/tests/extraction/test_fact_detector.py
server/tests/extraction/test_integration.py
```

---

### Step 29: Final commit — mark pipeline complete

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add -A server/
git commit -m "feat(extraction): complete 3-layer extraction pipeline + document processing + API routes"
```

---

## Summary

| Component | File | Layer | Tests |
|-----------|------|-------|-------|
| FactDetector | `server/app/extraction/fact_detector.py` | Layer 1 (regex, zero cost) | `test_fact_detector.py` — 42 tests (zh + en + negative + context + edge) |
| Extractor | `server/app/extraction/extractor.py` | Layer 2 (LLM structured) | `test_extractor.py` — 11 tests (basic, context, contradictions, conversation) |
| DeepScanner | `server/app/extraction/deep_scanner.py` | Layer 3 (batch deep scan) | `test_deep_scanner.py` — 9 tests (basic, empty, context, threshold) |
| DocumentProcessor | `server/app/extraction/document_processor.py` | Document pipeline | `test_document_processor.py` — 16 tests (chunker, text/pdf/webpage, status) |
| Conversation API | `server/app/api/conversations.py` | `/v1/conversations/` | `test_api_conversations.py` — 10 tests (submit, end, skipped, context) |
| Document API | `server/app/api/documents.py` | `/v1/documents/` | `test_api_documents.py` — 12 tests (create, list, get, delete, status) |
| Schemas | `server/app/schemas/extraction.py` | Shared types | Verified via import |
| Integration | `tests/extraction/test_integration.py` | E2E | 4 tests (full pipeline) |

**Total: ~104 tests, all with mock LLM/embedding, TDD approach, Chinese + English support.**