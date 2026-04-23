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