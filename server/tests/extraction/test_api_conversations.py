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