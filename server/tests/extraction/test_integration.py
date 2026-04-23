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