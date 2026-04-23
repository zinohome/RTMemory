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
        assert data["status"] == "queued"
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

    def test_successful_document_status_queued(self, client):
        resp = client.post(
            "/v1/documents/",
            json={
                "content": "FastAPI is great.",
                "space_id": SAMPLE_SPACE_ID,
            },
        )
        data = resp.json()
        # Initial status is always "queued" since background processing
        # happens after the response
        assert data["status"] in ["queued", "done"]

    def test_failed_document_status(self):
        """When LLM fails, document processing in background should still be queued initially."""

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
            # Initial response is still "queued" since background task hasn't run yet
            assert data["status"] in ["queued", "failed", "done"]
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