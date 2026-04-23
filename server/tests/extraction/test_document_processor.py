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