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
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas.extraction import (
    DocumentType,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

# Reuse the same extraction schema from Extractor
from app.extraction.extractor import EXTRACTION_OUTPUT_SCHEMA, Extractor


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
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    memories: list = field(default_factory=list)
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