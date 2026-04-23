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