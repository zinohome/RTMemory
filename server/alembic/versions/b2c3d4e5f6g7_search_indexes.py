"""Add search indexes — vector HNSW, tsvector, and GIN indexes.

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6g7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HNSW indexes for vector similarity search (pgvector)
    op.execute("CREATE INDEX IF NOT EXISTS ix_entities_embedding ON entities USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_summary_embedding ON documents USING hnsw (summary_embedding vector_cosine_ops)")

    # tsvector column for full-text search on memory content
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_tsvector tsvector")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_memories_content_tsvector
        ON memories USING GIN (content_tsvector)
    """)

    # Populate tsvector from existing content (idempotent)
    op.execute("""
        UPDATE memories SET content_tsvector = to_tsvector('english', COALESCE(content, ''))
        WHERE content_tsvector IS NULL AND content IS NOT NULL
    """)

    # Trigger to keep tsvector in sync
    op.execute("""
        CREATE OR REPLACE FUNCTION memories_tsvector_trigger() RETURNS trigger AS $$
        BEGIN
          NEW.content_tsvector := to_tsvector('english', COALESCE(NEW.content, ''));
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS memories_tsvector_update ON memories;
        CREATE TRIGGER memories_tsvector_update
            BEFORE INSERT OR UPDATE OF content ON memories
            FOR EACH ROW EXECUTE FUNCTION memories_tsvector_trigger()
    """)

    # tsvector on document content for full-text search
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsvector tsvector")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_documents_content_tsvector
        ON documents USING GIN (content_tsvector)
    """)
    op.execute("""
        UPDATE documents SET content_tsvector = to_tsvector('english', COALESCE(content, ''))
        WHERE content_tsvector IS NULL AND content IS NOT NULL
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION documents_tsvector_trigger() RETURNS trigger AS $$
        BEGIN
          NEW.content_tsvector := to_tsvector('english', COALESCE(NEW.content, ''));
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS documents_tsvector_update ON documents;
        CREATE TRIGGER documents_tsvector_update
            BEFORE INSERT OR UPDATE OF content ON documents
            FOR EACH ROW EXECUTE FUNCTION documents_tsvector_trigger()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS memories_tsvector_update ON memories")
    op.execute("DROP FUNCTION IF EXISTS memories_tsvector_trigger()")
    op.execute("DROP INDEX IF EXISTS ix_memories_content_tsvector")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS content_tsvector")

    op.execute("DROP TRIGGER IF EXISTS documents_tsvector_update ON documents")
    op.execute("DROP FUNCTION IF EXISTS documents_tsvector_trigger()")
    op.execute("DROP INDEX IF EXISTS ix_documents_content_tsvector")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS content_tsvector")

    op.execute("DROP INDEX IF EXISTS ix_documents_summary_embedding")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding")
    op.execute("DROP INDEX IF EXISTS ix_entities_embedding")