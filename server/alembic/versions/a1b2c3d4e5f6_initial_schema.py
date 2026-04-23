"""Initial schema — all 7 RTMemory tables.

Revision ID: 001_initial
Revises: None
Create Date: 2026-04-23
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. spaces
    op.create_table(
        "spaces",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", UUID(as_uuid=True), nullable=True),
        sa.Column("container_tag", sa.String(255), nullable=True),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # 2. entities
    op.create_table(
        "entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False, server_default="person"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_entities_space_id", "entities", ["space_id"])
    op.create_index("ix_entities_org_id", "entities", ["org_id"])
    op.create_index("ix_entities_entity_type", "entities", ["entity_type"])

    # 3. relations
    op.create_table(
        "relations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(255), nullable=False),
        sa.Column("value", sa.Text, nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("source_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_relations_source_entity_id", "relations", ["source_entity_id"])
    op.create_index("ix_relations_target_entity_id", "relations", ["target_entity_id"])
    op.create_index("ix_relations_space_id", "relations", ["space_id"])
    op.create_index("ix_relations_relation_type", "relations", ["relation_type"])
    op.create_index("ix_relations_is_current", "relations", ["is_current"])

    # 4. memories
    op.create_table(
        "memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("custom_id", sa.String(512), nullable=True),
        sa.Column("memory_type", sa.String(50), nullable=False, server_default="fact"),
        sa.Column("entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("relation_id", UUID(as_uuid=True), sa.ForeignKey("relations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("decay_rate", sa.Float, nullable=False, server_default="0.02"),
        sa.Column("is_forgotten", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("forget_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("forget_reason", sa.Text, nullable=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="SET NULL"), nullable=True),
        sa.Column("root_id", UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="SET NULL"), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_memories_space_id", "memories", ["space_id"])
    op.create_index("ix_memories_entity_id", "memories", ["entity_id"])
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"])
    op.create_index("ix_memories_is_forgotten", "memories", ["is_forgotten"])
    op.create_index("ix_memories_custom_id", "memories", ["custom_id"])

    # 5. documents
    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("doc_type", sa.String(50), nullable=False, server_default="text"),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("summary_embedding", Vector(768), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_documents_space_id", "documents", ["space_id"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_doc_type", "documents", ["doc_type"])

    # 6. chunks
    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # 7. memory_sources
    op.create_table(
        "memory_sources",
        sa.Column("memory_id", UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, primary_key=True),
        sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("relevance_score", sa.Float, nullable=False, server_default="0.0"),
    )
    op.create_index("ix_memory_sources_memory_id", "memory_sources", ["memory_id"])
    op.create_index("ix_memory_sources_document_id", "memory_sources", ["document_id"])


def downgrade() -> None:
    op.drop_table("memory_sources")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("memories")
    op.drop_table("relations")
    op.drop_table("entities")
    op.drop_table("spaces")
    op.execute("DROP EXTENSION IF EXISTS vector")