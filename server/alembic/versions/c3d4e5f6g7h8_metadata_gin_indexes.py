"""Add GIN indexes for JSONB metadata columns.

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-04-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6g7h8"
down_revision: Union[str, None] = "b2c3d4e5f6g7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # GIN index on entities.metadata for JSONB key lookups (e.g. metadata->>'user_id')
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entities_metadata_gin "
        "ON entities USING GIN (metadata)"
    )
    # GIN index on documents.metadata
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_metadata_gin "
        "ON documents USING GIN (metadata)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_entities_metadata_gin")
    op.execute("DROP INDEX IF EXISTS ix_documents_metadata_gin")