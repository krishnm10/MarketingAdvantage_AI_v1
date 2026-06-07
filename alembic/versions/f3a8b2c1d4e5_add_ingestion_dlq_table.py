"""add ingestion_dlq table

Revision ID: f3a8b2c1d4e5
Revises: e4f1a2b3c5d6
Create Date: 2026-06-03
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f3a8b2c1d4e5"
down_revision: Union[str, Sequence[str], None] = "e4f1a2b3c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ingestion_dlq",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", sa.String(length=128), nullable=False),
        sa.Column("error_type", sa.String(length=128), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="failed",
        ),
        sa.Column(
            "payload_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("window_index", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_ingestion_dlq_business_id",
        "ingestion_dlq",
        ["business_id"],
    )
    op.create_index(
        "ix_ingestion_dlq_file_id",
        "ingestion_dlq",
        ["file_id"],
    )
    op.create_index(
        "ix_ingestion_dlq_created_at",
        "ingestion_dlq",
        ["created_at"],
    )
    op.create_index(
        "ix_ingestion_dlq_business_file",
        "ingestion_dlq",
        ["business_id", "file_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_dlq_business_file", table_name="ingestion_dlq")
    op.drop_index("ix_ingestion_dlq_created_at", table_name="ingestion_dlq")
    op.drop_index("ix_ingestion_dlq_file_id", table_name="ingestion_dlq")
    op.drop_index("ix_ingestion_dlq_business_id", table_name="ingestion_dlq")
    op.drop_table("ingestion_dlq")
