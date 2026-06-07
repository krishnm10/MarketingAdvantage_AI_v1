"""add streaming checkpoint fields to ingested_file

Revision ID: e4f1a2b3c5d6
Revises: a1b2c3d4e5f6
Create Date: 2026-06-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e4f1a2b3c5d6"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingested_file",
        sa.Column(
            "last_processed_chunk_index",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "ingested_file",
        sa.Column(
            "last_processed_page",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "ingested_file",
        sa.Column(
            "last_processed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("ingested_file", "last_processed_at")
    op.drop_column("ingested_file", "last_processed_page")
    op.drop_column("ingested_file", "last_processed_chunk_index")
