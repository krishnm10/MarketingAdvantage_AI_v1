"""add parent_chunk_id to ingested_content

Revision ID: 9b2d4f6a1c77
Revises: 7a9c1f2d3e44
Create Date: 2026-03-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "9b2d4f6a1c77"
down_revision: Union[str, Sequence[str], None] = "7a9c1f2d3e44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingested_content",
        sa.Column("parent_chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingested_content", "parent_chunk_id")

