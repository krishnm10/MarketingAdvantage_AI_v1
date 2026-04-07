"""add page_number to ingested_content

Revision ID: 7a9c1f2d3e44
Revises: bc4efacf6f39
Create Date: 2026-03-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7a9c1f2d3e44"
down_revision: Union[str, Sequence[str], None] = "bc4efacf6f39"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingested_content",
        sa.Column("page_number", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingested_content", "page_number")

