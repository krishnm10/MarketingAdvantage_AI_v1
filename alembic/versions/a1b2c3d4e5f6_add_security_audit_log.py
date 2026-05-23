"""add security_audit_log

Revision ID: a1b2c3d4e5f6
Revises: 9b2d4f6a1c77
Create Date: 2026-05-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "9b2d4f6a1c77"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "security_audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column("business_id", sa.String(64), nullable=False, index=True),
        sa.Column("pipeline_id", sa.String(36), nullable=True),
        sa.Column("node_id", sa.String(36), nullable=True),
        sa.Column("position", sa.String(20), nullable=True),
        sa.Column("entities_detected", sa.Text(), nullable=True),
        sa.Column("action_taken", sa.String(20), nullable=False),
        sa.Column("severity_max", sa.String(10), nullable=True),
        sa.Column("chunk_ids_affected", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("meta_data", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("security_audit_log")
