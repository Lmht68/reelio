"""Create durable Textify coordination.

Revision ID: 0001_textify_coordination
Revises:
Create Date: 2026-10-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_textify_coordination"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create durable state for one admitted or pending Textify job."""
    op.create_table(
        "textify_coordination",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("url_identity", sa.LargeBinary(), nullable=False),
        sa.Column(
            "idempotency_key",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("capability_link", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="textify_coordination_pkey"),
        sa.UniqueConstraint(
            "url_identity",
            name="textify_coordination_url_identity_key",
        ),
        sa.UniqueConstraint(
            "idempotency_key",
            name="textify_coordination_idempotency_key_key",
        ),
        sa.CheckConstraint(
            "octet_length(url_identity) = 32",
            name="textify_coordination_url_identity_length_check",
        ),
        sa.CheckConstraint(
            "(get_byte(uuid_send(idempotency_key), 6) & 240) = 64 "
            "AND (get_byte(uuid_send(idempotency_key), 8) & 192) = 128",
            name="textify_coordination_idempotency_key_uuid4_check",
        ),
        sa.CheckConstraint(
            "(capability_link IS NULL AND updated_at = created_at) "
            "OR (capability_link IS NOT NULL AND capability_link ~ "
            "'^/api/v1/transcription-jobs/[0-9a-f]{8}-[0-9a-f]{4}-"
            "4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')",
            name="textify_coordination_capability_state_check",
        ),
        sa.CheckConstraint(
            "created_at <= updated_at AND updated_at < expires_at",
            name="textify_coordination_timestamp_order_check",
        ),
    )


def downgrade() -> None:
    """Remove durable Textify coordination storage."""
    op.drop_table("textify_coordination")
