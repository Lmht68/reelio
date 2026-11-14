"""Add anonymous owner-scoped Extraction Job idempotency.

Revision ID: 0004_anonymous_job_ownership
Revises: 0003_extraction_job
Create Date: 2026-11-14

"""

import secrets
from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_anonymous_job_ownership"
down_revision: str | Sequence[str] | None = "0003_extraction_job"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "extraction_job"
_OWNER_DIGEST_LENGTH_CONSTRAINT = "extraction_job_owner_digest_length_check"
_OWNER_IDEMPOTENCY_UNIQUE = "extraction_job_owner_idempotency_key_key"


def upgrade() -> None:
    """Bind every existing and future job to an anonymous owner and replay key."""
    op.add_column(
        _TABLE_NAME,
        sa.Column("owner_digest", postgresql.BYTEA(), nullable=True),
    )
    op.add_column(
        _TABLE_NAME,
        sa.Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=True),
    )

    extraction_job = sa.table(
        _TABLE_NAME,
        sa.column("job_id", postgresql.UUID(as_uuid=True)),
        sa.column("owner_digest", postgresql.BYTEA()),
        sa.column("idempotency_key", postgresql.UUID(as_uuid=True)),
    )
    connection = op.get_bind()
    job_ids = connection.scalars(sa.select(extraction_job.c.job_id)).all()
    for job_id in job_ids:
        connection.execute(
            extraction_job.update()
            .where(extraction_job.c.job_id == job_id)
            .values(
                owner_digest=secrets.token_bytes(32),
                idempotency_key=uuid4(),
            )
        )

    op.alter_column(_TABLE_NAME, "owner_digest", nullable=False)
    op.alter_column(_TABLE_NAME, "idempotency_key", nullable=False)
    op.create_check_constraint(
        _OWNER_DIGEST_LENGTH_CONSTRAINT,
        _TABLE_NAME,
        "octet_length(owner_digest) = 32",
    )
    op.create_unique_constraint(
        _OWNER_IDEMPOTENCY_UNIQUE,
        _TABLE_NAME,
        ["owner_digest", "idempotency_key"],
    )


def downgrade() -> None:
    """Remove anonymous owner-scoped idempotency from Extraction Jobs."""
    op.drop_constraint(_OWNER_IDEMPOTENCY_UNIQUE, _TABLE_NAME, type_="unique")
    op.drop_constraint(_OWNER_DIGEST_LENGTH_CONSTRAINT, _TABLE_NAME, type_="check")
    op.drop_column(_TABLE_NAME, "idempotency_key")
    op.drop_column(_TABLE_NAME, "owner_digest")
