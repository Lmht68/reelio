"""Create durable Source Extraction jobs.

Revision ID: 0003_extraction_job
Revises: 0002_coordination_expiration
Create Date: 2026-10-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_extraction_job"
down_revision: str | Sequence[str] | None = "0002_coordination_expiration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "extraction_job"
_QUEUED_INDEX = "extraction_job_queued_submitted_at_job_id_idx"


def upgrade() -> None:
    """Create committed Source Extraction job state and queued claim ordering."""
    op.create_table(
        _TABLE_NAME,
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("market", sa.String(length=2), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_version", sa.Text(), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("job_id", name="extraction_job_pkey"),
        sa.CheckConstraint(
            "(get_byte(uuid_send(job_id), 6) & 240) = 64 "
            "AND (get_byte(uuid_send(job_id), 8) & 192) = 128",
            name="extraction_job_job_id_uuid4_check",
        ),
        sa.CheckConstraint(
            "char_length(source_url) BETWEEN 1 AND 2048",
            name="extraction_job_source_url_length_check",
        ),
        sa.CheckConstraint(
            "market IS NULL OR market ~ '^[A-Z]{2}$'",
            name="extraction_job_market_check",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="extraction_job_status_check",
        ),
        sa.CheckConstraint(
            "submitted_at <= updated_at "
            "AND (started_at IS NULL OR submitted_at <= started_at) "
            "AND (finished_at IS NULL OR COALESCE(started_at, submitted_at) <= finished_at) "
            "AND (finished_at IS NULL OR finished_at <= updated_at)",
            name="extraction_job_timestamp_order_check",
        ),
        sa.CheckConstraint(
            "result IS NULL OR jsonb_typeof(result) = 'object'",
            name="extraction_job_result_object_check",
        ),
        sa.CheckConstraint(
            "(result_version IS NULL OR btrim(result_version) <> '') "
            "AND (error_code IS NULL OR btrim(error_code) <> '') "
            "AND (error_message IS NULL OR btrim(error_message) <> '')",
            name="extraction_job_terminal_text_check",
        ),
        sa.CheckConstraint(
            "(status = 'queued' AND updated_at = submitted_at "
            "AND started_at IS NULL AND finished_at IS NULL "
            "AND result_version IS NULL AND result IS NULL "
            "AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'running' AND updated_at = started_at "
            "AND started_at IS NOT NULL AND finished_at IS NULL "
            "AND result_version IS NULL AND result IS NULL "
            "AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'succeeded' AND updated_at = finished_at "
            "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "AND result_version IS NOT NULL AND result IS NOT NULL "
            "AND error_code IS NULL AND error_message IS NULL) "
            "OR (status = 'failed' AND updated_at = finished_at "
            "AND finished_at IS NOT NULL AND result_version IS NULL AND result IS NULL "
            "AND error_code IS NOT NULL AND error_message IS NOT NULL)",
            name="extraction_job_lifecycle_state_check",
        ),
    )
    op.create_index(
        _QUEUED_INDEX,
        _TABLE_NAME,
        ["submitted_at", "job_id"],
        unique=False,
        postgresql_where=sa.text("status = 'queued'"),
    )


def downgrade() -> None:
    """Remove durable Source Extraction job state and its queued claim index."""
    op.drop_index(_QUEUED_INDEX, table_name=_TABLE_NAME)
    op.drop_table(_TABLE_NAME)
