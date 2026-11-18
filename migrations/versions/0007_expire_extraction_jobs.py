"""Expire terminal Extraction Jobs and clear operational input.

Revision ID: 0007_extraction_job_expiration
Revises: 0006_extraction_job_retries
Create Date: 2026-11-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_extraction_job_expiration"
down_revision: str | Sequence[str] | None = "0006_extraction_job_retries"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "extraction_job"
_LIFECYCLE_CONSTRAINT = "extraction_job_lifecycle_state_check"
_SOURCE_URL_LENGTH_CONSTRAINT = "extraction_job_source_url_length_check"
_INPUT_STATE_CONSTRAINT = "extraction_job_input_state_check"
_EXPIRATION_INDEX = "extraction_job_expires_at_job_id_idx"


def _expiring_lifecycle_constraint() -> str:
    """Return the lifecycle invariant with terminal seven-day retention."""
    return (
        "attempt_count >= 0 AND attempt_count <= 3 AND ("
        "(status = 'queued' AND started_at IS NULL AND finished_at IS NULL "
        "AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND eligible_at IS NOT NULL AND expires_at IS NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL) "
        "OR (status = 'running' AND attempt_count >= 1 "
        "AND started_at IS NOT NULL AND finished_at IS NULL "
        "AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND eligible_at IS NULL AND expires_at IS NULL "
        "AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL "
        "AND started_at <= updated_at AND updated_at < lease_expires_at) "
        "OR (status = 'succeeded' AND attempt_count >= 1 "
        "AND updated_at = finished_at "
        "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
        "AND result_version IS NOT NULL AND result IS NOT NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND eligible_at IS NULL "
        "AND expires_at = finished_at + INTERVAL '7 days' "
        "AND lease_token IS NULL AND lease_expires_at IS NULL) "
        "OR (status = 'failed' AND attempt_count >= 1 "
        "AND updated_at = finished_at "
        "AND finished_at IS NOT NULL AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NOT NULL AND error_message IS NOT NULL "
        "AND eligible_at IS NULL "
        "AND expires_at = finished_at + INTERVAL '7 days' "
        "AND lease_token IS NULL AND lease_expires_at IS NULL))"
    )


def _retry_aware_lifecycle_constraint() -> str:
    """Return the prior lifecycle invariant for a safe downgrade."""
    return (
        "attempt_count >= 0 AND attempt_count <= 3 AND ("
        "(status = 'queued' AND started_at IS NULL AND finished_at IS NULL "
        "AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND eligible_at IS NOT NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL) "
        "OR (status = 'running' AND attempt_count >= 1 "
        "AND started_at IS NOT NULL AND finished_at IS NULL "
        "AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND eligible_at IS NULL "
        "AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL "
        "AND started_at <= updated_at AND updated_at < lease_expires_at) "
        "OR (status = 'succeeded' AND attempt_count >= 1 "
        "AND updated_at = finished_at "
        "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
        "AND result_version IS NOT NULL AND result IS NOT NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND eligible_at IS NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL) "
        "OR (status = 'failed' AND attempt_count >= 1 "
        "AND updated_at = finished_at "
        "AND finished_at IS NOT NULL AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NOT NULL AND error_message IS NOT NULL "
        "AND eligible_at IS NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL))"
    )


def upgrade() -> None:
    """Retain terminal jobs briefly without retaining their operational input."""
    op.add_column(_TABLE_NAME, sa.Column("transcript", sa.Text(), nullable=True))
    op.add_column(
        _TABLE_NAME,
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(_SOURCE_URL_LENGTH_CONSTRAINT, _TABLE_NAME, type_="check")
    op.alter_column(_TABLE_NAME, "source_url", existing_type=sa.Text(), nullable=True)
    op.create_check_constraint(
        _SOURCE_URL_LENGTH_CONSTRAINT,
        _TABLE_NAME,
        "source_url IS NULL OR char_length(source_url) BETWEEN 1 AND 2048",
    )
    op.execute(
        sa.text(
            "UPDATE extraction_job SET source_url = NULL, market = NULL, "
            "expires_at = finished_at + INTERVAL '7 days' "
            "WHERE status IN ('succeeded', 'failed')"
        )
    )
    op.create_check_constraint(
        _INPUT_STATE_CONSTRAINT,
        _TABLE_NAME,
        "((status IN ('queued', 'running') AND ("
        "(source_url IS NOT NULL AND transcript IS NULL) "
        "OR (source_url IS NULL AND transcript IS NOT NULL))) "
        "OR (status IN ('succeeded', 'failed') "
        "AND source_url IS NULL AND transcript IS NULL AND market IS NULL))",
    )
    op.drop_constraint(_LIFECYCLE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.create_check_constraint(
        _LIFECYCLE_CONSTRAINT,
        _TABLE_NAME,
        _expiring_lifecycle_constraint(),
    )
    op.create_index(
        _EXPIRATION_INDEX,
        _TABLE_NAME,
        ["expires_at", "job_id"],
        unique=False,
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )


def downgrade() -> None:
    """Restore the Source-only schema without discarding terminal results."""
    op.drop_index(_EXPIRATION_INDEX, table_name=_TABLE_NAME)
    op.drop_constraint(_LIFECYCLE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.drop_constraint(_INPUT_STATE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.drop_constraint(_SOURCE_URL_LENGTH_CONSTRAINT, _TABLE_NAME, type_="check")
    op.execute(sa.text("UPDATE extraction_job SET source_url = 'redacted' WHERE source_url IS NULL"))
    op.alter_column(_TABLE_NAME, "source_url", existing_type=sa.Text(), nullable=False)
    op.create_check_constraint(
        _SOURCE_URL_LENGTH_CONSTRAINT,
        _TABLE_NAME,
        "char_length(source_url) BETWEEN 1 AND 2048",
    )
    op.create_check_constraint(
        _LIFECYCLE_CONSTRAINT,
        _TABLE_NAME,
        _retry_aware_lifecycle_constraint(),
    )
    op.drop_column(_TABLE_NAME, "transcript")
    op.drop_column(_TABLE_NAME, "expires_at")
