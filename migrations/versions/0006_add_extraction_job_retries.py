"""Add retry scheduling to durable Extraction Jobs.

Revision ID: 0006_extraction_job_retries
Revises: 0005_extraction_job_leases
Create Date: 2026-11-16

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_extraction_job_retries"
down_revision: str | Sequence[str] | None = "0005_extraction_job_leases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "extraction_job"
_LIFECYCLE_CONSTRAINT = "extraction_job_lifecycle_state_check"
_QUEUED_INDEX = "extraction_job_queued_submitted_at_job_id_idx"
_QUEUED_ELIGIBLE_INDEX = "extraction_job_queued_eligible_at_submitted_at_job_id_idx"


def _retry_aware_lifecycle_constraint() -> str:
    """Return the lifecycle invariant for fenced attempts and queued retries."""
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


def _lease_aware_lifecycle_constraint() -> str:
    """Return the prior lifecycle invariant for a safe downgrade."""
    return (
        "(status = 'queued' AND started_at IS NULL AND finished_at IS NULL "
        "AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL) "
        "OR (status = 'running' AND started_at IS NOT NULL AND finished_at IS NULL "
        "AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL "
        "AND started_at <= updated_at AND updated_at < lease_expires_at) "
        "OR (status = 'succeeded' AND updated_at = finished_at "
        "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
        "AND result_version IS NOT NULL AND result IS NOT NULL "
        "AND error_code IS NULL AND error_message IS NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL) "
        "OR (status = 'failed' AND updated_at = finished_at "
        "AND finished_at IS NOT NULL AND result_version IS NULL AND result IS NULL "
        "AND error_code IS NOT NULL AND error_message IS NOT NULL "
        "AND lease_token IS NULL AND lease_expires_at IS NULL)"
    )


def upgrade() -> None:
    """Add durable attempt counts and claim eligibility timestamps."""
    op.add_column(
        _TABLE_NAME,
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        _TABLE_NAME,
        sa.Column("eligible_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(_LIFECYCLE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.execute(
        sa.text(
            "UPDATE extraction_job SET attempt_count = "
            "CASE WHEN status = 'queued' THEN 0 ELSE 1 END, "
            "eligible_at = CASE WHEN status = 'queued' THEN updated_at ELSE NULL END"
        )
    )
    op.create_check_constraint(
        _LIFECYCLE_CONSTRAINT,
        _TABLE_NAME,
        _retry_aware_lifecycle_constraint(),
    )
    op.drop_index(_QUEUED_INDEX, table_name=_TABLE_NAME)
    op.create_index(
        _QUEUED_ELIGIBLE_INDEX,
        _TABLE_NAME,
        ["eligible_at", "submitted_at", "job_id"],
        unique=False,
        postgresql_where=sa.text("status = 'queued'"),
    )


def downgrade() -> None:
    """Remove retry scheduling while preserving existing lease ownership."""
    op.drop_index(_QUEUED_ELIGIBLE_INDEX, table_name=_TABLE_NAME)
    op.create_index(
        _QUEUED_INDEX,
        _TABLE_NAME,
        ["submitted_at", "job_id"],
        unique=False,
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.drop_constraint(_LIFECYCLE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.drop_column(_TABLE_NAME, "eligible_at")
    op.drop_column(_TABLE_NAME, "attempt_count")
    op.create_check_constraint(
        _LIFECYCLE_CONSTRAINT,
        _TABLE_NAME,
        _lease_aware_lifecycle_constraint(),
    )
