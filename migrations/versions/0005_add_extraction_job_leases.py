"""Add fenced renewable leases to running Extraction Jobs.

Revision ID: 0005_extraction_job_leases
Revises: 0004_anonymous_job_ownership
Create Date: 2026-11-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_extraction_job_leases"
down_revision: str | Sequence[str] | None = "0004_anonymous_job_ownership"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "extraction_job"
_LIFECYCLE_CONSTRAINT = "extraction_job_lifecycle_state_check"
_LEASE_TOKEN_CONSTRAINT = "extraction_job_lease_token_uuid4_check"
_RUNNING_LEASE_INDEX = "extraction_job_running_lease_expires_at_idx"


def _lease_aware_lifecycle_constraint() -> str:
    """Return the lifecycle invariant that fences every running attempt."""
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


def _legacy_lifecycle_constraint() -> str:
    """Return the pre-lease lifecycle invariant for a safe downgrade."""
    return (
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
        "AND error_code IS NOT NULL AND error_message IS NOT NULL)"
    )


def upgrade() -> None:
    """Add durable attempt ownership and make legacy running work reclaimable."""
    op.add_column(
        _TABLE_NAME,
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        _TABLE_NAME,
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(_LIFECYCLE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.execute(
        sa.text(
            "UPDATE extraction_job SET status = 'queued', started_at = NULL "
            "WHERE status = 'running'"
        )
    )
    op.create_check_constraint(
        _LIFECYCLE_CONSTRAINT,
        _TABLE_NAME,
        _lease_aware_lifecycle_constraint(),
    )
    op.create_check_constraint(
        _LEASE_TOKEN_CONSTRAINT,
        _TABLE_NAME,
        "lease_token IS NULL OR ((get_byte(uuid_send(lease_token), 6) & 240) = 64 "
        "AND (get_byte(uuid_send(lease_token), 8) & 192) = 128)",
    )
    op.create_index(
        _RUNNING_LEASE_INDEX,
        _TABLE_NAME,
        ["lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    """Return active work to the old unfenced queue before removing leases."""
    op.drop_index(_RUNNING_LEASE_INDEX, table_name=_TABLE_NAME)
    op.drop_constraint(_LIFECYCLE_CONSTRAINT, _TABLE_NAME, type_="check")
    op.execute(
        sa.text(
            "UPDATE extraction_job SET status = 'queued', started_at = NULL, "
            "updated_at = submitted_at WHERE status IN ('queued', 'running')"
        )
    )
    op.drop_constraint(_LEASE_TOKEN_CONSTRAINT, _TABLE_NAME, type_="check")
    op.drop_column(_TABLE_NAME, "lease_expires_at")
    op.drop_column(_TABLE_NAME, "lease_token")
    op.create_check_constraint(
        _LIFECYCLE_CONSTRAINT,
        _TABLE_NAME,
        _legacy_lifecycle_constraint(),
    )
