"""Index Textify coordination expiration.

Revision ID: 0002_coordination_expiration
Revises: 0001_textify_coordination

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_coordination_expiration"
down_revision: str | Sequence[str] | None = "0001_textify_coordination"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "textify_coordination_expires_at_id_idx"
_TABLE_NAME = "textify_coordination"


def upgrade() -> None:
    """Create the expiration cleanup index."""
    # Cleanup scans oldest expirations first. Appending the primary key makes equal
    # expiration times deterministic for cursor-based batches.
    op.create_index(
        _INDEX_NAME,
        _TABLE_NAME,
        ["expires_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove only the expiration cleanup index."""
    op.drop_index(_INDEX_NAME, table_name=_TABLE_NAME)
