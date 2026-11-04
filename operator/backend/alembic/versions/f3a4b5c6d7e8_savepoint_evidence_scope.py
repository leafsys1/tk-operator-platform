"""Separate operator evidence identity from full savepoint archive identity."""
from alembic import op
import sqlalchemy as sa

revision = "f3a4b5c6d7e8"
down_revision = "f2b3c4d5e6a7"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("savepoints", sa.Column("evidence_scope", sa.String(32), nullable=True))
    op.add_column("savepoints", sa.Column("op_name_snapshot", sa.String(200), nullable=True))
    op.add_column("savepoints", sa.Column("archive_snapshot_sha256", sa.String(80), nullable=True))


def downgrade():
    with op.batch_alter_table("savepoints") as batch:
        batch.drop_column("archive_snapshot_sha256")
        batch.drop_column("op_name_snapshot")
        batch.drop_column("evidence_scope")
