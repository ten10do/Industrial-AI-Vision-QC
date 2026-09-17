"""unified business operation audit journal

Revision ID: 0011_business_audit_log
Revises: 0010_inspection_image_provenance
Create Date: 2026-08-31

Append-only audit_log table. Every protected business mutation (inspection
creation, human review claim/resolve/correction, quality-rule changes,
telemetry ingestion) records the acting principal, roles, action, resource
and outcome in the same transaction as the write itself.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_business_audit_log"
down_revision = "0010_inspection_image_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("result", sa.String(length=16), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("actor_roles", sa.JSON(), nullable=True),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=128), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_audit_log_created", "audit_log", ["created_at"])
    op.create_index("ix_audit_log_actor", "audit_log", ["actor"])
    op.create_index("ix_audit_log_resource", "audit_log", ["resource_type"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_resource", table_name="audit_log")
    op.drop_index("ix_audit_log_actor", table_name="audit_log")
    op.drop_index("ix_audit_log_created", table_name="audit_log")
    op.drop_table("audit_log")
