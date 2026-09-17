"""model_evaluations: stored evaluation evidence for the quality gate

Revision ID: 0012_model_evaluations
Revises: 0011_business_audit_log
Create Date: 2026-09-17

Adds the append-only evaluation evidence store the Model Quality Gate reads.
One row is one offline evaluation of one model version on one dataset split,
submitted through the signed trusted-pipeline attestation path.

No column is added to model_registry: evaluation evidence is a stream
(multiple evaluations per version over time), not a field, so the promotion
gate reads the latest row for the version it is judging.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_model_evaluations"
down_revision = "0011_business_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_evaluations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("registry_id", sa.Uuid(), sa.ForeignKey("model_registry.id"), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_type", sa.String(length=32), nullable=True),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("dataset_name", sa.String(length=128), nullable=False),
        sa.Column("dataset_split", sa.String(length=64), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("false_accept_rate", sa.Float(), nullable=True),
        sa.Column("false_reject_rate", sa.Float(), nullable=True),
        sa.Column("latency_p95_ms", sa.Float(), nullable=True),
        sa.Column("true_positive", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("true_negative", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("false_positive", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("false_negative", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("report", sa.JSON(), nullable=False),
        sa.Column("report_sha256", sa.String(length=64), nullable=False),
        sa.Column("report_uri", sa.String(length=512), nullable=True),
        sa.Column("attested_by", sa.String(length=128), nullable=True),
        sa.Column("attestation_digest", sa.String(length=64), nullable=True),
        sa.Column("evaluation_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_evaluations_version", "model_evaluations", ["model_name", "model_version"])
    op.create_index("ix_model_evaluations_created", "model_evaluations", ["created_at"])
    op.create_index("ix_model_evaluations_registry", "model_evaluations", ["registry_id"])


def downgrade() -> None:
    op.drop_index("ix_model_evaluations_registry", table_name="model_evaluations")
    op.drop_index("ix_model_evaluations_created", table_name="model_evaluations")
    op.drop_index("ix_model_evaluations_version", table_name="model_evaluations")
    op.drop_table("model_evaluations")
