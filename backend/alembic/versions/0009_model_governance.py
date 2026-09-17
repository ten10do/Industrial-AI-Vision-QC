"""governance: model registry provenance + append-only audit journal

Revision ID: 0009_model_governance
Revises: 0008_model_registry
Create Date: 2026-08-31

Closes the self-attestation hole in the model registry:

- model_registry gains provenance columns (attested_by / attested_at /
  attestation_digest / artifact_hash_verified / domain_evidence /
  domain_evidence_verified), approval attribution (approved_by /
  approval_reason) and runtime activation tracking (activated_at /
  activation_target). Metrics, domain_validated and artifact_sha256 are now
  only writable through the signed trusted-pipeline attestation path.
- model_registry_audit is an append-only journal recording both applied and
  denied governance operations with the acting principal and the gate result.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_model_governance"
down_revision = "0008_model_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("attested_by", sa.String(length=128), nullable=True))
    op.add_column("model_registry", sa.Column("attested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("model_registry", sa.Column("attestation_digest", sa.String(length=64), nullable=True))
    op.add_column(
        "model_registry",
        sa.Column("artifact_hash_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("model_registry", sa.Column("domain_evidence", sa.JSON(), nullable=True))
    op.add_column(
        "model_registry",
        sa.Column("domain_evidence_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("model_registry", sa.Column("approved_by", sa.String(length=128), nullable=True))
    op.add_column("model_registry", sa.Column("approval_reason", sa.String(length=1024), nullable=True))
    op.add_column("model_registry", sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("model_registry", sa.Column("activation_target", sa.String(length=512), nullable=True))

    op.create_table(
        "model_registry_audit",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("registry_id", sa.Uuid(), sa.ForeignKey("model_registry.id"), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("from_status", sa.String(length=16), nullable=True),
        sa.Column("to_status", sa.String(length=16), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("actor_roles", sa.JSON(), nullable=True),
        sa.Column("approved_by", sa.String(length=128), nullable=True),
        sa.Column("reason", sa.String(length=1024), nullable=True),
        sa.Column("gate", sa.JSON(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_registry_audit_registry", "model_registry_audit", ["registry_id"])
    op.create_index("ix_model_registry_audit_created", "model_registry_audit", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_audit_created", table_name="model_registry_audit")
    op.drop_index("ix_model_registry_audit_registry", table_name="model_registry_audit")
    op.drop_table("model_registry_audit")
    for column in (
        "activation_target",
        "activated_at",
        "approval_reason",
        "approved_by",
        "domain_evidence_verified",
        "domain_evidence",
        "artifact_hash_verified",
        "attestation_digest",
        "attested_at",
        "attested_by",
    ):
        op.drop_column("model_registry", column)
