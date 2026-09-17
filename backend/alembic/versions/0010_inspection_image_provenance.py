"""inspection image provenance: sha256 + detected media type of stored artifact

Revision ID: 0010_inspection_image_provenance
Revises: 0009_model_governance
Create Date: 2026-08-31

P0 traceability fix. The inspections table records the raw image as the
traceability anchor. image_path now holds the real on-disk URI (never the
upload filename); image_sha256 and image_media_type let auditors verify the
served bytes against a server-side digest and magic-byte detection. All three
are written by the storage layer via InspectionService; a failed storage write
fails the inspection closed (FAILED, no inference, no RELEASE).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_inspection_image_provenance"
down_revision = "0009_model_governance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspections", sa.Column("image_sha256", sa.String(length=64), nullable=True))
    op.add_column("inspections", sa.Column("image_media_type", sa.String(length=32), nullable=True))
    op.create_index("ix_inspections_image_sha256", "inspections", ["image_sha256"])


def downgrade() -> None:
    op.drop_index("ix_inspections_image_sha256", table_name="inspections")
    op.drop_column("inspections", "image_media_type")
    op.drop_column("inspections", "image_sha256")
