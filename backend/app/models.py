from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Index, Integer, JSON, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .enums import BatchStatus, HumanDecision, InspectionStatus, QualityResult, ReviewTaskStatus, Severity


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Batch(TimestampMixin, Base):
    __tablename__ = "batches"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    production_line: Mapped[str] = mapped_column(String(64), nullable=False)
    product_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[BatchStatus] = mapped_column(SAEnum(BatchStatus, native_enum=False), nullable=False, default=BatchStatus.OPEN)

    products: Mapped[list["Product"]] = relationship(back_populates="batch")


class Product(TimestampMixin, Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    batch_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("batches.id"), nullable=True)
    production_line: Mapped[str] = mapped_column(String(64), nullable=False)
    station: Mapped[str] = mapped_column(String(64), nullable=False)

    batch: Mapped[Batch | None] = relationship(back_populates="products")
    inspections: Mapped[list["Inspection"]] = relationship(back_populates="product")


class Inspection(TimestampMixin, Base):
    __tablename__ = "inspections"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    product_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("products.id"), index=True, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, index=True, nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    status: Mapped[InspectionStatus] = mapped_column(
        SAEnum(InspectionStatus, native_enum=False), nullable=False, default=InspectionStatus.PENDING
    )
    quality_result: Mapped[QualityResult | None] = mapped_column(SAEnum(QualityResult, native_enum=False), nullable=True)
    # AI 原始判定（创建时写入，永不修改）；final_quality_result 为业务最终事实（5G）。
    # 非 REVIEW 的 inspection 在创建时 final = ai；REVIEW 在人工 resolve 后写入。
    final_quality_result: Mapped[QualityResult | None] = mapped_column(
        SAEnum(QualityResult, native_enum=False), nullable=True
    )
    severity: Mapped[Severity | None] = mapped_column(SAEnum(Severity, native_enum=False), nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rule_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # P0 traceability: the stored artifact's real URI, content digest and
    # detected media type. image_path must never hold the upload filename;
    # it points at bytes that actually exist on disk.
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_media_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ---- Phase 8 MLOps: deployment traceability (8D) ----
    deployment_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_registry_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_registry.id"), nullable=True)

    # ---- Phase 6 anomaly (PatchCore) + vision fusion ----
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomalous: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anomaly_map_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    anomaly_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    anomaly_regions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    fusion_class: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # ---- Phase 7 industrial state ----
    # Three-layer semantics (see docs): desired_command (what the system
    # wants the field layer to do), execution_status (whether the command was
    # actually sent/acked), industrial_state / industrial_final_state (the
    # product's real field state). NOT_INTEGRATED means the PLC was never
    # engaged; it is never faked as HELD / SAFE_HOLD / RELEASED / REJECTED.
    desired_command: Mapped[str | None] = mapped_column(String(16), nullable=True)
    execution_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    industrial_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    industrial_final_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    plc_command: Mapped[str | None] = mapped_column(String(16), nullable=True)
    plc_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    plc_adapter_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    plc_reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plc_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    mes_sync_status: Mapped[str | None] = mapped_column(String(16), nullable=True)

    product: Mapped[Product] = relationship(back_populates="inspections")
    defects: Mapped[list["Defect"]] = relationship(back_populates="inspection", cascade="all, delete-orphan")


class Defect(TimestampMixin, Base):
    __tablename__ = "defects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), index=True, nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_xyxy: Mapped[list] = mapped_column(JSON, nullable=False)
    bbox_normalized: Mapped[list] = mapped_column(JSON, nullable=False)
    defect_area_px: Mapped[float] = mapped_column(Float, nullable=False)
    defect_area_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[Severity | None] = mapped_column(SAEnum(Severity, native_enum=False), nullable=True)
    matched_rule: Mapped[str | None] = mapped_column(String(64), nullable=True)

    inspection: Mapped[Inspection] = relationship(back_populates="defects")


class QualityRule(TimestampMixin, Base):
    __tablename__ = "quality_rules"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    defect_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="class_name or '*' for any")
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_area_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    action: Mapped[QualityResult] = mapped_column(SAEnum(QualityResult, native_enum=False), nullable=False)
    severity: Mapped[Severity] = mapped_column(SAEnum(Severity, native_enum=False), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100, comment="lower is evaluated first")
    rule_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# 一个 inspection 最多一个 active（非 RESOLVED）review task（5B）。用部分唯一索引
# 在 DB 层强制；应用层再做幂等检查（见 ReviewTask 下方的 Index 定义）。


class ReviewTask(TimestampMixin, Base):
    """人工复核任务（5A）。AI 快照在创建时固化（ai_defects_snapshot），
    之后任何数据变化都不影响已归档的原始判断。"""

    __tablename__ = "review_tasks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    review_task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), index=True, nullable=False)

    status: Mapped[ReviewTaskStatus] = mapped_column(
        SAEnum(ReviewTaskStatus, native_enum=False), nullable=False, default=ReviewTaskStatus.PENDING
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100, comment="lower is processed first")
    assigned_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="optimistic lock")

    # ---- AI snapshot（创建时固化，5A / 5F）----
    ai_quality_result: Mapped[str] = mapped_column(String(16), nullable=False)
    ai_defects_snapshot: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    ai_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ai_rule_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_severity: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # ---- denormalized 产品上下文（队列筛选，免 join）----
    product_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    production_line: Mapped[str] = mapped_column(String(64), nullable=False)
    station: Mapped[str] = mapped_column(String(64), nullable=False)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ---- Phase 6 anomaly snapshot（UNKNOWN_ANOMALY 复核展示用）----
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomalous: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anomaly_regions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    anomaly_map_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    inspection: Mapped[Inspection] = relationship()
    decision: Mapped["ReviewDecision | None"] = relationship(back_populates="task", uselist=False)


# 一个 inspection 生命周期最多一个 review task（Phase 6 前置门禁）。
# 系统不支持二次 Review（RESOLVED 无 reopen 路径），故使用全局唯一约束；
# 已 RESOLVED 决策的修订走 review_corrections。
Index(
    "uq_review_task_inspection",
    ReviewTask.__table__.c.inspection_id,
    unique=True,
)


class ReviewDecision(TimestampMixin, Base):
    """人工决策记录（5A / 5F）。创建后不可修改；如需变更走 review_corrections 追加记录。"""

    __tablename__ = "review_decisions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    review_task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("review_tasks.id"), unique=True, nullable=False)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), index=True, nullable=False)
    reviewer: Mapped[str] = mapped_column(String(64), nullable=False)
    ai_quality_result: Mapped[str] = mapped_column(String(16), nullable=False)
    ai_defects_snapshot: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    human_decision: Mapped[HumanDecision] = mapped_column(SAEnum(HumanDecision, native_enum=False), nullable=False)
    human_label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    final_quality_result: Mapped[QualityResult] = mapped_column(SAEnum(QualityResult, native_enum=False), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    task: Mapped[ReviewTask] = relationship(back_populates="decision")
    corrections: Mapped[list["ReviewCorrection"]] = relationship(back_populates="decision")


class ReviewCorrection(TimestampMixin, Base):
    """审计修订记录（5F）：对已 RESOLVED 决策的追加修正，不覆盖原记录。"""

    __tablename__ = "review_corrections"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    review_decision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("review_decisions.id"), index=True, nullable=False)
    reviewer: Mapped[str] = mapped_column(String(64), nullable=False)
    field_changed: Mapped[str] = mapped_column(String(64), nullable=False)
    old_value: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    new_value: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    decision: Mapped[ReviewDecision] = relationship(back_populates="corrections")


class PlcEvent(TimestampMixin, Base):
    """PLC command event log (Phase 7). Answers "why was this product
    rejected?" with the full chain: Image -> AI -> Rule -> Final Quality
    Result -> PLC Command -> PLC ACK."""

    __tablename__ = "plc_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    command_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    product_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    inspection_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    command: Mapped[str] = mapped_column(String(16), nullable=False)
    # three-layer semantics: what we wanted (desired_command) vs what
    # actually happened (execution_status) vs the product field state
    desired_command: Mapped[str] = mapped_column(String(16), nullable=False)
    execution_status: Mapped[str] = mapped_column(String(32), nullable=False)
    industrial_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    adapter_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelRegistry(TimestampMixin, Base):
    """Model identity + lifecycle (Phase 8).

    status lifecycle: CANDIDATE -> STAGING -> PRODUCTION -> ARCHIVED.
    At most one PRODUCTION row per model_name is enforced in the DB by a
    partial unique index (status='PRODUCTION').

    Provenance columns (governance hardening): metrics, domain_validated and
    artifact_sha256 are privileged facts. They are only ever written through
    the signed trusted-pipeline attestation path, and each carries a
    server-computed verification flag that the promotion gate enforces.
    """

    __tablename__ = "model_registry"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)  # yolo | patchcore
    artifact_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    training_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="CANDIDATE")
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    domain_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ---- provenance (attested by the trusted pipeline, verified server-side) ----
    attested_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attestation_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_hash_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    domain_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    domain_evidence_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ---- human approval ----
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approval_reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # ---- runtime activation (registry -> deployment manifest) ----
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activation_target: Mapped[str | None] = mapped_column(String(512), nullable=True)


class ModelRegistryAudit(TimestampMixin, Base):
    """Append-only governance journal.

    Every mutation and every *denied* attempt lands here. Rows are never
    updated or deleted through the API: promotion, rollback, archive and
    activation each append one record before and one after the transition.
    """

    __tablename__ = "model_registry_audit"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    registry_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_registry.id"), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    action: Mapped[str] = mapped_column(String(32), nullable=False)  # register|attest|promote|rollback|archive|activate
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)  # APPLIED|DENIED|ERROR
    from_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(16), nullable=True)

    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    actor_roles: Mapped[list | None] = mapped_column(JSON, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    gate: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


Index("ix_model_registry_audit_registry", ModelRegistryAudit.__table__.c.registry_id)
Index("ix_model_registry_audit_created", ModelRegistryAudit.__table__.c.created_at)


class ModelEvaluation(TimestampMixin, Base):
    """Stored evaluation evidence for one model version (Error Analysis Pipeline).

    One row is one offline evaluation of one model version on one dataset
    split. It is the evidence the quality gate reads, so it is written only
    through the signed trusted-pipeline path and is append-only: a re-run
    appends a new row rather than editing the old one, which is what makes
    "the gate passed, and here is what it read" checkable after the fact.

    The scalar columns are a denormalised index of the report so that a list
    endpoint or a dashboard cell never has to parse the JSON payload; the
    payload itself (``report``) is the source of truth and is re-hashed by the
    server on submission.
    """

    __tablename__ = "model_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    registry_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_registry.id"), nullable=True)

    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    task: Mapped[str] = mapped_column(String(32), nullable=False)

    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_split: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)

    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_accept_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_reject_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    true_positive: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    true_negative: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_positive: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_negative: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # The full structured report and the SHA256 of its canonical form. The
    # fingerprint is recomputed server-side on submission: a caller cannot
    # store a report and quote a different digest for it.
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    report_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    report_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)

    attested_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attestation_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


Index("ix_model_evaluations_version", ModelEvaluation.__table__.c.model_name,
      ModelEvaluation.__table__.c.model_version)
Index("ix_model_evaluations_created", ModelEvaluation.__table__.c.created_at)
Index("ix_model_evaluations_registry", ModelEvaluation.__table__.c.registry_id)


class DatasetVersion(TimestampMixin, Base):
    """Dataset versioning: manifest + SHA256 (8K). A model must be able to
    trace back to an exact dataset version."""

    __tablename__ = "dataset_versions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    manifest_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)


class AuditLog(TimestampMixin, Base):
    """Business operation audit journal (unified, append-only by convention).

    Every protected business mutation records who (authenticated principal),
    what (action + resource), the outcome (applied | denied) and the request
    id. Rows are never updated or deleted through the API; the journal answers
    "who did what when" across inspections / review / quality rules / telemetry
    without tying itself to the model-registry lifecycle table.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    result: Mapped[str] = mapped_column(String(16), nullable=False)  # applied | denied
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    actor_roles: Mapped[list | None] = mapped_column(JSON, nullable=True)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


Index("ix_audit_log_created", AuditLog.__table__.c.created_at)
Index("ix_audit_log_actor", AuditLog.__table__.c.actor)
Index("ix_audit_log_resource", AuditLog.__table__.c.resource_type)
