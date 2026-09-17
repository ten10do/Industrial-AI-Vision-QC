from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import InspectionStatus, QualityResult, Severity


class RuleBase(BaseModel):
    defect_type: str = Field(min_length=1, description="class_name or '*' for any")
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    max_area_ratio: float = Field(ge=0.0, le=1.0, default=1.0)
    action: QualityResult
    severity: Severity
    priority: int = Field(default=100, ge=0, description="lower is evaluated first")
    enabled: bool = True


class RuleCreate(RuleBase):
    rule_version: int = Field(default=1, ge=1)


class RuleUpdate(BaseModel):
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    action: QualityResult | None = None
    severity: Severity | None = None
    priority: int | None = Field(default=None, ge=0)
    enabled: bool | None = None


class RuleOut(RuleBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    rule_version: int
    created_at: datetime
    updated_at: datetime


class DefectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_normalized: list[float]
    defect_area_px: float
    defect_area_ratio: float
    severity: Severity | None
    matched_rule: str | None


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    product_id: str
    production_line: str
    station: str
    created_at: datetime


class InspectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    inspection_id: str
    product_id: str
    batch_id: str | None = None
    image_url: str | None = None
    # P0 traceability: content digest + detected media type of the stored
    # artifact (never the upload filename). Lets auditors verify the served
    # image_url bytes against the recorded sha256.
    image_sha256: str | None = None
    image_media_type: str | None = None
    status: InspectionStatus
    quality_result: QualityResult | None
    final_quality_result: QualityResult | None = None
    severity: Severity | None
    model_name: str | None
    model_version: str | None
    rule_version: int | None
    inference_latency_ms: float | None
    error_message: str | None
    created_at: datetime
    defects: list[DefectOut] = Field(default_factory=list)
    # Phase 6 anomaly + fusion
    anomaly_score: float | None = None
    anomaly_threshold: float | None = None
    is_anomalous: bool | None = None
    anomaly_map_url: str | None = None
    anomaly_model_version: str | None = None
    anomaly_regions: list[dict] | None = None
    fusion_class: str | None = None
    # Phase 7 industrial
    desired_command: str | None = None
    execution_status: str | None = None
    industrial_state: str | None = None
    industrial_final_state: str | None = None
    plc_command: str | None = None
    plc_status: str | None = None
    plc_adapter_type: str | None = None
    plc_reason_code: str | None = None
    plc_latency_ms: float | None = None
    mes_sync_status: str | None = None
    # Phase 8 MLOps
    deployment_version: str | None = None


class InspectionDetail(InspectionOut):
    product: ProductOut


class ReviewDecisionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    review_task_id: uuid.UUID
    inspection_id: uuid.UUID
    reviewer: str
    ai_quality_result: str
    ai_defects_snapshot: list[dict]
    human_decision: str
    human_label: str | None
    final_quality_result: QualityResult
    reason: str | None
    created_at: datetime


class ReviewTaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    review_task_id: str
    inspection_id: uuid.UUID
    inspection: InspectionDetail | None = None
    status: str
    priority: int
    assigned_to: str | None
    claimed_at: datetime | None
    resolved_at: datetime | None
    version: int
    ai_quality_result: str
    ai_defects_snapshot: list[dict]
    ai_model_version: str | None
    ai_rule_version: int | None
    ai_severity: str | None
    product_id: str
    production_line: str
    station: str
    batch_id: str | None
    image_url: str | None
    decision: ReviewDecisionOut | None = None
    # Phase 6 anomaly snapshot (6G)
    anomaly_score: float | None = None
    anomaly_threshold: float | None = None
    is_anomalous: bool | None = None
    anomaly_regions: list[dict] | None = None
    anomaly_map_url: str | None = None
    created_at: datetime
    updated_at: datetime


class ReviewMetricsOut(BaseModel):
    pending_review_count: int
    pending: int
    in_review: int
    resolved: int
    average_review_wait_time_s: float | None
    review_rate: float | None
    defect_confirmation_rate: float | None
    ai_human_label_agreement_rate: float | None
    override_rate: float | None
    corrected_label_count: int
    pass_overrides: int


class TrainingCandidate(BaseModel):
    """Retraining candidate manifest (8J / 9 semantic fix).

    The three source identities are kept distinct:
      source_dataset_version      - training dataset identity (manifest 8K)
      source_model_version        - which model version judged this sample
      source_deployment_version   - which online AI stack (8D) was live
    """
    inspection_id: str
    image_url: str | None
    ai_label: str | None
    human_label: str | None
    ai_confidence: float | None
    agreement: bool
    review_reason: str | None
    anomaly_score: float | None = None
    source_dataset_version: str | None = None
    source_model_version: str | None = None
    source_deployment_version: str | None = None
    timestamp: datetime


class ErrorBody(BaseModel):
    error: dict[str, Any]
