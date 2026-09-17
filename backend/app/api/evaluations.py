"""Evaluation evidence and Model Quality Gate API.

POST /api/v1/models/{id}/evaluations     submit evaluation evidence   [pipeline]
GET  /api/v1/models/{id}/evaluations     list stored evidence         [viewer]
GET  /api/v1/models/{id}/quality-gate    dry-run the quality gate     [viewer]
GET  /api/v1/evaluations/{id}            one stored evaluation        [viewer]
GET  /api/v1/model-quality-gate/policy   the policy in force          [viewer]

Two boundaries worth stating explicitly, because both are load-bearing:

* **Submission is signed.** Evidence arrives only through the trusted-pipeline
  HMAC path used by the metrics attestation. The signature covers the report's
  SHA256, and the server recomputes that digest from the body it received, so
  a report edited in flight stops verifying.
* **The dry run does not write.** ``GET .../quality-gate`` answers "what would
  the gate say right now?" without touching the journal, mirroring the existing
  ``POST .../gate`` dry run. The gate evaluations that *do* land in the
  governance journal are the ones taken during a real promotion attempt.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_session
from ..evaluation.report import report_summary
from ..mlops.attestation import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    evaluation_attestation_payload,
    verify_attestation_signature,
)
from ..mlops.quality_gate import (
    VERDICT_HOLD,
    VERDICT_NOT_ENFORCED,
    VERDICT_PASS,
    safe_get_quality_gate_policy,
)
from ..models import ModelEvaluation
from ..security.auth import (
    ROLE_ADMIN,
    ROLE_APPROVER,
    ROLE_ENGINEER,
    ROLE_PIPELINE,
    ROLE_VIEWER,
    require_roles,
    request_id as _request_id,
)
from ..services.evaluation_service import (
    EvaluationSubmissionError,
    get_evaluation_service,
)
from ..services.registry_service import get_registry_service

router = APIRouter(prefix="/api/v1", tags=["evaluation"])

RequireViewer = Depends(require_roles(ROLE_VIEWER, ROLE_ENGINEER, ROLE_PIPELINE, ROLE_APPROVER, ROLE_ADMIN))
RequirePipeline = Depends(require_roles(ROLE_PIPELINE, ROLE_ADMIN))


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _raise_submission_error(exc: EvaluationSubmissionError) -> None:
    status = 403 if exc.code == "insufficient_role" else 422
    raise HTTPException(status_code=status, detail=_err(exc.code, exc.message)) from exc


class EvaluationIn(BaseModel):
    """`extra="forbid"` on purpose: a caller that posts a stray field is told,
    instead of having it silently dropped and later wondering why the stored
    evidence differs from what it sent."""

    model_config = ConfigDict(extra="forbid")

    report: dict
    report_uri: str | None = None


def _evaluation_out(row: ModelEvaluation) -> dict:
    return {
        "id": str(row.id),
        "registry_id": str(row.registry_id) if row.registry_id is not None else None,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "model_type": row.model_type,
        "task": row.task,
        "dataset": {"name": row.dataset_name, "split": row.dataset_split},
        "sample_count": row.sample_count,
        "threshold": row.threshold,
        "metrics": {
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "false_accept_rate": row.false_accept_rate,
            "false_reject_rate": row.false_reject_rate,
            "latency_p95_ms": row.latency_p95_ms,
        },
        "confusion_matrix": [
            [row.true_negative, row.false_positive],
            [row.false_negative, row.true_positive],
        ],
        "report_sha256": row.report_sha256,
        "report_uri": row.report_uri,
        "attested_by": row.attested_by,
        "evaluation_time": row.evaluation_time.isoformat() if row.evaluation_time else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("/models/{entry_id}/evaluations")
async def submit_evaluation(
    entry_id: uuid.UUID,
    body: EvaluationIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor=RequirePipeline,
) -> dict:
    svc = get_registry_service()
    entry = await svc.get(session, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))

    report = body.report
    model = report.get("model") if isinstance(report, dict) else None
    if not isinstance(model, dict) or not model.get("name") or not model.get("version"):
        raise HTTPException(
            status_code=422,
            detail=_err("report_model_identity_missing", "report.model.name and report.model.version are required"),
        )

    claimed = str(report.get("fingerprint") or "")
    payload = evaluation_attestation_payload(
        model_name=str(model["name"]),
        model_version=str(model["version"]),
        report_sha256=claimed,
    )
    signature_check = verify_attestation_signature(
        get_settings().pipeline_hmac_secret,
        payload,
        request.headers.get(SIGNATURE_HEADER),
        request.headers.get(TIMESTAMP_HEADER),
    )
    if not signature_check.ok:
        raise HTTPException(
            status_code=422,
            detail=_err("evaluation_attestation_invalid", signature_check.reason),
        )

    try:
        row = await get_evaluation_service().submit(
            session,
            actor=actor,
            report=report,
            entry=entry,
            report_uri=body.report_uri,
            attestation_digest=signature_check.digest,
            request_id=_request_id(request),
        )
        await session.commit()
    except EvaluationSubmissionError as exc:
        await session.rollback()
        _raise_submission_error(exc)
    return _evaluation_out(row)


@router.get("/models/{entry_id}/evaluations", dependencies=[RequireViewer])
async def list_evaluations(
    entry_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
) -> dict:
    svc = get_registry_service()
    entry = await svc.get(session, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    rows = await get_evaluation_service().list(session, registry_id=entry.id, limit=limit)
    return {
        "model": f"{entry.model_name}@{entry.model_version}",
        "count": len(rows),
        "evaluations": [_evaluation_out(row) for row in rows],
    }


@router.get("/models/{entry_id}/quality-gate", dependencies=[RequireViewer])
async def quality_gate(
    entry_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Read-only: reports the verdict without writing to the journal."""
    svc = get_registry_service()
    entry = await svc.get(session, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    result = await get_evaluation_service().quality_gate_for(
        session, entry, record_audit=False
    )
    return {"model": f"{entry.model_name}@{entry.model_version}", "quality_gate": result.to_dict()}


@router.get("/evaluations/{evaluation_id}", dependencies=[RequireViewer])
async def get_evaluation(
    evaluation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    row = await get_evaluation_service().get(session, evaluation_id)
    if row is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "evaluation not found"))
    return {**_evaluation_out(row), "summary": report_summary(row.report)}


@router.get("/model-quality-gate/policy", dependencies=[RequireViewer])
async def quality_gate_policy() -> dict:
    """The policy in force, including whether it enforces anything yet.

    Exposed so that an unenforced scope is never a hidden state: a reader can
    see that the gate is configured, see its digest, and see that its
    ``enforced_model_types`` list is still empty.
    """
    policy, error = safe_get_quality_gate_policy()
    if policy is None:
        return {
            "available": False,
            "error": error,
            "enforcement_enabled": False,
            "note": "the quality gate policy could not be loaded; every enforced promotion fails closed",
        }
    return {
        "available": True,
        "enforcement_enabled": bool(policy.enforcement.enforced_model_types),
        "verdicts": [VERDICT_PASS, VERDICT_HOLD, VERDICT_NOT_ENFORCED],
        "policy": policy.to_dict(),
    }
