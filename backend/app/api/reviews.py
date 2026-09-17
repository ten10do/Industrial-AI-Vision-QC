"""Review Queue API (Phase 5, 5C-5F).

RBAC: only a reviewer (or admin) may mutate the review queue; read access
is open to reviewers / release managers / viewers / engineers / approvers /
pipeline / admin. The reviewer identity is the authenticated principal's
subject; a client-supplied `reviewer` string must match it (403 otherwise).
DB is the source of truth; WS events are notifications only.
"""

from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..enums import HumanDecision, ReviewTaskStatus
from ..mlops.manifest import dataset_version_for_model
from ..models import Inspection, ReviewDecision, ReviewTask
from ..schemas import ReviewMetricsOut, ReviewTaskOut, TrainingCandidate
from ..security.auth import (
    ROLE_ADMIN,
    ROLE_APPROVER,
    ROLE_ENGINEER,
    ROLE_PIPELINE,
    ROLE_RELEASE_MANAGER,
    ROLE_REVIEWER,
    ROLE_VIEWER,
    Principal,
    require_any_authenticated,
    require_roles,
    request_id as _request_id,
)
from ..services.audit_service import record as audit_record
from ..services.review_service import ReviewConflictError, ReviewService, ReviewValidationError
from .serializers import to_review_task_out

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reviews"])

# Separation of duties: only a reviewer (or admin) may mutate the review
# queue; read access is open to reviewers, release managers, viewers and admins.
RequireReviewerWrite = Depends(require_roles(ROLE_REVIEWER, ROLE_ADMIN))
RequireReviewRead = Depends(
    require_roles(ROLE_REVIEWER, ROLE_RELEASE_MANAGER, ROLE_VIEWER, ROLE_ENGINEER, ROLE_APPROVER, ROLE_PIPELINE, ROLE_ADMIN)
)


def get_review_service() -> ReviewService:
    return ReviewService()


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _matching_reviewer(body_reviewer: str | None, actor: Principal) -> str:
    """The human-review identity is the authenticated principal's subject.
    A client-supplied reviewer string is accepted only when it matches the
    token; a mismatch is a 403 impersonation attempt, never a silent no-op."""
    if body_reviewer is not None and body_reviewer != actor.subject:
        raise HTTPException(
            status_code=403,
            detail=_err(
                "reviewer_mismatch",
                f"reviewer in body ({body_reviewer}) does not match the authenticated principal ({actor.subject})",
            ),
        )
    return actor.subject


class ClaimIn(BaseModel):
    # Kept for client compatibility; the authoritative reviewer identity is
    # the authenticated principal. A non-matching value is rejected (403).
    reviewer: str | None = Field(default=None, min_length=1, max_length=64)


class ResolveIn(BaseModel):
    reviewer: str | None = Field(default=None, min_length=1, max_length=64)
    human_decision: HumanDecision
    human_label: str | None = None
    reason: str | None = Field(default=None, max_length=1024)


class CorrectionIn(BaseModel):
    reviewer: str | None = Field(default=None, min_length=1, max_length=64)
    field_changed: str = Field(min_length=1, max_length=64)
    new_value: dict
    reason: str | None = Field(default=None, max_length=1024)


@router.get("/reviews", response_model=list[ReviewTaskOut], dependencies=[RequireReviewRead])
async def list_reviews(
    status: ReviewTaskStatus | None = None,
    priority: int | None = Query(default=None, ge=1, le=1000),
    defect_type: str | None = None,
    production_line: str | None = None,
    station: str | None = None,
    batch_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
    service: ReviewService = Depends(get_review_service),
) -> list[ReviewTaskOut]:
    tasks = await service.list_tasks(
        session, status=status, priority=priority, defect_type=defect_type,
        production_line=production_line, station=station, batch_id=batch_id,
        limit=limit, offset=offset,
    )
    return [to_review_task_out(t) for t in tasks]


@router.get("/reviews/{task_id}", response_model=ReviewTaskOut, dependencies=[RequireReviewRead])
async def get_review(
    task_id: str,
    session: AsyncSession = Depends(get_session),
    service: ReviewService = Depends(get_review_service),
) -> ReviewTaskOut:
    task = await service.get_task(session, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "review task not found"))
    return to_review_task_out(task)


@router.post("/reviews/{task_id}/claim", response_model=ReviewTaskOut)
async def claim_review(
    task_id: str,
    request: Request,
    body: ClaimIn,
    session: AsyncSession = Depends(get_session),
    service: ReviewService = Depends(get_review_service),
    actor: Principal = RequireReviewerWrite,
) -> ReviewTaskOut:
    reviewer = _matching_reviewer(body.reviewer, actor)
    try:
        task = await service.claim(session, task_id, reviewer)
    except ReviewConflictError as exc:
        audit_record(
            session, action="review.claim", actor=actor, result="denied",
            resource_type="review_task", resource_id=task_id,
            detail={"code": exc.code, "error": str(exc)}, request_id=_request_id(request),
        )
        await session.commit()
        raise HTTPException(status_code=409, detail=_err(exc.code, str(exc))) from exc
    audit_record(
        session, action="review.claim", actor=actor, result="applied",
        resource_type="review_task", resource_id=task_id, request_id=_request_id(request),
    )
    await session.commit()
    return to_review_task_out(task)


@router.post("/reviews/{task_id}/resolve", response_model=ReviewTaskOut)
async def resolve_review(
    task_id: str,
    request: Request,
    body: ResolveIn,
    session: AsyncSession = Depends(get_session),
    service: ReviewService = Depends(get_review_service),
    actor: Principal = RequireReviewerWrite,
) -> ReviewTaskOut:
    reviewer = _matching_reviewer(body.reviewer, actor)
    try:
        task = await service.resolve(
            session, task_id, reviewer, body.human_decision, body.human_label, body.reason
        )
    except ReviewConflictError as exc:
        audit_record(
            session, action="review.resolve", actor=actor, result="denied",
            resource_type="review_task", resource_id=task_id,
            detail={"code": exc.code, "error": str(exc)}, request_id=_request_id(request),
        )
        await session.commit()
        raise HTTPException(status_code=409, detail=_err(exc.code, str(exc))) from exc
    except ReviewValidationError as exc:
        audit_record(
            session, action="review.resolve", actor=actor, result="denied",
            resource_type="review_task", resource_id=task_id,
            detail={"code": exc.code, "error": str(exc)}, request_id=_request_id(request),
        )
        await session.commit()
        raise HTTPException(status_code=422, detail=_err(exc.code, str(exc))) from exc
    audit_record(
        session, action="review.resolve", actor=actor, result="applied",
        resource_type="review_task", resource_id=task_id,
        detail={"human_decision": body.human_decision.value if hasattr(body.human_decision, "value") else str(body.human_decision)},
        request_id=_request_id(request),
    )
    await session.commit()
    return to_review_task_out(task)


@router.post("/reviews/{task_id}/corrections", response_model=dict)
async def add_correction(
    task_id: str,
    request: Request,
    body: CorrectionIn,
    session: AsyncSession = Depends(get_session),
    service: ReviewService = Depends(get_review_service),
    actor: Principal = RequireReviewerWrite,
) -> dict:
    """5F: append an audit correction after resolve; the original decision is
    never overwritten."""
    reviewer = _matching_reviewer(body.reviewer, actor)
    try:
        correction = await service.add_correction(
            session, task_id, reviewer, body.field_changed, body.new_value, body.reason
        )
    except ReviewConflictError as exc:
        audit_record(
            session, action="review.correction", actor=actor, result="denied",
            resource_type="review_task", resource_id=task_id,
            detail={"code": exc.code, "error": str(exc)}, request_id=_request_id(request),
        )
        await session.commit()
        raise HTTPException(status_code=409, detail=_err(exc.code, str(exc))) from exc
    except ReviewValidationError as exc:
        audit_record(
            session, action="review.correction", actor=actor, result="denied",
            resource_type="review_task", resource_id=task_id,
            detail={"code": exc.code, "error": str(exc)}, request_id=_request_id(request),
        )
        await session.commit()
        raise HTTPException(status_code=422, detail=_err(exc.code, str(exc))) from exc
    audit_record(
        session, action="review.correction", actor=actor, result="applied",
        resource_type="review_task", resource_id=task_id,
        detail={"field_changed": correction.field_changed}, request_id=_request_id(request),
    )
    await session.commit()
    return {
        "status": "ok",
        "correction_id": str(correction.id),
        "field_changed": correction.field_changed,
        "reason": correction.reason,
    }


@router.get("/reviews-metrics", response_model=ReviewMetricsOut, dependencies=[Depends(require_roles(ROLE_REVIEWER, ROLE_RELEASE_MANAGER, ROLE_ADMIN))])
async def review_metrics(
    session: AsyncSession = Depends(get_session),
    service: ReviewService = Depends(get_review_service),
) -> ReviewMetricsOut:
    """5K: review metrics with explicit semantics (see service docstring)."""
    data = await service.metrics(session)
    return ReviewMetricsOut(**data)


@router.get("/training-candidates", response_model=list[TrainingCandidate], dependencies=[Depends(require_roles(ROLE_REVIEWER, ROLE_RELEASE_MANAGER, ROLE_ADMIN))])
async def training_candidates(
    kind: str = Query(default="all", pattern="^(all|corrected|disagreed|low_confidence)$"),
    format: str = Query(default="json", pattern="^(json|csv)$"),
    limit: int = Query(default=500, ge=1, le=2000),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """5J: export manifest of resolved reviews for future training. No
    automatic retraining is triggered. Images stay referenced via StorageService."""
    from ..enums import QualityResult as QR

    stmt = (
        select(ReviewDecision, Inspection, ReviewTask)
        .join(Inspection, Inspection.id == ReviewDecision.inspection_id)
        .join(ReviewTask, ReviewTask.id == ReviewDecision.review_task_id)
        .options(
            selectinload(Inspection.product),
            selectinload(Inspection.defects),
        )
        .order_by(ReviewDecision.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()

    candidates: list[TrainingCandidate] = []
    for decision, inspection, task in rows:
        ai_defects = decision.ai_defects_snapshot or []
        top = max(ai_defects, key=lambda d: d.get("confidence", 0), default=None)
        ai_label = top["class_name"] if top else None
        ai_conf = top["confidence"] if top else None
        human_decision = decision.human_decision.value if hasattr(decision.human_decision, "value") else decision.human_decision
        agreement = human_decision == "CONFIRM_DEFECT"
        low_conf = ai_conf is not None and ai_conf < 0.6
        include = {
            "all": True,
            "corrected": human_decision in ("CORRECT_DEFECT", "OTHER_DEFECT"),
            "disagreed": human_decision in ("PASS", "CORRECT_DEFECT", "OTHER_DEFECT"),
            "low_confidence": low_conf,
        }[kind]
        if not include:
            continue
        candidates.append(
            TrainingCandidate(
                inspection_id=inspection.inspection_id,
                image_url=f"/api/v1/inspections/{inspection.inspection_id}/image",
                ai_label=ai_label,
                human_label=decision.human_label,
                ai_confidence=ai_conf,
                agreement=agreement,
                review_reason=decision.reason,
                anomaly_score=task.anomaly_score,
                # semantic fix (Phase 9): dataset identity != deployment
                # identity. source_dataset_version is the TRAINING dataset
                # from the pinned manifest; source_deployment_version is the
                # ONLINE AI stack that was live when the sample was judged.
                source_dataset_version=dataset_version_for_model(inspection.model_name),
                source_model_version=inspection.model_version,
                source_deployment_version=inspection.deployment_version,
                timestamp=decision.created_at,
            )
        )

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=["inspection_id", "image_url", "ai_label", "human_label",
                        "ai_confidence", "agreement", "review_reason",
                        "anomaly_score", "source_dataset_version",
                        "source_model_version", "source_deployment_version",
                        "timestamp"],
        )
        writer.writeheader()
        for c in candidates:
            writer.writerow(c.model_dump())
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=training-candidates-{kind}.csv"},
        )

    import json

    return Response(
        content=json.dumps([c.model_dump(mode="json") for c in candidates], ensure_ascii=False, indent=2),
        media_type="application/json",
    )
