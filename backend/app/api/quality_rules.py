from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import QualityRule
from ..schemas import RuleCreate, RuleOut, RuleUpdate
from ..security.auth import ROLE_ADMIN, ROLE_RELEASE_MANAGER, Principal, require_any_authenticated, require_roles, request_id as _request_id
from ..services.audit_service import record as audit_record

router = APIRouter(prefix="/api/v1/quality-rules", tags=["quality-rules"])

# Quality rules decide PASS/FAIL/REVIEW. Only the release manager (or admin)
# may change them; reads are open to every authenticated principal.
RequireReleaseManagerWrite = Depends(require_roles(ROLE_RELEASE_MANAGER, ROLE_ADMIN))


@router.get("", response_model=list[RuleOut], dependencies=[Depends(require_any_authenticated())])
async def list_rules(
    enabled_only: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[QualityRule]:
    stmt = select(QualityRule).order_by(QualityRule.priority, QualityRule.defect_type)
    if enabled_only:
        stmt = stmt.where(QualityRule.enabled.is_(True))
    result = await session.execute(stmt)
    return list(result.scalars())


@router.post("", response_model=RuleOut, status_code=201)
async def create_rule(
    request: Request,
    rule: RuleCreate,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireReleaseManagerWrite,
) -> QualityRule:
    row = QualityRule(**rule.model_dump())
    session.add(row)
    audit_record(
        session, action="quality_rule.create", actor=actor, result="applied",
        resource_type="quality_rule",
        detail={"defect_type": rule.defect_type, "action": rule.action.value if hasattr(rule.action, "value") else str(rule.action)},
        request_id=_request_id(request),
    )
    await session.commit()
    await session.refresh(row)
    return row


@router.patch("/{rule_id}", response_model=RuleOut)
async def update_rule(
    rule_id: uuid.UUID,
    request: Request,
    patch: RuleUpdate,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireReleaseManagerWrite,
) -> QualityRule:
    result = await session.execute(select(QualityRule).where(QualityRule.id == rule_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "rule not found"))
    changed = patch.model_dump(exclude_unset=True)
    for field, value in changed.items():
        setattr(row, field, value)
    audit_record(
        session, action="quality_rule.update", actor=actor, result="applied",
        resource_type="quality_rule", resource_id=str(rule_id),
        detail={"changed": changed}, request_id=_request_id(request),
    )
    await session.commit()
    await session.refresh(row)
    return row


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": uuid.uuid4().hex[:12]}}
