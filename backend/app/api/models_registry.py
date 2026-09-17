"""Model Registry API (Phase 8, hardened).

POST   /api/v1/models                    register a model identity      [engineer]
POST   /api/v1/models/{id}/attest        signed pipeline attestation    [pipeline]
GET    /api/v1/models?status=&model_type=list                           [viewer]
GET    /api/v1/models/runtime-sync       registry vs running stack      [viewer]
GET    /api/v1/models/production/{name}  current production for a model [viewer]
GET    /api/v1/models/{id}               detail                         [viewer]
GET    /api/v1/models/{id}/audit         governance journal             [viewer]
POST   /api/v1/models/{id}/gate          dry-run promotion gate         [viewer]
POST   /api/v1/models/{id}/promote       run the gate; promote if passed [approver]
POST   /api/v1/models/{id}/archive                                       [engineer / approver for PRODUCTION]
POST   /api/v1/models/{id}/activate      pin PRODUCTION into the runtime [admin]
POST   /api/v1/models/rollback           switch production back          [approver]

Removed: DELETE /api/v1/models/{id}. A governance journal whose rows can be
hard-deleted is not a journal. Registry rows are archived, never deleted; the
only place a row disappears is a manual DBA operation, which is itself a
change of record outside the API.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_session
from ..mlops.attestation import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    attestation_payload,
    canonical_json,
    sha256_hex,
    verify_attestation_signature,
)
from ..mlops.deployment_sync import runtime_sync
from ..mlops.promotion_gate import evaluate
from ..security.auth import (
    ROLE_ADMIN,
    ROLE_APPROVER,
    ROLE_ENGINEER,
    ROLE_PIPELINE,
    ROLE_VIEWER,
    Principal,
    request_id as _request_id,
    require_roles,
)
from ..services.evaluation_service import get_evaluation_service
from ..services.registry_service import (
    APPROVAL_ERROR_CODES,
    RegistryError,
    RegistryService,
    get_registry_service,
    provenance_for,
)

router = APIRouter(prefix="/api/v1/models", tags=["models"])

RequireViewer = Depends(require_roles(ROLE_VIEWER, ROLE_ENGINEER, ROLE_PIPELINE, ROLE_APPROVER, ROLE_ADMIN))
RequireEngineer = Depends(require_roles(ROLE_ENGINEER, ROLE_PIPELINE, ROLE_APPROVER, ROLE_ADMIN))
RequirePipeline = Depends(require_roles(ROLE_PIPELINE, ROLE_ADMIN))
RequireApprover = Depends(require_roles(ROLE_APPROVER, ROLE_ADMIN))
RequireAdmin = Depends(require_roles(ROLE_ADMIN))


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


CONFLICT_ERROR_CODES = frozenset({"duplicate_version", "rollback_target_is_current_production"})


def _raise_registry_error(exc: RegistryError) -> None:
    if exc.code == "not_found":
        status = 404
    elif exc.code in APPROVAL_ERROR_CODES:
        status = 403
    elif exc.code in CONFLICT_ERROR_CODES:
        status = 409
    else:
        status = 422
    raise HTTPException(status_code=status, detail=_err(exc.code, exc.message)) from exc


class RegisterIn(BaseModel):
    """Identity only. Metrics, the domain verdict and the artifact hash are
    privileged facts that arrive through the signed attestation endpoint.

    extra="forbid" makes a caller who still posts `metrics` or
    `domain_validated` receive a 422 rather than a silent no-op: the old
    contract is gone, and silence here would be indistinguishable from a
    successful write.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_type: str = Field(min_length=1)  # yolo | patchcore
    artifact_uri: str = Field(min_length=1)
    dataset_version: str | None = None
    training_run_id: str | None = None
    notes: str | None = None


class DomainEvidenceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    dataset_version: str | None = None
    eval_report_uri: str = Field(min_length=1)
    eval_report_sha256: str = Field(min_length=64, max_length=64)
    validated_by: str = Field(min_length=1)
    validated_at: str | None = None
    sample_count: int | None = None


class AttestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    metrics: dict = Field(default_factory=dict)
    domain_validated: bool = False
    domain_evidence: DomainEvidenceIn | None = None


class PromoteIn(BaseModel):
    """`thresholds` is tightening-only: any value looser than the server
    policy is a violation. `approved_by` must name a human other than the
    authenticated caller."""

    model_config = ConfigDict(extra="forbid")

    required_domain: str = "steel"
    thresholds: dict | None = None
    approved_by: str | None = None
    reason: str | None = None


class RollbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str
    approved_by: str | None = None
    reason: str | None = None


class ArchiveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved_by: str | None = None
    reason: str | None = None


class ActivateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_stack_version: str = Field(min_length=1)
    approved_by: str | None = None
    reason: str | None = None


def _out(m) -> dict:
    return {
        "id": str(m.id),
        "model_name": m.model_name,
        "model_version": m.model_version,
        "model_type": m.model_type,
        "artifact_uri": m.artifact_uri,
        "artifact_sha256": m.artifact_sha256,
        "dataset_version": m.dataset_version,
        "training_run_id": m.training_run_id,
        "status": m.status,
        "metrics": m.metadata_json or {},
        "domain_validated": m.domain_validated,
        "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
        "notes": m.notes,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "provenance": provenance_for(m),
        "approval": {
            "approved_by": m.approved_by,
            "reason": m.approval_reason,
            "activated_at": m.activated_at.isoformat() if m.activated_at else None,
            "activation_target": m.activation_target,
        },
    }


@router.post("")
async def register(
    body: RegisterIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireEngineer,
) -> dict:
    svc = get_registry_service()
    try:
        m = await svc.register_identity(
            session, actor=actor, model_name=body.model_name, model_version=body.model_version,
            model_type=body.model_type, artifact_uri=body.artifact_uri,
            dataset_version=body.dataset_version, training_run_id=body.training_run_id,
            notes=body.notes, request_id=_request_id(request),
        )
        await session.commit()
    except RegistryError as exc:
        _raise_registry_error(exc)
    return _out(m)


@router.post("/{entry_id}/attest")
async def attest(
    entry_id: uuid.UUID,
    body: AttestIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequirePipeline,
) -> dict:
    """Record evaluation facts from the trusted pipeline.

    The body must be HMAC-signed with IVQC_PIPELINE_HMAC_SECRET over the
    canonical JSON of the attestation payload, together with a timestamp
    inside the configured skew window. The signature is what makes "the
    pipeline said mAP50=0.82" a verifiable statement rather than a claim.
    """
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))

    # Signature is verified over the raw body the client sent, so a client
    # signs exactly what it posts and nothing it could not predict.
    raw = await request.json()
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail=_err("invalid_body", "attestation body must be a JSON object"))
    payload = attestation_payload(
        model_name=m.model_name,
        model_version=m.model_version,
        training_run_id=m.training_run_id,
        body=raw,
    )
    settings = get_settings()
    check = verify_attestation_signature(
        settings.pipeline_hmac_secret,
        payload,
        request.headers.get(SIGNATURE_HEADER),
        request.headers.get(TIMESTAMP_HEADER),
    )
    if not check.ok:
        status = 503 if check.reason.startswith("attestation_not_configured") else 401
        raise HTTPException(status_code=status, detail=_err("attestation_rejected", check.reason))

    try:
        m = await svc.attest(
            session, m, actor=actor, metrics=body.metrics, domain_validated=body.domain_validated,
            domain_evidence=body.domain_evidence.model_dump() if body.domain_evidence else None,
            artifact_sha256=body.artifact_sha256, attestation_digest=check.digest,
            request_id=_request_id(request),
        )
        await session.commit()
    except RegistryError as exc:
        _raise_registry_error(exc)
    return {
        **_out(m),
        "attestation": {
            "signature": check.reason,
            "digest": check.digest,
            "signed_payload_sha256": sha256_hex(canonical_json(payload)),
        },
    }


@router.get("", dependencies=[RequireViewer])
async def list_models(status: str | None = None, model_type: str | None = None,
                      session: AsyncSession = Depends(get_session)) -> list[dict]:
    svc = get_registry_service()
    return [_out(m) for m in await svc.list(session, status=status, model_type=model_type)]


@router.get("/runtime-sync", dependencies=[RequireViewer])
async def get_runtime_sync(session: AsyncSession = Depends(get_session)) -> dict:
    """Is the registry PRODUCTION pointer what the running stack actually
    serves? DRIFT means it is not. UNVERIFIED means we could not prove it,
    which is reported as UNVERIFIED rather than as IN_SYNC."""
    rows = await get_registry_service().production_by_channel(session)
    report = await runtime_sync(rows)
    return report.to_dict()


@router.get("/production/{model_name}", dependencies=[RequireViewer])
async def production(model_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    m = await get_registry_service().get_production(session, model_name)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("no_production", f"no production model for {model_name}"))
    return _out(m)


@router.get("/{entry_id}", dependencies=[RequireViewer])
async def detail(entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> dict:
    m = await get_registry_service().get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    return _out(m)


@router.get("/{entry_id}/audit", dependencies=[RequireViewer])
async def audit(entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> list[dict]:
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    rows = await svc.audit_trail(session, entry_id)
    return [
        {
            "id": str(r.id),
            "action": r.action,
            "outcome": r.outcome,
            "from_status": r.from_status,
            "to_status": r.to_status,
            "actor": r.actor,
            "actor_roles": r.actor_roles,
            "approved_by": r.approved_by,
            "reason": r.reason,
            "gate": r.gate,
            "payload": r.payload,
            "request_id": r.request_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def _evaluate_for(entry, body: PromoteIn):
    return evaluate(
        entry.model_type,
        metrics=entry.metadata_json or {},
        thresholds=body.thresholds,
        domain_validated=entry.domain_validated,
        required_domain=body.required_domain,
        provenance=provenance_for(entry),
    )


@router.post("/{entry_id}/gate", dependencies=[RequireViewer])
async def gate(
    entry_id: uuid.UUID,
    body: PromoteIn,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Dry run. Reports both gates and writes nothing.

    ``promotion_gate`` reads the attested aggregate metrics;
    ``quality_gate`` reads the stored evaluation evidence. A promotion needs
    both, so a dry run that showed only one of them would be misleading.
    """
    m = await get_registry_service().get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    g = _evaluate_for(m, body)
    qg = await get_evaluation_service().quality_gate_for(session, m, record_audit=False)
    return {
        "model": f"{m.model_name}@{m.model_version}",
        "gate": g.to_dict(),
        "quality_gate": qg.to_dict(),
        # Backward compatibility: `passed` keeps meaning "the promotion gate
        # passed". The combined answer is `promotion_allowed`.
        "promotion_allowed": bool(g.passed and qg.allows_promotion),
    }


@router.post("/{entry_id}/promote")
async def promote(
    entry_id: uuid.UUID,
    body: PromoteIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireApprover,
) -> dict:
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    g = _evaluate_for(m, body)
    entry_quality_gate = await get_evaluation_service().quality_gate_for(
        session, m, actor=actor, request_id=_request_id(request)
    )
    try:
        m = await svc.promote(
            session, m, gate=g, required_domain=body.required_domain, actor=actor,
            approved_by=body.approved_by, reason=body.reason, request_id=_request_id(request),
            quality_gate=entry_quality_gate,
        )
        await session.commit()
    except RegistryError as exc:
        await session.commit()  # persist the DENIED audit row written by the service
        _raise_registry_error(exc)
    return {**_out(m), "gate": g.to_dict(), "quality_gate": entry_quality_gate.to_dict()}


@router.post("/{entry_id}/archive")
async def archive(
    entry_id: uuid.UUID,
    request: Request,
    body: ArchiveIn | None = None,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireEngineer,
) -> dict:
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    body = body or ArchiveIn()
    try:
        await svc.archive(
            session, m, actor=actor, approved_by=body.approved_by, reason=body.reason,
            request_id=_request_id(request),
        )
        await session.commit()
    except RegistryError as exc:
        await session.commit()
        _raise_registry_error(exc)
    return _out(m)


@router.post("/rollback")
async def rollback(
    body: RollbackIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireApprover,
) -> dict:
    svc = get_registry_service()
    try:
        m = await svc.rollback(
            session, body.model_name, body.model_version, actor=actor,
            approved_by=body.approved_by, reason=body.reason, request_id=_request_id(request),
        )
        await session.commit()
    except RegistryError as exc:
        await session.commit()
        _raise_registry_error(exc)
    return _out(m)


@router.post("/{entry_id}/activate")
async def activate(
    entry_id: uuid.UUID,
    body: ActivateIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireAdmin,
) -> dict:
    """Pin a PRODUCTION row into the deployment manifest and the runtime env
    file the inference service reads. The artifact hash is re-verified before
    the write. The running process still needs a restart to pick it up; the
    response says so instead of pretending the swap happened."""
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    try:
        m, result = await svc.activate(
            session, m, actor=actor, approved_by=body.approved_by, reason=body.reason,
            new_stack_version=body.new_stack_version, request_id=_request_id(request),
        )
        await session.commit()
    except RegistryError as exc:
        await session.commit()
        _raise_registry_error(exc)
    return {**_out(m), "activation": result}
