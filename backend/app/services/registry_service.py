"""Model Registry service (8A / 8C) with a hard governance boundary.

Lifecycle: CANDIDATE -> STAGING -> PRODUCTION -> ARCHIVED. At most one
PRODUCTION row per model_name (enforced by a partial unique index). Rollback
switches the registry pointer, it never rebuilds the system (8L).

Trust model (the reason this module looks the way it does)
----------------------------------------------------------
The service layer is reachable only from server code, so it may write
privileged facts, but it must still *prove* them: every artifact hash is
recomputed from the artifact, and every domain verdict must be backed by an
eval report this server has hashed. The HTTP layer is untrusted and can only
reach privileged fields through the signed attestation path.

Every mutation and every refusal appends to model_registry_audit. Nothing in
this module deletes a row: archival is the terminal state, which is what keeps
the audit trail meaningful.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..mlops.attestation import verify_artifact_hash, verify_domain_evidence
from ..mlops.deployment_sync import ActivationError, apply_activation
from ..mlops.gate_policy import get_policy
from ..mlops.promotion_gate import GateResult, default_provenance
from ..models import ModelRegistry, ModelRegistryAudit
from ..security.auth import ROLE_ADMIN, ROLE_APPROVER, ROLE_PIPELINE, Principal

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..mlops.quality_gate import QualityGateResult

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


# Codes the API layer answers with 403 (authorization / approval shape) rather
# than 422 (business rule).
APPROVAL_ERROR_CODES = frozenset(
    {"approval_required", "self_approval_forbidden", "reason_required", "insufficient_role"}
)


def provenance_for(entry: ModelRegistry) -> dict:
    """Build the provenance block the promotion gate enforces."""
    return {
        **default_provenance(),
        "metrics_attested": bool(entry.attested_by),
        "attested_by": entry.attested_by,
        "attested_at": entry.attested_at.isoformat() if entry.attested_at else None,
        "artifact_hash_verified": bool(entry.artifact_hash_verified),
        "domain_evidence_verified": bool(entry.domain_evidence_verified),
        "domain": (entry.domain_evidence or {}).get("domain"),
    }


def _approval_rules() -> dict:
    try:
        return dict(get_policy().approval or {})
    except Exception:  # noqa: BLE001 - a policy failure is surfaced by the gate
        return {}


def _validate_approval(actor: Principal, approved_by: str | None, reason: str | None, action: str) -> None:
    rules = _approval_rules()
    forbid_self = bool(rules.get("forbid_self_approval", True))
    min_reason = int(rules.get("min_reason_chars", 8) or 0)
    if not approved_by or not str(approved_by).strip():
        raise RegistryError("approval_required", f"{action} requires an approved_by approver identity")
    if forbid_self and str(approved_by).strip() == actor.subject:
        raise RegistryError("self_approval_forbidden", f"{action} cannot be self-approved by {actor.subject}")
    if not reason or len(str(reason).strip()) < min_reason:
        raise RegistryError("reason_required", f"{action} requires a reason of at least {min_reason} characters")


class RegistryService:
    # ---- audit ----

    async def audit(
        self,
        session: AsyncSession,
        *,
        action: str,
        outcome: str,
        entry: ModelRegistry | None = None,
        actor: Principal | None = None,
        approved_by: str | None = None,
        reason: str | None = None,
        gate: GateResult | dict | None = None,
        payload: dict | None = None,
        from_status: str | None = None,
        to_status: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> ModelRegistryAudit:
        gate_dict = gate.to_dict() if isinstance(gate, GateResult) else gate
        row = ModelRegistryAudit(
            registry_id=entry.id if entry is not None else None,
            model_name=model_name or (entry.model_name if entry else None),
            model_version=model_version or (entry.model_version if entry else None),
            action=action,
            outcome=outcome,
            from_status=from_status or (entry.status if entry else None),
            to_status=to_status,
            actor=actor.subject if actor else None,
            actor_roles=sorted(actor.roles) if actor else None,
            approved_by=approved_by,
            reason=reason,
            gate=gate_dict,
            payload=payload,
            request_id=request_id,
        )
        session.add(row)
        await session.flush()
        return row

    # ---- registration ----

    async def register_identity(
        self,
        session: AsyncSession,
        *,
        actor: Principal,
        model_name: str,
        model_version: str,
        model_type: str,
        artifact_uri: str | None = None,
        dataset_version: str | None = None,
        training_run_id: str | None = None,
        notes: str | None = None,
        request_id: str | None = None,
    ) -> ModelRegistry:
        """Untrusted registration: identity only. No metrics, no domain
        verdict, no artifact hash. Those arrive via ``attest``."""
        dup = await session.execute(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.model_version == model_version,
            )
        )
        if dup.scalar_one_or_none() is not None:
            raise RegistryError("duplicate_version", f"{model_name}@{model_version} already registered")

        entry = ModelRegistry(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            artifact_uri=artifact_uri,
            artifact_sha256=None,
            dataset_version=dataset_version,
            training_run_id=training_run_id,
            status="CANDIDATE",
            metadata_json=None,
            domain_validated=False,
            notes=notes,
        )
        session.add(entry)
        await session.flush()
        await self.audit(
            session, action="register", outcome="APPLIED", entry=entry, actor=actor,
            payload={"model_type": model_type, "artifact_uri": artifact_uri,
                     "dataset_version": dataset_version, "training_run_id": training_run_id},
            request_id=request_id,
        )
        return entry

    async def attest(
        self,
        session: AsyncSession,
        entry: ModelRegistry,
        *,
        actor: Principal,
        metrics: dict | None,
        domain_validated: bool,
        domain_evidence: dict | None = None,
        artifact_sha256: str | None = None,
        attestation_digest: str | None = None,
        request_id: str | None = None,
    ) -> ModelRegistry:
        """Write privileged facts. The caller must already hold the pipeline
        role and a valid signature; here every claim is re-verified."""
        if not actor.has_any(ROLE_PIPELINE, ROLE_ADMIN):
            raise RegistryError(
                "insufficient_role",
                f"principal {actor.subject} may not attest metrics/domain/artifact facts",
            )

        hash_check = verify_artifact_hash(entry.artifact_uri, artifact_sha256 or entry.artifact_sha256)
        if artifact_sha256:
            entry.artifact_sha256 = artifact_sha256
        entry.artifact_hash_verified = hash_check.verified

        if domain_validated:
            evidence_check = verify_domain_evidence(domain_evidence, str((domain_evidence or {}).get("domain") or ""))
        else:
            evidence_check = verify_domain_evidence(None, "")
        entry.domain_evidence = domain_evidence or None
        entry.domain_evidence_verified = evidence_check.verified if domain_validated else False
        entry.domain_validated = bool(domain_validated)

        entry.metadata_json = dict(metrics or {})
        entry.attested_by = actor.subject
        entry.attested_at = datetime.now(timezone.utc)
        entry.attestation_digest = attestation_digest
        await session.flush()

        await self.audit(
            session, action="attest", outcome="APPLIED", entry=entry, actor=actor,
            payload={
                "metrics": dict(metrics or {}),
                "domain_validated": bool(domain_validated),
                "artifact_verification": hash_check.to_dict(),
                "domain_evidence_verification": evidence_check.to_dict(),
                "attestation_digest": attestation_digest,
            },
            request_id=request_id,
        )
        if not hash_check.verified:
            logger.warning("attestation recorded with UNVERIFIED artifact hash for %s@%s: %s",
                           entry.model_name, entry.model_version, hash_check.detail)
        return entry

    async def register(
        self,
        session: AsyncSession,
        *,
        actor: Principal,
        model_name: str,
        model_version: str,
        model_type: str,
        artifact_uri: str | None = None,
        artifact_sha256: str | None = None,
        dataset_version: str | None = None,
        training_run_id: str | None = None,
        metrics: dict | None = None,
        domain_validated: bool = False,
        domain_evidence: dict | None = None,
        notes: str | None = None,
        request_id: str | None = None,
    ) -> ModelRegistry:
        """In-process trusted entry point (training/eval scripts).

        Convenience wrapper over register_identity + attest. It performs the
        same verification, so a caller cannot smuggle an unverified hash in
        here either; it would simply be recorded as unverified and blocked by
        the gate.
        """
        entry = await self.register_identity(
            session, actor=actor, model_name=model_name, model_version=model_version,
            model_type=model_type, artifact_uri=artifact_uri, dataset_version=dataset_version,
            training_run_id=training_run_id, notes=notes, request_id=request_id,
        )
        if metrics is not None or domain_validated or artifact_sha256:
            entry = await self.attest(
                session, entry, actor=actor, metrics=metrics, domain_validated=domain_validated,
                domain_evidence=domain_evidence, artifact_sha256=artifact_sha256, request_id=request_id,
            )
        return entry

    # ---- reads ----

    async def list(self, session: AsyncSession, *, status: str | None = None,
                   model_type: str | None = None) -> list[ModelRegistry]:
        stmt = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
        if status:
            stmt = stmt.where(ModelRegistry.status == status)
        if model_type:
            stmt = stmt.where(ModelRegistry.model_type == model_type)
        return list((await session.execute(stmt)).scalars().all())

    async def get(self, session: AsyncSession, entry_id: uuid.UUID) -> ModelRegistry | None:
        return await session.get(ModelRegistry, entry_id)

    async def get_by_version(self, session: AsyncSession, model_name: str, model_version: str) -> ModelRegistry | None:
        r = await session.execute(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.model_version == model_version,
            )
        )
        return r.scalar_one_or_none()

    async def get_production(self, session: AsyncSession, model_name: str) -> ModelRegistry | None:
        r = await session.execute(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.status == "PRODUCTION",
            )
        )
        return r.scalar_one_or_none()

    async def production_by_channel(self, session: AsyncSession) -> dict[str, ModelRegistry]:
        rows = (await session.execute(
            select(ModelRegistry).where(ModelRegistry.status == "PRODUCTION")
        )).scalars().all()
        out: dict[str, ModelRegistry] = {}
        for row in rows:
            out.setdefault(row.model_type, row)
        return out

    async def audit_trail(self, session: AsyncSession, entry_id: uuid.UUID, limit: int = 100) -> list[ModelRegistryAudit]:
        rows = await session.execute(
            select(ModelRegistryAudit)
            .where(ModelRegistryAudit.registry_id == entry_id)
            .order_by(ModelRegistryAudit.created_at.asc())
            .limit(limit)
        )
        return list(rows.scalars().all())

    # ---- lifecycle ----

    async def promote(
        self,
        session: AsyncSession,
        entry: ModelRegistry,
        *,
        gate: GateResult,
        required_domain: str,
        actor: Principal,
        approved_by: str | None = None,
        reason: str | None = None,
        request_id: str | None = None,
        quality_gate: "QualityGateResult | None" = None,
    ) -> ModelRegistry:
        """Promote to PRODUCTION.

        Three independent conditions must hold, and none of them can be waived
        by the caller:

        1. the server-evaluated promotion gate passes (attested metrics,
           verified artifact hash, verified domain evidence);
        2. the evaluation quality gate does not HOLD (see
           ``app.mlops.quality_gate``). It is computed here when the caller
           does not supply one, so there is no code path that reaches
           PRODUCTION without it;
        3. a human approver, who is not the caller, signs off with a reason.

        A HOLD from the quality gate is a hard safety gate. Human approval
        cannot bypass it: the approval is validated *after* the gate, and a
        held gate raises before any approval is honoured.
        """
        _validate_approval(actor, approved_by, reason, "promote")

        if quality_gate is None:
            from .evaluation_service import get_evaluation_service

            quality_gate = await get_evaluation_service().quality_gate_for(
                session, entry, actor=actor, request_id=request_id
            )

        if quality_gate.held:
            await self.audit(
                session, action="promote", outcome="DENIED", entry=entry, actor=actor,
                approved_by=approved_by, reason=reason, gate=gate,
                payload={
                    "required_domain": required_domain,
                    "block": "quality_gate_hold",
                    "quality_gate": quality_gate.to_dict(),
                },
                request_id=request_id,
            )
            raise RegistryError(
                "quality_gate_held",
                "quality gate HOLD: " + "; ".join(
                    f"{rule['rule']} ({rule['message']})" for rule in quality_gate.failed_rules
                ) or "quality gate HOLD",
            )

        if not gate.passed:
            await self.audit(
                session, action="promote", outcome="DENIED", entry=entry, actor=actor,
                approved_by=approved_by, reason=reason, gate=gate,
                payload={"required_domain": required_domain}, request_id=request_id,
            )
            raise RegistryError("promotion_gate_failed", "; ".join(gate.blocked or ["gate failed"]))
        if entry.status == "ARCHIVED":
            await self.audit(
                session, action="promote", outcome="DENIED", entry=entry, actor=actor,
                approved_by=approved_by, reason=reason, gate=gate,
                payload={"required_domain": required_domain, "block": "archived"}, request_id=request_id,
            )
            raise RegistryError("bad_state", "cannot promote an ARCHIVED model")
        if not entry.domain_validated:
            await self.audit(
                session, action="promote", outcome="DENIED", entry=entry, actor=actor,
                approved_by=approved_by, reason=reason, gate=gate,
                payload={"required_domain": required_domain, "block": "domain"}, request_id=request_id,
            )
            raise RegistryError("domain_mismatch", f"model not validated for domain {required_domain}")

        now = datetime.now(timezone.utc)
        previous = entry.status
        await session.execute(
            update(ModelRegistry)
            .where(ModelRegistry.model_name == entry.model_name, ModelRegistry.status == "PRODUCTION")
            .values(status="ARCHIVED")
        )
        entry.status = "PRODUCTION"
        entry.promoted_at = now
        entry.approved_by = approved_by
        entry.approval_reason = reason
        await session.flush()
        await self.audit(
            session, action="promote", outcome="APPLIED", entry=entry, actor=actor,
            approved_by=approved_by, reason=reason, gate=gate,
            from_status=previous, to_status="PRODUCTION",
            payload={
                "required_domain": required_domain,
                "quality_gate": quality_gate.to_dict(),
                "quality_gate_verdict": quality_gate.verdict,
            },
            request_id=request_id,
        )
        return entry

    async def rollback(
        self,
        session: AsyncSession,
        model_name: str,
        model_version: str,
        *,
        actor: Principal,
        approved_by: str | None = None,
        reason: str | None = None,
        request_id: str | None = None,
    ) -> ModelRegistry:
        """Rollback: switch the production pointer back to `model_version`
        without rebuilding anything (8L).

        The target must have been PRODUCTION before. This closes the path where
        an arbitrary CANDIDATE could be made production by calling rollback,
        which skipped the gate entirely.
        """
        _validate_approval(actor, approved_by, reason, "rollback")

        target = await self.get_by_version(session, model_name, model_version)

        def _deny(code: str, message: str) -> RegistryError:
            return RegistryError(code, message)

        if target is None:
            await self.audit(
                session, action="rollback", outcome="DENIED", actor=actor, approved_by=approved_by,
                reason=reason, payload={"model_name": model_name, "model_version": model_version},
                model_name=model_name, model_version=model_version, request_id=request_id,
            )
            raise _deny("not_found", f"{model_name}@{model_version} not in registry")
        if target.status == "PRODUCTION":
            await self.audit(
                session, action="rollback", outcome="DENIED", entry=target, actor=actor,
                approved_by=approved_by, reason=reason,
                payload={"block": "target_is_current_production"}, request_id=request_id,
            )
            raise _deny("rollback_target_is_current_production",
                        f"{model_name}@{model_version} is already the production model")
        if target.promoted_at is None:
            await self.audit(
                session, action="rollback", outcome="DENIED", entry=target, actor=actor,
                approved_by=approved_by, reason=reason,
                payload={"block": "target_never_promoted"}, request_id=request_id,
            )
            raise _deny(
                "rollback_target_never_promoted",
                f"{model_name}@{model_version} has never passed promotion; rollback cannot bypass the gate",
            )

        now = datetime.now(timezone.utc)
        previous_status = target.status
        await session.execute(
            update(ModelRegistry)
            .where(ModelRegistry.model_name == model_name, ModelRegistry.status == "PRODUCTION")
            .values(status="ARCHIVED")
        )
        target.status = "PRODUCTION"
        target.promoted_at = now
        target.approved_by = approved_by
        target.approval_reason = reason
        await session.flush()
        await self.audit(
            session, action="rollback", outcome="APPLIED", entry=target, actor=actor,
            approved_by=approved_by, reason=reason, from_status=previous_status, to_status="PRODUCTION",
            request_id=request_id,
        )
        return target

    async def archive(
        self,
        session: AsyncSession,
        entry: ModelRegistry,
        *,
        actor: Principal,
        approved_by: str | None = None,
        reason: str | None = None,
        request_id: str | None = None,
    ) -> ModelRegistry:
        """Archive. Archiving a PRODUCTION row needs an approver, otherwise a
        single engineer could silently unpublish the running model."""
        if entry.status == "PRODUCTION":
            if not actor.has_any(ROLE_APPROVER, ROLE_ADMIN):
                await self.audit(
                    session, action="archive", outcome="DENIED", entry=entry, actor=actor,
                    approved_by=approved_by, reason=reason,
                    payload={"block": "production_requires_approver"}, request_id=request_id,
                )
                raise RegistryError("insufficient_role", "archiving a PRODUCTION model requires the approver role")
            _validate_approval(actor, approved_by, reason, "archive")

        previous_status = entry.status
        entry.status = "ARCHIVED"
        await session.flush()
        await self.audit(
            session, action="archive", outcome="APPLIED", entry=entry, actor=actor,
            approved_by=approved_by, reason=reason, from_status=previous_status, to_status="ARCHIVED",
            request_id=request_id,
        )
        return entry

    async def activate(
        self,
        session: AsyncSession,
        entry: ModelRegistry,
        *,
        actor: Principal,
        approved_by: str | None = None,
        reason: str | None = None,
        new_stack_version: str,
        request_id: str | None = None,
    ) -> tuple[ModelRegistry, dict]:
        """Write a PRODUCTION row into the deployment manifest + runtime env.

        This is the step that closes the loop the database cannot: a
        PRODUCTION row alone never changed what the inference process loads.
        The running process still needs a restart, which is why the result
        reports requires_restart=true instead of claiming success.
        """
        _validate_approval(actor, approved_by, reason, "activate")
        try:
            result = apply_activation(entry, new_stack_version=new_stack_version)
        except ActivationError as exc:
            await self.audit(
                session, action="activate", outcome="DENIED", entry=entry, actor=actor,
                approved_by=approved_by, reason=reason,
                payload={"block": "activation_error", "detail": str(exc)}, request_id=request_id,
            )
            raise RegistryError("activation_failed", str(exc)) from exc

        entry.activated_at = datetime.now(timezone.utc)
        entry.activation_target = result["manifest_path"]
        await session.flush()
        await self.audit(
            session, action="activate", outcome="APPLIED", entry=entry, actor=actor,
            approved_by=approved_by, reason=reason, payload=result, request_id=request_id,
        )
        return entry, result


def get_registry_service() -> RegistryService:
    return RegistryService()


def policy_path_configured() -> str:
    return get_settings().promotion_policy_path
