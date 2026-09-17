"""Business operation audit (unified journal).

`record()` appends a row to audit_log inside the caller's transaction, so an
audit entry commits atomically with the business write it describes. Denied
operations are recorded with result="denied" so the journal shows attempts as
well as outcomes. Authentication failures (401/403 raised by the auth
dependency) never reach a session and are covered by the structured request
log instead; this journal starts where a principal is known.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import AuditLog
from ..security.auth import Principal


def record(
    session: AsyncSession,
    *,
    action: str,
    actor: Principal | None,
    result: str = "applied",
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> AuditLog:
    """Append an audit row to the open transaction (caller commits)."""
    entry = AuditLog(
        action=action,
        result=result,
        actor=actor.subject if actor else None,
        actor_roles=sorted(actor.roles) if actor else None,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        detail=detail,
        request_id=request_id or "unknown",
    )
    session.add(entry)
    return entry


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
