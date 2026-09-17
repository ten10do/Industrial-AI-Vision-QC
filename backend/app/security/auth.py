"""Bearer-token authentication and role-based authorization.

Properties this module must preserve:

1. Fail-closed. An empty credential store means nobody can mutate the model
   registry. There is no "development bypass" role and no anonymous fallback.
2. Timing-safe lookup. Tokens are indexed by SHA256 so a lookup cannot leak a
   prefix through comparison time.
3. Attribution. Every protected handler receives a Principal; the principal's
   subject is what lands in the audit journal.

Roles
-----
viewer          read registry, gate dry-runs, audit trail
engineer        register model identity, archive non-production rows
pipeline        attest metrics / domain validation / artifact hash (signed
                payload); also the internal service identity for telemetry
approver        promote, rollback, archive a PRODUCTION row
admin           activate a promoted model into the runtime deployment
                manifest; acts as superuser for every business endpoint
operator        production-line operator: create inspections, read
                inspections / images / products / realtime status
reviewer        human review: claim / resolve / correct review tasks, review
                metrics, training-candidate export
release-manager quality-rule configuration and model release approvals

Separation of duties (business endpoints)
-----------------------------------------
- Every business mutation requires a bearer token. There is no anonymous
  write path.
- The human-review identity is derived from the authenticated principal's
  subject; a client-supplied `reviewer` string that does not match it is
  rejected (403), so a reviewer cannot be impersonated through the body.
- Telemetry ingestion requires the pipeline role (internal service identity)
  plus admin; an operator token cannot spoof pipeline counters.
- Read-only business endpoints accept any authenticated principal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Request

from ..config import get_settings

logger = logging.getLogger(__name__)

ROLE_VIEWER = "viewer"
ROLE_ENGINEER = "engineer"
ROLE_PIPELINE = "pipeline"
ROLE_APPROVER = "approver"
ROLE_ADMIN = "admin"
# business roles (separation of duties for the shop-floor endpoints)
ROLE_OPERATOR = "operator"
ROLE_REVIEWER = "reviewer"
ROLE_RELEASE_MANAGER = "release-manager"

ALL_ROLES = (
    ROLE_VIEWER,
    ROLE_ENGINEER,
    ROLE_PIPELINE,
    ROLE_APPROVER,
    ROLE_ADMIN,
    ROLE_OPERATOR,
    ROLE_REVIEWER,
    ROLE_RELEASE_MANAGER,
)


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]
    authn: str = "bearer"

    def has_any(self, *roles: str) -> bool:
        return bool(self.roles.intersection(roles))

    def to_dict(self) -> dict:
        return {"subject": self.subject, "roles": sorted(self.roles), "authn": self.authn}


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_store(raw: str, source: str) -> dict[str, Principal]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"IVQC_API_TOKENS from {source} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"IVQC_API_TOKENS from {source} must be a JSON object")
    store: dict[str, Principal] = {}
    for token, spec in data.items():
        if isinstance(spec, str):  # shorthand: {"token": "admin"}
            spec = {"subject": spec, "roles": [spec]}
        if not isinstance(spec, dict):
            raise RuntimeError("IVQC_API_TOKENS: each token entry must be a mapping")
        subject = str(spec.get("subject") or "unattributed")
        roles = spec.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        unknown = [r for r in roles if r not in ALL_ROLES]
        if unknown:
            raise RuntimeError(f"IVQC_API_TOKENS: unknown roles {unknown}; allowed {list(ALL_ROLES)}")
        store[_token_digest(str(token))] = Principal(
            subject=subject, roles=frozenset(roles), authn="bearer"
        )
    return store


@lru_cache(maxsize=1)
def _load_store() -> dict[str, Principal]:
    settings = get_settings()
    raw = settings.api_tokens
    source = "IVQC_API_TOKENS"
    if not raw.strip() and settings.api_tokens_file:
        path = Path(settings.api_tokens_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise RuntimeError(f"IVQC_API_TOKENS_FILE points at a missing file: {path}")
        raw = path.read_text(encoding="utf-8")
        source = f"file:{path}"
    store = _parse_store(raw, source)
    if not store:
        logger.warning(
            "no API credentials configured; every model-governance endpoint will deny (fail-closed)"
        )
    return store


def reset_auth_store() -> None:
    _load_store.cache_clear()


def lookup(token: str) -> Principal | None:
    store = _load_store()
    target = _token_digest(token)
    for digest, principal in store.items():
        if hmac.compare_digest(digest, target):
            return principal
    return None


def principal_from_request(request: Request) -> Principal | None:
    header = request.headers.get("authorization")
    if not header or not header.lower().startswith("bearer "):
        return None
    return lookup(header.split(" ", 1)[1].strip())


def require_roles(*roles: str):
    """FastAPI dependency enforcing authentication plus at least one role."""

    def dependency(request: Request) -> Principal:
        principal = principal_from_request(request)
        if principal is None:
            store_configured = bool(_load_store())
            raise HTTPException(
                status_code=401,
                detail=_err(
                    "unauthenticated" if store_configured else "auth_not_configured",
                    "a bearer token is required for this operation"
                    if store_configured
                    else "no API credentials are configured on this server; governance endpoints are closed",
                ),
            )
        if not principal.has_any(*roles):
            raise HTTPException(
                status_code=403,
                detail=_err(
                    "forbidden",
                    f"principal {principal.subject} lacks one of roles {list(roles)}",
                ),
            )
        return principal

    return dependency


def request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", "unknown")


def require_any_authenticated():
    """FastAPI dependency: any valid bearer token (no role required beyond
    authentication). Used by read-only business endpoints where every
    authenticated operator / reviewer / viewer may look."""
    return require_roles(*ALL_ROLES)
