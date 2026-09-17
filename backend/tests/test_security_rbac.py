"""P0 regression: business endpoints are fail-closed behind authentication
and role-based access control (operator / reviewer / release-manager
separation of duties).

Every request must carry a valid bearer token; a missing one answers 401.
A token whose roles do not cover the operation answers 403. The human-review
identity is derived from the authenticated principal's subject; a body
`reviewer` that differs from it is a 403 impersonation attempt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enums import QualityResult  # noqa: E402
from app.models import QualityRule  # noqa: E402

from test_inspection_api import SAMPLE_JPG, StubInference, contract, detection  # noqa: E402
from test_reviews import _task_for_inspection  # noqa: E402


async def _seed_review_rule(db_session):
    db_session.add(
        QualityRule(
            defect_type="crazing", min_confidence=0.3, max_area_ratio=1.0,
            action=QualityResult.REVIEW, severity="medium", priority=10, rule_version=1,
        )
    )
    await db_session.commit()


async def _create_review_task(client, db_session, stub_infer):
    """One REVIEW inspection -> one PENDING review task (operator can do this)."""
    await _seed_review_rule(db_session)
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.42, 0.06)])))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-RBAC-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201
    return await _task_for_inspection(db_session, resp.json()["inspection_id"])


# --- unauthenticated: every protected endpoint answers 401 (fail-closed) ---


@pytest.mark.asyncio
async def test_unauthenticated_401_on_inspection_read(client_unauthenticated):
    r = await client_unauthenticated.get("/api/v1/inspections")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


@pytest.mark.asyncio
async def test_unauthenticated_401_on_review_queue(client_unauthenticated):
    r = await client_unauthenticated.get("/api/v1/reviews")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_unauthenticated_401_on_training_candidates(client_unauthenticated):
    r = await client_unauthenticated.get("/api/v1/training-candidates")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_unauthenticated_401_on_telemetry(client_unauthenticated):
    r = await client_unauthenticated.post("/api/v1/realtime/telemetry", json={"captured_total": 1})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_unauthenticated_401_on_quality_rules_write(client_unauthenticated):
    r = await client_unauthenticated.post(
        "/api/v1/quality-rules",
        json={"defect_type": "crazing", "action": "REVIEW", "severity": "medium"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_unauthenticated_401_on_realtime_status(client_unauthenticated):
    r = await client_unauthenticated.get("/api/v1/realtime/status")
    assert r.status_code == 401


# --- separation of duties: operator cannot act as reviewer / release manager ---


@pytest.mark.asyncio
async def test_operator_forbidden_on_review_claim(client, db_session, stub_infer):
    task = await _create_review_task(client, db_session, stub_infer)
    r = await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", json={})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


@pytest.mark.asyncio
async def test_operator_forbidden_on_review_metrics(client):
    r = await client.get("/api/v1/reviews-metrics")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_operator_forbidden_on_training_candidates(client):
    r = await client.get("/api/v1/training-candidates")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_operator_forbidden_on_quality_rule_write(client):
    r = await client.post(
        "/api/v1/quality-rules",
        json={"defect_type": "crazing", "action": "REVIEW", "severity": "medium"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_operator_forbidden_on_telemetry(client):
    r = await client.post("/api/v1/realtime/telemetry", json={"captured_total": 1})
    assert r.status_code == 403


# --- reviewer identity comes from the authenticated principal ---


@pytest.mark.asyncio
async def test_claim_identity_is_authenticated_principal(client, db_session, stub_infer, auth):
    task = await _create_review_task(client, db_session, stub_infer)
    r = await client.post(f"/api/v1/reviews/{task.review_task_id}/claim", headers=auth("reviewer_a"), json={})
    assert r.status_code == 200
    assert r.json()["assigned_to"] == "tester-reviewer-a"  # from token, not body


@pytest.mark.asyncio
async def test_reviewer_mismatch_body_is_rejected(client, db_session, stub_infer, auth):
    """A body reviewer that differs from the token subject is a 403
    impersonation attempt, never a silent no-op."""
    task = await _create_review_task(client, db_session, stub_infer)
    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/claim",
        headers=auth("reviewer_a"),
        json={"reviewer": "alice"},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "reviewer_mismatch"


@pytest.mark.asyncio
async def test_matching_body_reviewer_accepted(client, db_session, stub_infer, auth):
    """Backward compatibility: a body reviewer equal to the token subject is
    accepted and still records the authenticated identity."""
    task = await _create_review_task(client, db_session, stub_infer)
    r = await client.post(
        f"/api/v1/reviews/{task.review_task_id}/claim",
        headers=auth("reviewer_a"),
        json={"reviewer": "tester-reviewer-a"},
    )
    assert r.status_code == 200
    assert r.json()["assigned_to"] == "tester-reviewer-a"


# --- positive paths: operator reads, release manager writes ---


@pytest.mark.asyncio
async def test_operator_can_read_quality_rules(client):
    r = await client.get("/api/v1/quality-rules")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_release_manager_can_write_quality_rules(client, auth):
    r = await client.post(
        "/api/v1/quality-rules",
        headers=auth("release-manager"),
        json={"defect_type": "crazing", "action": "REVIEW", "severity": "medium"},
    )
    assert r.status_code == 201


# --- fail-closed guard: every business HTTP route must carry an auth dependency ---


@pytest.mark.asyncio
async def test_all_business_http_routes_require_auth(app):
    """No business HTTP route may exist without an auth dependency. A future
    route added without `require_roles` / `require_any_authenticated` fails
    here, so exposure cannot regress silently. Infra endpoints (/health,
    /ready, /docs, /openapi) are public by design; the WebSocket route is
    authenticated in its handler via ?token= (browsers cannot attach an
    Authorization header to a WS handshake)."""
    def calls_in(node, out):
        c = getattr(node, "call", None)
        if c is not None:
            out.append(c)
        for d in getattr(node, "dependencies", []) or []:
            calls_in(d, out)

    def iter_routes(container):
        for r in container:
            if type(r).__name__ == "_IncludedRouter":
                yield from iter_routes(r.original_router.routes)
            else:
                yield r

    public_prefixes = ("/health", "/ready", "/docs", "/redoc", "/openapi.json", "/api/v1/ws/")
    missing: list[tuple[str, list[str]]] = []
    for r in iter_routes(app.routes):
        path = getattr(r, "path", "?")
        if path.startswith(public_prefixes):
            continue
        if not getattr(r, "methods", None):  # WebSocketRoute: handler-level auth
            continue
        calls: list = []
        for d in getattr(r, "dependencies", []) or []:
            calls_in(d, calls)
        calls_in(getattr(r, "dependant", None), calls)
        authed = any(
            c.__qualname__.startswith("require_roles") or "require_any_authenticated" in c.__qualname__
            for c in calls
        )
        if not authed:
            missing.append((path, sorted(getattr(r, "methods"))))
    assert missing == [], f"unprotected business routes: {missing}"
