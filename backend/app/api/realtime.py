from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..metrics import metrics
from ..security.auth import ROLE_ADMIN, ROLE_PIPELINE, Principal, lookup, require_any_authenticated, require_roles
from ..services.audit_service import record as audit_record
from ..ws import manager

rt_router = APIRouter(prefix="/api/v1/realtime", tags=["realtime"])
ws_router = APIRouter(prefix="/api/v1", tags=["realtime"])

# Telemetry ingestion is an internal service channel: only the orchestrator
# (pipeline role) or admin may report pipeline counters. Operator tokens
# cannot spoof them.
RequireTelemetryWrite = Depends(require_roles(ROLE_PIPELINE, ROLE_ADMIN))


@rt_router.get("/status", dependencies=[Depends(require_any_authenticated())])
async def realtime_status() -> dict:
    data = await metrics.snapshot()
    data["ws_client_count"] = await manager.total_client_count()
    return data


@rt_router.post("/telemetry")
async def update_telemetry(
    body: dict,
    request: Request,
    session: AsyncSession = Depends(get_session),
    actor: Principal = RequireTelemetryWrite,
) -> dict:
    """Internal endpoint used by the orchestrator to report the pipeline view
    (produced captures, queue depth, workers). Terminal counters
    (completed/failed/pass/review/fail) are owned by the backend, which is the
    source of truth for persisted inspections, and cannot be overridden here.
    Requires the pipeline (internal service) identity or admin.
    """
    allowed = {
        "captured_total",
        "queued_current",
        "processing_current",
        "simulator_running",
        "simulator_interval_ms",
        "worker_count",
        "queue_size",
        "queue_peak_depth",
    }
    await metrics.update_telemetry({k: v for k, v in body.items() if k in allowed})
    audit_record(
        session=session, action="telemetry.write", actor=actor, result="applied",
        resource_type="telemetry",
        detail={k: v for k, v in body.items() if k in allowed},
        request_id=request.headers.get("X-Request-ID"),
    )
    await session.commit()
    return {"status": "ok", "request_id": uuid.uuid4().hex[:12]}


@ws_router.websocket("/ws/inspections")
async def ws_inspections(websocket: WebSocket) -> None:
    """Subscribe to inspection events. The browser cannot attach an
    Authorization header, so the bearer token travels as ?token= in the
    WebSocket URL (same token as the REST API); a missing or invalid token
    closes the socket before any event is pushed."""
    token = websocket.query_params.get("token")
    principal = lookup(token) if token else None
    if principal is None:
        await websocket.close(code=4401)
        return
    await manager.connect(websocket)
    try:
        while True:
            # Receiving lets us detect disconnect promptly; payload ignored.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)
