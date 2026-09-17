from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics import metrics  # noqa: E402


@pytest.fixture(autouse=True)
async def _reset_metrics():
    await metrics.reset()
    yield
    await metrics.reset()


@pytest.mark.asyncio
async def test_metrics_snapshot_fields(client):
    await metrics.record_completed("PASS", 120.5, 11.0)
    await metrics.record_completed("REVIEW", 80.0, 10.0)
    await metrics.record_completed("FAIL", 200.0, 12.0)
    await metrics.record_failed()
    resp = await client.get("/api/v1/realtime/status")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "completed_total", "failed_total", "pass_total", "review_total", "fail_total",
        "total_inspected", "yield_rate",
        "captured_total", "queued_current", "processing_current", "queue_depth", "throughput",
        "snapshot_at", "telemetry_at",
        "current_throughput", "average_processing_latency_ms",
        "p50_latency_ms", "p95_latency_ms", "ws_client_count",
    ):
        assert key in body, f"missing metric {key}"
    # backend-side invariants (single coherent quality snapshot)
    assert body["completed_total"] == 3
    assert body["failed_total"] == 1
    assert body["total_inspected"] == 4
    assert body["pass_total"] == 1 and body["review_total"] == 1 and body["fail_total"] == 1
    assert body["pass_total"] + body["review_total"] + body["fail_total"] == body["completed_total"]
    assert body["yield_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert body["average_processing_latency_ms"] == pytest.approx(133.5, abs=0.1)
    assert body["snapshot_at"] and "T" in body["snapshot_at"]
    assert body["telemetry_at"] is None  # no telemetry pushed yet


@pytest.mark.asyncio
async def test_quality_snapshot_never_mixes_timestamps(client, auth):
    """Gate 1 regression: the quality snapshot (pass/review/fail/completed/
    failed/total_inspected) must stay internally consistent even when the
    pipeline captured_total lags the DB counters (telemetry interval).
    This prevents states like 'Total Inspected=2840 but PASS+REVIEW+FAIL=2843'
    appearing in one quality snapshot."""
    for i in range(2840):
        q = "PASS" if i % 3 == 0 else ("REVIEW" if i % 3 == 1 else "FAIL")
        await metrics.record_completed(q, 60.0, 12.0)
    await client.post("/api/v1/realtime/telemetry", headers=auth("pipeline"), json={
        "captured_total": 2800, "queued_current": 0, "processing_current": 0,
        "simulator_running": True, "worker_count": 2, "queue_size": 20,
    })
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["completed_total"] == 2840
    assert body["total_inspected"] == 2840
    assert body["pass_total"] + body["review_total"] + body["fail_total"] == body["completed_total"]
    assert body["total_inspected"] == body["completed_total"] + body["failed_total"]
    # captured is a separate runtime metric, never folded into total_inspected
    assert body["captured_total"] == 2800
    assert body["total_inspected"] != body["captured_total"]
    assert body["snapshot_at"] is not None and body["telemetry_at"] is not None


@pytest.mark.asyncio
async def test_metric_invariants_via_telemetry(client, auth):
    """Quality facts: pass + review + fail == completed and
    total_inspected == completed + failed. Terminal counters come from the
    backend (source of truth); pipeline captured/queued/processing come from
    telemetry and are never mixed into the quality conservation."""
    for _ in range(9):
        await metrics.record_completed("PASS", 100.0, 10.0)
    for _ in range(12):
        await metrics.record_completed("REVIEW", 100.0, 10.0)
    for _ in range(6):
        await metrics.record_completed("FAIL", 100.0, 10.0)
    for _ in range(3):
        await metrics.record_failed()

    await client.post("/api/v1/realtime/telemetry", headers=auth("pipeline"), json={
        "captured_total": 30, "queued_current": 0, "processing_current": 0,
        "simulator_running": False, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 9,
    })
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["completed_total"] == 27 and body["failed_total"] == 3
    assert body["total_inspected"] == 30
    assert body["pass_total"] + body["review_total"] + body["fail_total"] == body["completed_total"]
    assert body["captured_total"] == 30  # happens to equal total; still independent


@pytest.mark.asyncio
async def test_running_telemetry_fields(client, auth):
    """Runtime telemetry is reported independently with its own timestamp."""
    for _ in range(28):
        await metrics.record_completed("PASS", 100.0, 10.0)
    for _ in range(5):
        await metrics.record_failed()
    await client.post("/api/v1/realtime/telemetry", headers=auth("pipeline"), json={
        "captured_total": 40, "queued_current": 5, "processing_current": 2,
        "simulator_running": True, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 8,
    })
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["captured_total"] == 40
    assert body["queued_current"] == 5 and body["queue_depth"] == 5
    assert body["processing_current"] == 2
    assert body["simulator_running"] is True
    assert body["telemetry_at"] is not None


@pytest.mark.asyncio
async def test_telemetry_update(client, auth):
    resp = await client.post("/api/v1/realtime/telemetry", headers=auth("pipeline"), json={
        "captured_total": 42, "queued_current": 7, "processing_current": 2,
        "simulator_running": True, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 9,
    })
    assert resp.status_code == 200
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["captured_total"] == 42
    assert body["queued_current"] == 7
    assert body["worker_count"] == 2
    assert body["queue_peak_depth"] == 9
    # telemetry_at must be a wall-clock ISO timestamp (regression: monotonic
    # seconds were once fed to fromtimestamp -> 1970 garbage)
    assert body["telemetry_at"] is not None and body["telemetry_at"].startswith("20")
    assert body["snapshot_at"] is not None and body["snapshot_at"].startswith("20")
