from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from simulator.camera_simulator import CameraSimulator  # noqa: E402
from simulator.config import OrchestratorConfig, SimulatorConfig  # noqa: E402
from simulator.orchestrator import InspectionOrchestrator  # noqa: E402

IMG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\xff\xd9"

# P0: the backend is fail-closed, so every pipeline run must present bearer
# credentials. Tests use dedicated operator / pipeline identities.
OP_TOKEN = "test-operator-token"
PIPE_TOKEN = "test-pipeline-token"


def cfg(**overrides) -> OrchestratorConfig:
    base = dict(
        backend_url="http://test",
        api_token=OP_TOKEN,
        pipeline_token=PIPE_TOKEN,
    )
    base.update(overrides)
    return OrchestratorConfig(**base)


def capture(product: str = "P-000001", key: str = "cap-1") -> dict:
    return {
        "product_id": product,
        "capture_id": key,
        "timestamp": "2026-08-04T00:00:00Z",
        "image_bytes": IMG,
        "filename": "img.jpg",
        "production_line": "line-a",
        "station": "qc-01",
        "batch_id": "batch-1",
    }


def make_capture(product: str = "P-000001", key: str = "cap-1"):
    from simulator.camera_simulator import Capture

    return Capture(**capture(product, key))


def make_src(tmp_path, count: int = 3):
    for i in range(count):
        (tmp_path / f"img_{i}.jpg").write_bytes(IMG)
    return tmp_path


async def run_orchestrator(transport, cfg: OrchestratorConfig, src, max_images=3):
    client = httpx.AsyncClient(transport=transport, timeout=10)
    orch = InspectionOrchestrator(cfg, client=client)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), orch.queue)
    await orch.run(sim, max_images=max_images)
    return orch


def ok_handler(body=None):
    body = body or {
        "inspection_id": "insp-x",
        "product_id": "P",
        "status": "completed",
        "quality_result": "REVIEW",
        "severity": "medium",
        "defects": [],
        "inference_latency_ms": 11.0,
        "model_version": "phase1-baseline",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(201, json=body)

    return handler


@pytest.mark.asyncio
async def test_orchestrator_success_flow(tmp_path):
    src = make_src(tmp_path, 3)
    orch_cfg = cfg(workers=1, retry_max=0, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(ok_handler()), orch_cfg, src, max_images=3)
    m = orch.metrics
    assert m.captured_total == 3
    assert m.completed_total == 3
    assert m.failed_total == 0
    assert m.review_total == 3
    assert len(m.e2e_latencies) == 3
    assert len(m.inference_latencies) == 3


@pytest.mark.asyncio
async def test_orchestrator_retry_then_success(tmp_path):
    """Two transient 500s then success: retry_max=2 recovers, processed counted once."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(500, json={"error": {"code": "boom"}})
        return httpx.Response(201, json={
            "inspection_id": "insp-x", "product_id": "P", "status": "completed",
            "quality_result": "PASS", "severity": "low", "defects": [], "inference_latency_ms": 9.0,
        })

    src = make_src(tmp_path, 1)
    orch_cfg = cfg(workers=1, retry_max=2, retry_base_ms=10, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), orch_cfg, src, max_images=1)
    assert calls["n"] == 3  # 2 failed attempts + 1 success
    assert orch.metrics.completed_total == 1
    assert orch.metrics.failed_total == 0


@pytest.mark.asyncio
async def test_orchestrator_no_infinite_retry(tmp_path):
    """Persistent 500: bounded retries, capture marked failed, system continues."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        return httpx.Response(500, json={"error": {"code": "boom"}})

    src = make_src(tmp_path, 3)
    orch_cfg = cfg(workers=1, retry_max=2, retry_base_ms=10, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), orch_cfg, src, max_images=3)
    assert orch.metrics.failed_total == 3
    assert orch.metrics.completed_total == 0
    assert calls["n"] == 3 * 3  # 3 captures x (1 + 2 retries)


@pytest.mark.asyncio
async def test_orchestrator_timeout_retry_idempotent(tmp_path):
    """Timeout on first attempt then success: idempotency key means the retried
    POST returns the SAME inspection; no duplicate counting."""
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={
            "inspection_id": "insp-same", "product_id": "P", "status": "completed",
            "quality_result": "FAIL", "severity": "high", "defects": [], "inference_latency_ms": 10.0,
        })

    src = make_src(tmp_path, 1)
    orch_cfg = cfg(workers=1, retry_max=2, retry_base_ms=10, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), orch_cfg, src, max_images=1)
    assert calls["n"] == 2
    assert orch.metrics.completed_total == 1  # one logical inspection despite 2 HTTP calls
    assert orch.metrics.fail_total == 1


@pytest.mark.asyncio
async def test_failed_persisted_inspection_not_counted_as_success(tmp_path):
    """Retry after a persisted FAILED record (idempotent replay returns 200 +
    status FAILED) must be treated as a terminal failure, never a success."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(504, json={"error": {"code": "inference_failed"}})
        return httpx.Response(200, json={
            "inspection_id": "insp-failed", "product_id": "P", "status": "FAILED",
            "quality_result": None, "severity": None, "defects": [], "error_message": "inference down",
        })

    src = make_src(tmp_path, 1)
    orch_cfg = cfg(workers=1, retry_max=2, retry_base_ms=10, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), orch_cfg, src, max_images=1)
    assert calls["n"] == 2  # 504 then idempotent replay, no further retry
    assert orch.metrics.failed_total == 1
    assert orch.metrics.completed_total == 0


@pytest.mark.asyncio
async def test_conservation_law_during_run(tmp_path):
    """Running invariant captured == queued + processing + completed + failed
    must hold at every sample point; drained invariant and the quality-sum
    invariant must hold at the end."""
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        await asyncio.sleep(0.05)  # slow consumer -> queue builds up
        return httpx.Response(201, json={
            "inspection_id": "insp-x", "product_id": "P", "status": "completed",
            "quality_result": "REVIEW", "severity": "medium", "defects": [], "inference_latency_ms": 8.0,
        })

    src = make_src(tmp_path, 10)
    orch_cfg = cfg(workers=1, retry_max=0, telemetry_interval_seconds=999, queue_size=5)
    client = httpx.AsyncClient(transport=httpx.MockTransport(slow_handler), timeout=10)
    orch = InspectionOrchestrator(orch_cfg, client=client)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=5, loop=False), orch.queue)
    task = asyncio.create_task(orch.run(sim, max_images=10))

    saw_running = False
    for _ in range(60):
        await asyncio.sleep(0.05)
        if sim.captured_count > 0 and not task.done():
            orch.metrics.captured_total = sim.captured_count
            orch.metrics.update_flow(orch.queue.qsize(), orch.processing_count)
            # sampling is not atomic; two bounded windows can skew by +/-1:
            #  +1 the simulator increments its counter just before the queue
            #     put lands (produced but not yet queued)
            #  -1 a worker dequeued but has not yet incremented processing
            #     (the item is counted in both queued and processing reads)
            # the invariant is exact at quiescent points and in the drained
            # state, which is asserted below.
            skew = orch.metrics.captured_total - (
                orch.metrics.queued_current + orch.metrics.processing_current
                + orch.metrics.completed_total + orch.metrics.failed_total
            )
            assert -1 <= skew <= 1, f"running conservation broken, skew={skew}"
            saw_running = True
    await task

    m = orch.metrics
    assert saw_running
    assert m.captured_total == 10
    assert m.captured_total == m.completed_total + m.failed_total, "drained conservation broken"
    assert m.pass_total + m.review_total + m.fail_total == m.completed_total, "quality sum broken"
    await client.aclose()


@pytest.mark.asyncio
async def test_queue_bounded_and_no_drop(tmp_path):
    src = make_src(tmp_path, 5)
    orch_cfg = cfg(workers=0, retry_max=0, telemetry_interval_seconds=999)
    orch = InspectionOrchestrator(orch_cfg)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), orch.queue)
    sim.start()
    await asyncio.sleep(0.3)
    assert orch.queue.qsize() <= orch_cfg.queue_size, "queue must never exceed maxsize"
    assert sim.captured_count <= orch_cfg.queue_size  # producer blocked at capacity
    await sim.stop()


@pytest.mark.asyncio
async def test_simulator_exhaustion_stops_pipeline(tmp_path):
    """loop=False + source exhausted -> pipeline terminates with all captures accounted."""
    src = make_src(tmp_path, 2)
    orch_cfg = cfg(workers=1, retry_max=0, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(ok_handler()), orch_cfg, src, max_images=None)
    m = orch.metrics
    assert m.captured_total == 2
    assert m.completed_total == 2
    assert m.failed_total == 0


@pytest.mark.asyncio
async def test_orchestrator_attaches_bearer_tokens(tmp_path):
    """P0: inspection POSTs carry the operator bearer token; the telemetry
    push carries the pipeline (internal service) bearer token. No request may
    leave the orchestrator without an Authorization header."""
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(201, json={
            "inspection_id": "insp-x", "product_id": "P", "status": "completed",
            "quality_result": "PASS", "severity": "low", "defects": [], "inference_latency_ms": 9.0,
        })

    src = make_src(tmp_path, 2)
    orch_cfg = cfg(workers=1, retry_max=0, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), orch_cfg, src, max_images=2)
    assert orch.metrics.completed_total == 2

    inspection_headers = [h for p, h in seen if not p.endswith("/realtime/telemetry")]
    assert inspection_headers, "no inspection POST observed"
    assert all(h == f"Bearer {OP_TOKEN}" for h in inspection_headers), inspection_headers

    # telemetry interval is 999s, so the loop never fired during run(); call
    # the one-shot push directly to verify the pipeline identity header.
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10)
    orch2 = InspectionOrchestrator(orch_cfg, client=client)
    await orch2._push_telemetry()
    await client.aclose()
    telemetry_headers = [h for p, h in seen if p.endswith("/realtime/telemetry")]
    assert telemetry_headers, "no telemetry POST observed"
    assert all(h == f"Bearer {PIPE_TOKEN}" for h in telemetry_headers), telemetry_headers


@pytest.mark.asyncio
async def test_orchestrator_fails_fast_without_tokens(tmp_path):
    """P0: the backend is fail-closed; a pipeline run without credentials must
    refuse to start instead of silently failing every capture."""
    src = make_src(tmp_path, 2)
    orch_cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=0,
                                  telemetry_interval_seconds=999)  # no tokens
    orch = InspectionOrchestrator(orch_cfg)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), orch.queue)
    with pytest.raises(RuntimeError, match="fail-closed"):
        await orch.run(sim, max_images=2)
