"""Inspection orchestrator: bounded queue + workers pushing captures into the
backend over HTTP.

Constraints:
- The simulator never calls the model, PostgreSQL or the rule engine directly.
  Every capture enters the system through the existing Backend API.
- Backpressure: a bounded asyncio.Queue; the producer blocks when full
  (policy "block"), so no image is ever silently dropped.
- Limited retry with exponential backoff for transient HTTP errors
  (max retry_max, no infinite retry). Idempotency keys (capture_id) prevent
  duplicate inspections when a retried request hit the backend successfully.
- Auth (P0): the backend is fail-closed, so the orchestrator presents an
  operator bearer token for inspection creation and a pipeline bearer token
  (internal service identity) for telemetry. run() refuses to start without
  both; a missing token must fail loudly, never silently drop telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from statistics import median

import httpx

from .camera_simulator import Capture, CameraSimulator
from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorMetrics:
    """Canonical pipeline counters (Phase 4 metric semantics).

    Conservation law (running):
        captured_total == queued_current + processing_current + completed_total + failed_total
    After drain (queued == processing == 0):
        captured_total == completed_total + failed_total
    And: pass_total + review_total + fail_total == completed_total

    failed_total counts SYSTEM processing failures (inference down, backend
    error, retries exhausted). fail_total counts PRODUCT quality FAIL after a
    completed inspection. The two must never be mixed.
    """

    captured_total: int = 0
    queued_current: int = 0
    processing_current: int = 0
    completed_total: int = 0
    failed_total: int = 0
    pass_total: int = 0
    review_total: int = 0
    fail_total: int = 0
    queue_peak_depth: int = 0
    e2e_latencies: list[float] = field(default_factory=list)
    inference_latencies: list[float] = field(default_factory=list)

    def update_flow(self, queue_depth: int, processing: int) -> None:
        self.queued_current = queue_depth
        self.processing_current = processing

    def conservation_ok(self) -> bool:
        return (
            self.captured_total
            == self.queued_current + self.processing_current + self.completed_total + self.failed_total
        )

    def snapshot(self) -> dict:
        e2e = self.e2e_latencies
        inf = self.inference_latencies
        return {
            "captured_total": self.captured_total,
            "queued_current": self.queued_current,
            "processing_current": self.processing_current,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "pass_total": self.pass_total,
            "review_total": self.review_total,
            "fail_total": self.fail_total,
            "queue_peak_depth": self.queue_peak_depth,
            "e2e_avg_ms": round(sum(e2e) / len(e2e), 2) if e2e else None,
            "e2e_p50_ms": round(median(e2e), 2) if e2e else None,
            "e2e_p95_ms": round(sorted(e2e)[max(0, int(len(e2e) * 0.95) - 1)], 2) if e2e else None,
            "inference_avg_ms": round(sum(inf) / len(inf), 2) if inf else None,
        }


class InspectionOrchestrator:
    def __init__(self, config: OrchestratorConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self.metrics = OrchestratorMetrics()
        self._workers: list[asyncio.Task] = []
        self._processing = 0
        self._client = client or httpx.AsyncClient(timeout=config.request_timeout_seconds)
        self._lock = asyncio.Lock()
        self._started_at = 0.0
        self._simulator: CameraSimulator | None = None

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def processing_count(self) -> int:
        return self._processing

    async def run(self, simulator: CameraSimulator, max_images: int | None = None) -> None:
        if not self.config.api_token or not self.config.pipeline_token:
            raise RuntimeError(
                "the backend is fail-closed (P0): set IVQC_API_TOKEN (operator "
                "identity) and IVQC_PIPELINE_TOKEN (pipeline identity) in "
                "OrchestratorConfig before pushing inspections"
            )
        if simulator.queue is not self.queue:
            raise RuntimeError(
                "the simulator must share the orchestrator's bounded queue "
                "(CameraSimulator(cfg, orchestrator.queue))"
            )
        if max_images is not None:
            simulator.max_captures = max_images  # precise production limit
        self._started_at = time.perf_counter()
        self._simulator = simulator
        self._sim_interval_ms = simulator.config.interval_ms
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.config.workers)]
        telemetry_task = asyncio.create_task(self._telemetry_loop())
        try:
            simulator.start()
            while True:
                await asyncio.sleep(0.2)
                if simulator.state.value == "stopped" and self.queue.empty():
                    break
        finally:
            await simulator.stop()
            # graceful shutdown: sentinels let workers finish in-flight items,
            # so captured == processed + failed holds exactly
            for _ in self._workers:
                await self.queue.put(None)
            for w in self._workers:
                try:
                    await w
                except asyncio.CancelledError:
                    pass
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError:
                pass
            # final drained snapshot: captured == completed + failed
            async with self._lock:
                self.metrics.captured_total = simulator.captured_count
                self.metrics.update_flow(0, 0)
                if not self.metrics.conservation_ok():
                    logger.warning(
                        "final metric conservation broken: captured=%d completed=%d failed=%d",
                        self.metrics.captured_total, self.metrics.completed_total, self.metrics.failed_total,
                    )
            await self._client.aclose()

    async def _worker(self, index: int) -> None:
        while True:
            capture = await self.queue.get()
            if capture is None:  # sentinel: no more work
                self.queue.task_done()
                return
            async with self._lock:
                self._processing += 1
            started = time.perf_counter()
            try:
                await self._submit_with_retry(capture)
                elapsed = (time.perf_counter() - started) * 1000.0
                self.metrics.e2e_latencies.append(elapsed)
            except Exception as exc:
                logger.warning("worker %d failed capture %s: %s", index, capture.capture_id, exc)
                async with self._lock:
                    self.metrics.failed_total += 1
            finally:
                async with self._lock:
                    self._processing -= 1
                self.queue.task_done()
                self.metrics.queue_peak_depth = max(self.metrics.queue_peak_depth, self.queue.qsize())

    async def _submit_with_retry(self, capture: Capture) -> None:
        last_error: Exception | None = None
        for attempt in range(self.config.retry_max + 1):
            if attempt > 0:
                await asyncio.sleep(self.config.retry_base_ms / 1000.0 * (2 ** (attempt - 1)))
            try:
                await self._submit(capture)
                return
            except _CaptureFailed:
                raise  # terminal: inspection already persisted as FAILED, never retry
            except Exception as exc:
                last_error = exc
                retryable = isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)) or (
                    isinstance(exc, _BackendError) and exc.status_code >= 500
                )
                if not retryable or attempt == self.config.retry_max:
                    raise
        if last_error:
            raise last_error

    async def _submit(self, capture: Capture) -> None:
        files = {"file": (capture.filename, capture.image_bytes, "application/octet-stream")}
        data = {
            "product_id": capture.product_id,
            "production_line": capture.production_line,
            "station": capture.station,
            "batch_id": capture.batch_id,
            "idempotency_key": capture.capture_id,
        }
        try:
            response = await self._client.post(
                f"{self.config.backend_url}/api/v1/inspections",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {self.config.api_token}"},
            )
        except httpx.TimeoutException as exc:
            raise  # retryable
        except httpx.HTTPError as exc:
            raise _BackendError(0, f"backend unreachable: {exc}") from exc
        if response.status_code not in (200, 201):
            raise _BackendError(response.status_code, response.text[:200])
        body = response.json()
        # A 200 idempotent replay can carry a FAILED inspection (e.g. a prior
        # attempt hit the backend while inference was down). That is a terminal
        # outcome, not a success, and must never be retried again.
        if str(body.get("status", "")).upper() == "FAILED":
            raise _CaptureFailed(f"inspection persisted as FAILED: {body.get('inspection_id')}")
        async with self._lock:
            self.metrics.completed_total += 1
            quality = body.get("quality_result")
            if quality == "PASS":
                self.metrics.pass_total += 1
            elif quality == "REVIEW":
                self.metrics.review_total += 1
            elif quality == "FAIL":
                self.metrics.fail_total += 1
            if body.get("inference_latency_ms") is not None:
                self.metrics.inference_latencies.append(body["inference_latency_ms"])

    async def _push_telemetry(self) -> None:
        """One-shot telemetry report carrying the pipeline (internal service)
        bearer identity. Kept separate from the loop so tests can exercise the
        header deterministically."""
        await self._client.post(
            f"{self.config.backend_url}/api/v1/realtime/telemetry",
            headers={"Authorization": f"Bearer {self.config.pipeline_token}"},
            json={
                "captured_total": self.metrics.captured_total,
                "queued_current": self.metrics.queued_current,
                "processing_current": self.metrics.processing_current,
                "completed_total": self.metrics.completed_total,
                "failed_total": self.metrics.failed_total,
                "pass_total": self.metrics.pass_total,
                "review_total": self.metrics.review_total,
                "fail_total": self.metrics.fail_total,
                "simulator_running": True,
                "simulator_interval_ms": getattr(self, "_sim_interval_ms", None),
                "worker_count": self.config.workers,
                "queue_size": self.config.queue_size,
                "queue_peak_depth": self.metrics.queue_peak_depth,
            },
        )

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.telemetry_interval_seconds)
            if self._simulator is not None:
                # captured_total = produced captures; conservation law holds
                # while running: captured == queued + processing + completed + failed
                async with self._lock:
                    self.metrics.captured_total = self._simulator.captured_count
                    self.metrics.update_flow(self.queue.qsize(), self._processing)
                    if not self.metrics.conservation_ok():
                        logger.warning(
                            "metric conservation broken: captured=%d queued=%d processing=%d completed=%d failed=%d",
                            self.metrics.captured_total, self.metrics.queued_current,
                            self.metrics.processing_current, self.metrics.completed_total,
                            self.metrics.failed_total,
                        )
            try:
                await self._push_telemetry()
            except Exception:
                logger.debug("telemetry push failed", exc_info=True)


class _BackendError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"backend error {status_code}: {body[:120]}")


class _CaptureFailed(Exception):
    """The inspection was persisted as FAILED by the backend (terminal)."""
