"""Realtime metrics with a Redis backend and a single-process fallback."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

MAX_LATENCY_SAMPLES = 1000
THROUGHPUT_WINDOW_SECONDS = 60.0


class RealtimeMetrics:
    """Canonical counters shared through Redis when the app is scaled out."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._redis: Any | None = None
        self._prefix = "ivqc"
        self._reset_memory()

    def _reset_memory(self) -> None:
        self._processed = 0
        self._failed = 0
        self._pass_count = 0
        self._review_count = 0
        self._fail_count = 0
        self._latency_samples: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._inference_samples: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._completions: deque[float] = deque()
        self._started_at = time.monotonic()
        self._telemetry_updated_at: float | None = None
        self._telemetry: dict = {
            "captured_total": 0,
            "queued_current": 0,
            "processing_current": 0,
            "simulator_running": False,
            "simulator_interval_ms": None,
            "worker_count": None,
            "queue_size": None,
            "queue_peak_depth": 0,
        }

    async def configure(self, redis_client: Any | None, prefix: str = "ivqc") -> None:
        self._redis = redis_client
        self._prefix = prefix
        if redis_client is not None:
            await redis_client.set(self._key("started_at"), time.time(), nx=True)

    def _key(self, suffix: str) -> str:
        return f"{self._prefix}:metrics:{suffix}"

    async def record_completed(self, quality_result: str, latency_ms: float, inference_ms: float | None) -> None:
        if self._redis is not None:
            try:
                now = time.time()
                member = f"{now:.6f}:{uuid.uuid4().hex}"
                pipe = self._redis.pipeline(transaction=True)
                pipe.hincrby(self._key("counters"), "completed_total", 1)
                pipe.hincrby(self._key("counters"), f"{quality_result.lower()}_total", 1)
                pipe.rpush(self._key("latency"), latency_ms)
                pipe.ltrim(self._key("latency"), -MAX_LATENCY_SAMPLES, -1)
                if inference_ms is not None:
                    pipe.rpush(self._key("inference"), inference_ms)
                    pipe.ltrim(self._key("inference"), -MAX_LATENCY_SAMPLES, -1)
                pipe.zadd(self._key("completions"), {member: now})
                pipe.zremrangebyscore(self._key("completions"), "-inf", now - THROUGHPUT_WINDOW_SECONDS)
                await pipe.execute()
            except Exception:
                logger.exception("shared metrics update failed after inspection completion")
            return
        async with self._lock:
            self._processed += 1
            self._latency_samples.append(latency_ms)
            if inference_ms is not None:
                self._inference_samples.append(inference_ms)
            self._completions.append(time.monotonic())
            if quality_result == "PASS":
                self._pass_count += 1
            elif quality_result == "REVIEW":
                self._review_count += 1
            else:
                self._fail_count += 1

    async def record_failed(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.hincrby(self._key("counters"), "failed_total", 1)
            except Exception:
                logger.exception("shared metrics update failed after inspection failure")
            return
        async with self._lock:
            self._failed += 1

    async def update_telemetry(self, telemetry: dict) -> None:
        if self._redis is not None:
            values = {key: json.dumps(value) for key, value in telemetry.items()}
            values["updated_at"] = json.dumps(time.time())
            await self._redis.hset(self._key("telemetry"), mapping=values)
            return
        async with self._lock:
            self._telemetry.update(telemetry)
            self._telemetry_updated_at = time.time()

    async def reset(self) -> None:
        if self._redis is not None:
            await self._redis.delete(*[self._key(name) for name in (
                "counters", "latency", "inference", "completions", "telemetry", "started_at"
            )])
            await self._redis.set(self._key("started_at"), time.time())
            return
        async with self._lock:
            self._reset_memory()

    async def snapshot(self) -> dict:
        if self._redis is not None:
            return await self._redis_snapshot()
        async with self._lock:
            now = time.monotonic()
            recent = [stamp for stamp in self._completions if now - stamp <= THROUGHPUT_WINDOW_SECONDS]
            return self._build_snapshot(
                completed=self._processed,
                failed=self._failed,
                passed=self._pass_count,
                reviewed=self._review_count,
                failed_q=self._fail_count,
                samples=list(self._latency_samples),
                inference=list(self._inference_samples),
                throughput=len(recent) / THROUGHPUT_WINDOW_SECONDS,
                uptime=now - self._started_at,
                telemetry=dict(self._telemetry),
                telemetry_updated_at=self._telemetry_updated_at,
            )

    async def _redis_snapshot(self) -> dict:
        now = time.time()
        pipe = self._redis.pipeline(transaction=True)
        pipe.zremrangebyscore(self._key("completions"), "-inf", now - THROUGHPUT_WINDOW_SECONDS)
        pipe.hgetall(self._key("counters"))
        pipe.lrange(self._key("latency"), 0, -1)
        pipe.lrange(self._key("inference"), 0, -1)
        pipe.zcount(self._key("completions"), now - THROUGHPUT_WINDOW_SECONDS, "+inf")
        pipe.hgetall(self._key("telemetry"))
        pipe.get(self._key("started_at"))
        _, counters, latency, inference, recent_count, raw_telemetry, started_at = await pipe.execute()
        telemetry = dict(self._telemetry)
        telemetry_updated_at = None
        for key, value in raw_telemetry.items():
            if key == "updated_at":
                telemetry_updated_at = float(json.loads(value))
            else:
                telemetry[key] = json.loads(value)
        return self._build_snapshot(
            completed=int(counters.get("completed_total", 0)),
            failed=int(counters.get("failed_total", 0)),
            passed=int(counters.get("pass_total", 0)),
            reviewed=int(counters.get("review_total", 0)),
            failed_q=int(counters.get("fail_total", 0)),
            samples=[float(value) for value in latency],
            inference=[float(value) for value in inference],
            throughput=int(recent_count) / THROUGHPUT_WINDOW_SECONDS,
            uptime=max(0.0, now - float(started_at or now)),
            telemetry=telemetry,
            telemetry_updated_at=telemetry_updated_at,
        )

    @staticmethod
    def _build_snapshot(*, completed: int, failed: int, passed: int, reviewed: int,
                        failed_q: int, samples: list[float], inference: list[float],
                        throughput: float, uptime: float, telemetry: dict,
                        telemetry_updated_at: float | None) -> dict:
        if passed + reviewed + failed_q != completed:
            logger.error("quality invariant broken: pass=%d review=%d fail=%d completed=%d",
                         passed, reviewed, failed_q, completed)
        snapshot_at = datetime.now(timezone.utc).isoformat()
        telemetry_at = (
            datetime.fromtimestamp(telemetry_updated_at, tz=timezone.utc).isoformat()
            if telemetry_updated_at is not None else None
        )
        return {
            "completed_total": completed,
            "failed_total": failed,
            "pass_total": passed,
            "review_total": reviewed,
            "fail_total": failed_q,
            "total_inspected": completed + failed,
            "yield_rate": round(passed / completed, 6) if completed else None,
            "captured_total": telemetry["captured_total"],
            "queued_current": telemetry["queued_current"],
            "processing_current": telemetry["processing_current"],
            "queue_depth": telemetry["queued_current"],
            "throughput": round(throughput, 3),
            "queue_peak_depth": telemetry["queue_peak_depth"],
            "simulator_running": telemetry["simulator_running"],
            "simulator_interval_ms": telemetry["simulator_interval_ms"],
            "worker_count": telemetry["worker_count"],
            "queue_size": telemetry["queue_size"],
            "snapshot_at": snapshot_at,
            "telemetry_at": telemetry_at,
            "current_throughput": round(throughput, 3),
            "average_processing_latency_ms": round(sum(samples) / len(samples), 2) if samples else None,
            "p50_latency_ms": round(median(samples), 2) if samples else None,
            "p95_latency_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 2) if samples else None,
            "average_inference_latency_ms": round(sum(inference) / len(inference), 2) if inference else None,
            "uptime_seconds": round(uptime, 1),
            "ws_client_count": 0,
        }


metrics = RealtimeMetrics()
