from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from pydantic import ValidationError
from vision_contract import VisionResult

from ..config import get_settings

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Base class for inference service failures."""


class InferenceTimeoutError(InferenceError):
    pass


class InferenceConnectionError(InferenceError):
    pass


class InferenceHTTPError(InferenceError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"inference service returned HTTP {status_code}: {body[:200]}")


class InferenceContractError(InferenceError):
    pass


class InferenceClient:
    """HTTP-only client for the inference service.

    The backend never imports model code; everything crosses the wire as the
    standardized Vision Contract.

    The underlying httpx client is cached per event loop and reused, so the
    connection stays pooled across requests. Creating a client per request
    costs ~150 ms on Windows and inflates E2E latency.
    """

    _clients_by_loop: dict[int, httpx.AsyncClient] = {}

    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.inference_service_url).rstrip("/")
        self.timeout = timeout_seconds if timeout_seconds is not None else settings.inference_timeout_seconds

    def _client(self) -> httpx.AsyncClient:
        import asyncio

        loop = asyncio.get_running_loop()
        key = id(loop)
        client = self._clients_by_loop.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._clients_by_loop[key] = client
        return client

    @classmethod
    async def close_all(cls) -> None:
        """Close every event-loop-local pool during application shutdown."""
        clients = list(cls._clients_by_loop.values())
        cls._clients_by_loop.clear()
        for client in clients:
            await client.aclose()

    async def infer(self, image_bytes: bytes, filename: str = "image.jpg", request_id: str | None = None) -> VisionResult:
        rid = request_id or f"req-{uuid.uuid4().hex[:12]}"
        try:
            response = await self._client().post(
                f"{self.base_url}/v1/infer",
                files={"file": (filename, image_bytes, "application/octet-stream")},
                headers={"X-Request-ID": rid},
            )
        except httpx.TimeoutException as exc:
            logger.warning("inference timeout request_id=%s", rid)
            raise InferenceTimeoutError(f"inference service timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            logger.warning("inference connection error request_id=%s: %s", rid, exc)
            raise InferenceConnectionError(f"cannot reach inference service at {self.base_url}") from exc

        if response.status_code != 200:
            logger.warning("inference non-200 request_id=%s status=%s", rid, response.status_code)
            raise InferenceHTTPError(response.status_code, response.text)

        try:
            payload: dict[str, Any] = response.json()
            result = VisionResult.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            logger.warning("inference contract invalid request_id=%s: %s", rid, exc)
            raise InferenceContractError(f"invalid vision contract from inference service: {exc}") from exc

        logger.info("inference ok request_id=%s detections=%d fusion=%s latency=%.1fms", rid, len(result.detections), result.fusion_class, result.inference_latency_ms)
        return result
