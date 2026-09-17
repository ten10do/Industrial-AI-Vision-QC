"""Cross-worker WebSocket delivery backed by Redis Pub/Sub."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

_SEND_TIMEOUT = 5.0
_CLIENT_COUNT_TTL_SECONDS = 90


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._redis: Any | None = None
        self._pubsub: Any | None = None
        self._subscriber_task: asyncio.Task | None = None
        self._publish_tasks: set[asyncio.Task] = set()
        self._prefix = "ivqc"
        self._instance_id = uuid.uuid4().hex

    @property
    def client_count(self) -> int:
        return len(self._connections)

    @property
    def _channel(self) -> str:
        return f"{self._prefix}:events"

    @property
    def _count_key(self) -> str:
        return f"{self._prefix}:ws:clients:{self._instance_id}"

    async def start(self, redis_client: Any | None, prefix: str = "ivqc") -> None:
        self._redis = redis_client
        self._prefix = prefix
        if redis_client is None:
            return
        self._pubsub = redis_client.pubsub()
        await self._pubsub.subscribe(self._channel)
        await self._update_shared_count()
        self._subscriber_task = asyncio.create_task(self._listen(), name="ivqc-ws-event-subscriber")

    async def _listen(self) -> None:
        try:
            while True:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is not None:
                    try:
                        await self.broadcast(json.loads(message["data"]))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        logger.warning("invalid realtime event received from shared bus")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("shared realtime event subscriber stopped unexpectedly")

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        await self._update_shared_count()

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        await self._update_shared_count()

    async def _update_shared_count(self) -> None:
        if self._redis is not None:
            await self._redis.setex(self._count_key, _CLIENT_COUNT_TTL_SECONDS, self.client_count)

    async def total_client_count(self) -> int:
        if self._redis is None:
            return self.client_count
        await self._update_shared_count()
        total = 0
        async for key in self._redis.scan_iter(match=f"{self._prefix}:ws:clients:*"):
            value = await self._redis.get(key)
            total += int(value or 0)
        return total

    async def publish(self, event: dict) -> int:
        if self._redis is not None:
            return int(await self._redis.publish(self._channel, json.dumps(event, separators=(",", ":"))))
        return await self.broadcast(event)

    async def broadcast(self, event: dict) -> int:
        """Deliver to this worker's clients; one dead client cannot fan out failure."""
        async with self._lock:
            targets = list(self._connections)
        delivered = 0
        for websocket in targets:
            try:
                await asyncio.wait_for(websocket.send_json(event), timeout=_SEND_TIMEOUT)
                delivered += 1
            except Exception:
                logger.warning("ws broadcast failed for a client, dropping it")
                await self.disconnect(websocket)
        return delivered

    def schedule(self, event: dict) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self.publish(event))
        except RuntimeError:
            logger.warning("no running event loop, broadcast skipped")
            return
        self._publish_tasks.add(task)
        task.add_done_callback(self._publish_done)

    def _publish_done(self, task: asyncio.Task) -> None:
        self._publish_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("shared realtime publish failed: %s", task.exception())

    async def shutdown(self) -> None:
        if self._publish_tasks:
            await asyncio.gather(*list(self._publish_tasks), return_exceptions=True)
        if self._subscriber_task is not None:
            self._subscriber_task.cancel()
            await asyncio.gather(self._subscriber_task, return_exceptions=True)
            self._subscriber_task = None
        if self._pubsub is not None:
            await self._pubsub.unsubscribe(self._channel)
            await self._pubsub.aclose()
            self._pubsub = None
        async with self._lock:
            targets = list(self._connections)
            self._connections.clear()
        for websocket in targets:
            try:
                await websocket.close()
            except Exception:
                pass
        if self._redis is not None:
            await self._redis.delete(self._count_key)
        self._redis = None


manager = ConnectionManager()


def schedule_broadcast(event: dict) -> None:
    """Publish after commit without blocking the inspection/review request."""
    manager.schedule(event)
