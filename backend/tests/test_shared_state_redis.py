from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from app.copilot.conversation import ConversationStore, Turn
from app.metrics import RealtimeMetrics
from app.ws import ConnectionManager

REDIS_URL = os.environ.get("IVQC_TEST_REDIS_URL")


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self) -> None:
        pass

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.skipif(not REDIS_URL, reason="IVQC_TEST_REDIS_URL is not configured")
@pytest.mark.asyncio
async def test_redis_shares_metrics_sessions_and_websocket_events_across_instances():
    import redis.asyncio as redis

    client = redis.from_url(REDIS_URL, decode_responses=True)
    prefix = f"ivqc-test-{uuid.uuid4().hex}"
    first_metrics, second_metrics = RealtimeMetrics(), RealtimeMetrics()
    first_store, second_store = ConversationStore(), ConversationStore()
    first_ws, second_ws = ConnectionManager(), ConnectionManager()
    socket_one, socket_two = FakeWS(), FakeWS()
    try:
        await first_metrics.configure(client, prefix)
        await second_metrics.configure(client, prefix)
        await first_metrics.reset()
        await first_metrics.record_completed("PASS", 10.0, 5.0)
        await second_metrics.record_completed("REVIEW", 20.0, 6.0)
        snapshot = await first_metrics.snapshot()
        assert snapshot["completed_total"] == 2
        assert snapshot["pass_total"] == snapshot["review_total"] == 1

        await first_store.configure(client, prefix)
        await second_store.configure(client, prefix)
        conversation = await first_store.get_or_create(None)
        await first_store.append(conversation.id, Turn(role="user", content="跨副本会话"))
        restored = await second_store.get(conversation.id)
        assert restored is not None and restored.turns[0].content == "跨副本会话"

        await first_ws.start(client, prefix)
        await second_ws.start(client, prefix)
        await first_ws.connect(socket_one)
        await second_ws.connect(socket_two)
        event = {"event_type": "inspection.completed", "inspection_id": "shared-1"}
        await first_ws.publish(event)
        for _ in range(100):
            if socket_one.sent and socket_two.sent:
                break
            await asyncio.sleep(0.01)
        assert socket_one.sent == socket_two.sent == [event]
        assert await first_ws.total_client_count() == 2
    finally:
        await first_ws.shutdown()
        await second_ws.shutdown()
        keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()
