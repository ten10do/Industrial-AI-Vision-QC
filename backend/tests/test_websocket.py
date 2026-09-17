"""Unit tests for the WebSocket ConnectionManager (3E).

Broadcast isolation is tested with fake sockets: a dead client must be dropped
without affecting other clients and without raising out of broadcast().
The real end-to-end WebSocket path (server + network client) is covered by the
integration pipeline E2E test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ws import ConnectionManager  # noqa: E402


class FakeWS:
    def __init__(self, fail_send: bool = False) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.fail_send = fail_send

    async def accept(self) -> None:
        pass

    async def send_json(self, data: dict) -> None:
        if self.fail_send:
            raise ConnectionError("dead socket")
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_disconnect_lifecycle():
    mgr = ConnectionManager()
    ws = FakeWS()
    await mgr.connect(ws)
    assert mgr.client_count == 1
    await mgr.disconnect(ws)
    assert mgr.client_count == 0


@pytest.mark.asyncio
async def test_broadcast_delivers_to_all_clients():
    mgr = ConnectionManager()
    ws1, ws2 = FakeWS(), FakeWS()
    await mgr.connect(ws1)
    await mgr.connect(ws2)
    delivered = await mgr.broadcast({"event_type": "inspection.completed", "inspection_id": "i1"})
    assert delivered == 2
    assert ws1.sent == ws2.sent == [{"event_type": "inspection.completed", "inspection_id": "i1"}]


@pytest.mark.asyncio
async def test_dead_client_isolated():
    """A dead client must be dropped without affecting healthy clients."""
    mgr = ConnectionManager()
    dead, alive = FakeWS(fail_send=True), FakeWS()
    await mgr.connect(dead)
    await mgr.connect(alive)
    delivered = await mgr.broadcast({"inspection_id": "i2"})
    assert delivered == 1
    assert alive.sent == [{"inspection_id": "i2"}]
    assert mgr.client_count == 1
    assert dead not in mgr._connections


@pytest.mark.asyncio
async def test_broadcast_never_raises_even_if_all_clients_fail():
    mgr = ConnectionManager()
    await mgr.connect(FakeWS(fail_send=True))
    await mgr.connect(FakeWS(fail_send=True))
    delivered = await mgr.broadcast({"inspection_id": "i3"})
    assert delivered == 0
    assert mgr.client_count == 0


@pytest.mark.asyncio
async def test_shutdown_closes_all():
    mgr = ConnectionManager()
    ws = FakeWS()
    await mgr.connect(ws)
    await mgr.shutdown()
    assert ws.closed
    assert mgr.client_count == 0
