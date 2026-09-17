from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main
from app.copilot.conversation import conversation_store
from app.inference.client import InferenceClient
from app.config import Settings
from app import shared_state


@pytest.mark.asyncio
async def test_lifespan_closes_shared_and_network_resources(monkeypatch):
    shared = object()
    open_state = AsyncMock(return_value=shared)
    close_state = AsyncMock()
    start_ws = AsyncMock()
    stop_ws = AsyncMock()
    configure_metrics = AsyncMock()
    configure_conversations = AsyncMock()
    close_inference = AsyncMock()
    dispose_engine = AsyncMock()
    monkeypatch.setattr(main, "open_shared_state", open_state)
    monkeypatch.setattr(main, "close_shared_state", close_state)
    monkeypatch.setattr(main.manager, "start", start_ws)
    monkeypatch.setattr(main.manager, "shutdown", stop_ws)
    monkeypatch.setattr(main.metrics, "configure", configure_metrics)
    monkeypatch.setattr(conversation_store, "configure", configure_conversations)
    monkeypatch.setattr(main.InferenceClient, "close_all", close_inference)
    monkeypatch.setattr(main, "engine", SimpleNamespace(dispose=dispose_engine))
    app = SimpleNamespace(state=SimpleNamespace())

    async with main.lifespan(app):
        assert app.state.shared_state is shared

    start_ws.assert_awaited_once()
    stop_ws.assert_awaited_once()
    close_inference.assert_awaited_once()
    close_state.assert_awaited_once()
    dispose_engine.assert_awaited_once()
    assert configure_metrics.await_args_list[-1].args == (None,)
    assert configure_conversations.await_args_list[-1].args == (None,)


@pytest.mark.asyncio
async def test_inference_client_close_all_closes_cached_pools():
    first = AsyncMock()
    second = AsyncMock()
    InferenceClient._clients_by_loop = {1: first, 2: second}

    await InferenceClient.close_all()

    first.aclose.assert_awaited_once()
    second.aclose.assert_awaited_once()
    assert InferenceClient._clients_by_loop == {}


@pytest.mark.asyncio
async def test_production_requires_shared_state(monkeypatch):
    monkeypatch.setattr(
        shared_state, "get_settings", lambda: Settings(environment="production", redis_url="")
    )
    with pytest.raises(RuntimeError, match="IVQC_REDIS_URL"):
        await shared_state.open_shared_state()
