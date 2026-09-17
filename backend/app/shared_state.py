"""Lifecycle-owned Redis connection for cross-worker state and messaging."""
from __future__ import annotations

import logging
from typing import Any

from .config import get_settings

_client: Any | None = None
logger = logging.getLogger(__name__)


async def open_shared_state() -> Any | None:
    """Connect once per worker; development may intentionally use memory."""
    global _client
    settings = get_settings()
    if not settings.redis_url:
        if settings.environment.lower() in {"production", "staging"}:
            raise RuntimeError("IVQC_REDIS_URL is required in production/staging")
        logger.warning("Redis is not configured; shared state is limited to one process")
        return None
    import redis.asyncio as redis

    client = redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        raise
    _client = client
    return client


async def close_shared_state() -> None:
    global _client
    client, _client = _client, None
    if client is not None:
        await client.aclose()


def get_shared_state() -> Any | None:
    return _client
