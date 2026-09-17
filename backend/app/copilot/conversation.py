"""Bounded Copilot context persisted in Redis when shared state is enabled."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_TURNS = 10
MAX_CONVERSATIONS = 200
TTL_SECONDS = 3600 * 6


@dataclass
class Turn:
    role: str
    content: str
    tool_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "tools": self.tool_summary}


@dataclass
class Conversation:
    id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [turn.to_dict() for turn in self.turns],
        }


class ConversationStore:
    def __init__(self) -> None:
        self._store: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()
        self._redis: Any | None = None
        self._prefix = "ivqc"

    async def configure(self, redis_client: Any | None, prefix: str = "ivqc") -> None:
        self._redis = redis_client
        self._prefix = prefix

    def _meta_key(self, conversation_id: str) -> str:
        return f"{self._prefix}:copilot:{conversation_id}:meta"

    def _turns_key(self, conversation_id: str) -> str:
        return f"{self._prefix}:copilot:{conversation_id}:turns"

    @property
    def _index_key(self) -> str:
        return f"{self._prefix}:copilot:index"

    def _evict_memory(self) -> None:
        now = time.time()
        stale = [key for key, value in self._store.items() if now - value.updated_at > TTL_SECONDS]
        for key in stale:
            del self._store[key]
        if len(self._store) > MAX_CONVERSATIONS:
            oldest = sorted(self._store, key=lambda key: self._store[key].updated_at)
            for key in oldest[:len(self._store) - MAX_CONVERSATIONS]:
                del self._store[key]

    async def get_or_create(self, conversation_id: str | None) -> Conversation:
        conversation_id = conversation_id or uuid.uuid4().hex
        existing = await self.get(conversation_id)
        if existing is not None:
            return existing
        conversation = Conversation(id=conversation_id)
        if self._redis is None:
            async with self._lock:
                self._evict_memory()
                return self._store.setdefault(conversation_id, conversation)
        created = str(conversation.created_at)
        await self._redis.hset(self._meta_key(conversation_id), mapping={
            "id": conversation_id, "created_at": created, "updated_at": created,
        })
        await self._touch(conversation_id, conversation.created_at)
        return conversation

    async def append(self, conversation_id: str, turn: Turn) -> None:
        if self._redis is None:
            async with self._lock:
                conversation = self._store.setdefault(conversation_id, Conversation(id=conversation_id))
                conversation.add(turn)
                self._evict_memory()
            return
        updated_at = time.time()
        pipe = self._redis.pipeline(transaction=True)
        pipe.rpush(self._turns_key(conversation_id), json.dumps(turn.to_dict(), ensure_ascii=False))
        pipe.ltrim(self._turns_key(conversation_id), -MAX_TURNS, -1)
        pipe.hset(self._meta_key(conversation_id), mapping={"updated_at": updated_at})
        pipe.expire(self._meta_key(conversation_id), TTL_SECONDS)
        pipe.expire(self._turns_key(conversation_id), TTL_SECONDS)
        pipe.zadd(self._index_key, {conversation_id: updated_at})
        await pipe.execute()
        await self._evict_redis(updated_at)

    async def get(self, conversation_id: str) -> Conversation | None:
        if self._redis is None:
            async with self._lock:
                self._evict_memory()
                return self._store.get(conversation_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.hgetall(self._meta_key(conversation_id))
        pipe.lrange(self._turns_key(conversation_id), 0, -1)
        meta, raw_turns = await pipe.execute()
        if not meta:
            return None
        turns = []
        for raw in raw_turns:
            value = json.loads(raw)
            turns.append(Turn(role=value["role"], content=value["content"], tool_summary=value.get("tools", [])))
        return Conversation(
            id=meta["id"], created_at=float(meta["created_at"]),
            updated_at=float(meta["updated_at"]), turns=turns,
        )

    async def _touch(self, conversation_id: str, updated_at: float) -> None:
        pipe = self._redis.pipeline(transaction=True)
        pipe.expire(self._meta_key(conversation_id), TTL_SECONDS)
        pipe.expire(self._turns_key(conversation_id), TTL_SECONDS)
        pipe.zadd(self._index_key, {conversation_id: updated_at})
        await pipe.execute()
        await self._evict_redis(updated_at)

    async def _evict_redis(self, now: float) -> None:
        await self._redis.zremrangebyscore(self._index_key, "-inf", now - TTL_SECONDS)
        count = int(await self._redis.zcard(self._index_key))
        if count <= MAX_CONVERSATIONS:
            return
        stale = await self._redis.zrange(self._index_key, 0, count - MAX_CONVERSATIONS - 1)
        if stale:
            pipe = self._redis.pipeline(transaction=True)
            for conversation_id in stale:
                pipe.delete(self._meta_key(conversation_id), self._turns_key(conversation_id))
                pipe.zrem(self._index_key, conversation_id)
            await pipe.execute()

    async def reset(self) -> None:
        if self._redis is None:
            async with self._lock:
                self._store.clear()
            return
        keys = [key async for key in self._redis.scan_iter(match=f"{self._prefix}:copilot:*")]
        if keys:
            await self._redis.delete(*keys)


conversation_store = ConversationStore()
