"""Redis cache helpers with automatic JSON serialisation."""
import hashlib
import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_redis: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    """
    Explicitly close the shared client. FastAPI and Celery never call this (the
    process lifetime bounds the connection, which is fine) -- it exists for short
    one-shot scripts like eval/capture_fixtures.py, where skipping it means the
    connection's __del__ fires after asyncio.run() has already closed the loop it
    needs, logging a harmless but noisy "Event loop is closed" on exit.
    """
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
    return _redis


def _make_key(prefix: str, *parts: str) -> str:
    raw = ":".join(parts)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{h}"


async def cache_get(key: str) -> Optional[Any]:
    try:
        r = get_redis()
        raw = await r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache GET failed for key %s: %s", key, exc)
        return None


async def cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        r = get_redis()
        await r.set(key, json.dumps(value), ex=ttl)
    except Exception as exc:
        logger.warning("Cache SET failed for key %s: %s", key, exc)


async def cache_delete(key: str) -> None:
    try:
        r = get_redis()
        await r.delete(key)
    except Exception as exc:
        logger.warning("Cache DELETE failed for key %s: %s", key, exc)


# Convenience key builders
def source_key(source_type: str, query: str) -> str:
    return _make_key(f"source:{source_type}", query)


def llm_extract_key(content: str) -> str:
    return _make_key("llm:extract", content)


def llm_ambiguity_key(topic: str) -> str:
    return _make_key("llm:ambiguity", topic.strip().lower())


def llm_summary_key(node_id: str, depth: str) -> str:
    return f"llm:summary:{node_id}:{depth}"
