"""
WebSocket endpoint for real-time graph-update streaming.
Clients subscribe by topic_id or graph_id.

Real-time events are produced by Celery tasks (tasks.py), which run in the
celery_worker container -- a different process from FastAPI. broadcast() therefore
cannot write directly to the in-memory `_channels` registry below: that registry only
exists in whichever process imports this module, so a broadcast from the worker used
to write into a dict no browser-facing process ever reads, and every WebSocket event
was silently lost. Instead, broadcast() PUBLISHes to a Redis channel, and FastAPI runs
a background subscriber (started from main.py's lifespan) that receives those
messages and fans them out to this process's local WebSocket connections. Redis is
already a required dependency, so this adds no new infrastructure, and it works for
any number of FastAPI/worker replicas.
"""
import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.cache import get_redis

logger = logging.getLogger(__name__)
router = APIRouter()

_PUBSUB_CHANNEL = "ws:broadcast"

# In-memory channel registry: channel_id -> set of WebSocket connections *in this process*.
_channels: Dict[str, Set[WebSocket]] = defaultdict(set)


@router.websocket("/channel/{channel_id}")
async def ws_channel(websocket: WebSocket, channel_id: str):
    await websocket.accept()
    _channels[channel_id].add(websocket)
    logger.info("WS client connected to channel '%s'", channel_id)
    try:
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _channels[channel_id].discard(websocket)
        logger.info("WS client disconnected from channel '%s'", channel_id)


async def _deliver_locally(channel_id: str, payload: dict) -> None:
    """Send to every WebSocket connected to this channel *in this process only*."""
    message = json.dumps(payload)
    dead: set[WebSocket] = set()
    for ws in list(_channels.get(channel_id, set())):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _channels[channel_id] -= dead


async def broadcast(channel_id: str, payload: dict) -> None:
    """
    Publish a payload for a channel. Delivered to every process's connected
    WebSocket clients via the Redis subscriber started in main.py's lifespan.
    """
    envelope = json.dumps({"channel_id": channel_id, "payload": payload})
    try:
        redis = get_redis()
        await redis.publish(_PUBSUB_CHANNEL, envelope)
    except Exception as exc:
        logger.warning(
            "Redis publish failed for channel '%s': %s -- falling back to "
            "local-only delivery (clients on other processes will miss this event)",
            channel_id, exc,
        )
        await _deliver_locally(channel_id, payload)


async def run_subscriber() -> None:
    """
    Background task, started once from FastAPI's lifespan: receives every broadcast()
    call from any process -- including this one -- and fans it out to this process's
    local WebSocket connections. Runs until cancelled at shutdown.
    """
    redis = get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(_PUBSUB_CHANNEL)
    logger.info("WebSocket pub/sub subscriber started on '%s'", _PUBSUB_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                envelope = json.loads(message["data"])
                await _deliver_locally(envelope["channel_id"], envelope["payload"])
            except Exception as exc:
                logger.warning("Failed to process WS pub/sub message: %s", exc)
    except asyncio.CancelledError:
        raise
    finally:
        await pubsub.unsubscribe(_PUBSUB_CHANNEL)
        await pubsub.close()
