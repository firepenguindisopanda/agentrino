import json
import logging
import os
from datetime import datetime, timezone

from upstash_redis import Redis

logger = logging.getLogger("agentrino.cache")

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "30"))
RECENT_MESSAGES_TTL = int(os.getenv("RECENT_MESSAGES_TTL", "3600"))

# Lazily-initialized client: the Redis cache is an optional accelerator in front
# of MongoDB, so we neither build the client nor fail at import time. If the
# cache is unreachable, callers fall back to the database instead of erroring.
_redis_client = None


def _get_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
            raise ValueError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set")
        _redis_client = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
    return _redis_client


def _serialize_message(message: dict) -> str:
    return json.dumps(message, default=str)


def _deserialize_message(raw: str) -> dict:
    data = json.loads(raw)
    if "created_at" in data and isinstance(data["created_at"], str):
        try:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        except ValueError:
            data["created_at"] = datetime.now(timezone.utc)
    return data


def cache_recent_message(conversation_id: str, message: dict) -> None:
    """Best-effort cache write. A cache failure must never break a request."""
    key = f"recent_messages:{conversation_id}"
    try:
        client = _get_client()
        client.lpush(key, _serialize_message(message))
        client.ltrim(key, 0, RECENT_MESSAGES_LIMIT - 1)
        client.expire(key, RECENT_MESSAGES_TTL)
    except Exception as exc:  # noqa: BLE001 - cache is optional
        logger.warning("Cache write failed for %s: %s", key, exc)


def get_recent_messages(conversation_id: str) -> list[dict]:
    """Best-effort cache read. On any failure, return empty so callers hit the DB."""
    key = f"recent_messages:{conversation_id}"
    try:
        data = _get_client().lrange(key, 0, -1)
    except Exception as exc:  # noqa: BLE001 - cache is optional
        logger.warning("Cache read failed for %s: %s", key, exc)
        return []
    if not data:
        return []
    messages = [_deserialize_message(item) for item in data]
    messages.reverse()
    return messages
