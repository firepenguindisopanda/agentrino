import json
import os
from datetime import datetime, timezone

import redis


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "30"))
RECENT_MESSAGES_TTL = int(os.getenv("RECENT_MESSAGES_TTL", "3600"))


def _redis_client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


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
    key = f"recent_messages:{conversation_id}"
    client = _redis_client()
    client.lpush(key, _serialize_message(message))
    client.ltrim(key, 0, RECENT_MESSAGES_LIMIT - 1)
    client.expire(key, RECENT_MESSAGES_TTL)


def get_recent_messages(conversation_id: str) -> list[dict]:
    key = f"recent_messages:{conversation_id}"
    client = _redis_client()
    data = client.lrange(key, 0, -1)
    if not data:
        return []
    messages = [_deserialize_message(item) for item in data]
    messages.reverse()
    return messages
