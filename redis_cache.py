import json
import os
from datetime import datetime, timezone

from upstash_redis import Redis

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
    raise ValueError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set")

RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "30"))
RECENT_MESSAGES_TTL = int(os.getenv("RECENT_MESSAGES_TTL", "3600"))

redis_client = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)


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
    redis_client.lpush(key, _serialize_message(message))
    redis_client.ltrim(key, 0, RECENT_MESSAGES_LIMIT - 1)
    redis_client.expire(key, RECENT_MESSAGES_TTL)


def get_recent_messages(conversation_id: str) -> list[dict]:
    key = f"recent_messages:{conversation_id}"
    data = redis_client.lrange(key, 0, -1)
    if not data:
        return []
    messages = [_deserialize_message(item) for item in data]
    messages.reverse()
    return messages
