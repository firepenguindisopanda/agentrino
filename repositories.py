from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from mongo import db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_object_id(value: str | ObjectId) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _serialize_id(document: dict[str, Any]) -> dict[str, Any]:
    if not document:
        return document
    doc = dict(document)
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def ensure_indexes() -> None:
    await db.agents.create_index([("name", ASCENDING)], unique=True)
    await db.conversations.create_index([("agent_id", ASCENDING)])
    await db.messages.create_index(
        [("conversation_id", ASCENDING), ("created_at", DESCENDING)]
    )


async def list_agents() -> list[dict[str, Any]]:
    agents = []
    async for agent in db.agents.find({}).sort("name", ASCENDING):
        agents.append(_serialize_id(agent))
    return agents


async def get_agent(agent_id: str) -> dict[str, Any] | None:
    agent = await db.agents.find_one({"_id": _to_object_id(agent_id)})
    return _serialize_id(agent) if agent else None


async def create_agent(agent: dict[str, Any]) -> dict[str, Any]:
    payload = {
        **agent,
        "created_at": _now(),
        "updated_at": _now(),
    }
    result = await db.agents.insert_one(payload)
    payload["id"] = str(result.inserted_id)
    payload.pop("_id", None)
    return payload


async def create_conversation(
    agent_id: str, title: str | None = None
) -> dict[str, Any]:
    payload = {
        "agent_id": _to_object_id(agent_id),
        "title": title,
        "created_at": _now(),
        "updated_at": _now(),
    }
    result = await db.conversations.insert_one(payload)
    payload["id"] = str(result.inserted_id)
    payload["agent_id"] = str(payload["agent_id"])
    payload.pop("_id", None)
    return payload


async def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    convo = await db.conversations.find_one({"_id": _to_object_id(conversation_id)})
    if not convo:
        return None
    convo = _serialize_id(convo)
    convo["agent_id"] = str(convo["agent_id"])
    return convo


async def update_conversation_timestamp(conversation_id: str) -> None:
    await db.conversations.update_one(
        {"_id": _to_object_id(conversation_id)},
        {"$set": {"updated_at": _now()}},
    )


async def list_messages(conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
    messages = []
    cursor = (
        db.messages.find({"conversation_id": _to_object_id(conversation_id)})
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    async for message in cursor:
        doc = _serialize_id(message)
        doc["conversation_id"] = str(doc["conversation_id"])
        messages.append(doc)
    messages.reverse()
    return messages


async def create_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "conversation_id": _to_object_id(conversation_id),
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": _now(),
    }
    result = await db.messages.insert_one(payload)
    payload["id"] = str(result.inserted_id)
    payload["conversation_id"] = str(payload["conversation_id"])
    payload.pop("_id", None)
    return payload
