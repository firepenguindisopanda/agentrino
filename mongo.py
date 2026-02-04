import os
from pymongo.errors import PyMongoError

try:
    from pymongo.asynchronous.mongo_client import AsyncMongoClient
except ImportError:  # pragma: no cover - fallback for older pymongo
    from pymongo import AsyncMongoClient  # type: ignore


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_chatter")

client = AsyncMongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]


async def ping_mongo() -> bool:
    try:
        await client.admin.command("ping")
        return True
    except PyMongoError:
        return False
