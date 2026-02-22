import asyncio
from datetime import datetime, timezone

from mongo import db


async def migrate_conversations():
    """Add session_id, is_archived, and last_activity_at fields to existing conversations."""

    now = datetime.now(timezone.utc)

    # First, handle conversations without session_id - give them a temporary placeholder
    # This is only for old conversations that need migration
    # We'll mark them as archived since they can't be associated with a session

    result = await db.conversations.update_many(
        {"session_id": {"$exists": False}},
        {
            "$set": {
                "session_id": "migrated_legacy",
                "is_archived": False,
            }
        },
    )

    print(f"Marked {result.modified_count} legacy conversations")

    # Now set last_activity_at for conversations that don't have it
    # Use updated_at value if it exists, otherwise use now
    conversations = db.conversations.find({"last_activity_at": {"$exists": False}})

    count = 0
    async for conv in conversations:
        last_activity = conv.get("updated_at") or now
        await db.conversations.update_one({"_id": conv["_id"]}, {"$set": {"last_activity_at": last_activity}})
        count += 1

    print(f"Set last_activity_at for {count} conversations")

    # Create the new indexes (if they don't exist)
    print("Creating indexes...")
    try:
        await db.conversations.create_index([("session_id", 1), ("agent_id", 1)], unique=True, sparse=True)
        print("Created unique index on (session_id, agent_id)")
    except Exception as e:
        print(f"Index (session_id, agent_id) may already exist: {e}")

    try:
        await db.conversations.create_index([("session_id", 1)])
        print("Created index on session_id")
    except Exception as e:
        print(f"Index on session_id may already exist: {e}")

    try:
        await db.conversations.create_index([("last_activity_at", 1)], expireAfterSeconds=604800)
        print("Created TTL index on last_activity_at (7 days)")
    except Exception as e:
        print(f"TTL index may already exist: {e}")

    print("Migration complete!")
    print("\nNOTE: Legacy conversations have been marked with session_id='migrated_legacy'")
    print("They will appear in 'Manage Chats' but users won't be able to continue them.")
    print("Recommend deleting old conversations or running cleanup.")


if __name__ == "__main__":
    asyncio.run(migrate_conversations())
