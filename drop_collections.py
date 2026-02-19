import asyncio

from mongo import db


async def drop_collections() -> None:
    await db.agents.drop()
    await db.conversations.drop()
    await db.messages.drop()
    print("Dropped collections: agents, conversations, messages")


if __name__ == "__main__":
    asyncio.run(drop_collections())
