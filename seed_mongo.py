import asyncio
from datetime import datetime, timezone

from mongo import db
from prompt_templates import get_agent_prompt

AGENTS = [
    {
        "name": "Oracle",
        "description": "Strategic technical advisor for architecture decisions, debugging, and comparative analysis.",
        "agent_type": "oracle",
        "color": "violet",
        "icon": "🔮",
    },
    {
        "name": "Travel Agent",
        "description": "Trip planning, destinations, itineraries, and travel advice.",
        "agent_type": "travel",
        "color": "blue",
        "icon": "🌍",
    },
    {
        "name": "Construction Agent",
        "description": "Construction planning, materials, safety, and best practices.",
        "agent_type": "construction",
        "color": "orange",
        "icon": "🏗️",
    },
    {
        "name": "Finance Agent",
        "description": "Personal finance, budgeting, investing, and planning.",
        "agent_type": "finance",
        "color": "green",
        "icon": "💰",
    },
    {
        "name": "General Assistant",
        "description": "General-purpose help across many topics.",
        "agent_type": "general",
        "color": "purple",
        "icon": "🤖",
    },
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def seed_agents() -> None:
    for agent in AGENTS:
        system_prompt = get_agent_prompt(agent["agent_type"])

        payload = {
            "name": agent["name"],
            "description": agent["description"],
            "agent_type": agent["agent_type"],
            "system_prompt": system_prompt,
            "color": agent["color"],
            "icon": agent["icon"],
            "updated_at": utc_now(),
        }

        await db.agents.update_one(
            {"name": agent["name"]},
            {
                "$set": payload,
                "$setOnInsert": {"created_at": utc_now()},
            },
            upsert=True,
        )

    print("Seeded Mongo agents successfully.")


if __name__ == "__main__":
    asyncio.run(seed_agents())
