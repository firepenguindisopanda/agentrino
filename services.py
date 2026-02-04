from typing import AsyncGenerator

import langgraph_agent
import redis_cache
import repositories


async def list_agents():
    return await repositories.list_agents()


async def get_agent(agent_id: str):
    return await repositories.get_agent(agent_id)


async def create_conversation(agent_id: str, title: str | None = None):
    return await repositories.create_conversation(agent_id, title=title)


async def get_conversation(conversation_id: str):
    return await repositories.get_conversation(conversation_id)


async def list_messages(conversation_id: str, limit: int = 50):
    cached = redis_cache.get_recent_messages(conversation_id)
    if cached:
        return cached[-limit:]
    return await repositories.list_messages(conversation_id, limit=limit)


async def append_message(conversation_id: str, role: str, content: str, metadata: dict | None = None):
    message = await repositories.create_message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata=metadata,
    )
    redis_cache.cache_recent_message(conversation_id, message)
    await repositories.update_conversation_timestamp(conversation_id)
    return message


async def stream_response(
    conversation_id: str,
    agent: dict,
    user_content: str,
) -> AsyncGenerator[str, None]:
    # Fetch recent history for context (before adding current message)
    # We use a limit, e.g., 20 messages.
    raw_history = await list_messages(conversation_id, limit=20)
    formatted_history = []
    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            formatted_history.append({"role": role, "content": content})

    await append_message(conversation_id, "user", user_content)
    system_prompt = agent.get("system_prompt") if agent else None
    collected = []
    async for chunk in langgraph_agent.stream_agent(
        user_content, system_prompt=system_prompt, history=formatted_history
    ):
        collected.append(chunk)
        yield chunk
    if collected:
        await append_message(conversation_id, "assistant", "".join(collected))


async def complete_response(conversation_id: str, agent: dict, user_content: str) -> str:
    # Fetch recent history for context
    raw_history = await list_messages(conversation_id, limit=20)
    formatted_history = []
    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            formatted_history.append({"role": role, "content": content})

    await append_message(conversation_id, "user", user_content)
    system_prompt = agent.get("system_prompt") if agent else None
    response = await langgraph_agent.invoke_agent(user_content, system_prompt=system_prompt, history=formatted_history)
    await append_message(conversation_id, "assistant", response)
    return response
