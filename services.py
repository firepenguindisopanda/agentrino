import asyncio
import logging
from typing import AsyncGenerator

import langgraph_agent
import llm
import redis_cache
import repositories

logger = logging.getLogger(__name__)

MAX_CONVERSATIONS_PER_SESSION = 10


async def maybe_generate_title(conversation_id: str, user_content: str) -> None:
    """Best-effort: give an untitled conversation a title from its first message.

    Never raises; a failure here must not affect the chat response.
    """
    try:
        conversation = await repositories.get_conversation(conversation_id)
        if not conversation or conversation.get("title"):
            return
        title = await asyncio.to_thread(llm.generate_title, user_content)
        if title:
            await repositories.set_conversation_title(conversation_id, title)
    except Exception:
        logger.warning("Failed to generate conversation title", exc_info=True)


async def list_agents():
    return await repositories.list_agents()


async def get_agent(agent_id: str):
    return await repositories.get_agent(agent_id)


async def create_agent(agent: dict):
    return await repositories.create_agent(agent)


async def update_agent(agent_id: str, updates: dict):
    return await repositories.update_agent(agent_id, updates)


async def delete_agent(agent_id: str):
    return await repositories.delete_agent(agent_id)


async def create_conversation(agent_id: str, session_id: str, title: str | None = None):
    return await repositories.create_conversation(agent_id, session_id, title=title)


async def get_or_create_conversation(agent_id: str, session_id: str):
    existing = await repositories.get_conversation_by_session_agent(session_id, agent_id)
    if existing:
        return existing

    count = await repositories.count_active_conversations(session_id)
    if count >= MAX_CONVERSATIONS_PER_SESSION:
        raise ValueError(
            f"Maximum number of conversations ({MAX_CONVERSATIONS_PER_SESSION}) reached. Please delete an existing conversation."
        )

    return await repositories.create_conversation(agent_id, session_id)


async def list_conversations(session_id: str, include_archived: bool = False):
    return await repositories.list_conversations_by_session(session_id, include_archived)


async def archive_conversation(conversation_id: str):
    return await repositories.archive_conversation(conversation_id)


async def delete_conversation(conversation_id: str):
    return await repositories.delete_conversation(conversation_id)


async def get_conversation(conversation_id: str):
    return await repositories.get_conversation(conversation_id)


async def list_messages(conversation_id: str, limit: int = 50):
    cached = redis_cache.get_recent_messages(conversation_id)
    if cached:
        return cached[-limit:]

    # Fetch from MongoDB and populate cache
    messages = await repositories.list_messages(conversation_id, limit=limit)
    for msg in messages:
        redis_cache.cache_recent_message(conversation_id, msg)
    return messages


async def append_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
    rag_used: bool = False,
    rag_docs_count: int = 0,
):
    if not metadata:
        metadata = {}
    if rag_used:
        metadata["rag_used"] = rag_used
        metadata["rag_docs_count"] = rag_docs_count

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
    meta: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Stream the assistant reply. If ``meta`` is given it is filled with
    ``rag_used`` / ``rag_docs_count`` once retrieval has run, so the caller can
    report them in the terminating SSE event."""
    # Fetch recent history for context (before adding current message)
    # We use a limit, e.g., 20 messages.
    raw_history = await list_messages(conversation_id, limit=20)
    formatted_history = []
    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            formatted_history.append({"role": role, "content": content})

    is_first_turn = not formatted_history
    await append_message(conversation_id, "user", user_content)
    system_prompt = agent.get("system_prompt") if agent else None

    stream_generator, rag_used, rag_docs_count = await langgraph_agent.stream_agent(
        user_content, system_prompt=system_prompt, history=formatted_history
    )
    if meta is not None:
        meta.update({"rag_used": rag_used, "rag_docs_count": rag_docs_count})

    collected = []
    async for chunk in stream_generator:
        collected.append(chunk)
        yield chunk

    if collected:
        await append_message(
            conversation_id, "assistant", "".join(collected), rag_used=rag_used, rag_docs_count=rag_docs_count
        )

    if is_first_turn:
        await maybe_generate_title(conversation_id, user_content)


async def complete_response(conversation_id: str, agent: dict, user_content: str) -> str:
    # Fetch recent history for context
    raw_history = await list_messages(conversation_id, limit=20)
    formatted_history = []
    for msg in raw_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            formatted_history.append({"role": role, "content": content})

    is_first_turn = not formatted_history
    await append_message(conversation_id, "user", user_content)
    system_prompt = agent.get("system_prompt") if agent else None
    response = await langgraph_agent.invoke_agent(user_content, system_prompt=system_prompt, history=formatted_history)
    await append_message(
        conversation_id,
        "assistant",
        response["content"],
        rag_used=response["rag_used"],
        rag_docs_count=response["rag_docs_count"],
    )

    if is_first_turn:
        await maybe_generate_title(conversation_id, user_content)

    return response["content"]
