from typing import AsyncGenerator, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

import llm
import rag


class AgentState(TypedDict):
    prompt: str
    system_prompt: Optional[str]
    history: Optional[List[dict]]
    context: Optional[str]
    response: Optional[str]
    rag_used: Optional[bool]
    rag_docs_count: Optional[int]


class RagResponse(TypedDict):
    content: str
    rag_used: bool
    rag_docs_count: int


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    async def retrieve(state: AgentState) -> AgentState:
        prompt = state["prompt"]
        context_str, context_docs = rag.build_rag_prompt(prompt, include_context=True)
        rag_used = len(context_docs) > 0 if context_docs else False
        return {
            "context": context_str,
            "rag_used": rag_used,
            "rag_docs_count": len(context_docs) if context_docs else 0,
        }

    async def respond(state: AgentState) -> AgentState:
        prompt = state["prompt"]
        system_prompt = state.get("system_prompt")
        history = state.get("history")
        context = state.get("context")
        rag_used = state.get("rag_used", False)
        rag_docs_count = state.get("rag_docs_count", 0)

        if context:
            full_prompt = f"{context}\n\nAnswer the user's question above based on the context provided."
        else:
            full_prompt = prompt

        response = llm.get_response_text(full_prompt, system_prompt=system_prompt, history=history)
        return {
            "response": response,
            "rag_used": rag_used,
            "rag_docs_count": rag_docs_count,
        }

    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    return graph


_GRAPH = _build_graph().compile()


async def invoke_agent(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None) -> RagResponse:
    output = await _GRAPH.ainvoke({"prompt": prompt, "system_prompt": system_prompt, "history": history})
    return {
        "content": output["response"],
        "rag_used": output.get("rag_used", False),
        "rag_docs_count": output.get("rag_docs_count", 0),
    }


async def stream_agent(
    prompt: str, system_prompt: str | None = None, history: list[dict] | None = None
) -> tuple[AsyncGenerator[str, None], bool, int]:
    context_str, context_docs = rag.build_rag_prompt(prompt, include_context=True)
    rag_used = len(context_docs) > 0 if context_docs else False
    rag_docs_count = len(context_docs) if context_docs else 0

    if context_str:
        full_prompt = f"{context_str}\n\nAnswer the user's question above based on the context provided."
    else:
        full_prompt = prompt

    async def content_stream():
        for chunk in llm.stream_response(full_prompt, system_prompt=system_prompt, history=history):
            yield chunk

    return content_stream(), rag_used, rag_docs_count
