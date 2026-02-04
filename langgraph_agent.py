from typing import AsyncGenerator, TypedDict, Optional, List

from langgraph.graph import END, StateGraph

import llm


class AgentState(TypedDict):
    prompt: str
    system_prompt: Optional[str]
    history: Optional[List[dict]]
    response: Optional[str]


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    async def respond(state: AgentState) -> AgentState:
        prompt = state["prompt"]
        system_prompt = state.get("system_prompt")
        history = state.get("history")
        response = llm.get_response_text(prompt, system_prompt=system_prompt, history=history)
        return {"response": response}

    graph.add_node("respond", respond)
    graph.set_entry_point("respond")
    graph.add_edge("respond", END)
    return graph


_GRAPH = _build_graph().compile()


async def invoke_agent(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None) -> str:
    output = await _GRAPH.ainvoke({"prompt": prompt, "system_prompt": system_prompt, "history": history})
    return output["response"]


async def stream_agent(
    prompt: str, system_prompt: str | None = None, history: list[dict] | None = None
) -> AsyncGenerator[str, None]:
    for chunk in llm.stream_response(prompt, system_prompt=system_prompt, history=history):
        yield chunk
