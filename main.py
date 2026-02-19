import json
import os
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

import repositories
import schemas
import services


app = FastAPI()

# CORS configuration - can be customized via environment variable
# Format: comma-separated URLs, e.g., "http://localhost:3000,https://myapp.vercel.app"
_default_origins = [
    "http://localhost:3001",
    "http://localhost:3000",
]
_cors_env = os.getenv("CORS_ORIGINS", "")
_extra_origins = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
origins = _default_origins + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    await repositories.ensure_indexes()


@app.get("/agents", response_model=List[schemas.AgentOut])
async def list_agents():
    return await services.list_agents()


@app.post("/conversations", response_model=schemas.ConversationOut)
async def create_conversation(request: schemas.ConversationCreate):
    agent = await services.get_agent(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return await services.create_conversation(request.agent_id, request.session_id, title=request.title)


@app.post("/conversations/get-or-create", response_model=schemas.ConversationOut)
async def get_or_create_conversation(request: schemas.ConversationGetOrCreate):
    agent = await services.get_agent(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    try:
        return await services.get_or_create_conversation(request.agent_id, request.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/conversations", response_model=List[schemas.ConversationOut])
async def list_conversations(session_id: str = Query(...), include_archived: bool = Query(False)):
    return await services.list_conversations(session_id, include_archived)


@app.patch("/conversations/{conversation_id}/archive", response_model=schemas.ConversationOut)
async def archive_conversation(conversation_id: str):
    success = await services.archive_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation = await services.get_conversation(conversation_id)
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    success = await services.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@app.get("/conversations/{conversation_id}", response_model=schemas.ConversationOut)
async def get_conversation(conversation_id: str):
    conversation = await services.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/conversations/{conversation_id}/messages", response_model=List[schemas.MessageOut])
async def get_messages(conversation_id: str, limit: int = Query(50, ge=1, le=200)):
    conversation = await services.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return await services.list_messages(conversation_id, limit=limit)


@app.post("/conversations/{conversation_id}/messages", response_model=schemas.MessageOut)
async def add_message(conversation_id: str, request: schemas.MessageCreate):
    conversation = await services.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return await services.append_message(conversation_id, "user", request.content)


@app.post("/agents/{agent_id}/conversations/{conversation_id}/stream")
async def stream_agent_conversation(
    agent_id: str,
    conversation_id: str,
    request: schemas.ChatStreamRequest,
    stream: bool = Query(True),
):
    agent = await services.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    conversation = await services.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation["agent_id"] != agent_id:
        raise HTTPException(status_code=400, detail="Conversation does not belong to agent")

    if stream:

        async def event_stream():
            async for chunk in services.stream_response(conversation_id, agent, request.content):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    reply = await services.complete_response(conversation_id, agent, request.content)
    return {"reply": reply}
