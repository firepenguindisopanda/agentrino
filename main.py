import json
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

import pinecone_service
import langgraph_agent
from pinecone_service import PINECONE_INDEX_NAME
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
    pinecone_service.ensure_index()


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

            yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    reply = await services.complete_response(conversation_id, agent, request.content)
    return {"reply": reply}


@app.post("/agents/oracle/analyze")
async def analyze_with_oracle(request: schemas.ChatStreamRequest):
    """
    Dedicated endpoint for the Oracle agent to generate a 4-option comparative analysis.
    Returns structured JSON conforming to OracleAnalysisResponse.
    """
    from prompt_templates import get_agent_prompt

    system_prompt = get_agent_prompt("oracle") + """

    IMPORTANT INSTRUCTION: CRITICAL! You must wrap your entire response in a JSON object conforming exactly to this schema:
    {
      "bottom_line": "string",
      "options": [
        {
          "title": "string",
          "description": "string",
          "pros": ["string"],
          "cons": ["string"],
          "effort": "string",
          "recommended": boolean
        }
      ],
      "action_plan": ["string"],
      "watch_out_for": ["string"]
    }
    """

    try:
        response_dict = await langgraph_agent.invoke_oracle_agent(request.content, system_prompt=system_prompt)
        return response_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/documents", response_model=schemas.DocumentAddResponse)
async def add_documents(request: schemas.DocumentAdd):
    from uuid import uuid4

    doc_id = str(uuid4())
    pinecone_service.delete_documents([doc_id])
    return schemas.DocumentAddResponse(ids=[doc_id])


@app.delete("/documents", response_model=schemas.DocumentDeleteResponse)
async def delete_documents(request: schemas.DocumentDelete):
    pinecone_service.delete_documents(request.ids)
    return schemas.DocumentDeleteResponse(deleted=True)


ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


@app.post("/admin/auth")
async def admin_login(request: schemas.AdminLogin):
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="Admin not configured")
    if request.password == ADMIN_PASSWORD:
        return {"authenticated": True}
    raise HTTPException(status_code=401, detail="Invalid password")


def verify_admin_auth(password: str | None) -> bool:
    if not password:
        return False
    return password == ADMIN_PASSWORD


@app.post("/admin/documents/upload")
async def upload_document(file: UploadFile = File(...), x_admin_password: str = Header(None)):
    if not verify_admin_auth(x_admin_password):
        raise HTTPException(status_code=401, detail="Not authenticated")

    import file_processor

    try:
        documents = file_processor.process_file(file.file, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not documents:
        raise HTTPException(status_code=400, detail="No content extracted from file")

    from uuid import uuid4

    doc_ids = []
    docs_to_upsert = []

    for doc in documents:
        doc_id = str(uuid4())
        doc_ids.append(doc_id)

        docs_to_upsert.append(
            {
                "id": doc_id,
                "text": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content,
                },
            }
        )

    pinecone_service.add_documents(docs_to_upsert)

    return {
        "filename": file.filename,
        "chunks": len(doc_ids),
        "ids": doc_ids,
    }


@app.get("/admin/documents")
async def list_documents(x_admin_password: str = Header(None)):
    if not verify_admin_auth(x_admin_password):
        raise HTTPException(status_code=401, detail="Not authenticated")

    index = pinecone_service.get_index()

    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
    except Exception:
        total_vectors = 0

    index_info = pinecone_service.get_index_info()

    return {
        "total_documents": total_vectors,
        **index_info,
    }


@app.delete("/admin/documents/{doc_id}")
async def delete_document(doc_id: str, x_admin_password: str = Header(None)):
    if not verify_admin_auth(x_admin_password):
        raise HTTPException(status_code=401, detail="Not authenticated")

    pinecone_service.delete_documents([doc_id])
    return {"deleted": True}
