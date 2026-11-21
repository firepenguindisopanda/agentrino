import json
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

load_dotenv()

import admin_sessions
import langgraph_agent
import model_router
import pinecone_service
import repositories
import schemas
import services
import telemetry
from mongo import ping_mongo

telemetry.configure_logging(os.getenv("LOG_LEVEL", "INFO"), os.getenv("LOG_FORMAT", "json"))
logger = logging.getLogger("agentrino")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown.

    Startup work (index creation, Pinecone setup) is wrapped so a temporarily
    unreachable dependency (e.g. a paused MongoDB Atlas cluster) degrades the
    service instead of crashing it. The app still boots and serves /health,
    and reconnects to MongoDB on the next request once it is reachable again.
    """
    try:
        await repositories.ensure_indexes()
        logger.info("MongoDB indexes ensured")
    except Exception as exc:  # noqa: BLE001 - never let a DB blip kill startup
        logger.error("Could not ensure MongoDB indexes at startup: %s", exc)

    try:
        pinecone_service.ensure_index()
        logger.info("Pinecone index ready")
    except Exception as exc:  # noqa: BLE001 - Pinecone is non-critical at boot
        logger.error("Could not ensure Pinecone index at startup: %s", exc)

    yield


app = FastAPI(title="Agentrino API", version="0.2.0", lifespan=lifespan)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Probes and the dashboard's own polling are excluded from RED metrics: every
# open tab pings /health, so counting it would let probe traffic (and a known,
# separately-reported DB outage) swamp the error rate and latency of real API
# calls. Health is surfaced by /health itself and the frontend banner.
_UNMETERED_ROUTES = frozenset({"/", "/health", "/admin/metrics"})


@app.exception_handler(InvalidId)
async def invalid_object_id(_request: Request, _exc: InvalidId):
    """A malformed resource ID can never match a document: that's a 404, not a
    server error (previously it surfaced as an unhandled 500)."""
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Correlation ID + RED metrics for every request.

    Accepts a well-formed inbound X-Request-ID (else generates one), binds it to
    the logging context, echoes it on the response, and records method, route
    *template* (bounded cardinality), status and duration. For streaming
    responses the duration is time-to-headers; stream length is tracked by the
    LLM call telemetry instead.
    """
    inbound = request.headers.get("x-request-id", "")
    request_id = inbound if _REQUEST_ID_RE.match(inbound) else uuid.uuid4().hex
    token = telemetry.request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Convert here (inside CORS) rather than letting Starlette's outer
            # error handler do it: that path skips CORS headers, so browsers
            # report a bare network failure instead of a 500 they can act on.
            logger.exception("unhandled error", extra={"event": "unhandled_exception", "path": request.url.path})
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error", "request_id": request_id}
            )
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        route = request.scope.get("route")
        route_template = getattr(route, "path", None) or "unmatched"
        if request.method != "OPTIONS" and route_template not in _UNMETERED_ROUTES:
            telemetry.record_request(request.method, route_template, status, duration_ms)
        if status >= 500:
            logger.error(
                "request failed",
                extra={
                    "event": "http_request",
                    "method": request.method,
                    "route": route_template,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
        telemetry.request_id_var.reset(token)

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
    expose_headers=["X-Request-ID"],
)


@app.get("/")
async def root():
    """Liveness probe: confirms the process is up and serving. Always 200."""
    return {"service": "agentrino", "status": "ok"}


@app.get("/health")
async def health():
    """Readiness/health check with dependency status.

    Returns 200 when MongoDB is reachable, 503 (degraded) otherwise, so
    monitoring can distinguish 'process alive' (/) from 'fully ready' (/health).
    """
    mongo_ok = await ping_mongo()
    payload = {
        "status": "ok" if mongo_ok else "degraded",
        "checks": {"mongodb": "up" if mongo_ok else "down"},
    }
    return JSONResponse(status_code=200 if mongo_ok else 503, content=payload)


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
        request_id = telemetry.request_id_var.get()

        async def event_stream():
            # The body runs after the middleware returns, so re-bind the ID.
            token = telemetry.request_id_var.set(request_id)
            meta: dict = {}
            try:
                async for chunk in services.stream_response(conversation_id, agent, request.content, meta=meta):
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                yield f"event: done\ndata: {json.dumps({'done': True, **meta})}\n\n"
            except Exception:
                logger.exception("stream failed", extra={"event": "stream_failed", "agent_id": agent_id})
                payload = {
                    "error": "The model failed while generating a reply. Please try again.",
                    "request_id": request_id,
                }
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            finally:
                telemetry.request_id_var.reset(token)

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
        return await langgraph_agent.invoke_oracle_agent(request.content, system_prompt=system_prompt)
    except (json.JSONDecodeError, ValidationError):
        logger.warning("oracle returned invalid analysis", exc_info=True, extra={"event": "oracle_invalid"})
        raise HTTPException(
            status_code=502, detail="The Oracle returned an incomplete analysis. Please try again or rephrase."
        )
    except Exception as exc:
        timed_out = "timeout" in type(exc).__name__.lower()
        logger.error("oracle failed", exc_info=True, extra={"event": "oracle_failed", "timed_out": timed_out})
        if timed_out:
            raise HTTPException(status_code=504, detail="The Oracle took too long to respond. Please try again.")
        raise HTTPException(status_code=502, detail="The Oracle is unavailable right now. Please try again.")



ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

_bearer = HTTPBearer(auto_error=False, description="Admin session token from POST /admin/auth")
_UNAUTHENTICATED = HTTPException(
    status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"}
)


def _client_key(request: Request) -> str:
    # Behind a proxy (e.g. Render) this is the proxy's address, so the throttle
    # becomes global. That still caps password guessing; it just means failed
    # attempts from anyone briefly delay everyone.
    return request.client.host if request.client else "unknown"


def require_admin(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """Dependency for admin-only routes: a live session token, sent as a Bearer header."""
    token = credentials.credentials if credentials else None
    if admin_sessions.sessions.expires_at(token) is None:
        raise _UNAUTHENTICATED
    return token


@app.post("/admin/auth", response_model=schemas.AdminSession)
async def admin_login(body: schemas.AdminLogin, request: Request):
    """Exchange the admin password for an expiring session token."""
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin access is not configured on this server")
    client = _client_key(request)
    retry_after = admin_sessions.login_throttle.retry_after(client)
    if retry_after:
        logger.warning("admin login throttled", extra={"event": "admin_login_throttled"})
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
    if not secrets.compare_digest(body.password.encode(), ADMIN_PASSWORD.encode()):
        admin_sessions.login_throttle.record_failure(client)
        logger.warning("admin login failed", extra={"event": "admin_login_failed"})
        raise HTTPException(status_code=401, detail="Invalid password")
    admin_sessions.login_throttle.reset(client)
    session = admin_sessions.sessions.create()
    logger.info("admin login", extra={"event": "admin_login"})
    return {"authenticated": True, "token": session.token, "expires_at": session.expires_at}


@app.get("/admin/session", response_model=schemas.AdminSessionStatus)
async def admin_session_status(token: str = Depends(require_admin)):
    """Lets the console confirm a stored token is still valid (e.g. after a server restart)."""
    return {"authenticated": True, "expires_at": admin_sessions.sessions.expires_at(token)}


@app.post("/admin/logout", status_code=204)
async def admin_logout(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    """Revoke the caller's session. Idempotent: an unknown or expired token is fine."""
    admin_sessions.sessions.revoke(credentials.credentials if credentials else None)
    return Response(status_code=204)


@app.get("/admin/metrics")
async def admin_metrics(
    _admin: str = Depends(require_admin),
    window_minutes: int = Query(60, ge=5, le=24 * 60),
):
    """Live operational analytics: HTTP RED metrics, LLM call telemetry and the
    effective task-to-tier-to-model routing table. In-memory; resets on restart."""
    return {**telemetry.summary(window_minutes), "routing": model_router.routing_table()}


@app.post("/admin/documents/upload")
async def upload_document(file: UploadFile = File(...), _admin: str = Depends(require_admin)):
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
async def list_documents(_admin: str = Depends(require_admin)):
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
async def delete_document(doc_id: str, _admin: str = Depends(require_admin)):
    pinecone_service.delete_documents([doc_id])
    return {"deleted": True}


# ---- Agent management (admin-protected) ----


@app.post("/agents", response_model=schemas.AgentOut, status_code=201)
async def create_agent(request: schemas.AgentCreate, _admin: str = Depends(require_admin)):
    return await services.create_agent(request.model_dump())


@app.patch("/agents/{agent_id}", response_model=schemas.AgentOut)
async def update_agent(agent_id: str, request: schemas.AgentUpdate, _admin: str = Depends(require_admin)):
    updates = request.model_dump(exclude_unset=True)
    agent = await services.update_agent(agent_id, updates)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, _admin: str = Depends(require_admin)):
    success = await services.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"deleted": True}
