# Agentrino API

FastAPI backend for Agentrino — a multi-agent chat application with streaming
responses, retrieval-augmented generation (RAG), and a structured-analysis
"Oracle" agent.

## Stack

- **FastAPI** — HTTP API with Server-Sent Events (SSE) streaming
- **MongoDB** (async `pymongo`) — conversations & messages
- **Pinecone** — vector store for RAG
- **Upstash Redis** — recent-message cache (optional; degrades to the DB)
- **LangGraph + NVIDIA-hosted LLM** — agent orchestration
- **uv** — dependency & environment management

## Quick start

```bash
cp .env.example .env     # fill in your keys
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## Health & operations

| Route      | Purpose                                                        |
| ---------- | -------------------------------------------------------------- |
| `GET /`    | Liveness — always `200` while the process is up.               |
| `GET /health` | Readiness — `200` when MongoDB is reachable, `503` if not.  |

Startup is resilient: if MongoDB or Pinecone is unreachable at boot (e.g. a
paused Atlas free cluster), the service logs the error and still starts, then
reconnects on the next request — it no longer crashes with "Exited with status 3".

A scheduled GitHub Action (`.github/workflows/keepalive.yml`) pings `/health`
so the Atlas free cluster never hits its ~60-day inactivity auto-pause.

## Testing

```bash
uv run pytest -q
```

Tests run without real secrets — `tests/conftest.py` supplies dummy env vars and
all external clients are lazily initialized.

## Deployment

Deployed to Render from the `Dockerfile`. See [SETUP.md](./SETUP.md) for the full
environment variable list and deployment guide.
