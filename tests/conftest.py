"""Shared test fixtures/config.

Provides safe dummy environment defaults so the FastAPI app is importable in
tests without real secrets. Modules like llm/pinecone_service/redis_cache raise
at import time if their env vars are missing, so these must be set before
`main` is imported (conftest is loaded before test modules).
"""

import os

_ENV_DEFAULTS = {
    "NVIDIA_API_KEY": "test-nvidia-key",
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_INDEX": "test-index",
    "MONGO_URI": "mongodb://localhost:27017",
    "MONGO_DB_NAME": "agent_chatter_test",
    "UPSTASH_REDIS_REST_URL": "https://example.upstash.io",
    "UPSTASH_REDIS_REST_TOKEN": "test-token",
}

for _key, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)
