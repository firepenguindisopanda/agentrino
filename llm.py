import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

import model_router
import telemetry
from model_router import Task

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

logger = logging.getLogger(__name__)

# Lazily-initialized client so importing this module has no side effects and
# does not fail when NVIDIA_API_KEY is absent (e.g. during tests). The key is
# still required the first time the LLM is actually called.
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=NVIDIA_API_KEY)
    return _client


# Kept for backwards compatibility with callers that read these constants.
MODEL = model_router.DEFAULT_MODEL
MAX_TOKENS = model_router.DEFAULT_MAX_TOKENS[model_router.Tier.STANDARD]
ORACLE_MAX_TOKENS = model_router.DEFAULT_MAX_TOKENS[model_router.Tier.STRONG]
ORACLE_TIMEOUT = 60  # seconds


def _estimate_tokens(chars: int) -> int:
    # ~4 characters per token for English text; flagged as an estimate.
    return max(0, round(chars / 4))


def _build_messages(prompt: str, system_prompt: str | None, history: list[dict] | None) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        # history should be a list of {"role": "user"|"assistant", "content": "..."}
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _record(route: model_router.ModelRoute, status: str, started: float, **fields) -> None:
    rec = telemetry.LLMCallRecord(
        ts=time.time(),
        task=route.task,
        tier=route.tier,
        model=route.model,
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        **fields,
    )
    telemetry.record_llm_call(rec)
    log = logger.warning if status == "error" else logger.info
    log(
        "llm call %s",
        status,
        extra={
            "event": "llm_call",
            "task": rec.task,
            "tier": rec.tier,
            "model": rec.model,
            "status": status,
            "latency_ms": rec.latency_ms,
            "ttft_ms": rec.ttft_ms,
            "output_tokens": rec.output_tokens,
            "error_type": rec.error_type,
        },
    )


def generate_title(user_message: str) -> str:
    """Generate a short, human-readable title from a user's first message.

    Uses the same streaming path as chat (a non-streaming call returns empty
    content on reasoning-style models), then cleans the result. Returns an
    empty string if nothing usable comes back. Blocking; run it off the loop.
    """
    system_prompt = (
        "You write a short title for a chat conversation based on the user's first "
        "message. Reply with the title only: 3 to 6 words, Title Case, no surrounding "
        "quotes, and no trailing punctuation."
    )
    raw = get_response_text(user_message[:2000], system_prompt=system_prompt, task=Task.TITLE)
    # Keep the first non-empty line and strip wrapping quotes / trailing punctuation.
    title = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    title = title.strip().strip('"').strip("'").rstrip(".!?,;:").strip()
    return title[:120]


def stream_response(
    prompt: str,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
    task: Task | str = Task.CHAT,
):
    """Yield text deltas from the model routed for ``task``. Blocking iterator;
    async callers must drive it from a worker thread.

    Every call is recorded (latency, time-to-first-token, output size, status),
    including streams the client abandons part-way (status ``cancelled``).
    """
    route = model_router.resolve(task)
    started = time.perf_counter()
    ttft_ms = None
    chars = 0
    usage_tokens = None
    status = "ok"
    error_type = None
    try:
        response = _get_client().chat.completions.create(
            model=route.model,
            messages=_build_messages(prompt, system_prompt, history),
            max_tokens=route.max_tokens,
            stream=True,
        )
        for chunk in response:
            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None) is not None:
                usage_tokens = usage.completion_tokens
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started) * 1000, 1)
                    chars += len(delta.content)
                    yield delta.content
    except GeneratorExit:
        status = "cancelled"
        raise
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        _record(
            route,
            status,
            started,
            ttft_ms=ttft_ms,
            output_chars=chars,
            output_tokens=usage_tokens if usage_tokens is not None else _estimate_tokens(chars),
            tokens_estimated=usage_tokens is None,
            error_type=error_type,
        )


def get_response_text(
    prompt: str,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
    task: Task | str = Task.CHAT,
) -> str:
    return "".join(stream_response(prompt, system_prompt=system_prompt, history=history, task=task))


def get_oracle_response_structured(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None) -> dict:
    """Blocking structured call for the Oracle. Raises ``json.JSONDecodeError``
    or ``pydantic.ValidationError`` when the model returns an unusable payload."""
    from schemas import OracleAnalysisResponse

    route = model_router.resolve(Task.ORACLE)
    started = time.perf_counter()
    try:
        response = _get_client().chat.completions.create(
            model=route.model,
            messages=_build_messages(prompt, system_prompt, history),
            max_tokens=route.max_tokens,
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=ORACLE_TIMEOUT,
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        # Validate against the Pydantic schema (enforces 4 options, required fields, etc.)
        validated = OracleAnalysisResponse(**parsed)
    except Exception as exc:
        _record(route, "error", started, error_type=type(exc).__name__)
        raise

    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    _record(
        route,
        "ok",
        started,
        output_chars=len(content),
        output_tokens=completion_tokens if completion_tokens is not None else _estimate_tokens(len(content)),
        tokens_estimated=completion_tokens is None,
    )
    return validated.model_dump()
