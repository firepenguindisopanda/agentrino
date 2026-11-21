import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Lazily-initialized client so importing this module has no side effects and
# does not fail when NVIDIA_API_KEY is absent (e.g. during tests). The key is
# still required the first time the LLM is actually called.
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
        _client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    return _client

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 4512
ORACLE_MAX_TOKENS = 8192
ORACLE_TIMEOUT = 60  # seconds
TITLE_MAX_TOKENS = 24
TITLE_TIMEOUT = 20  # seconds


def generate_title(user_message: str) -> str:
    """Generate a short, human-readable title from a user's first message.

    Returns a cleaned title string (empty string if the model returns nothing
    usable). This is a blocking network call; run it off the event loop.
    """
    system_prompt = (
        "You write a short title for a chat conversation based on the user's first "
        "message. Reply with the title only: 3 to 6 words, Title Case, no surrounding "
        "quotes, and no trailing punctuation."
    )
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message[:2000]},
        ],
        max_tokens=TITLE_MAX_TOKENS,
        temperature=0.3,
        timeout=TITLE_TIMEOUT,
    )
    raw = response.choices[0].message.content or ""
    # Keep the first non-empty line and strip wrapping quotes / trailing punctuation.
    title = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    title = title.strip().strip('"').strip("'").rstrip(".!?,;:").strip()
    return title[:120]


def stream_response(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        # history should be a list of {"role": "user"|"assistant", "content": "..."}
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content


def get_response_text(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None) -> str:
    out = []
    for chunk_content in stream_response(prompt, system_prompt=system_prompt, history=history):
        out.append(chunk_content)
    return "".join(out)


def get_oracle_response_structured(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None) -> dict:
    import json

    from schemas import OracleAnalysisResponse

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=ORACLE_MAX_TOKENS,
        response_format={"type": "json_object"},
        temperature=0.1,
        timeout=ORACLE_TIMEOUT,
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    # Validate against the Pydantic schema (enforces 4 options, required fields, etc.)
    validated = OracleAnalysisResponse(**parsed)
    return validated.model_dump()
