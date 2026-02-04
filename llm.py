import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable not set")

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 4512


def stream_response(prompt: str, system_prompt: str | None = None, history: list[dict] | None = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        # history should be a list of {"role": "user"|"assistant", "content": "..."}
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
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
