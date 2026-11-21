"""Tests for task-based model routing, telemetry, and the API's error/observability
behavior. No network or database access: all collaborators are stubbed."""

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

import langgraph_agent
import llm
import main
import model_router
import services
import telemetry
from model_router import Task

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    telemetry.reset()
    for key in list(__import__("os").environ):
        if key.startswith(("LLM_MODEL", "LLM_TIER_", "LLM_MAX_TOKENS_")):
            monkeypatch.delenv(key, raising=False)
    yield
    telemetry.reset()


# ---------------------------------------------------------------- routing


def test_default_routing_uses_cheapest_adequate_tier():
    assert model_router.resolve(Task.TITLE).tier == "light"
    assert model_router.resolve(Task.CHAT).tier == "standard"
    assert model_router.resolve(Task.ORACLE).tier == "strong"


def test_unconfigured_tiers_fall_back_to_single_model(monkeypatch):
    models = {r["model"] for r in model_router.routing_table()}
    assert models == {model_router.DEFAULT_MODEL}
    monkeypatch.setenv("LLM_MODEL", "base-model")
    assert {r["model"] for r in model_router.routing_table()} == {"base-model"}


def test_env_overrides_tier_model_and_budget(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_STRONG", "big-model")
    monkeypatch.setenv("LLM_MAX_TOKENS_STRONG", "1000")
    monkeypatch.setenv("LLM_TIER_CHAT", "strong")
    route = model_router.resolve("chat")
    assert (route.tier, route.model, route.max_tokens) == ("strong", "big-model", 1000)


def test_invalid_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("LLM_TIER_TITLE", "gigantic")
    monkeypatch.setenv("LLM_MAX_TOKENS_LIGHT", "-5")
    route = model_router.resolve(Task.TITLE)
    assert route.tier == "light"
    assert route.max_tokens == model_router.DEFAULT_MAX_TOKENS[model_router.Tier.LIGHT]


# ---------------------------------------------------------------- llm telemetry


def _chunk(content):
    delta = type("Delta", (), {"content": content})
    choice = type("Choice", (), {"delta": delta()})
    return type("Chunk", (), {"choices": [choice()], "usage": None})()


def _fake_client(monkeypatch, chunks=None, error=None, captured=None):
    def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        if error:
            raise error
        return iter([_chunk(c) for c in chunks])

    completions = type("C", (), {"create": staticmethod(create)})
    chat = type("Chat", (), {"completions": completions()})
    monkeypatch.setattr(llm, "_get_client", lambda: type("Client", (), {"chat": chat()})())


def test_stream_uses_routed_model_and_records_call(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_LIGHT", "tiny-model")
    captured = {}
    _fake_client(monkeypatch, chunks=["Hello ", "World"], captured=captured)

    assert llm.get_response_text("hi", task=Task.TITLE) == "Hello World"
    assert captured["model"] == "tiny-model"

    rec = telemetry.summary()["llm"]["recent"][0]
    assert rec["task"] == "title" and rec["tier"] == "light" and rec["status"] == "ok"
    assert rec["output_chars"] == 11 and rec["tokens_estimated"] is True
    assert rec["ttft_ms"] is not None


def test_stream_error_is_recorded_and_reraised(monkeypatch):
    _fake_client(monkeypatch, error=RuntimeError("provider down"))
    with pytest.raises(RuntimeError):
        llm.get_response_text("hi")
    summary = telemetry.summary()["llm"]
    assert summary["errors"] == 1
    assert summary["recent"][0]["error_type"] == "RuntimeError"


def test_abandoned_stream_is_recorded_as_cancelled(monkeypatch):
    _fake_client(monkeypatch, chunks=["a", "b", "c"])
    gen = llm.stream_response("hi")
    next(gen)
    gen.close()
    assert telemetry.summary()["llm"]["cancelled"] == 1


# ---------------------------------------------------------------- event loop


def test_iterate_in_thread_runs_blocking_iterator_off_the_loop():
    loop_thread = []
    seen_threads = []

    def blocking():
        for i in range(3):
            seen_threads.append(threading.get_ident())
            yield str(i)

    async def run():
        loop_thread.append(threading.get_ident())
        return [c async for c in langgraph_agent.iterate_in_thread(blocking())]

    assert asyncio.run(run()) == ["0", "1", "2"]
    assert loop_thread[0] not in seen_threads


# ---------------------------------------------------------------- aggregation


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert telemetry.percentile(values, 50) == 50
    assert telemetry.percentile(values, 95) == 95
    assert telemetry.percentile(values, 99) == 99
    assert telemetry.percentile([], 50) is None
    assert telemetry.percentile([7.0], 99) == 7.0


def test_summary_aggregates_http_red_metrics():
    for ms in (10, 20, 30):
        telemetry.record_request("GET", "/agents", 200, ms)
    telemetry.record_request("POST", "/agents/oracle/analyze", 502, 900)
    http = telemetry.summary()["http"]
    assert http["requests"] == 4
    assert http["errors"] == 1
    assert http["error_rate"] == 0.25
    assert http["status_classes"] == {"2xx": 3, "5xx": 1}
    assert http["routes"][0]["route"] == "/agents"
    assert len(http["series"]) == 60
    assert sum(b["count"] for b in http["series"]) == 4


# ---------------------------------------------------------------- API


def test_responses_carry_request_id_and_honor_inbound_id():
    generated = client.get("/")
    assert len(generated.headers["X-Request-ID"]) == 32
    echoed = client.get("/", headers={"X-Request-ID": "trace-abc_123"})
    assert echoed.headers["X-Request-ID"] == "trace-abc_123"
    unsafe = client.get("/", headers={"X-Request-ID": "bad id\nwith newline"})
    assert unsafe.headers["X-Request-ID"] != "bad id\nwith newline"


def test_probe_and_dashboard_routes_are_not_metered():
    client.get("/")
    client.get("/health")
    client.get("/admin/metrics")
    assert telemetry.summary()["http"]["requests"] == 0


def test_requests_are_recorded_by_route_template():
    client.get("/conversations/abc123", headers={})
    routes = {r["route"] for r in telemetry.summary()["http"]["routes"]}
    assert "/conversations/{conversation_id}" in routes
    assert not any("abc123" in r for r in routes)


def test_metrics_endpoint_requires_admin(monkeypatch):
    assert client.get("/admin/metrics").status_code == 401
    monkeypatch.setattr(main, "ADMIN_PASSWORD", "s3cret")
    # The raw password is no longer accepted as a credential on data routes.
    assert client.get("/admin/metrics", headers={"X-Admin-Password": "s3cret"}).status_code == 401
    token = client.post("/admin/auth", json={"password": "s3cret"}).json()["token"]
    ok = client.get("/admin/metrics", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    body = ok.json()
    assert {"http", "llm", "routing", "uptime_seconds"} <= body.keys()
    assert {r["task"] for r in body["routing"]} == {"title", "chat", "oracle"}


@pytest.mark.parametrize(
    "error, status",
    [
        (json.JSONDecodeError("bad", "", 0), 502),
        (TimeoutError("slow"), 504),
        (RuntimeError("secret internal detail"), 502),
    ],
)
def test_oracle_errors_map_to_safe_responses(monkeypatch, error, status):
    async def boom(*_a, **_k):
        raise error

    monkeypatch.setattr(langgraph_agent, "invoke_oracle_agent", boom)
    res = client.post("/agents/oracle/analyze", json={"content": "monolith or microservices?"})
    assert res.status_code == status
    assert "secret internal detail" not in res.text


def _stub_chat(monkeypatch, gen_factory):
    async def get_agent(_id):
        return {"id": "a1", "system_prompt": "x"}

    async def get_conversation(_id):
        return {"id": "c1", "agent_id": "a1"}

    monkeypatch.setattr(services, "get_agent", get_agent)
    monkeypatch.setattr(services, "get_conversation", get_conversation)
    monkeypatch.setattr(services, "stream_response", gen_factory)


def test_stream_done_event_includes_rag_metadata(monkeypatch):
    async def ok(_cid, _agent, _content, meta=None):
        meta.update({"rag_used": True, "rag_docs_count": 3})
        yield "hi"

    _stub_chat(monkeypatch, ok)
    res = client.post("/agents/a1/conversations/c1/stream", json={"content": "hello"})
    assert 'data: {"text": "hi"}' in res.text
    assert '"rag_docs_count": 3' in res.text and "event: done" in res.text


def test_stream_failure_emits_sse_error_event(monkeypatch):
    async def failing(_cid, _agent, _content, meta=None):
        yield "partial"
        raise RuntimeError("provider exploded")

    _stub_chat(monkeypatch, failing)
    res = client.post("/agents/a1/conversations/c1/stream", json={"content": "hello"})
    assert "event: error" in res.text
    assert "provider exploded" not in res.text
    assert "event: done" not in res.text


def test_oversized_messages_are_rejected_before_reaching_the_model(monkeypatch):
    called = []

    async def spy(*_a, **_k):
        called.append(True)
        return {}

    monkeypatch.setattr(langgraph_agent, "invoke_oracle_agent", spy)
    res = client.post("/agents/oracle/analyze", json={"content": "x" * 8001})
    assert res.status_code == 422
    assert not called


def test_unhandled_errors_return_json_500_with_cors_and_request_id(monkeypatch):
    async def boom():
        raise RuntimeError("db exploded")

    monkeypatch.setattr(services, "list_agents", boom)
    res = client.get("/agents", headers={"Origin": "http://localhost:3001"})
    assert res.status_code == 500
    assert res.json()["detail"] == "Internal server error"
    assert res.json()["request_id"] == res.headers["X-Request-ID"]
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3001"
    assert "db exploded" not in res.text
