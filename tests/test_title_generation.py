"""Tests for auto-generated conversation titles.

Covers the LLM output cleaning and the best-effort title service, without any
network or database access (all collaborators are stubbed).
"""

import asyncio

import llm
import services


def _run(coro):
    return asyncio.run(coro)


def _chunk(content):
    delta = type("Delta", (), {"content": content})
    choice = type("Choice", (), {"delta": delta()})
    return type("Chunk", (), {"choices": [choice()]})()


def _patch_client(monkeypatch, content):
    # generate_title streams, so return an iterable of streaming-style chunks.
    chunks = [_chunk(content)] if content is not None else [_chunk(None)]

    def create(**_kwargs):
        return iter(chunks)

    completions = type("Completions", (), {"create": staticmethod(create)})
    chat = type("Chat", (), {"completions": completions()})
    client = type("Client", (), {"chat": chat()})
    monkeypatch.setattr(llm, "_get_client", lambda: client())


def test_generate_title_strips_quotes_punctuation_and_extra_lines(monkeypatch):
    _patch_client(monkeypatch, '"Trip To Japan."\nSome extra reasoning')
    assert llm.generate_title("Help me plan a trip to Japan") == "Trip To Japan"


def test_generate_title_handles_empty_content(monkeypatch):
    _patch_client(monkeypatch, None)
    assert llm.generate_title("hello") == ""


def test_maybe_generate_title_skips_when_already_titled(monkeypatch):
    calls = {"llm": False, "set": False}

    async def fake_get_conversation(_cid):
        return {"id": _cid, "title": "Existing Title"}

    def fake_generate_title(_msg):
        calls["llm"] = True
        return "New Title"

    async def fake_set_title(_cid, _title):
        calls["set"] = True

    monkeypatch.setattr(services.repositories, "get_conversation", fake_get_conversation)
    monkeypatch.setattr(services.llm, "generate_title", fake_generate_title)
    monkeypatch.setattr(services.repositories, "set_conversation_title", fake_set_title)

    _run(services.maybe_generate_title("c1", "anything"))
    assert calls == {"llm": False, "set": False}


def test_maybe_generate_title_sets_title_when_missing(monkeypatch):
    saved = {}

    async def fake_get_conversation(_cid):
        return {"id": _cid, "title": None}

    def fake_generate_title(_msg):
        return "Trip To Japan"

    async def fake_set_title(cid, title):
        saved["cid"] = cid
        saved["title"] = title

    monkeypatch.setattr(services.repositories, "get_conversation", fake_get_conversation)
    monkeypatch.setattr(services.llm, "generate_title", fake_generate_title)
    monkeypatch.setattr(services.repositories, "set_conversation_title", fake_set_title)

    _run(services.maybe_generate_title("c1", "Help me plan a trip to Japan"))
    assert saved == {"cid": "c1", "title": "Trip To Japan"}


def test_maybe_generate_title_does_not_save_empty_title(monkeypatch):
    calls = {"set": False}

    async def fake_get_conversation(_cid):
        return {"id": _cid, "title": None}

    def fake_generate_title(_msg):
        return ""

    async def fake_set_title(_cid, _title):
        calls["set"] = True

    monkeypatch.setattr(services.repositories, "get_conversation", fake_get_conversation)
    monkeypatch.setattr(services.llm, "generate_title", fake_generate_title)
    monkeypatch.setattr(services.repositories, "set_conversation_title", fake_set_title)

    _run(services.maybe_generate_title("c1", "hi"))
    assert calls["set"] is False


def test_maybe_generate_title_swallows_errors(monkeypatch):
    async def fake_get_conversation(_cid):
        raise RuntimeError("db down")

    monkeypatch.setattr(services.repositories, "get_conversation", fake_get_conversation)
    # Must not raise despite the failure.
    _run(services.maybe_generate_title("c1", "anything"))
