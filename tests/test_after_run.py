"""Auto-retain behaviour in ``after_run``."""

from __future__ import annotations

import pytest
from agent_framework import AgentSession

from conftest import RecordingAsyncClient, make_context
from infolang_agent_framework import InfoLangContextProvider


async def _run_after(provider: InfoLangContextProvider, context, session=None):
    session = session or AgentSession(session_id="thread-1")
    state: dict = {}
    await provider.after_run(agent=None, session=session, context=context, state=state)
    return state


async def test_retain_stores_full_turn(session):
    client = RecordingAsyncClient()
    provider = InfoLangContextProvider(client, namespace="acme", tags="chat")
    context = make_context(["My name is Ada"], assistant_texts=["Nice to meet you, Ada"])

    state = await _run_after(provider, context, session)

    name, args, kwargs = client.calls[0]
    assert name == "remember"
    assert args[0] == "User: My name is Ada\nAssistant: Nice to meet you, Ada"
    assert kwargs["namespace"] == "acme"
    assert kwargs["source"] == "infolang"
    assert kwargs["tags"] == "chat"
    assert state["last_memory_id"] == "mem-1"


async def test_retain_user_only_when_no_response():
    client = RecordingAsyncClient()
    provider = InfoLangContextProvider(client)
    context = make_context(["Just a note"])  # no assistant response

    await _run_after(provider, context)

    _name, args, _kwargs = client.calls[0]
    assert args[0] == "User: Just a note"


async def test_retain_assistant_only_when_no_input():
    client = RecordingAsyncClient()
    provider = InfoLangContextProvider(client)
    context = make_context([], assistant_texts=["Standalone answer"])

    await _run_after(provider, context)

    _name, args, _kwargs = client.calls[0]
    assert args[0] == "Assistant: Standalone answer"


async def test_retain_skips_when_turn_empty():
    client = RecordingAsyncClient()
    provider = InfoLangContextProvider(client)
    context = make_context([], assistant_texts=[])

    state = await _run_after(provider, context)

    assert client.calls == []
    assert "last_memory_id" not in state


async def test_retain_disabled():
    client = RecordingAsyncClient()
    provider = InfoLangContextProvider(client, auto_retain=False)
    context = make_context(["hi"], assistant_texts=["hello"])

    await _run_after(provider, context)

    assert client.calls == []


async def test_retain_session_scoped_namespace():
    client = RecordingAsyncClient()
    provider = InfoLangContextProvider(client, session_scoped=True)
    context = make_context(["hi"], assistant_texts=["hello"])
    session = AgentSession(session_id="t-42")

    await _run_after(provider, context, session)

    _name, _args, kwargs = client.calls[0]
    assert kwargs["namespace"] == "t-42"


async def test_retain_error_swallowed_by_default():
    class Boom(RecordingAsyncClient):
        async def remember(self, text: str, **kwargs):  # type: ignore[override]
            raise RuntimeError("write failed")

    provider = InfoLangContextProvider(Boom())
    context = make_context(["hi"], assistant_texts=["hello"])

    # Should not raise.
    await _run_after(provider, context)


async def test_retain_error_raised_when_configured():
    class Boom(RecordingAsyncClient):
        async def remember(self, text: str, **kwargs):  # type: ignore[override]
            raise RuntimeError("write failed")

    provider = InfoLangContextProvider(Boom(), raise_on_error=True)
    context = make_context(["hi"], assistant_texts=["hello"])

    with pytest.raises(RuntimeError, match="write failed"):
        await _run_after(provider, context)


async def test_retain_handles_result_without_memory_id():
    from infolang import RememberResult

    client = RecordingAsyncClient(remember_result=RememberResult())
    provider = InfoLangContextProvider(client)
    context = make_context(["hi"], assistant_texts=["hello"])

    state = await _run_after(provider, context)

    assert client.calls[0][0] == "remember"
    assert "last_memory_id" not in state
