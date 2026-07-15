"""Auto-recall behaviour in ``before_run``."""

from __future__ import annotations

import pytest
from agent_framework import AgentSession

from conftest import RecordingAsyncClient, make_context, make_recall_result
from infolang_agent_framework import InfoLangContextProvider


async def _run_before(provider: InfoLangContextProvider, context, session=None):
    session = session or AgentSession(session_id="thread-1")
    state: dict = {}
    await provider.before_run(agent=None, session=session, context=context, state=state)
    return state


async def test_recall_injects_instructions(session):
    client = RecordingAsyncClient(
        recall_result=make_recall_result("Rust is the favorite language", "Prefers dark mode")
    )
    provider = InfoLangContextProvider(client, namespace="acme")
    context = make_context(["What is my favorite language?"])

    state = await _run_before(provider, context, session)

    joined = "\n".join(context.instructions)
    assert "Rust is the favorite language" in joined
    assert "Prefers dark mode" in joined
    assert provider._context_instruction in joined
    assert state["recalled_chunk_ids"] == ["c0", "c1"]

    name, args, kwargs = client.calls[0]
    assert name == "recall"
    assert args[0] == "What is my favorite language?"
    assert kwargs["namespace"] == "acme"
    assert kwargs["top_k"] == 5


async def test_recall_uses_investigate_when_configured(session):
    client = RecordingAsyncClient(investigate_result=make_recall_result("hit"))
    provider = InfoLangContextProvider(client, namespace="acme", use_investigate=True)
    context = make_context(["question"])

    await _run_before(provider, context, session)

    name, _args, kwargs = client.calls[0]
    assert name == "investigate"
    assert kwargs["namespace_hint"] == "acme"


async def test_no_input_messages_skips_recall():
    client = RecordingAsyncClient(recall_result=make_recall_result("x"))
    provider = InfoLangContextProvider(client)
    context = make_context([])

    await _run_before(provider, context)

    assert client.calls == []
    assert context.instructions == []


async def test_blank_input_skips_recall():
    client = RecordingAsyncClient(recall_result=make_recall_result("x"))
    provider = InfoLangContextProvider(client)
    context = make_context(["   "])

    await _run_before(provider, context)

    assert client.calls == []


async def test_no_chunks_injects_nothing():
    client = RecordingAsyncClient(recall_result=make_recall_result())
    provider = InfoLangContextProvider(client)
    context = make_context(["hello"])

    state = await _run_before(provider, context)

    assert context.instructions == []
    assert "recalled_chunk_ids" not in state


async def test_min_score_filters_weak_chunks():
    client = RecordingAsyncClient(
        recall_result=make_recall_result("strong", "weak", scores=[0.95, 0.40], ids=["s", "w"])
    )
    provider = InfoLangContextProvider(client, min_score=0.85)
    context = make_context(["q"])

    state = await _run_before(provider, context)

    joined = "\n".join(context.instructions)
    assert "strong" in joined
    assert "weak" not in joined
    assert state["recalled_chunk_ids"] == ["s"]


async def test_min_score_with_missing_score_treated_as_zero():
    result = make_recall_result("no-score")
    result.chunks[0].score = None
    client = RecordingAsyncClient(recall_result=result)
    provider = InfoLangContextProvider(client, min_score=0.5)
    context = make_context(["q"])

    await _run_before(provider, context)

    assert context.instructions == []


async def test_top_k_caps_injected_chunks():
    client = RecordingAsyncClient(
        recall_result=make_recall_result("a", "b", "c", ids=["a", "b", "c"])
    )
    provider = InfoLangContextProvider(client, top_k=2)
    context = make_context(["q"])

    state = await _run_before(provider, context)

    assert state["recalled_chunk_ids"] == ["a", "b"]
    _name, _args, kwargs = client.calls[0]
    assert kwargs["top_k"] == 2


async def test_auto_recall_disabled():
    client = RecordingAsyncClient(recall_result=make_recall_result("x"))
    provider = InfoLangContextProvider(client, auto_recall=False)
    context = make_context(["q"])

    await _run_before(provider, context)

    assert client.calls == []


async def test_session_scoped_namespace_on_recall():
    client = RecordingAsyncClient(recall_result=make_recall_result("x"))
    provider = InfoLangContextProvider(client, namespace="base", session_scoped=True)
    context = make_context(["q"], session_id="thread-9")
    session = AgentSession(session_id="thread-9")

    await _run_before(provider, context, session)

    _name, _args, kwargs = client.calls[0]
    assert kwargs["namespace"] == "base.thread-9"


async def test_recall_error_swallowed_by_default(caplog):
    class Boom(RecordingAsyncClient):
        async def recall(self, query: str, **kwargs):  # type: ignore[override]
            raise RuntimeError("backend down")

    provider = InfoLangContextProvider(Boom())
    context = make_context(["q"])

    # Should not raise.
    await _run_before(provider, context)
    assert context.instructions == []


async def test_recall_error_raised_when_configured():
    class Boom(RecordingAsyncClient):
        async def recall(self, query: str, **kwargs):  # type: ignore[override]
            raise RuntimeError("backend down")

    provider = InfoLangContextProvider(Boom(), raise_on_error=True)
    context = make_context(["q"])

    with pytest.raises(RuntimeError, match="backend down"):
        await _run_before(provider, context)
