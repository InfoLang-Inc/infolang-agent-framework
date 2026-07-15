"""End-to-end style test: memory retained in one thread is recalled in another,
with no tools added to the context (zero tool calls)."""

from __future__ import annotations

from typing import Any

from agent_framework import AgentSession
from infolang import RecallResult, RememberResult
from infolang.types import Chunk

from conftest import make_context
from infolang_agent_framework import InfoLangContextProvider


class InMemoryInfoLang:
    """Tiny in-memory async InfoLang stand-in with naive word-overlap recall."""

    def __init__(self) -> None:
        self.store: list[tuple[str | None, str, str]] = []  # (namespace, id, text)
        self._counter = 0

    async def remember(
        self, text: str, *, namespace: str | None = None, **_: Any
    ) -> RememberResult:
        self._counter += 1
        mem_id = f"m{self._counter}"
        self.store.append((namespace, mem_id, text))
        return RememberResult(id=mem_id)

    async def recall(
        self, query: str, *, namespace: str | None = None, top_k: int | None = None, **_: Any
    ) -> RecallResult:
        terms = {w.lower().strip("?.,!") for w in query.split()}
        chunks: list[Chunk] = []
        for ns, mem_id, text in self.store:
            if namespace is not None and ns != namespace:
                continue
            words = {w.lower().strip("?.,!") for w in text.split()}
            overlap = len(terms & words)
            if overlap:
                chunks.append(Chunk(id=mem_id, text=text, score=0.9))
        return RecallResult(chunks=chunks[: (top_k or len(chunks))])


async def test_memory_recalled_across_threads_without_tools():
    backend = InMemoryInfoLang()
    provider = InfoLangContextProvider(backend, namespace="acme")

    # --- Thread A: user shares a durable fact, agent responds. -----------
    thread_a = AgentSession(session_id="thread-A")
    ctx_a = make_context(
        ["My favorite programming language is Rust"],
        assistant_texts=["Noted — Rust it is."],
    )
    await provider.before_run(agent=None, session=thread_a, context=ctx_a, state={})
    # Nothing to recall yet, and no tools injected.
    assert ctx_a.instructions == []
    assert ctx_a.tools == []
    await provider.after_run(agent=None, session=thread_a, context=ctx_a, state={})

    assert backend.store  # the turn was retained

    # --- Thread B (fresh session): the fact is recalled. -----------------
    thread_b = AgentSession(session_id="thread-B")
    ctx_b = make_context(["What is my favorite programming language?"])
    state_b: dict = {}
    await provider.before_run(agent=None, session=thread_b, context=ctx_b, state=state_b)

    injected = "\n".join(ctx_b.instructions)
    assert "Rust" in injected
    assert state_b["recalled_chunk_ids"] == ["m1"]
    # The whole point: automatic memory with zero tool calls in the transcript.
    assert ctx_b.tools == []


async def test_session_scoped_memory_isolated_between_threads():
    backend = InMemoryInfoLang()
    provider = InfoLangContextProvider(backend, namespace="acme", session_scoped=True)

    thread_a = AgentSession(session_id="A")
    ctx_a = make_context(["secret alpha token"], assistant_texts=["ok"])
    await provider.after_run(agent=None, session=thread_a, context=ctx_a, state={})

    # A different thread must not see thread A's session-scoped memory.
    thread_b = AgentSession(session_id="B")
    ctx_b = make_context(["what is the alpha token"])
    await provider.before_run(agent=None, session=thread_b, context=ctx_b, state={})

    assert ctx_b.instructions == []
