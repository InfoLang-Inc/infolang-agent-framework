"""Shared fixtures and fakes for the InfoLang context-provider tests.

Tests use the *real* Agent Framework types (``Message``, ``SessionContext``,
``AgentSession``, ``AgentResponse``) so behaviour is verified against the
installed framework, while InfoLang network calls are replaced with in-memory
recording fakes (offline by default).
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import AgentResponse, AgentSession, Message, SessionContext
from infolang import RecallResult, RememberResult
from infolang.types import Chunk


def make_recall_result(
    *texts: str,
    scores: list[float] | None = None,
    ids: list[str] | None = None,
    tags: list[str | None] | None = None,
) -> RecallResult:
    """Build a RecallResult with one chunk per text."""
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                id=(ids[index] if ids else f"c{index}"),
                text=text,
                score=(scores[index] if scores else 0.9),
                tags=(tags[index] if tags else None),
            )
        )
    return RecallResult(chunks=chunks)


class RecordingAsyncClient:
    """Async InfoLang stand-in that records calls and returns canned results."""

    def __init__(
        self,
        *,
        recall_result: RecallResult | None = None,
        investigate_result: RecallResult | None = None,
        remember_result: RememberResult | None = None,
    ) -> None:
        self.recall_result = recall_result or make_recall_result()
        self.investigate_result = investigate_result or make_recall_result()
        self.remember_result = remember_result or RememberResult(id="mem-1")
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.closed = False

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((name, args, kwargs))

    async def recall(self, query: str, **kwargs: Any) -> RecallResult:
        self._record("recall", (query,), kwargs)
        return self.recall_result

    async def investigate(self, query: str, **kwargs: Any) -> RecallResult:
        self._record("investigate", (query,), kwargs)
        return self.investigate_result

    async def remember(self, text: str, **kwargs: Any) -> RememberResult:
        self._record("remember", (text,), kwargs)
        return self.remember_result

    async def forget(self, memory_id: str, **kwargs: Any) -> None:
        self._record("forget", (memory_id,), kwargs)

    async def aclose(self) -> None:
        self.closed = True


def make_context(
    user_texts: list[str] | None = None,
    *,
    session_id: str | None = None,
    assistant_texts: list[str] | None = None,
) -> SessionContext:
    """Build a SessionContext with user input and an optional agent response."""
    input_messages = [Message("user", [text]) for text in (user_texts or [])]
    context = SessionContext(session_id=session_id, input_messages=input_messages)
    if assistant_texts is not None:
        context._response = AgentResponse(  # type: ignore[assignment]
            messages=[Message("assistant", [text]) for text in assistant_texts]
        )
    return context


@pytest.fixture
def async_client() -> RecordingAsyncClient:
    return RecordingAsyncClient()


@pytest.fixture
def session() -> AgentSession:
    return AgentSession(session_id="thread-1")
