"""InfoLang context provider for the Microsoft Agent Framework.

This module implements :class:`InfoLangContextProvider`, a subclass of the
Agent Framework ``ContextProvider`` that wires InfoLang semantic memory into an
agent's turn lifecycle:

* ``before_run`` recalls relevant memory for the incoming turn and injects it as
  additional instructions (no tool call is emitted).
* ``after_run`` remembers the completed turn (user input + agent response) so it
  can be recalled on later turns, including from other sessions/threads.

The interface intentionally mirrors the *installed* Agent Framework
``ContextProvider`` contract (``before_run`` / ``after_run``), not the older
``invoking`` / ``invoked`` naming, which is not present in this framework
version.

Only the published InfoLang public SDK (``infolang``) is used; no runtime or
engine internals are touched.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from agent_framework import AgentSession, ContextProvider, Message, SessionContext
from infolang import AsyncInfoLang, InfoLang, RecallResult, RememberResult

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from infolang.types import Chunk

logger = logging.getLogger("infolang_agent_framework")

DEFAULT_SOURCE_ID = "infolang"
DEFAULT_CONTEXT_INSTRUCTION = (
    "The following context was recalled from InfoLang memory. "
    "Use it when it is relevant to the user's request:"
)

InfoLangClient = InfoLang | AsyncInfoLang


class InfoLangContextProvider(ContextProvider):
    """Auto-recall / auto-retain InfoLang memory for Agent Framework agents.

    The provider participates in the context-engineering pipeline. On each run it
    recalls memory relevant to the incoming messages and injects it as
    instructions before the model is invoked, then persists the turn afterwards.
    Both behaviours are independently toggleable, and the provider also exposes
    ``recall`` / ``investigate`` / ``remember`` / ``forget`` for explicit use.

    Args:
        client: A pre-built :class:`infolang.InfoLang` or
            :class:`infolang.AsyncInfoLang` client. When omitted, a client is
            constructed from ``api_key`` (or the ``INFOLANG_API_KEY`` environment
            variable) plus any extra ``client_kwargs``.
        api_key: InfoLang API key used to build a client when ``client`` is not
            supplied. Mutually exclusive with ``client``.
        namespace: InfoLang namespace (memory bank) for reads and writes. With a
            managed API key this is honored on both reads and writes.
        workspace: InfoLang workspace (tenant) identifier.
        source_id: Provider source id used for message/tool attribution.
        top_k: Maximum number of chunks to recall and inject per turn.
        auto_recall: When True, recall + inject context in ``before_run``.
        auto_retain: When True, remember the turn in ``after_run``.
        use_investigate: When True, auto-recall uses ``investigate`` instead of
            ``recall``.
        session_scoped: When True, the effective namespace is suffixed with the
            session id, giving each thread its own memory bank. Defaults to False
            so memory is shared across threads (the common "recall across
            threads" pattern).
        min_score: Optional confidence floor; recalled chunks scoring below this
            are dropped before injection.
        tags: Optional tag string attached to memories written by auto-retain.
        context_instruction: Header prepended to the injected context block.
        raise_on_error: When True, InfoLang errors during auto-recall/retain are
            re-raised. When False (default) they are logged and swallowed so a
            memory-backend hiccup never breaks an agent run.
        **client_kwargs: Extra keyword args forwarded to the InfoLang client
            constructor when ``client`` is not supplied (e.g. ``base_url``).

    Raises:
        ValueError: If ``client`` is supplied together with ``api_key`` or extra
            client kwargs.
    """

    def __init__(
        self,
        client: InfoLangClient | None = None,
        *,
        api_key: str | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        source_id: str = DEFAULT_SOURCE_ID,
        top_k: int = 5,
        auto_recall: bool = True,
        auto_retain: bool = True,
        use_investigate: bool = False,
        session_scoped: bool = False,
        min_score: float | None = None,
        tags: str | None = None,
        context_instruction: str = DEFAULT_CONTEXT_INSTRUCTION,
        raise_on_error: bool = False,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(source_id)

        if client is not None and (api_key is not None or client_kwargs):
            raise ValueError(
                "Pass either a pre-built `client` or client construction args "
                "(`api_key`/`base_url`/...), not both."
            )

        if client is None:
            client = self._build_client(
                api_key=api_key,
                namespace=namespace,
                workspace=workspace,
                **client_kwargs,
            )
            self._owns_client = True
        else:
            self._owns_client = False

        self._client: InfoLangClient = client
        # Real synchronous clients block on network I/O, so their calls are
        # offloaded to a worker thread. Async clients (and async mocks) are
        # awaited directly.
        self._prefers_thread = isinstance(client, InfoLang) and not isinstance(
            client, AsyncInfoLang
        )

        self._base_namespace = namespace
        self._top_k = top_k
        self._auto_recall = auto_recall
        self._auto_retain = auto_retain
        self._use_investigate = use_investigate
        self._session_scoped = session_scoped
        self._min_score = min_score
        self._tags = tags
        self._context_instruction = context_instruction
        self._raise_on_error = raise_on_error

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        namespace: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> InfoLangContextProvider:
        """Build a provider backed by an async client from an API key.

        Args:
            api_key: InfoLang API key.
            namespace: Namespace (memory bank) for reads and writes.
            workspace: Workspace (tenant) identifier.
            **kwargs: Additional keyword args forwarded to the provider
                constructor (e.g. ``top_k``, ``session_scoped``, ``source_id``).

        Returns:
            A configured :class:`InfoLangContextProvider`.
        """
        return cls(
            api_key=api_key,
            namespace=namespace,
            workspace=workspace,
            **kwargs,
        )

    @staticmethod
    def _build_client(
        *,
        api_key: str | None,
        namespace: str | None,
        workspace: str | None,
        **client_kwargs: Any,
    ) -> AsyncInfoLang:
        if api_key is not None:
            return AsyncInfoLang.from_api_key(
                api_key, namespace=namespace, workspace=workspace, **client_kwargs
            )
        # No explicit key: let the SDK resolve credentials from the environment.
        return AsyncInfoLang(namespace=namespace, workspace=workspace, **client_kwargs)

    # -- lifecycle hooks -------------------------------------------------

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Recall relevant memory and inject it as instructions before the model runs."""
        if not self._auto_recall:
            return
        query = self._messages_text(context.input_messages)
        if not query:
            return
        try:
            result = await self._recall_for_context(query, session)
            chunks = self._select_chunks(result)
            if not chunks:
                return
            context.extend_instructions(self.source_id, self._format_context(chunks))
            state["recalled_chunk_ids"] = [chunk.id for chunk in chunks]
        except Exception:
            logger.exception("InfoLang auto-recall failed (source_id=%s)", self.source_id)
            if self._raise_on_error:
                raise

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Remember the completed turn after the model runs."""
        if not self._auto_retain:
            return
        text = self._format_turn(context)
        if not text:
            return
        try:
            result = await self._call(
                "remember",
                text,
                namespace=self._namespace_for(session),
                source=self.source_id,
                tags=self._tags,
            )
            memory_id = getattr(result, "memory_id", None)
            if memory_id is not None:
                state["last_memory_id"] = memory_id
        except Exception:
            logger.exception("InfoLang auto-retain failed (source_id=%s)", self.source_id)
            if self._raise_on_error:
                raise

    # -- explicit memory API ---------------------------------------------

    async def recall(
        self, query: str, *, session: AgentSession | None = None, **kwargs: Any
    ) -> RecallResult:
        """Recall memory for ``query`` using the provider's default namespace/top_k."""
        kwargs.setdefault("namespace", self._namespace_for(session))
        kwargs.setdefault("top_k", self._top_k)
        return cast(RecallResult, await self._call("recall", query, **kwargs))

    async def investigate(
        self, query: str, *, session: AgentSession | None = None, **kwargs: Any
    ) -> RecallResult:
        """Investigate memory for ``query`` (agent-style recall)."""
        kwargs.setdefault("namespace_hint", self._namespace_for(session))
        kwargs.setdefault("top_k", self._top_k)
        return cast(RecallResult, await self._call("investigate", query, **kwargs))

    async def remember(
        self, text: str, *, session: AgentSession | None = None, **kwargs: Any
    ) -> RememberResult:
        """Store ``text`` in memory using the provider's default namespace."""
        kwargs.setdefault("namespace", self._namespace_for(session))
        kwargs.setdefault("source", self.source_id)
        return cast(RememberResult, await self._call("remember", text, **kwargs))

    async def forget(
        self, memory_id: str, *, session: AgentSession | None = None, **kwargs: Any
    ) -> None:
        """Delete a memory by id from the provider's default namespace."""
        kwargs.setdefault("namespace", self._namespace_for(session))
        await self._call("forget", memory_id, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying client if this provider created it.

        Only clients built by the provider are closed; a client passed in by the
        caller is left for the caller to manage. Owned clients are always async.
        """
        if self._owns_client:
            await cast(AsyncInfoLang, self._client).aclose()

    async def __aenter__(self) -> InfoLangContextProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- internals -------------------------------------------------------

    async def _recall_for_context(self, query: str, session: AgentSession | None) -> RecallResult:
        namespace = self._namespace_for(session)
        if self._use_investigate:
            return cast(
                RecallResult,
                await self._call("investigate", query, namespace_hint=namespace, top_k=self._top_k),
            )
        return cast(
            RecallResult,
            await self._call("recall", query, namespace=namespace, top_k=self._top_k),
        )

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a client method, transparently handling sync and async clients."""
        fn = getattr(self._client, method_name)
        if self._prefers_thread:
            return await asyncio.to_thread(lambda: fn(*args, **kwargs))
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _namespace_for(self, session: AgentSession | None) -> str | None:
        if not self._session_scoped:
            return self._base_namespace
        session_id = session.session_id if session is not None else None
        if not session_id:
            return self._base_namespace
        if self._base_namespace:
            return f"{self._base_namespace}.{session_id}"
        return session_id

    def _select_chunks(self, result: RecallResult) -> list[Chunk]:
        chunks = list(result.chunks)
        if self._min_score is not None:
            chunks = [c for c in chunks if (c.score or 0.0) >= self._min_score]
        return chunks[: self._top_k]

    def _format_context(self, chunks: Sequence[Chunk]) -> str:
        lines = [self._context_instruction]
        for chunk in chunks:
            lines.append(f"- {chunk.text.strip()}")
        return "\n".join(lines)

    def _format_turn(self, context: SessionContext) -> str:
        user_text = self._messages_text(context.input_messages)
        assistant_text = ""
        response = context.response
        if response is not None and getattr(response, "messages", None):
            assistant_text = self._messages_text(response.messages)
        parts: list[str] = []
        if user_text:
            parts.append(f"User: {user_text}")
        if assistant_text:
            parts.append(f"Assistant: {assistant_text}")
        return "\n".join(parts)

    @staticmethod
    def _messages_text(messages: Sequence[Message] | None) -> str:
        if not messages:
            return ""
        parts: list[str] = []
        for message in messages:
            text = (message.text or "").strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
