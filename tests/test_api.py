"""Explicit memory API, construction, and lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agent_framework import AgentSession, ContextProvider
from infolang import AsyncInfoLang, InfoLang, RememberResult

from conftest import RecordingAsyncClient, make_recall_result
from infolang_agent_framework import InfoLangContextProvider


def test_is_context_provider(async_client):
    provider = InfoLangContextProvider(async_client)
    assert isinstance(provider, ContextProvider)
    assert provider.source_id == "infolang"


def test_custom_source_id(async_client):
    provider = InfoLangContextProvider(async_client, source_id="mem-a")
    assert provider.source_id == "mem-a"


# -- explicit API -------------------------------------------------------


async def test_recall_passthrough_defaults(async_client):
    provider = InfoLangContextProvider(async_client, namespace="acme", top_k=7)

    result = await provider.recall("question")

    assert result is async_client.recall_result
    name, args, kwargs = async_client.calls[0]
    assert name == "recall"
    assert args[0] == "question"
    assert kwargs["namespace"] == "acme"
    assert kwargs["top_k"] == 7


async def test_recall_override_kwargs(async_client):
    provider = InfoLangContextProvider(async_client, namespace="acme")

    await provider.recall("q", namespace="other", top_k=2)

    _name, _args, kwargs = async_client.calls[0]
    assert kwargs["namespace"] == "other"
    assert kwargs["top_k"] == 2


async def test_investigate_passthrough(async_client):
    provider = InfoLangContextProvider(async_client, namespace="acme")

    await provider.investigate("q")

    name, _args, kwargs = async_client.calls[0]
    assert name == "investigate"
    assert kwargs["namespace_hint"] == "acme"
    assert kwargs["top_k"] == 5


async def test_remember_passthrough(async_client):
    provider = InfoLangContextProvider(async_client, namespace="acme")

    result = await provider.remember("a fact", tags="notes")

    assert result is async_client.remember_result
    name, args, kwargs = async_client.calls[0]
    assert name == "remember"
    assert args[0] == "a fact"
    assert kwargs["namespace"] == "acme"
    assert kwargs["source"] == "infolang"
    assert kwargs["tags"] == "notes"


async def test_forget_passthrough(async_client):
    provider = InfoLangContextProvider(async_client, namespace="acme")

    result = await provider.forget("mem-1")

    assert result is None
    name, args, kwargs = async_client.calls[0]
    assert name == "forget"
    assert args[0] == "mem-1"
    assert kwargs["namespace"] == "acme"


async def test_api_uses_session_scoped_namespace(async_client):
    provider = InfoLangContextProvider(async_client, namespace="base", session_scoped=True)
    session = AgentSession(session_id="s-7")

    await provider.recall("q", session=session)

    _name, _args, kwargs = async_client.calls[0]
    assert kwargs["namespace"] == "base.s-7"


# -- sync client offloads to a worker thread ----------------------------


async def test_sync_client_offloaded_to_thread():
    sync_client = InfoLang(api_key="il_test")
    sync_client.recall = MagicMock(return_value=make_recall_result("from-sync"))  # type: ignore[method-assign]
    provider = InfoLangContextProvider(sync_client)
    assert provider._prefers_thread is True

    result = await provider.recall("q")

    assert result.chunks[0].text == "from-sync"
    sync_client.recall.assert_called_once()
    sync_client.close()


# -- construction & validation ------------------------------------------


def test_construct_from_api_key_builds_async_client():
    provider = InfoLangContextProvider(api_key="il_test", namespace="ns", workspace="ws")
    try:
        assert isinstance(provider._client, AsyncInfoLang)
        assert provider._owns_client is True
        assert provider._client.namespace == "ns"
        assert provider._client.workspace == "ws"
    finally:
        provider._client.close() if isinstance(provider._client, InfoLang) else None


async def test_construct_from_env(monkeypatch):
    monkeypatch.setenv("INFOLANG_API_KEY", "il_env_key")
    provider = InfoLangContextProvider()
    assert isinstance(provider._client, AsyncInfoLang)
    assert provider._owns_client is True
    await provider.aclose()


def test_from_api_key_classmethod():
    provider = InfoLangContextProvider.from_api_key("il_test", namespace="ns", top_k=3)
    assert provider._top_k == 3
    assert provider._client.namespace == "ns"
    assert provider._owns_client is True


def test_client_plus_api_key_is_error(async_client):
    with pytest.raises(ValueError, match="not both"):
        InfoLangContextProvider(async_client, api_key="il_test")


def test_client_plus_client_kwargs_is_error(async_client):
    with pytest.raises(ValueError, match="not both"):
        InfoLangContextProvider(async_client, base_url="http://localhost:8766")


# -- lifecycle ----------------------------------------------------------


async def test_aclose_closes_owned_client(monkeypatch):
    rec = RecordingAsyncClient()
    monkeypatch.setattr(
        InfoLangContextProvider, "_build_client", staticmethod(lambda **kwargs: rec)
    )
    provider = InfoLangContextProvider(api_key="il_test")

    await provider.aclose()

    assert rec.closed is True


async def test_async_context_manager_closes_owned_client(monkeypatch):
    rec = RecordingAsyncClient()
    monkeypatch.setattr(
        InfoLangContextProvider, "_build_client", staticmethod(lambda **kwargs: rec)
    )

    async with InfoLangContextProvider(api_key="il_test") as provider:
        assert provider._owns_client is True

    assert rec.closed is True


async def test_aclose_noop_when_client_not_owned(async_client):
    provider = InfoLangContextProvider(async_client)

    await provider.aclose()

    assert async_client.closed is False


async def test_remember_result_type(async_client):
    async_client.remember_result = RememberResult(id="abc")
    provider = InfoLangContextProvider(async_client)

    result = await provider.remember("x")

    assert result.memory_id == "abc"


async def test_synchronous_non_infolang_client_result_returned_directly():
    """A plain sync client (not an InfoLang instance) returns values directly."""

    class PlainSyncClient:
        def recall(self, query, **kwargs):
            return make_recall_result("plain-sync")

    provider = InfoLangContextProvider(PlainSyncClient())
    assert provider._prefers_thread is False

    result = await provider.recall("q")

    assert result.chunks[0].text == "plain-sync"


async def test_session_scoped_without_session_falls_back_to_base(async_client):
    provider = InfoLangContextProvider(async_client, namespace="base", session_scoped=True)

    # No session passed -> effective namespace is just the base namespace.
    await provider.recall("q")

    _name, _args, kwargs = async_client.calls[0]
    assert kwargs["namespace"] == "base"
