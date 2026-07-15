"""Opt-in live tests against the real InfoLang API.

Deselected by default (``-m 'not live'`` in pyproject). Run explicitly with::

    INFOLANG_API_KEY=il_live_... pytest -m live

These tests exercise a real remember -> recall round trip through the published
InfoLang SDK. They require network access and a valid managed API key.
"""

from __future__ import annotations

import os
import uuid

import pytest
from agent_framework import AgentSession

from conftest import make_context
from infolang_agent_framework import InfoLangContextProvider

pytestmark = pytest.mark.live

API_KEY = os.environ.get("INFOLANG_API_KEY")


@pytest.mark.skipif(not API_KEY, reason="INFOLANG_API_KEY not set")
async def test_live_remember_then_recall():
    assert API_KEY is not None
    namespace = f"waf-test-{uuid.uuid4().hex[:8]}"
    provider = InfoLangContextProvider.from_api_key(API_KEY, namespace=namespace)
    try:
        fact = "The WP40 live test canary phrase is heliotrope."
        result = await provider.remember(fact)
        assert result.memory_id

        recalled = await provider.recall("What is the WP40 canary phrase?")
        assert any("heliotrope" in chunk.text.lower() for chunk in recalled.chunks)
    finally:
        await provider.aclose()


@pytest.mark.skipif(not API_KEY, reason="INFOLANG_API_KEY not set")
async def test_live_auto_recall_injects_context():
    assert API_KEY is not None
    namespace = f"waf-test-{uuid.uuid4().hex[:8]}"
    provider = InfoLangContextProvider.from_api_key(API_KEY, namespace=namespace)
    try:
        turn = make_context(
            ["My deployment region is eu-west-1"],
            assistant_texts=["Understood."],
        )
        session = AgentSession(session_id="live-thread")
        await provider.after_run(agent=None, session=session, context=turn, state={})

        follow_up = make_context(["Which region do I deploy to?"])
        await provider.before_run(
            agent=None, session=AgentSession(session_id="other"), context=follow_up, state={}
        )
        assert any("eu-west-1" in line for line in follow_up.instructions)
    finally:
        await provider.aclose()
