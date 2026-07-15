"""InfoLang semantic-memory context provider for the Microsoft Agent Framework.

Quickstart::

    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient
    from infolang_agent_framework import InfoLangContextProvider

    memory = InfoLangContextProvider.from_api_key("il_live_...", namespace="acme")
    agent = ChatAgent(chat_client=OpenAIChatClient(), context_providers=memory)

    # Memory is recalled and retained automatically each turn — no tool calls.
    await agent.run("Remember that my favorite language is Rust.")
    await agent.run("What is my favorite language?")
"""

from __future__ import annotations

from ._version import __version__
from .provider import (
    DEFAULT_CONTEXT_INSTRUCTION,
    DEFAULT_SOURCE_ID,
    InfoLangContextProvider,
)

__all__ = [
    "__version__",
    "InfoLangContextProvider",
    "DEFAULT_SOURCE_ID",
    "DEFAULT_CONTEXT_INSTRUCTION",
]
