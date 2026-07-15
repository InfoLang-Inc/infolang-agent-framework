"""Cross-thread memory with the Microsoft Agent Framework and InfoLang.

A fact taught in one thread is recalled automatically in another — with no tool
calls in the transcript.

Run with::

    export INFOLANG_API_KEY=il_live_...
    export OPENAI_API_KEY=sk-...
    pip install infolang-agent-framework  # OpenAI chat client ships with agent-framework
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from infolang_agent_framework import InfoLangContextProvider


async def main() -> None:
    # Reads INFOLANG_API_KEY from the environment.
    memory = InfoLangContextProvider(namespace="quickstart-demo")

    agent = Agent(
        OpenAIChatClient(),
        instructions="You are a concise, helpful assistant.",
        context_providers=memory,
    )

    try:
        thread_a = agent.create_session()
        print("thread A >>", "My favorite programming language is Rust.")
        await agent.run("My favorite programming language is Rust.", session=thread_a)

        thread_b = agent.create_session()
        question = "What is my favorite programming language?"
        print("thread B >>", question)
        reply = await agent.run(question, session=thread_b)
        print("agent     <<", reply)
    finally:
        await memory.aclose()


if __name__ == "__main__":
    asyncio.run(main())
