# infolang-agent-framework

InfoLang semantic memory for the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework).

`InfoLangContextProvider` is an Agent Framework **context provider**: it plugs
into the agent turn lifecycle and gives any agent long-term memory with **zero
tool calls in the transcript**.

- **Auto-recall** — before each model call, relevant memory is recalled from
  InfoLang and injected as instructions.
- **Auto-retain** — after each turn, the user input and agent response are
  remembered so they can be recalled later, including from other threads.
- **Explicit API** — `recall`, `investigate`, `remember`, and `forget` are also
  available for direct use.

It talks only to the **published InfoLang Python SDK** (`infolang`), which wraps
the InfoLang runtime REST API. No engine internals are used.

## Install

```bash
pip install infolang-agent-framework
```

This pulls in `agent-framework` (>=1.0,<2) and `infolang` (>=0.2,<0.3). Requires
Python 3.11+.

## Quickstart

```python
import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from infolang_agent_framework import InfoLangContextProvider


async def main() -> None:
    memory = InfoLangContextProvider.from_api_key(
        "il_live_...",         # or set INFOLANG_API_KEY and call InfoLangContextProvider()
        namespace="acme-support",
    )
    agent = Agent(
        OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=memory,
    )

    # Thread 1: teach the agent something.
    thread_a = agent.create_session()
    await agent.run("My favorite programming language is Rust.", session=thread_a)

    # Thread 2: a brand-new conversation still recalls it — no tool calls.
    thread_b = agent.create_session()
    reply = await agent.run("What's my favorite language?", session=thread_b)
    print(reply)  # -> "...Rust..."

    await memory.aclose()


asyncio.run(main())
```

Because recall is injected as **instructions** (not a tool), the model never has
to decide to call a memory tool — memory "just works" and the transcript stays
clean, which suits enterprise governance.

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `client` | `None` | Pre-built `InfoLang` or `AsyncInfoLang` client. If omitted, one is built from `api_key`/env. |
| `api_key` | `None` | InfoLang API key (or set `INFOLANG_API_KEY`). |
| `namespace` | `None` | Memory bank for reads and writes. |
| `workspace` | `None` | Tenant identifier. |
| `source_id` | `"infolang"` | Provider id used for message/tool attribution. |
| `top_k` | `5` | Max chunks recalled and injected per turn. |
| `auto_recall` | `True` | Recall + inject context in `before_run`. |
| `auto_retain` | `True` | Remember the turn in `after_run`. |
| `use_investigate` | `False` | Use `investigate` instead of `recall` for auto-recall. |
| `session_scoped` | `False` | Suffix the namespace with the session id so each thread has isolated memory. |
| `min_score` | `None` | Drop recalled chunks scoring below this floor. |
| `tags` | `None` | Tag string attached to auto-retained memories. |
| `context_instruction` | see source | Header prepended to the injected context block. |
| `raise_on_error` | `False` | Re-raise InfoLang errors during auto-recall/retain instead of logging and continuing. |

### Namespaces and threads

By default all threads share one namespace, so a fact learned in one
conversation is recalled in another (the example above). Set
`session_scoped=True` to give each `AgentSession` its own memory bank
(`{namespace}.{session_id}`) when you need per-thread isolation.

> Namespace overrides on reads **and** writes require a managed API key. Dev
> keys are pinned to their own namespace.

## Explicit memory API

```python
memory = InfoLangContextProvider.from_api_key("il_live_...", namespace="acme")

await memory.remember("The Q3 launch date is Sept 30.")
result = await memory.recall("When does Q3 launch?")
for chunk in result.chunks:
    print(chunk.score, chunk.text)

hits = await memory.investigate("launch timeline")
await memory.forget(result.chunks[0].id)
```

Sync `InfoLang` clients are supported too — their blocking calls are offloaded
to a worker thread so they don't stall the event loop.

## Using a pre-built client

```python
from infolang import AsyncInfoLang
from infolang_agent_framework import InfoLangContextProvider

client = AsyncInfoLang.from_api_key("il_live_...", namespace="acme", workspace="tenant-1")
memory = InfoLangContextProvider(client)  # provider does not own/close this client
```

## Development

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

ruff check . && ruff format --check .
mypy
pytest --cov --cov-report=term-missing
```

Tests are **offline by default** — InfoLang is replaced with in-memory fakes.
Live tests that hit the real API are opt-in:

```bash
INFOLANG_API_KEY=il_live_... pytest -m live
```

## Compatibility note

This package targets the Agent Framework's installed `ContextProvider`
extension point, which uses `before_run` / `after_run` hooks (Agent Framework
1.x). Earlier design drafts referred to `invoking` / `invoked`; those names are
not present in the shipped framework.

## License

Apache-2.0. See [LICENSE](LICENSE).
