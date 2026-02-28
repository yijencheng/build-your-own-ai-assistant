# Koroku: Build Your Own AI Agent From Scratch

This is the companion code repo for the [Openclawd From Scratch](https://openclawd.com) series.

## 1 - It's Alive

The difference between a chatbot and an agent is simple: agents can act. In this first installment, we build a simple agent from scratch using the Gemini Python SDK that can call tools in a loop — and even write its own tools to extend its functionality.

We cover:

- **The tool calling loop** — how the model decides to call tools, how we execute them, and how we feed results back
- **A tool factory** — an `AgentTool` base class and `AgentRuntime` that makes defining new tools as simple as writing a Pydantic model with an `execute` method
- **Self-extension** — the agent can write its own tools and hot-reload them at runtime

Read the full article: [It's Alive!](https://ivanleo.com/blog/its-alive)

## 2 - Phone Home

The agent can act — now it needs to talk to the outside world. We add a hook/event system so the agent's behaviour is composable, then expose it over HTTP and connect it to Telegram.

We cover:

- **Hooks & events** — an `on_model_response` / `on_tool_call` / `on_tool_result` event system that decouples I/O from the core agent loop (`agent_hooks.py`)
- **HTTP server** — a FastAPI `POST /chat` endpoint that drives the agent from curl or any HTTP client (`server.py`)
- **Telegram integration** — receive messages via webhook, run the agent, and reply back through the Telegram Bot API (`telegram_server.py`, `telegram_reply_server.py`)

Read the full article: [ET Phone Home](https://ivanleo.com/blog/phone-home)

## 3 - Total Recall

Conversations disappear when the process dies. In this installment we give the agent persistent memory — a SQLite-backed session store, conversation compaction via LLM summarisation, and system instructions with guardrails.

We cover:

- **Session persistence** — serialise every message (text, function calls, function responses, thought signatures) into a SQLite database and replay history on startup (`agent_session.py`)
- **Compaction & memory** — when the conversation exceeds a threshold, summarise it with a separate LLM call and write timestamped summaries to a `memory/` folder the agent can read back later (`agent_compaction.py`)
- **System instructions & guardrails** — give the agent an identity, point it at its memory folder, and enforce a per-turn tool-call budget to prevent runaway loops (`agent_compaction.py`)

Read the full article: [Total Recall](https://ivanleo.com/blog/total-recall)

## 4 - Cloud Atlas

Our agent works locally — now we deploy it to the cloud so it runs 24/7. We use [Modal](https://modal.com) to run the agent as a queue worker that spins up on demand, with sandboxed code execution and snapshot-based persistence.

We cover:

- **Queue-driven worker** — a Modal `Worker` class that polls a `modal.Queue` for Telegram messages and spins up/down automatically based on demand (`worker.py`)
- **Sandboxed execution** — all tool calls (Bash, ReadFile, Write, Edit) run inside a `modal.Sandbox`, isolating the agent's environment from the host (`agent_tools.py`)
- **Sandbox snapshotting** — on shutdown the worker snapshots the sandbox state and restores it on next cold start, so installed packages and file changes persist across scale-downs (`worker.py`)
- **Lightweight webhook server** — a separate Modal function receives Telegram webhooks, enqueues messages, and spawns the worker — keeping the API surface minimal (`server.py`)

## Workshop

Each step builds on the last — run any step with `python workshop/<n>/agent.py`.

1. **First API call** — make a basic `generate_content` call to the Gemini API with a tool declaration (`workshop/1/agent.py`)
2. **First tool call** — execute the model's `read_file` function call and feed the result back (`workshop/2/agent.py`)
3. **Agentic loop** — wrap tool calling in an async loop so the agent can chain multiple tool calls per conversation turn (`workshop/3/agent.py`)
4. **Tool factory** — introduce `AgentTool` base class, `ToolResult`, `AgentContext`, and `AgentRuntime` so new tools are just Pydantic models with an `execute` method (`workshop/4/`)
5. **More tools** — add `Write`, `Edit`, and `Bash` tools so the agent can write files, apply edits, and run shell commands (`workshop/5/agent_tools.py`)
6. **Hot-reload** — track `st_mtime` on the tools file and `importlib.reload` it automatically, enabling the agent to write new tools and use them immediately (`workshop/6/`)
7. **Hooks** — refactor from a procedural agent to an `Agent` class with an event system (`on_model_response`, `on_tool_call`, `on_tool_result`) and rich-formatted output via hooks (`workshop/7/`)
8. **Server** — expose the agent over HTTP with a FastAPI server so it can be invoked via `POST /chat` (`workshop/8/simple_server.py`)
9. **Telegram** — hook the agent up to the Telegram Bot API, using hooks to send responses back to the chat (`workshop/9/telegram_server.py`)
10. **Session persistence** — store every message in a SQLite database via `SessionManager` so conversations survive restarts and can be replayed (`workshop/10/session.py`)
11. **Compaction & memory** — when the conversation grows too long, summarise it with a separate LLM call and persist the summary to timestamped `memory/` files (`workshop/11/session.py`)
12. **System instructions & guardrails** — add a system prompt that gives the agent identity and memory awareness, plus a tool-call budget that stops runaway loops (`workshop/12/agent.py`)

## Just for Fun

Why did the programmer quit his job?
Because he didn't get arrays.
