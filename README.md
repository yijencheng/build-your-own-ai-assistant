# Koroku: Build Your Own AI Agent From Scratch

This is the companion code repo for the [Openclawd From Scratch](https://openclawd.com) series.

## 1 - It's Alive

The difference between a chatbot and an agent is simple: agents can act. In this first installment, we build a simple agent from scratch using the Gemini Python SDK that can call tools in a loop — and even write its own tools to extend its functionality.

We cover:

- **The tool calling loop** — how the model decides to call tools, how we execute them, and how we feed results back
- **A tool factory** — an `AgentTool` base class and `AgentRuntime` that makes defining new tools as simple as writing a Pydantic model with an `execute` method
- **Self-extension** — the agent can write its own tools and hot-reload them at runtime

Read the full article: [It's Alive!](https://ivanleo.com/blog/its-alive)
