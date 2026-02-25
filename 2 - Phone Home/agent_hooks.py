import asyncio
import importlib
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeAlias, overload

import agent_tools
from agent_tools import AgentContext, AgentTool, ToolResult
from google.genai import Client, types
from rich import print as rprint

HookType: TypeAlias = Literal[
    "on_model_response",
    "on_tool_call",
    "on_tool_result",
]


class ModelResponseHook(Protocol):
    def __call__(
        self, *, message: types.Content, context: AgentContext
    ) -> Awaitable[None]: ...


class ToolCallHook(Protocol):
    def __call__(
        self, *, call: types.FunctionCall, context: AgentContext
    ) -> Awaitable[None]: ...


class ToolResultHook(Protocol):
    def __call__(
        self,
        *,
        result: ToolResult,
        call_name: str,
        call_args: dict[str, Any],
        context: AgentContext,
    ) -> Awaitable[None]: ...


AnyHook: TypeAlias = ModelResponseHook | ToolCallHook | ToolResultHook


class Agent:
    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        context: AgentContext = AgentContext(),
    ):
        self.model = model
        self.client = Client()
        self.context = context

        self.tools_module = agent_tools
        self.tools_file = Path(self.tools_module.__file__).resolve()
        self.last_modified = self._mtime(self.tools_file)
        self.tools = self._load_tools(self.tools_module)
        self._hooks: dict[HookType, list[AnyHook]] = {
            "on_model_response": [],
            "on_tool_call": [],
            "on_tool_result": [],
        }

    @staticmethod
    def _mtime(path: str | Path) -> float:
        return Path(path).stat().st_mtime

    @staticmethod
    def _load_tools(module: Any) -> dict[str, type[AgentTool]]:
        return {tool.__name__: tool for tool in module.TOOLS}

    def maybe_reload_runtime(self) -> bool:
        current = self._mtime(self.tools_file)
        if current == self.last_modified:
            return False

        try:
            self.tools_module = importlib.reload(self.tools_module)
            self.tools_file = Path(self.tools_module.__file__).resolve()
            self.tools = self._load_tools(self.tools_module)
            self.last_modified = self._mtime(self.tools_file)
            loaded_tools = ", ".join(sorted(self.tools.keys())) or "(none)"
            print(
                f"[Tool Reload] Loaded tools from {self.tools_file.name}: {loaded_tools}"
            )
            return True
        except Exception as exc:
            print(f"[Tool Reload] Failed, keeping previous runtime: {exc}")
            return False

    def get_tools(self) -> list[types.Tool]:
        self.maybe_reload_runtime()
        return [tool_cls.to_genai_schema() for tool_cls in self.tools.values()]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool_cls = self.tools.get(tool_name)
        if tool_cls is None:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": f"Unknown tool: {tool_name}"},
            )

        try:
            tool_input = tool_cls.model_validate(args or {})
            return await tool_input.execute(self.context)
        except Exception as exc:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": str(exc)},
            )

    @overload
    def on(
        self, event: Literal["on_model_response"], handler: ModelResponseHook
    ) -> "Agent": ...

    @overload
    def on(self, event: Literal["on_tool_call"], handler: ToolCallHook) -> "Agent": ...

    @overload
    def on(
        self, event: Literal["on_tool_result"], handler: ToolResultHook
    ) -> "Agent": ...

    def on(self, event: HookType, handler: AnyHook) -> "Agent":
        self._hooks[event].append(handler)
        return self

    async def emit(self, event: HookType, **kwargs: Any) -> None:
        for handler in self._hooks[event]:
            try:
                await handler(**kwargs)
            except Exception as exc:
                print(f"[Hook Error] Event '{event}' failed: {exc}")

    async def run(self, conversation: list[types.Content]) -> types.Content | None:
        """Run one assistant step. Returns tool-response message, or None when done."""
        completion = await self.client.aio.models.generate_content(
            model=self.model,
            contents=conversation,
            config=types.GenerateContentConfig(tools=self.get_tools()),
        )

        message = completion.candidates[0].content
        conversation.append(message)
        await self.emit("on_model_response", message=message, context=self.context)

        function_calls = [
            part.function_call for part in message.parts if part.function_call
        ]

        if not function_calls:
            return None

        tool_responses: list[types.Part] = []
        for call in function_calls:
            call_args = call.args or {}
            await self.emit("on_tool_call", call=call, context=self.context)
            result = await self.execute_tool(call.name, call_args)
            await self.emit(
                "on_tool_result",
                result=result,
                call_name=call.name,
                call_args=call_args,
                context=self.context,
            )
            tool_responses.append(result.to_genai_message())

        return types.UserContent(parts=tool_responses)


async def print_llm_response(message: types.Content, context: AgentContext):
    for part in message.parts:
        if part.text:
            print(f"* {part.text}")


async def print_tool_result(
    *,
    result: ToolResult,
    call_name: str,
    call_args: dict[str, Any],
    context: AgentContext,
) -> None:
    status = "[green]✓[/green]" if not result.error else "[red]✗[/red]"
    rprint(f"{status} [bold]{call_name}[/bold] {call_args}")


async def main() -> None:
    print("Type 'exit' or 'quit' to stop.")
    conversation: list[types.Content] = []
    agent = Agent()
    agent.on("on_tool_result", print_tool_result)
    agent.on("on_model_response", print_llm_response)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        conversation.append(
            types.UserContent(parts=[types.Part.from_text(text=user_input)])
        )
        print("Assistant")

        while True:
            next_message = await agent.run(conversation)
            if next_message is None:
                break
            conversation.append(next_message)


if __name__ == "__main__":
    asyncio.run(main())
