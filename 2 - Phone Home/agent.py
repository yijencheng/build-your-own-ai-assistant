import asyncio
import importlib
from pathlib import Path
from typing import Any

import agent_tools
from agent_tools import AgentContext, AgentTool, ToolResult
from google.genai import Client, types


class Agent:
    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.model = model
        self.client = Client()
        self.context = AgentContext()

        self.tools_module = agent_tools
        self.tools_file = Path(self.tools_module.__file__).resolve()
        self.last_modified = self._mtime(self.tools_file)
        self.tools = self._load_tools(self.tools_module)

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

    async def run(self, conversation: list[types.Content]) -> types.Content | None:
        """Run one assistant step. Returns tool-response message, or None when done."""
        completion = await self.client.aio.models.generate_content(
            model=self.model,
            contents=conversation,
            config=types.GenerateContentConfig(tools=self.get_tools()),
        )

        message = completion.candidates[0].content
        conversation.append(message)

        function_calls = [
            part.function_call for part in message.parts if part.function_call
        ]

        if not function_calls:
            text_parts = [part.text for part in message.parts if part.text]
            if text_parts:
                print(f"\nAssistant: {''.join(text_parts)}")
            return None

        tool_responses: list[types.Part] = []
        for call in function_calls:
            call_args = call.args or {}
            print(f"[Agent Action] Running '{call.name}' with args: {call_args}")
            result = await self.execute_tool(call.name, call_args)
            print(f"[Agent Action] Result: {result.response}")
            tool_responses.append(result.to_genai_message())

        return types.UserContent(parts=tool_responses)


async def main() -> None:
    print("Type 'exit' or 'quit' to stop.")
    conversation: list[types.Content] = []
    agent = Agent()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        conversation.append(
            types.UserContent(parts=[types.Part.from_text(text=user_input)])
        )

        while True:
            next_message = await agent.run(conversation)
            if next_message is None:
                break
            conversation.append(next_message)


if __name__ == "__main__":
    asyncio.run(main())
