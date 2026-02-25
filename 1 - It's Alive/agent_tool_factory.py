from agent_tools import TOOLS, AgentTool, ToolResult, AgentContext
from typing import Any, Literal, Type, TypeAlias, TypedDict
from google.genai import types, Client
import asyncio


class AgentRuntime:
    """Minimal runtime that exposes registered tools for the model."""

    def __init__(self, context: AgentContext):
        self.tools: dict[str, type[AgentTool]] = {tool.__name__: tool for tool in TOOLS}
        self.context = context

    def get_tools(self) -> list[types.Tool]:
        return [tool_cls.to_genai_schema() for tool_cls in self.tools.values()]

    def register_tool(self, tool_cls: Type[AgentTool]):
        self.tools[tool_cls.__name__] = tool_cls

    async def execute_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> ToolResult | types.Part:
        tool_cls = self.tools.get(tool_name)
        if tool_cls is None:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"Error": f"Unknown tool: {tool_name}"},
            )

        try:
            tool_input = tool_cls.model_validate(args)
            return await tool_input.execute(self.context)
        except Exception as exc:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": str(exc)},
            )


class FunctionResponseRunResult(TypedDict):
    kind: Literal["function_response"]
    message: types.UserContent


RunResult: TypeAlias = None | FunctionResponseRunResult


async def run(
    client: Client, contents: list[types.Content], runtime: AgentRuntime
) -> tuple[types.Content, RunResult]:
    completion = await client.aio.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(tools=runtime.get_tools()),
    )

    message = completion.candidates[0].content

    function_calls = [
        part.function_call for part in message.parts if part.function_call
    ]
    if not function_calls:
        return message, None

    tool_responses: list[types.Part] = []
    for call in function_calls:
        result = await runtime.execute_tool(call.name, call.args)
        print(f"Tool Call: [{call.name}:{call.args}]\n:{result.response}")
        tool_responses.append(result.to_genai_message())

    if not tool_responses:
        return message, None
    return message, {
        "kind": "function_response",
        "message": types.UserContent(parts=tool_responses),
    }


async def main() -> None:
    client = Client()
    contents: list[types.Content] = []
    context = AgentContext()
    runtime = AgentRuntime(context=context)

    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        contents.append(
            types.UserContent(parts=[types.Part.from_text(text=user_input)])
        )
        while True:
            assistant_message, tool_result = await run(client, contents, runtime)
            contents.append(assistant_message)
            if tool_result is None:
                for part in assistant_message.parts:
                    if part.text:
                        print(f"\nAssistant: {part.text}")
                break
            assert tool_result["kind"] == "function_response"
            contents.append(tool_result["message"])


if __name__ == "__main__":
    asyncio.run(main())
