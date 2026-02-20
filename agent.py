import asyncio
from typing import Optional

from google.genai import Client, types as genai_types

from adapters import ConsoleMemoryMessageAdaptor
from agent_types import (
    BASIC_TOOLS,
    FunctionPart,
    FunctionResponse,
    Message,
    MessageAdaptor,
    TextPart,
    ToolDefinition,
    ToolRuntime,
)
from runtime import BasicFileToolRuntime


class Agent:
    def __init__(
        self,
        tools: Optional[list[type[ToolDefinition]]] = None,
        tool_runtime: Optional[ToolRuntime] = None,
        message_adaptor: Optional[MessageAdaptor] = None,
    ):
        """
        We define a client, history
        """
        self.client = Client()
        tool_models = tools or []
        self.tools = [tool.to_tool_schema() for tool in tool_models]
        self.tool_runtime = tool_runtime
        if self.tools and self.tool_runtime is None:
            raise ValueError("tool_runtime is required when tools are configured")
        self.message_adaptor = message_adaptor or ConsoleMemoryMessageAdaptor()

    async def run(
        self,
        turn_input: Message,
        *,
        max_round_budget: int = 8,
    ) -> Message:
        if max_round_budget == 0:
            return

        if not isinstance(turn_input, Message):
            raise ValueError(f"Invalid type of {type(turn_input)}")

        self.message_adaptor.save_message(turn_input)

        for _ in range(max_round_budget):
            history = self.message_adaptor.get_history()
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[item.to_gemini_content() for item in history],
                config=genai_types.GenerateContentConfig(
                    tools=self.tools,
                    thinking_config=genai_types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level=genai_types.ThinkingLevel.MEDIUM,
                    ),
                    temperature=0.0,
                ),
            )

            message = Message.from_assistant_response(response)
            self.message_adaptor.print_new_message(message)
            self.message_adaptor.save_message(message)
            tool_calls = [
                part for part in message.content if isinstance(part, FunctionPart)
            ]
            if not tool_calls:
                return message

            for tool_call in tool_calls:
                tool_response = await self.execute_tool(tool_call)
                self.message_adaptor.print_new_message(tool_response)
                self.message_adaptor.save_message(tool_response)

        raise RuntimeError("Exceeded max_round_budget while resolving tool calls.")

    async def execute_tool(self, tool_call: FunctionPart) -> FunctionResponse:
        if self.tool_runtime is None:
            raise ValueError(
                f"Tool runtime not configured for tool call: {tool_call.name}"
            )
        return await self.tool_runtime.execute(tool_call.name, tool_call.args)


async def main():
    agent = Agent(tools=BASIC_TOOLS, tool_runtime=BasicFileToolRuntime())

    while True:
        user_input = input("You: ")
        await agent.run(
            Message(role="user", content=[TextPart(text=user_input)], usage=None)
        )


if __name__ == "__main__":
    asyncio.run(main())
