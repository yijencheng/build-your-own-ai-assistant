from fastapi import FastAPI
from pydantic import BaseModel
from google.genai import types

from agent_hooks import Agent
from agent_tools import AgentContext, ToolResult
from typing import Any
from rich import print as rprint

app = FastAPI()


class TelegramChat(BaseModel):
    id: int


class TelegramMessage(BaseModel):
    chat: TelegramChat
    text: str = ""


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage


# In-memory conversation store
conversation = []


class ChatRequest(BaseModel):
    message: str


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


agent = Agent()
agent.on("on_tool_result", print_tool_result)
agent.on("on_model_response", print_llm_response)


@app.post("/chat")
async def chat(req: TelegramUpdate):
    print(f"You: {req.message.text}")
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=req.message.text)])
    )

    while True:
        next_message = await agent.run(conversation)
        if next_message is None:
            break
        conversation.append(next_message)

    return {"response": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
