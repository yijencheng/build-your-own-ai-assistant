from fastapi import FastAPI
from pydantic import BaseModel
from google.genai import types
from agent_tools import AgentContext, ToolResult
from agent import Agent, print_tool_result, print_llm_response
from telegram import Bot
from typing import Any
import os
import dotenv

dotenv.load_dotenv()

app = FastAPI()

# In-memory conversation store
conversation = []


class ChatRequest(BaseModel):
    message: str


class TelegramChat(BaseModel):
    id: int


class TelegramMessage(BaseModel):
    chat: TelegramChat
    text: str = ""


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage


async def send_telegram_llm_response(
    *, message: types.Content, context: AgentContext
) -> None:
    for part in message.parts:
        if part.text:
            await context.send_message(part.text)


async def send_telegram_tool_response(
    *,
    result: ToolResult,
    call_name: str,
    call_args: dict[str, Any],
    context: AgentContext,
) -> None:
    status = "✓" if not result.error else "✗"
    await context.send_message(f"{status} {call_name} {call_args}")


@app.post("/chat")
async def chat(req: TelegramUpdate):
    print(f"You: {req.message.text}")
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=req.message.text)])
    )
    chat_id = req.message.chat.id

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    context = AgentContext(telegram_client=bot, chat_id=chat_id)
    agent = Agent(context=context)
    agent.on("on_tool_result", print_tool_result)
    agent.on("on_model_response", print_llm_response)
    agent.on("on_tool_result", send_telegram_tool_response)
    agent.on("on_model_response", send_telegram_llm_response)

    while True:
        next_message = await agent.run(conversation)
        if next_message is None:
            break
        conversation.append(next_message)

    return {"response": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
