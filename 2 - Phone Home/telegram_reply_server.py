import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from google.genai import types
from pydantic import BaseModel
from telegram import Bot

from agent_hooks import Agent
from agent_tools import AgentContext, ToolResult

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env")


app = FastAPI()
bot = Bot(token=TELEGRAM_BOT_TOKEN)


class TelegramContext(AgentContext):
    def __init__(self, *, telegram_client: Bot, chat_id: int):
        self.telegram_client = telegram_client
        self.chat_id = chat_id

    async def send_message(self, text: str) -> None:
        payload = text.strip()
        if not payload:
            return
        await self.telegram_client.send_message(chat_id=self.chat_id, text=payload)


class TelegramChat(BaseModel):
    id: int


class TelegramMessage(BaseModel):
    chat: TelegramChat
    text: str = ""


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage


# In-memory conversation store
conversation: list[types.Content] = []


async def send_llm_response(*, message: types.Content, context: AgentContext) -> None:
    if not isinstance(context, TelegramContext):
        return

    for part in message.parts:
        if part.text:
            await context.send_message(part.text)


async def send_tool_result(
    *,
    result: ToolResult,
    call_name: str,
    call_args: dict[str, Any],
    context: AgentContext,
) -> None:
    if not isinstance(context, TelegramContext):
        return

    status = "✓" if not result.error else "✗"
    await context.send_message(f"{status} {call_name} {call_args}")


@app.post("/chat")
async def chat(req: TelegramUpdate) -> dict[str, str]:
    chat_id = req.message.chat.id
    user_text = req.message.text.strip()
    if not user_text:
        return {"response": "ok"}

    print(f"Telegram[{chat_id}]: {user_text}")
    conversation.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    context = TelegramContext(telegram_client=bot, chat_id=chat_id)
    agent = Agent(context=context)
    agent.on("on_model_response", send_llm_response)
    agent.on("on_tool_result", send_tool_result)

    while True:
        next_message = await agent.run(conversation)
        if next_message is None:
            break
        conversation.append(next_message)

    return {"response": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
