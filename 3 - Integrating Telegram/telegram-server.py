import os
from contextvars import ContextVar
from collections import defaultdict
from typing import Any
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI
from google.genai import types
from telegram import Bot, Update

from agent import Agent, ToolExecutionPayload

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env")

WHITELIST_TELEGRAM_IDS = {
    int(chat_id.strip())
    for chat_id in os.getenv("WHITELIST_TELEGRAM_IDS", "").split(",")
    if chat_id.strip()
}

app = FastAPI()
agent = Agent()
bot = Bot(token=TELEGRAM_BOT_TOKEN)
CURRENT_CHAT_ID: ContextVar[int | None] = ContextVar("current_chat_id", default=None)

# In-memory conversation store keyed by Telegram chat id.
conversations: dict[int, list[types.Content]] = defaultdict(list)


def print_llm_response(response: types.Content) -> None:
    chat_id = CURRENT_CHAT_ID.get()
    if chat_id is None:
        return

    for part in response.parts:
        if part.text:
            text = part.text.strip()
            if text:
                asyncio.create_task(bot.send_message(chat_id=chat_id, text=text))


def print_llm_tool(response: ToolExecutionPayload) -> None:
    chat_id = CURRENT_CHAT_ID.get()
    if chat_id is None:
        return

    execution = response["execution"]
    if not execution.get("success"):
        asyncio.create_task(
            bot.send_message(chat_id=chat_id, text=f"X [Error Encountered]\n{response}")
        )
        return

    asyncio.create_task(
        bot.send_message(chat_id=chat_id, text=f"✓ {response['name']} : {response['args']}")
    )
    result = execution.get("result")
    if result is not None:
        asyncio.create_task(bot.send_message(chat_id=chat_id, text=str(result)))


agent.on("model_response", print_llm_response)
agent.on("tool_executed", print_llm_tool)


@app.post("/chat")
async def chat(update_payload: dict[str, Any]) -> dict[str, bool]:
    update = Update.de_json(update_payload, bot)
    if update is None or update.effective_chat is None:
        return {"ok": True}

    chat_id = update.effective_chat.id
    if WHITELIST_TELEGRAM_IDS and chat_id not in WHITELIST_TELEGRAM_IDS:
        print(f"Blocked Telegram[{chat_id}] (not in WHITELIST_TELEGRAM_IDS)")
        return {"ok": True}

    message = update.effective_message
    if message is None:
        return {"ok": True}

    user_text = (message.text or "").strip()
    if not user_text:
        return {"ok": True}

    print(f"Telegram[{chat_id}]: {user_text}")
    conversation = conversations[chat_id]
    conversation.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    token = CURRENT_CHAT_ID.set(chat_id)

    try:
        while True:
            next_message = agent.run(conversation)
            if next_message is None:
                break
            conversation.append(next_message)
    finally:
        CURRENT_CHAT_ID.reset(token)

    return {"ok": True}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
