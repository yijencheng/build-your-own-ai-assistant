import modal
from fastapi import FastAPI, HTTPException
from google.genai import types
from pydantic import BaseModel
import os
import httpx

from agent import Agent, SessionManager
from agent_tools import AgentContext, ToolResult

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]", "google-genai", "aiosqlite", "pydantic", "rich", "httpx"
    )
    .add_local_dir("4 - Cloud Atlas", remote_path="/root")
)
volume = modal.Volume.from_name("koroku-data", create_if_missing=True)
app = modal.App("open-claw")
api = FastAPI()
message_queue = modal.Queue.from_name("koroku-message-queue", create_if_missing=True)
telegram_secret = modal.Secret.from_name("telegram-bot")
gemini_secret = modal.Secret.from_name("gemini-api")


class ChatResponse(BaseModel):
    success: bool


class TelegramChat(BaseModel):
    id: int


class TelegramMessage(BaseModel):
    chat: TelegramChat
    text: str = ""


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage


class TelegramContext(AgentContext):
    def __init__(self, *, bot_token: str, chat_id: int):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send_message(self, text: str) -> None:
        payload = text.strip()
        if not payload:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(url, json={"chat_id": self.chat_id, "text": payload})


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
    call_args: dict[str, object],
    context: AgentContext,
) -> None:
    if not isinstance(context, TelegramContext):
        return
    status = "✓" if not result.error else "✗"
    await context.send_message(f"{status} {call_name} {call_args}")


def install_persisted_packages():
    import subprocess
    req_path = "/data/requirements.txt"
    if os.path.exists(req_path):
        subprocess.run(["pip", "install", "-r", req_path], check=False)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[telegram_secret, gemini_secret],
    max_containers=1,
    timeout=600,
)
async def process_message() -> None:
    install_persisted_packages()
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

    session_manager = SessionManager(root_dir="/data")

    message = await message_queue.get.aio(block=False)
    if message is None:
        return

    chat_id = int(message.get("chat_id", 0))
    text = str(message.get("text", "")).strip()
    if text.lower().startswith("/stop"):
        return

    context: AgentContext = TelegramContext(bot_token=bot_token, chat_id=chat_id)
    agent = Agent(session_manager=session_manager, context=context, root_dir="/data")
    agent.on("on_model_response", send_llm_response)
    agent.on("on_tool_result", send_tool_result)
    await agent.initialize()

    next_message: types.Content | None = types.UserContent(
        parts=[types.Part.from_text(text=text)]
    )
    while next_message is not None:
        # TODO: Check here if there are any new messages and then pull everything off the queue
        next_message = await agent.run(next_message)
    await volume.commit.aio()


@api.post("/chat", response_model=ChatResponse)
async def chat(req: TelegramUpdate) -> ChatResponse:
    text = req.message.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await message_queue.put.aio({"chat_id": req.message.chat.id, "text": text})

    await process_message.spawn.aio()

    return ChatResponse(success=True)


@app.function(image=image, secrets=[telegram_secret, gemini_secret])
@modal.asgi_app()
def fastapi_app():
    return api
