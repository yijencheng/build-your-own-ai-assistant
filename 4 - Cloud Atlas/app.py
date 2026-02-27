import modal
import os
import queue
import uuid

from telegram_types import TelegramUpdate
from google.genai import types

from agent import Agent, SessionManager
from agent_tools import AgentContext, ToolResult

SCALEDOWN_WINDOW = 300

telegram_secret = modal.Secret.from_name("telegram-bot")
gemini_secret = modal.Secret.from_name("gemini-api")

image = (
    modal.Image.debian_slim()
    .pip_install("pydantic>=2,<3", "fastapi[standard]>=0.115,<1", "google-genai", "aiosqlite", "rich", "httpx")
    .add_local_dir("4 - Cloud Atlas", remote_path="/root")
)
sandbox_image = modal.Image.debian_slim().apt_install("curl", "procps")

app = modal.App(name="koroku", image=image)
sandbox_app = modal.App.lookup("koroku-sandbox", create_if_missing=True)
snapshot_dict = modal.Dict.from_name("koroku-snapshots", create_if_missing=True)
message_queue = modal.Queue.from_name("koroku-messages", create_if_missing=True)


@app.cls(
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=SCALEDOWN_WINDOW,
    secrets=[telegram_secret, gemini_secret],
    max_containers=1,
)
class WebApp:
    """
    Webapp here stays alive for 5 minutes
    """

    @modal.enter()
    def startup(self):
        print("STARTING")
        snapshot_id = snapshot_dict.get("SNAPSHOT", None)
        if snapshot_id:
            print(f"Restoring sandbox from snapshot: {snapshot_id}")
            snapshot = modal.SandboxSnapshot.from_id(snapshot_id)
            self.sandbox = modal.Sandbox._experimental_from_snapshot(snapshot)
        else:
            print("Creating new sandbox")
            self.sandbox = modal.Sandbox.create(
                app=sandbox_app,
                image=sandbox_image,
                timeout=SCALEDOWN_WINDOW,
                _experimental_enable_snapshot=True,
            )

        self.sandbox.exec("mkdir", "-p", "uploads").wait()
        p = self.sandbox.exec("ls", "-R", "uploads")
        print(p.stdout.read())
        p.wait()

    @modal.exit()
    def shutdown(self):
        print("SHUTTING DOWN")
        print("Snapshotting sandbox...")
        snapshot = self.sandbox._experimental_snapshot()
        snapshot_dict["SNAPSHOT"] = snapshot.object_id
        print(f"Saved snapshot: {snapshot.object_id}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def chat(self, update: TelegramUpdate):
        print(f"Recieved {update}")
        await message_queue.put.aio(update.model_dump_json())
        await self.process_message.spawn.aio()
        return {"success": "ok"}

    async def drain_queue(self) -> tuple[list[str], bool, int | None]:
        parts: list[str] = []
        stopped = False
        chat_id: int | None = None
        while True:
            try:
                raw = await message_queue.get.aio(timeout=0)
            except queue.Empty:
                break
            message = TelegramUpdate.model_validate_json(raw)
            print(f"Processing message: {message}")
            chat_id = message.message.chat.id

            text = message.message.text or message.message.caption or ""
            if text.strip().lower().startswith("/stop"):
                stopped = True
                break

            if message.message.photo:
                photo = message.message.photo[-1]
                dest_path = f"uploads/{uuid.uuid4()}.jpg"
                await self.download_telegram_file(photo.file_id, dest_path)
                parts.append(f"<uploaded file to {dest_path}>")

            if message.message.document:
                doc = message.message.document
                ext = os.path.splitext(doc.file_name or "")[1] or ".bin"
                dest_path = f"uploads/{uuid.uuid4()}{ext}"
                await self.download_telegram_file(doc.file_id, dest_path)
                parts.append(f"<uploaded file to {dest_path}>")

            audio = message.message.audio or message.message.voice
            if audio:
                ext = (
                    os.path.splitext(audio.file_name or "")[1]
                    if audio.file_name
                    else ".ogg"
                )
                dest_path = f"uploads/{uuid.uuid4()}{ext}"
                await self.download_telegram_file(audio.file_id, dest_path)
                parts.append(f"<uploaded file to {dest_path}>")

            if message.message.caption:
                parts.append(message.message.caption)
            elif message.message.text:
                parts.append(message.message.text)

        return parts, stopped, chat_id

    async def download_telegram_file(self, file_id: str, dest_path: str) -> None:
        bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
        await (
            await self.sandbox.exec.aio("mkdir", "-p", os.path.dirname(dest_path))
        ).wait.aio()
        script = (
            f'curl -sS "https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"'
            f' | grep -o \'"file_path":"[^"]*"\' | cut -d\'"\' -f4'
            f' | xargs -I{{}} curl -sS -o {dest_path} "https://api.telegram.org/file/bot{bot_token}/{{}}"'
        )
        p = await self.sandbox.exec.aio("bash", "-c", script)
        await p.wait.aio()

    @modal.method()
    async def process_message(self):
        parts, stopped, chat_id = await self.drain_queue()
        if stopped or not parts:
            return

        context = AgentContext(
            sandbox=self.sandbox,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=chat_id,
        )

        async def send_llm_response(
            *, message: types.Content, context: AgentContext
        ) -> None:
            for part in message.parts:
                if part.text:
                    await context.send_telegram_message(part.text)

        async def send_tool_result(
            *,
            result: ToolResult,
            call_name: str,
            call_args: dict[str, object],
            context: AgentContext,
        ) -> None:
            status = "✓" if not result.error else "✗"
            await context.send_telegram_message(f"{status} {call_name} {call_args}")

        session_manager = SessionManager()
        agent = Agent(session_manager=session_manager, context=context)
        agent.on("on_model_response", send_llm_response)
        agent.on("on_tool_result", send_tool_result)
        await agent.initialize()

        next_message: types.Content | None = types.UserContent(parts=parts)
        while True:
            if next_message is None:
                break

            next_message = await agent.run(next_message)

            new_parts, stopped, _ = await self.drain_queue()
            if stopped:
                await context.send_telegram_message("Pausing execution")
                break
            if new_parts:
                next_message = types.UserContent(parts=new_parts)
