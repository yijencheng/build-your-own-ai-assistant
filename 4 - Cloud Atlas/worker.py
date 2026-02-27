import os
import queue
import time

import modal
from google.genai import Client, types
from typing import Any

from agent import Agent, SessionManager
from agent_tools import AgentContext, ToolResult


SCALEDOWN_WINDOW = 300
SANDBOX_TIMEOUT = 3600

gemini_secret = modal.Secret.from_name("gemini-api")
telegram_secret = modal.Secret.from_name("telegram-bot")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "pydantic>=2,<3",
        "google-genai",
        "aiosqlite",
        "aiohttp>=3.10.10",
        "rich",
        "httpx",
    )
    .add_local_dir("4 - Cloud Atlas", remote_path="/root")
)
sandbox_image = modal.Image.debian_slim().apt_install("curl", "procps")

app = modal.App(name="koroku-worker", image=image)
snapshot_dict = modal.Dict.from_name("koroku-snapshots", create_if_missing=True)
message_queue = modal.Queue.from_name("koroku-messages", create_if_missing=True)
data_volume = modal.Volume.from_name("koroku-data", create_if_missing=True)


@app.cls(
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=SANDBOX_TIMEOUT,
    max_containers=1,
    secrets=[gemini_secret, telegram_secret],
    volumes={"/data": data_volume},
)
class Worker:
    def _create_or_restore_sandbox(self) -> None:
        snapshot_id = snapshot_dict.get("SNAPSHOT", None)
        if snapshot_id:
            print(f"Restoring sandbox from snapshot: {snapshot_id}")
            snapshot = modal.SandboxSnapshot.from_id(snapshot_id)
            self.sandbox = modal.Sandbox._experimental_from_snapshot(snapshot)
        else:
            print("Creating new sandbox")

            self.sandbox = modal.Sandbox.create(
                app=app,
                image=sandbox_image,
                timeout=SANDBOX_TIMEOUT,
                _experimental_enable_snapshot=True,
            )

    def _ensure_sandbox(self) -> None:
        if self.sandbox.poll() is not None:
            print("Sandbox is dead, recreating...")
            self._create_or_restore_sandbox()

    @modal.enter()
    def startup(self) -> None:
        print("WORKER STARTING")
        self._create_or_restore_sandbox()
        self.client = Client()
        self.chat_id: int | None = None

    @modal.exit()
    def shutdown(self) -> None:
        print("WORKER SHUTTING DOWN")
        if self.sandbox.poll() is None:
            snapshot = self.sandbox._experimental_snapshot()
            snapshot_dict["SNAPSHOT"] = snapshot.object_id
            print(f"Saved snapshot: {snapshot.object_id}")
        self.sandbox.terminate()

    async def _pull_queue_parts(self) -> dict:
        queue_items: list[dict] = []
        while True:
            try:
                item = await message_queue.get.aio(timeout=0)
            except queue.Empty:
                break
            queue_items.append(item)

        if not queue_items:
            return {"ok": False, "message": "empty", "items": []}

        texts: list[str] = []
        for item in queue_items:
            if item.get("chat_id") is not None:
                self.chat_id = int(item["chat_id"])

            text = (item.get("text") or "").strip()
            if not text:
                continue

            if "/stop" in text.lower():
                return {"ok": False, "message": "stop requested", "items": []}
            if "/clear" in text.lower():
                return {"ok": False, "message": "clear requested", "items": []}

            texts.append(text)

        if not texts:
            return {"ok": False, "message": "invalid input", "items": []}

        parts: list[types.Part] = [types.Part.from_text(text=text) for text in texts]
        return {"ok": True, "message": "ready", "items": parts}

    @modal.method()
    async def process_queue(self) -> dict:
        self._ensure_sandbox()
        pull = await self._pull_queue_parts()
        if pull["message"] == "stop requested":
            context = AgentContext(
                telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
                telegram_chat_id=self.chat_id,
            )
            await context.send_telegram_message("Stopping model now")
        if not pull["ok"]:
            return pull

        user_message = types.UserContent(parts=pull["items"])

        context = AgentContext(
            sandbox=self.sandbox,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=self.chat_id,
        )
        session_manager = SessionManager(db_path="/data/agent.db")
        agent = Agent(session_manager=session_manager, context=context)

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
            call_args: dict[str, Any],
            context: AgentContext,
        ) -> None:
            status = "✓" if not result.error else "✗"
            await context.send_telegram_message(
                f"{status} {call_name} {call_args}\n{result.response}"
            )

        agent.on("on_model_response", send_llm_response)
        agent.on("on_tool_result", send_tool_result)
        await agent.initialize()
        await context.send_typing_indicator()

        next_message: types.Content | None = user_message
        while True:
            if next_message is None:
                return {"ok": True, "message": "done", "items": []}

            next_message = await agent.run(next_message)
            if next_message is None:
                return {"ok": True, "message": "done", "items": []}

            pull = await self._pull_queue_parts()
            if pull["ok"]:
                next_message = types.UserContent(parts=pull["items"])
                continue

            if pull["message"] == "clear requested":
                await session_manager.delete()
                return pull
            if pull["message"] == "stop requested":
                await context.send_telegram_message("Stopping model now")
                return pull
            if pull["message"] in {"empty", "invalid input"}:
                continue

            return pull


@app.local_entrypoint()
def main(poll_seconds: int = 5):
    worker = Worker()
    while True:
        result = worker.process_queue.remote()
        print(result)
        time.sleep(poll_seconds)
