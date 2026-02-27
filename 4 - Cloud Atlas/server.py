import modal

from telegram_types import TelegramUpdate

image = modal.Image.debian_slim().pip_install(
    "pydantic>=2,<3",
    "fastapi[standard]>=0.115,<1",
).add_local_dir("4 - Cloud Atlas", remote_path="/root")

app = modal.App(name="koroku-server", image=image)
message_queue = modal.Queue.from_name("koroku-messages", create_if_missing=True)
Worker = modal.Cls.from_name("koroku-worker", "Worker")


@app.function()
@modal.fastapi_endpoint(method="POST", docs=True)
async def chat(update: TelegramUpdate):
    if not update.message.text:
        return {"ok": True}

    await message_queue.put.aio(
        {
            "update_id": update.update_id,
            "chat_id": update.message.chat.id,
            "text": update.message.text,
        }
    )

    worker = Worker()
    await worker.process_queue.spawn.aio()
    return {"ok": True}
