import modal
import queue

from telegram_types import TelegramUpdate

image = (
    modal.Image.debian_slim()
    .pip_install("pydantic>=2,<3", "fastapi[standard]>=0.115,<1")
    .add_local_dir("4 - Cloud Atlas", remote_path="/root")
)
sandbox_image = modal.Image.debian_slim().apt_install("curl", "procps")

app = modal.App(name="koroku", image=image)
sandbox_app = modal.App.lookup("koroku-sandbox", create_if_missing=True)
snapshot_dict = modal.Dict.from_name("koroku-snapshots", create_if_missing=True)
message_queue = modal.Queue.from_name("koroku-messages", create_if_missing=True)


@app.cls(scaledown_window=300, timeout=3600)
class WebApp:
    """
    Webapp here stays alive for 5 minutes
    """

    @modal.enter()
    def startup(self):
        print("STARTING")
        # snapshot_id = snapshot_dict.get("SNAPSHOT", None)
        # if snapshot_id:
        #     print(f"Restoring sandbox from snapshot: {snapshot_id}")
        #     snapshot = modal.SandboxSnapshot.from_id(snapshot_id)
        #     self.sandbox = modal.Sandbox._experimental_from_snapshot(snapshot)
        # else:
        #     print("Creating new sandbox")
        #     self.sandbox = modal.Sandbox.create(
        #         app=sandbox_app, image=sandbox_image, _experimental_enable_snapshot=True
        #     )

    @modal.exit()
    def shutdown(self):
        print("SHUTTING DOWN")
        # print("Snapshotting sandbox...")
        # snapshot = self.sandbox._experimental_snapshot()
        # snapshot_dict["SNAPSHOT"] = snapshot.object_id
        # print(f"Saved snapshot: {snapshot.object_id}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def chat(self, update: TelegramUpdate):
        print(f"Recieved {update}")
        await message_queue.put.aio(update.model_dump_json())
        await self.process_message.spawn.aio()
        return {"success": "ok"}

    async def drain_queue(self):
        messages = []
        while True:
            try:
                message = await message_queue.get.aio(timeout=0)
            except queue.Empty:
                break
            messages.append(TelegramUpdate.model_validate_json(message))
        return messages

    @modal.method()
    async def process_message(self):
        messages = await self.drain_queue()
        for message in messages:
            print(f"Processing message: {message}")
