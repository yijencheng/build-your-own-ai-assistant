import modal

from telegram_types import TelegramUpdate

image = modal.Image.debian_slim().add_local_dir("4 - Cloud Atlas", remote_path="/root")
sandbox_image = modal.Image.debian_slim().apt_install("curl", "procps")

app = modal.App(name="koroku", image=image)
sandbox_app = modal.App.lookup("koroku-sandbox", create_if_missing=True)
snapshot_dict = modal.Dict.from_name("koroku-snapshots", create_if_missing=True)


@app.cls(scaledown_window=300)
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
                app=sandbox_app, image=sandbox_image, _experimental_enable_snapshot=True
            )

    @modal.exit()
    def shutdown(self):
        print("SHUTTING DOWN")
        print("Snapshotting sandbox...")
        snapshot = self.sandbox._experimental_snapshot()
        snapshot_dict["SNAPSHOT"] = snapshot.object_id
        print(f"Saved snapshot: {snapshot.object_id}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def chat(self, update: TelegramUpdate):
        print(update)
        return {"success": "ok"}
