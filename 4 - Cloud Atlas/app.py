import modal

from telegram_types import TelegramUpdate

image = modal.Image.debian_slim().add_local_dir("4 - Cloud Atlas", remote_path="/root")

app = modal.App(name="koroku", image=image)


@app.cls()
class WebApp:
    @modal.enter()
    def startup(self):
        print("STARTING")

    @modal.exit()
    def shutdown(self):
        print("SHUTTING DOWN")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def chat(self, update: TelegramUpdate):
        print(update)
        return {"success": "ok"}
