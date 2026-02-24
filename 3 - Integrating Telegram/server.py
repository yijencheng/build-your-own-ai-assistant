from fastapi import FastAPI
from pydantic import BaseModel
from google.genai import types

from agent import Agent, ToolExecutionPayload

app = FastAPI()
agent = Agent()

# In-memory conversation store for a single chat session
conversation = []


class ChatRequest(BaseModel):
    message: str

def print_llm_response(response: types.Content) -> None:
    for part in response.parts:
        if part.text:
            print(f"* {part.text}")


def print_llm_tool(response: ToolExecutionPayload) -> None:
    execution = response["execution"]
    if not execution.get("success"):
        print("X [Error Encountered]")
        print(response)
        return

    print(f"✓ {response['name']} : {response['args']}")
    result = execution.get("result")
    if result is not None:
        print(result)


agent.on("model_response", print_llm_response)
agent.on("tool_executed", print_llm_tool)


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, bool]:
    print(f"You: {req.message}")
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=req.message)])
    )

    while True:
        next_message = agent.run(conversation)
        if next_message is None:
            break
        conversation.append(next_message)

    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
