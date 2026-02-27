from fastapi import FastAPI
from pydantic import BaseModel
from google.genai import types

from agent import Agent, print_tool_result, print_llm_response

app = FastAPI()

# In-memory conversation store
conversation = []


class ChatRequest(BaseModel):
    message: str


agent = Agent()
agent.on("on_tool_result", print_tool_result)
agent.on("on_model_response", print_llm_response)


@app.post("/chat")
async def chat(req: ChatRequest):
    print(f"You: {req.message}")
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=req.message)])
    )

    while True:
        next_message = await agent.run(conversation)
        if next_message is None:
            break
        conversation.append(next_message)

    return {"response": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
