import importlib
from pathlib import Path

import agent_tools
from google.genai import Client, types


TOOLS_FILE = Path(agent_tools.__file__).resolve()


def mtime(path: str | Path) -> float:
    return Path(path).stat().st_mtime


runtime = agent_tools.get_default_runtime()
last_modified = mtime(TOOLS_FILE)


def maybe_reload_runtime() -> bool:
    global runtime, last_modified, agent_tools, TOOLS_FILE

    current = mtime(TOOLS_FILE)
    if current != last_modified:
        agent_tools = importlib.reload(agent_tools)
        TOOLS_FILE = Path(agent_tools.__file__).resolve()
        runtime = agent_tools.get_default_runtime()
        last_modified = mtime(TOOLS_FILE)
        return True

    return False


def run(client: Client, conversation: list[types.Content]) -> types.Content | None:
    """
    Runs a single step of the agent.
    Returns a tool response message if tools were called, or None if the agent is done.
    """
    maybe_reload_runtime()

    completion = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversation,
        config=types.GenerateContentConfig(tools=runtime.get_tools()),
    )

    message = completion.candidates[0].content
    conversation.append(message)

    function_calls = [
        part.function_call for part in message.parts if part.function_call
    ]

    if not function_calls:
        text_parts = [part.text for part in message.parts if part.text]
        print(f"Amie: {''.join(text_parts)}")
        return None

    tool_responses: list[types.Part] = []
    for call in function_calls:
        print(f"[Agent Action] Running '{call.name}' with args: {call.args}")
        tool_result = runtime.execute_tool(call.name, call.args or {})
        tool_responses.append(
            types.Part.from_function_response(
                name=call.name,
                response={"result": tool_result},
            )
        )

    return types.Content(role="user", parts=tool_responses)


def main():
    print("Welcome to Amie! (Type 'exit' to quit)")
    conversation: list[types.Content] = []
    client = Client()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        conversation.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        while True:
            next_message = run(client, conversation)
            if next_message is None:
                break
            conversation.append(next_message)


if __name__ == "__main__":
    main()
