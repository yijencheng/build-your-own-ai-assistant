from agent_tools import AgentRuntime, get_default_runtime
from google.genai import types, Client


def run(
    client: Client, conversation: list[types.Content], runtime: AgentRuntime
) -> types.Content | None:
    """
    Runs a single step of the agent.
    Returns a tool response message if tools were called, or None if the agent is done.
    """
    # 1. Let the model think and generate a response
    completion = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversation,
        config=types.GenerateContentConfig(tools=runtime.get_tools()),
    )

    message = completion.candidates[0].content
    conversation.append(message)

    # 2. Check for function calls
    function_calls = [
        part.function_call for part in message.parts if part.function_call
    ]

    # 3. Base Case: The agent is done. Return None.
    if not function_calls:
        text_parts = [part.text for part in message.parts if part.text]
        print(f"Amie: {''.join(text_parts)}")
        return None

    # 4. Action Case: Execute tools
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

    # Return the formatted tool responses as a tool message
    return types.Content(role="user", parts=tool_responses)


# ==========================================
# The Main Application Loop
# ==========================================
conversation: list[types.Content] = []
client = Client()
runtime = get_default_runtime()
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    conversation.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

    # The Agent Loop is now beautifully simple
    while True:
        # Run one step of the agent
        next_message = run(client, conversation, runtime=runtime)

        # If it returns None, the agent has finished answering
        if next_message is None:
            break

        # Otherwise, it returned a tool result. Append it and loop!
        conversation.append(next_message)
