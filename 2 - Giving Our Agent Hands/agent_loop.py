from google.genai import Client, types


def read_file(path: str) -> str:
    """Reads a text file and returns its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Failed to read '{path}': {exc}"


# 1. Setup
print("Welcome to Amie! (Type 'exit' to quit)")
client = Client()

read_file_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_file",
            description="Read a text file and return its contents.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(type="STRING", description="File path")
                },
                required=["path"],
            ),
        )
    ]
)


def run(client: Client, conversation: list[types.Content]) -> types.Content | None:
    """
    Runs a single step of the agent.
    Returns a tool response message if tools were called, or None if the agent is done.
    """
    # 1. Let the model think and generate a response
    completion = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversation,
        config=types.GenerateContentConfig(tools=[read_file_tool]),
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

        # NOTE: If you wanted an approval queue, you could pause here!
        if call.name == "read_file":
            result = read_file(call.args.get("path", ""))

            tool_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

    # Return the formatted tool responses as a tool message
    return types.Content(role="user", parts=tool_responses)


# ==========================================
# The Main Application Loop
# ==========================================
conversation: list[types.Content] = []

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    conversation.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

    # The Agent Loop is now beautifully simple
    while True:
        # Run one step of the agent
        next_message = run(client, conversation)

        # If it returns None, the agent has finished answering
        if next_message is None:
            break

        # Otherwise, it returned a tool result. Append it and loop!
        conversation.append(next_message)
