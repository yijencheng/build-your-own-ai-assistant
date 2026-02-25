from google.genai import Client, types
from rich import print
from typing import Literal, TypeAlias, TypedDict

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


def read_file(path: str) -> str:
    """Reads a text file and returns its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Failed to read '{path}': {exc}"


class FunctionResponseRunResult(TypedDict):
    kind: Literal["function_response"]
    message: types.Content


RunResult: TypeAlias = None | FunctionResponseRunResult


async def run(client: Client, conversation: list[types.Content]) -> RunResult:
    """
    Runs a single step of the agent.
    Returns None or a typed function-response result.
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
        for part in message.parts:
            if part.text:
                print(f"* {part.text}")
        return None

    # 4. Action Case: Execute tools
    tool_responses: list[types.Part] = []
    for call in function_calls:
        print(f"[Agent Action] Running '{call.name}' with args: {call.args}")
        if call.name == "read_file":
            result = read_file((call.args or {}).get("path", ""))

            tool_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

    # Return the formatted tool responses as a tool message
    return {
        "kind": "function_response",
        "message": types.Content(role="user", parts=tool_responses),
    }


client = Client()

contents = [
    types.UserContent(
        parts=[types.Part.from_text(text="Can u read the ./README.md file")]
    )
]

completion = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=types.GenerateContentConfig(tools=[read_file_tool]),
)

fc = completion.candidates[0].content.parts[0].function_call

if fc:
    print(fc.name)
    print(fc.args)
    if fc.name == "read_file":
        path = (fc.args or {}).get("path", "./README.md")
        # Reuse model content to preserve function-call thought signatures.
        contents.append(completion.candidates[0].content)
        contents.append(
            types.UserContent(
                parts=[
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"path": path, "content": read_file(path)},
                    )
                ]
            )
        )

completion = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=types.GenerateContentConfig(tools=[read_file_tool]),
)
print(completion)
