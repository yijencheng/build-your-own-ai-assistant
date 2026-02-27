from google.genai import Client, types
from rich import print
from typing import Literal, TypedDict, TypeAlias
import asyncio

client = Client()


class FunctionResponseRunResult(TypedDict):
    kind: Literal["function_response"]
    message: types.UserContent


RunResult: TypeAlias = None | FunctionResponseRunResult

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


async def run(
    client: Client, contents: list[types.Content]
) -> tuple[types.Content, RunResult]:
    completion = await client.aio.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(tools=[read_file_tool]),
    )

    message = completion.candidates[0].content

    function_calls = [
        part.function_call for part in message.parts if part.function_call
    ]
    if not function_calls:
        return message, None

    tool_responses: list[types.Part] = []
    for call in function_calls:
        if not call or call.name != "read_file":
            continue

        path = (call.args or {}).get("path", "")
        print(f"[Agent Action] read_file(path={path!r})")
        result = read_file(path)
        tool_responses.append(
            types.Part.from_function_response(
                name=call.name,
                response={"path": path, "content": result},
            )
        )

    if not tool_responses:
        return message, None

    return message, {
        "kind": "function_response",
        "message": types.UserContent(parts=tool_responses),
    }


async def main() -> None:
    client = Client()
    contents: list[types.Content] = []

    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        contents.append(
            types.UserContent(parts=[types.Part.from_text(text=user_input)])
        )
        while True:
            assistant_message, tool_result = await run(client, contents)
            contents.append(assistant_message)
            if tool_result is None:
                for part in assistant_message.parts:
                    if part.text:
                        print(f"\nAssistant: {part.text}")
                break
            assert tool_result["kind"] == "function_response"
            contents.append(tool_result["message"])


if __name__ == "__main__":
    asyncio.run(main())
