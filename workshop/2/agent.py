from google.genai import Client, types
from rich import print

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


def read_file(path: str) -> str:
    """Reads a text file and returns its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Failed to read '{path}': {exc}"


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
