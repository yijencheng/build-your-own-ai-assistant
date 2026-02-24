from google.genai import Client, types


def read_file(path: str) -> str:
    if not path:
        return "No path provided."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Failed to read '{path}': {exc}"


print("Welcome to Amie!")
client = Client()
user_input = input("You: ").strip()

read_file_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_file",
            description="Read a text file and return its contents.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(type="STRING", description="File path"),
                },
                required=["path"],
            ),
        )
    ]
)

completion = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[{"role": "user", "parts": [{"text": user_input}]}],
    config=types.GenerateContentConfig(
        max_output_tokens=4096,
        tools=[read_file_tool],
    ),
)

candidate = completion.candidates[0]
print("Gemini:")
for part in candidate.content.parts:
    if part.text:
        print(part.text)
    if part.function_call and part.function_call.name == "read_file":
        args = part.function_call.args or {}
        print(f"[tool:called] {args}")
        path = str(args.get("path", ""))
        # result = read_file(path)
        print(f"[tool:read_file] reading.....{path}")
