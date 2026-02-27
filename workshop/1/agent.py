from google.genai import Client, types

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

client = Client()

completion = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[{"role": "user", "parts": [{"text": "Can u read the ./README.md file"}]}],
    config=types.GenerateContentConfig(tools=[read_file_tool]),
)

print(completion)
