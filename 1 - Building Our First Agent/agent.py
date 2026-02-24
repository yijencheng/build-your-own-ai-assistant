from google.genai import types, Client


print("Welcome to Amie!")
client = Client()
conversations: list[dict[str, object]] = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"q", "quit"}:
        break
    if not user_input:
        continue

    conversations.append({"role": "user", "parts": [{"text": user_input}]})

    completion = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversations,
        config=types.GenerateContentConfig(max_output_tokens=4096),
    )

    content = completion.candidates[0]
    print("Gemini: ")
    for part in content.content.parts:
        if part.text:
            print(part.text)

    conversations.append({"role": "model", "parts": content.content.parts})

print("Exiting...")
