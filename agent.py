from google.genai import types, Client
from pydantic import BaseModel
from typing import Any, Literal, Optional, Union
from adaptor import MessageAdaptor, ConsoleMemoryMessageAdaptor


class ToolDefinition(BaseModel):
    @classmethod
    def to_tool_schema(cls):
        json_schema = cls.model_json_schema()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=json_schema["title"],
                    description=json_schema.get(
                        "description", "Adhere to this provided schema please"
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        required=json_schema["required"],
                        properties=json_schema["properties"],
                    ),
                ),
            ]
        )


class getWeather(ToolDefinition):
    location: str


TOOLS: list[type[ToolDefinition]] = [getWeather]
TOOL_NAME_TO_MODEL: dict[str, type[ToolDefinition]] = {
    tool.__name__: tool for tool in TOOLS
}
TOOL_NAMES: set[str] = set(TOOL_NAME_TO_MODEL.keys())


class ThinkingPart(BaseModel):
    thought: Optional[bool] = None
    thought_signature: Optional[bytes] = None
    text: str


class TextPart(BaseModel):
    text: str


class ImagePart(BaseModel):
    image_bytes: bytes
    display_name: str
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]

    def to_function_response_part(self) -> types.FunctionResponsePart:
        return types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(
                mime_type=self.mime_type,
                display_name=self.display_name,
                data=self.image_bytes,
            )
        )


class FilePart(BaseModel):
    mime_type: Literal[
        "image/png",
        "image/jpeg",
        "image/webp",
        "video/mp4",
        "audio/mpeg",
        "audio/wav",
        "audio/ogg",
        "application/pdf",
        "text/plain",
    ]
    file_uri: Optional[str] = None
    file_bytes: Optional[bytes] = None

    def to_gemini_part(self) -> types.Part:
        if self.file_bytes is not None:
            return types.Part.from_bytes(
                data=self.file_bytes,
                mime_type=self.mime_type,
            )
        if self.file_uri:
            return types.Part.from_uri(
                file_uri=self.file_uri,
                mime_type=self.mime_type,
            )
        raise ValueError("FilePart requires either file_bytes or file_uri.")


class FunctionPart(BaseModel):
    name: str
    args: dict[str, Any]
    thought: Optional[bool] = None
    thought_signature: Optional[bytes] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class Message(BaseModel):
    role: Union[Literal["user"], Literal["assistant"], Literal["model"]]
    content: list[Union[TextPart, ThinkingPart, FunctionPart, FilePart]]
    usage: Optional[Usage]

    @classmethod
    def from_assistant_response(
        cls, response: types.GenerateContentResponse
    ) -> "Message":
        if len(response.candidates) != 1:
            raise ValueError("Invalid number of responses")

        parts = []
        for part in response.candidates[0].content.parts:
            if part.text and (part.thought or part.thought_signature):
                parts.append(
                    ThinkingPart(
                        thought=part.thought,
                        thought_signature=part.thought_signature,
                        text=part.text,
                    )
                )
            elif part.text:
                parts.append(TextPart(text=part.text))
            elif part.function_call:
                parts.append(
                    FunctionPart(
                        name=part.function_call.name,
                        args=part.function_call.args,
                        thought=part.thought,
                        thought_signature=part.thought_signature,
                    )
                )
        usage_metadata = response.usage_metadata
        input_tokens = usage_metadata.prompt_token_count or 0
        output_tokens = usage_metadata.candidates_token_count or max(
            0, (usage_metadata.total_token_count or 0) - input_tokens
        )

        return cls(
            role=response.candidates[0].content.role or "assistant",
            content=parts,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def to_gemini_content(self) -> types.Content:
        api_parts = []
        for part in self.content:
            if isinstance(part, ThinkingPart):
                api_parts.append(
                    types.Part(
                        text=part.text,
                        thought=part.thought,
                        thought_signature=part.thought_signature,
                    )
                )
            elif isinstance(part, FunctionPart):
                api_parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=None, args=part.args, name=part.name
                        ),
                        thought=part.thought,
                        thought_signature=part.thought_signature,
                    )
                )
            elif isinstance(part, TextPart):
                api_parts.append(types.Part(text=part.text))
            elif isinstance(part, FilePart):
                api_parts.append(part.to_gemini_part())
        return types.Content(role=self.role, parts=api_parts)


class FunctionResponse(BaseModel):
    name: str
    content: list[Union[TextPart, ImagePart]]
    thought_signature: Optional[bytes] = None

    def to_gemini_content(self) -> types.Content:
        response_text: list[str] = []
        multimodal_parts: list[types.FunctionResponsePart] = []
        for part in self.content:
            if isinstance(part, TextPart):
                response_text.append(part.text)
            elif isinstance(part, ImagePart):
                multimodal_parts.append(part.to_function_response_part())

        # Keep structured response simple; images are passed via nested parts.
        response_payload = {"text": "\n".join(response_text)} if response_text else {}
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=self.name,
                    response=response_payload,
                    parts=multimodal_parts,
                )
            ],
        )


class Agent:
    def __init__(
        self,
        compaction_threshold: int,
        tools: Optional[list[type[ToolDefinition]]] = None,
        message_adaptor: Optional[MessageAdaptor] = None,
    ):
        """
        We define a client, history
        """
        self.client = Client()
        self.compaction_threshold = compaction_threshold
        self.tool_models = tools or TOOLS
        self.tool_name_to_model = {tool.__name__: tool for tool in self.tool_models}
        self.tool_names = set(self.tool_name_to_model.keys())
        self.tools = [tool.to_tool_schema() for tool in self.tool_models]
        self.message_adaptor = message_adaptor or ConsoleMemoryMessageAdaptor()

    def run(
        self,
        turn_input: Message,
        *,
        max_round_budget: int = 8,
    ) -> Message:
        if max_round_budget == 0:
            return

        if not isinstance(turn_input, Message):
            raise ValueError(f"Invalid type of {type(turn_input)}")

        self.message_adaptor.save_message(turn_input)

        for _ in range(max_round_budget):
            history = self.message_adaptor.get_history()
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[item.to_gemini_content() for item in history],
                config=types.GenerateContentConfig(
                    tools=self.tools,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level=types.ThinkingLevel.MEDIUM,
                    ),
                    temperature=0.0,
                ),
            )

            message = Message.from_assistant_response(response)
            self.message_adaptor.print_new_message(message)
            self.message_adaptor.save_message(message)
            tool_calls = [
                part for part in message.content if isinstance(part, FunctionPart)
            ]
            if not tool_calls:
                return message

            for tool_call in tool_calls:
                tool_response = self.execute_tool(tool_call)
                self.message_adaptor.print_new_message(tool_response)
                self.message_adaptor.save_message(tool_response)

        raise RuntimeError("Exceeded max_round_budget while resolving tool calls.")

    def execute_tool(self, tool_call: FunctionPart):
        match tool_call.name:
            case "getWeather":
                weather_args = getWeather.model_validate(tool_call.args)
                return FunctionResponse(
                    name=tool_call.name,
                    content=[
                        TextPart(
                            text=f"Mock weather for {weather_args.location} is sunny with a touch of rain to come in the afternoon"
                        )
                    ],
                )
            case _:
                raise ValueError(f"Unknown tool: {tool_call.name}")


def main():
    agent = Agent(compaction_threshold=100000, tools=TOOLS)

    while True:
        user_input = input("You: ")
        agent.run(Message(role="user", content=[TextPart(text=user_input)], usage=None))


if __name__ == "__main__":
    main()
