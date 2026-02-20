from typing import Any, Literal, Optional, Protocol, Union

from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict


class ToolDefinition(BaseModel):
    @classmethod
    def to_tool_schema(cls) -> genai_types.Tool:
        json_schema = cls.model_json_schema()
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=json_schema["title"],
                    description=json_schema.get(
                        "description", "Adhere to this provided schema please"
                    ),
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        required=json_schema.get("required", []),
                        properties=json_schema["properties"],
                    ),
                ),
            ]
        )


class WriteTool(ToolDefinition):
    model_config = ConfigDict(title="write")
    path: str
    content: str


class LsTool(ToolDefinition):
    model_config = ConfigDict(title="ls")
    path: str = "."


class EditTool(ToolDefinition):
    model_config = ConfigDict(title="edit")
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False


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

    def to_function_response_part(self) -> genai_types.FunctionResponsePart:
        return genai_types.FunctionResponsePart(
            inline_data=genai_types.FunctionResponseBlob(
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

    def to_gemini_part(self) -> genai_types.Part:
        if self.file_bytes is not None:
            return genai_types.Part.from_bytes(
                data=self.file_bytes,
                mime_type=self.mime_type,
            )
        if self.file_uri:
            return genai_types.Part.from_uri(
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
        cls, response: genai_types.GenerateContentResponse
    ) -> "Message":
        if len(response.candidates) != 1:
            raise ValueError("Invalid number of responses")

        parts: list[Union[TextPart, ThinkingPart, FunctionPart, FilePart]] = []
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

    def to_gemini_content(self) -> genai_types.Content:
        api_parts: list[genai_types.Part] = []
        for part in self.content:
            if isinstance(part, ThinkingPart):
                api_parts.append(
                    genai_types.Part(
                        text=part.text,
                        thought=part.thought,
                        thought_signature=part.thought_signature,
                    )
                )
            elif isinstance(part, FunctionPart):
                api_parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            id=None, args=part.args, name=part.name
                        ),
                        thought=part.thought,
                        thought_signature=part.thought_signature,
                    )
                )
            elif isinstance(part, TextPart):
                api_parts.append(genai_types.Part(text=part.text))
            elif isinstance(part, FilePart):
                api_parts.append(part.to_gemini_part())
        return genai_types.Content(role=self.role, parts=api_parts)


class FunctionResponse(BaseModel):
    name: str
    content: list[Union[TextPart, ImagePart]]
    thought_signature: Optional[bytes] = None

    def to_gemini_content(self) -> genai_types.Content:
        response_text: list[str] = []
        multimodal_parts: list[genai_types.FunctionResponsePart] = []
        for part in self.content:
            if isinstance(part, TextPart):
                response_text.append(part.text)
            elif isinstance(part, ImagePart):
                multimodal_parts.append(part.to_function_response_part())

        response_payload = {"text": "\n".join(response_text)} if response_text else {}
        return genai_types.Content(
            role="tool",
            parts=[
                genai_types.Part.from_function_response(
                    name=self.name,
                    response=response_payload,
                    parts=multimodal_parts,
                )
            ],
        )


HistoryItem = Union[Message, FunctionResponse]


class MessageAdaptor(Protocol):
    def print_new_message(self, message: HistoryItem) -> None: ...

    def save_message(self, message: HistoryItem) -> None: ...

    def get_history(self) -> list[HistoryItem]: ...


class ToolRuntime(Protocol):
    async def execute(
        self, tool_name: str, raw_args: dict[str, Any]
    ) -> FunctionResponse: ...


BASIC_TOOLS: list[type[ToolDefinition]] = [WriteTool, LsTool, EditTool]
