from abc import ABC, abstractmethod
from typing import Any, Awaitable
from pydantic import BaseModel
from google.genai import types
import os


# Empty for now
class AgentContext:
    pass


class ToolResult(BaseModel):
    error: bool
    name: str
    response: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True}

    def to_genai_message(self):
        return types.Part.from_function_response(name=self.name, response=self.response)


class AgentTool(BaseModel, ABC):
    @classmethod
    def tool_name(cls) -> str:
        return cls.__name__

    def tool_result(self, *, error: bool, response: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=error, name=self.__class__.tool_name(), response=response
        )

    @classmethod
    def to_genai_schema(cls):
        json_schema = cls.model_json_schema()
        tool_name = cls.tool_name()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool_name,
                    description=json_schema.get(
                        "description", f"Call the {tool_name} tool"
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=json_schema["properties"],
                        required=json_schema.get("required", []),
                    ),
                )
            ]
        )

    @abstractmethod
    def execute(self, _context: AgentContext) -> Awaitable[ToolResult]:
        """Override this in subclasses to define tool logic."""
        raise NotImplementedError


class ReadFile(AgentTool):
    path: str

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not os.path.exists(self.path) or not os.path.isfile(self.path):
            return self.tool_result(
                error=True, response={"error": "File does not exist"}
            )

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return self.tool_result(
                    error=False,
                    response={
                        "file_content": f"""
File {self.path} was read

<content>
{f.read()}
</content>
"""
                    },
                )

        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to read '{self.path}': {e}"},
            )


TOOLS = [ReadFile]
