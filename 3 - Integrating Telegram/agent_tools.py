from pydantic import BaseModel
from google.genai import types
import os
import subprocess
from typing import Type


class AgentTool(BaseModel):
    @classmethod
    def to_genai_schema(cls):
        json_schema = cls.model_json_schema()
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=cls.__name__,
                    description=json_schema.get(
                        "description", f"Call the {cls.__name__} tool"
                    ),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=json_schema["properties"],
                        required=json_schema.get("required", []),
                    ),
                )
            ]
        )

    def execute(self):
        """Override this in subclasses to define tool logic."""
        raise NotImplementedError(
            f"Execute not implemented for {self.__class__.__name__}"
        )


class ReadFile(AgentTool):
    path: str

    def execute(self, **kwargs):
        if not os.path.exists(self.path) or not os.path.isfile(self.path):
            return {"ok": False, "error": f"Path does not exist: {self.path}"}

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return {"ok": True, "result": f.read()}
        except Exception as e:
            return {"ok": False, "error": f"Failed to read '{self.path}': {e}"}


class Write(AgentTool):
    path: str
    content: str

    def execute(self, **kwargs):
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(self.path, "w", encoding="utf-8") as f:
                f.write(self.content)

            return {"ok": True, "result": f"Wrote {len(self.content)} chars to {self.path}"}
        except Exception as e:
            return {"ok": False, "error": f"Failed to write '{self.path}': {e}"}


class Edit(AgentTool):
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False

    def execute(self, **kwargs):
        if not os.path.exists(self.path) or not os.path.isfile(self.path):
            return {"ok": False, "error": f"Path does not exist: {self.path}"}

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:
            return {"ok": False, "error": f"Failed to read '{self.path}': {e}"}

        if self.old_str not in original:
            return {"ok": False, "error": f"old_str not found in {self.path}"}

        occurrences = original.count(self.old_str)
        if not self.replace_all and occurrences > 1:
            return {
                "ok": False,
                "error": (
                    "old_str appears multiple times; set replace_all=True "
                    "or provide a more specific old_str"
                ),
            }

        if self.replace_all:
            updated = original.replace(self.old_str, self.new_str)
            replacements = occurrences
        else:
            updated = original.replace(self.old_str, self.new_str, 1)
            replacements = 1

        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(updated)
            return {"ok": True, "result": f"Applied {replacements} edit(s) to {self.path}"}
        except Exception as e:
            return {"ok": False, "error": f"Failed to write '{self.path}': {e}"}


class Bash(AgentTool):
    command: str
    working_dir: str = "."
    timeout_seconds: int = 30

    def execute(self, **kwargs):
        if self.timeout_seconds <= 0:
            return {"ok": False, "error": "timeout_seconds must be > 0"}

        if not os.path.isdir(self.working_dir):
            return {"ok": False, "error": f"Invalid working directory: {self.working_dir}"}

        try:
            completed = subprocess.run(
                self.command,
                shell=True,
                cwd=self.working_dir,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return {
                "ok": True,
                "result": {
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"Command timed out after {self.timeout_seconds}s",
            }
        except Exception as e:
            return {"ok": False, "error": f"Bash execution failed: {e}"}


class AgentRuntime:
    """Minimal runtime that exposes registered tools for the model."""

    def __init__(self):
        self.tools: dict[str, Type[AgentTool]] = {}

    def get_tools(self) -> list[types.Tool]:
        return [tool_cls.to_genai_schema() for tool_cls in self.tools.values()]

    def register_tool(self, tool_cls: Type[AgentTool]):
        self.tools[tool_cls.__name__] = tool_cls

    def execute_tool(self, tool_name: str, args: dict):
        tool_cls = self.tools.get(tool_name)
        if tool_cls is None:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            tool_input = tool_cls.model_validate(args or {})
            return tool_input.execute()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def get_default_runtime():
    default_runtime = AgentRuntime()
    default_runtime.register_tool(ReadFile)
    default_runtime.register_tool(Write)
    default_runtime.register_tool(Edit)
    default_runtime.register_tool(Bash)
    return default_runtime
