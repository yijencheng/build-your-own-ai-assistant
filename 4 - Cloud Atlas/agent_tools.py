from abc import ABC, abstractmethod
from typing import Any, Awaitable
from pydantic import BaseModel
from google.genai import types
import os
import subprocess
import modal
import httpx


class AgentContext:
    def __init__(
        self,
        *,
        sandbox: modal.Sandbox | None = None,
        telegram_bot_token: str | None = None,
        telegram_chat_id: int | None = None,
    ):
        self.sandbox = sandbox
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

    @property
    def has_sandbox(self) -> bool:
        return self.sandbox is not None

    @property
    def has_telegram(self) -> bool:
        return (
            self.telegram_bot_token is not None and self.telegram_chat_id is not None
        )

    async def send_telegram_message(self, text: str) -> None:
        if not self.has_telegram:
            return
        payload = text.strip()
        if not payload:
            return
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(url, json={"chat_id": self.telegram_chat_id, "text": payload})


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


class Write(AgentTool):
    path: str
    content: str

    async def execute(self, _context: AgentContext) -> ToolResult:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(self.path, "w", encoding="utf-8") as f:
                f.write(self.content)

            return self.tool_result(
                error=False,
                response={"result": f"succesfully wrote content to {self.path}"},
            )
        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to write '{self.path}': {e}"},
            )


class Edit(AgentTool):
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not os.path.exists(self.path) or not os.path.isfile(self.path):
            return self.tool_result(
                error=True,
                response={"error": f"Path does not exist: {self.path}"},
            )

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to read '{self.path}': {e}"},
            )

        if self.old_str not in original:
            return self.tool_result(
                error=True,
                response={"error": f"old_str not found in {self.path}"},
            )

        occurrences = original.count(self.old_str)
        if not self.replace_all and occurrences > 1:
            return self.tool_result(
                error=True,
                response={
                    "error": (
                        "old_str appears multiple times; set replace_all=True or provide a more specific old_str"
                    )
                },
            )

        if self.replace_all:
            updated = original.replace(self.old_str, self.new_str)
            replacements = occurrences
        else:
            updated = original.replace(self.old_str, self.new_str, 1)
            replacements = 1

        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(updated)
            return self.tool_result(
                error=False,
                response={"result": f"Applied {replacements} edit(s) to {self.path}"},
            )
        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to write '{self.path}': {e}"},
            )


class InstallPackages(AgentTool):
    """Install Python packages into the current environment. Packages are also saved to /data/requirements.txt so they persist across container reboots."""

    packages: list[str]

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not self.packages:
            return self.tool_result(
                error=True, response={"error": "No packages specified"}
            )

        cmd = ["pip", "install"] + self.packages
        try:
            completed = subprocess.run(
                cmd, text=True, capture_output=True, timeout=120, check=False
            )
            if completed.returncode != 0:
                return self.tool_result(
                    error=True,
                    response={"error": f"pip install failed:\n{completed.stderr}"},
                )

            # Persist to volume so they survive container restarts
            req_path = "/data/requirements.txt"
            existing: set[str] = set()
            if os.path.exists(req_path):
                with open(req_path, "r") as f:
                    existing = {line.strip() for line in f if line.strip()}
            with open(req_path, "a") as f:
                for pkg in self.packages:
                    if pkg not in existing:
                        f.write(f"{pkg}\n")

            return self.tool_result(
                error=False,
                response={
                    "result": f"Installed: {', '.join(self.packages)}\n{completed.stdout}"
                },
            )
        except subprocess.TimeoutExpired:
            return self.tool_result(
                error=True, response={"error": "pip install timed out after 120s"}
            )
        except Exception as e:
            return self.tool_result(
                error=True, response={"error": f"Install failed: {e}"}
            )


class Bash(AgentTool):
    command: str
    working_dir: str = "."
    timeout_seconds: int = 30

    async def execute(self, _context: AgentContext) -> ToolResult:
        if self.timeout_seconds <= 0:
            return self.tool_result(
                error=True,
                response={"error": "timeout_seconds must be > 0"},
            )

        if not os.path.isdir(self.working_dir):
            return self.tool_result(
                error=True,
                response={"error": f"Invalid working directory: {self.working_dir}"},
            )

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
            return self.tool_result(
                error=False,
                response={
                    "result": (
                        "Executed command succesfully\n\n"
                        "<output>\n"
                        f"{completed.stdout}{completed.stderr}"
                        "</output>"
                    )
                },
            )
        except subprocess.TimeoutExpired:
            return self.tool_result(
                error=True,
                response={
                    "error": f"Command timed out after {self.timeout_seconds}s",
                },
            )
        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Bash execution failed: {e}"},
            )


TOOLS = [ReadFile, Write, Edit, Bash, InstallPackages]
