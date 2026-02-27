from abc import ABC, abstractmethod
from typing import Any, Awaitable
from pydantic import BaseModel
from google.genai import types
import os
import shlex
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
        if not _context.has_sandbox:
            return self.tool_result(
                error=True, response={"error": "No sandbox available"}
            )

        try:
            process = await _context.sandbox.exec.aio("cat", self.path)
            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            await process.wait.aio()

            if process.returncode != 0:
                return self.tool_result(
                    error=True,
                    response={"error": f"Failed to read '{self.path}': {stderr}"},
                )

            return self.tool_result(
                error=False,
                response={
                    "file_content": f"""
File {self.path} was read

<content>
{stdout}
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
        if not _context.has_sandbox:
            return self.tool_result(
                error=True, response={"error": "No sandbox available"}
            )

        try:
            parent = os.path.dirname(self.path)
            if parent:
                await (await _context.sandbox.exec.aio("mkdir", "-p", parent)).wait.aio()

            process = await _context.sandbox.exec.aio(
                "bash", "-c", f"cat > {shlex.quote(self.path)}"
            )
            process.stdin.write(self.content.encode("utf-8"))
            process.stdin.write_eof()
            await process.wait.aio()

            if process.returncode != 0:
                stderr = await process.stderr.read.aio()
                return self.tool_result(
                    error=True,
                    response={"error": f"Failed to write '{self.path}': {stderr}"},
                )

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
        if not _context.has_sandbox:
            return self.tool_result(
                error=True, response={"error": "No sandbox available"}
            )

        try:
            process = await _context.sandbox.exec.aio("cat", self.path)
            original = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            await process.wait.aio()

            if process.returncode != 0:
                return self.tool_result(
                    error=True,
                    response={"error": f"Path does not exist: {self.path}"},
                )
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
            process = await _context.sandbox.exec.aio(
                "bash", "-c", f"cat > {shlex.quote(self.path)}"
            )
            process.stdin.write(updated.encode("utf-8"))
            process.stdin.write_eof()
            await process.wait.aio()

            if process.returncode != 0:
                stderr = await process.stderr.read.aio()
                return self.tool_result(
                    error=True,
                    response={"error": f"Failed to write '{self.path}': {stderr}"},
                )

            return self.tool_result(
                error=False,
                response={"result": f"Applied {replacements} edit(s) to {self.path}"},
            )
        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Failed to write '{self.path}': {e}"},
            )


class SaveEnvironment(AgentTool):
    """Snapshot the current sandbox filesystem so that any installed packages or environment changes persist across reboots. Call this after installing new packages."""

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not _context.has_sandbox:
            return self.tool_result(
                error=True, response={"error": "No sandbox available"}
            )

        try:
            new_image = _context.sandbox.snapshot_filesystem()
            image_id = new_image.object_id

            sandbox_images = modal.Dict.from_name(
                "koroku-sandbox-images", create_if_missing=True
            )
            sandbox_images["sandbox_image"] = image_id

            return self.tool_result(
                error=False,
                response={
                    "result": f"Environment saved (image: {image_id}). Future sandboxes will use this snapshot."
                },
            )
        except Exception as e:
            return self.tool_result(
                error=True, response={"error": f"Snapshot failed: {e}"}
            )


class Bash(AgentTool):
    command: str
    working_dir: str = "."
    timeout_seconds: int = 30

    async def execute(self, _context: AgentContext) -> ToolResult:
        if not _context.has_sandbox:
            return self.tool_result(
                error=True, response={"error": "No sandbox available"}
            )

        if self.timeout_seconds <= 0:
            return self.tool_result(
                error=True,
                response={"error": "timeout_seconds must be > 0"},
            )

        try:
            cmd = (
                f"cd {shlex.quote(self.working_dir)} && {self.command}"
                if self.working_dir != "."
                else self.command
            )
            process = await _context.sandbox.exec.aio(
                "bash", "-c", cmd, timeout=self.timeout_seconds
            )
            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            await process.wait.aio()

            return self.tool_result(
                error=False,
                response={
                    "result": (
                        "Executed command succesfully\n\n"
                        "<output>\n"
                        f"{stdout}{stderr}"
                        "</output>"
                    )
                },
            )
        except Exception as e:
            return self.tool_result(
                error=True,
                response={"error": f"Bash execution failed: {e}"},
            )


TOOLS = [ReadFile, Write, Edit, Bash, SaveEnvironment]
