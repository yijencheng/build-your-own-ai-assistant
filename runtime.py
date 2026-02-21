import inspect
import posixpath
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from agent_types import (
    BashTool,
    EditTool,
    FunctionResponse,
    LsTool,
    TextPart,
    ToolRuntime,
    WriteTool,
)

try:
    import modal
except ImportError:  # pragma: no cover - handled at runtime
    modal = None


ToolExecutor = Callable[
    [dict[str, Any], "RuntimeContext"], FunctionResponse | Awaitable[FunctionResponse]
]


@dataclass
class RuntimeContext:
    sandbox: Optional[Any] = None
    volume: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HandlerToolRuntime:
    def __init__(
        self,
        handlers: dict[str, ToolExecutor],
        context: Optional[RuntimeContext] = None,
    ):
        self.handlers = handlers
        self.context = context or RuntimeContext()

    async def execute(self, tool_name: str, raw_args: dict[str, Any]) -> FunctionResponse:
        handler = self.handlers.get(tool_name)
        if handler is None:
            return FunctionResponse(
                name=tool_name,
                content=[TextPart(text=f"Unknown tool: {tool_name}")],
            )

        result = handler(raw_args, self.context)
        if inspect.isawaitable(result):
            return await result
        return result


class BasicFileToolRuntime:
    def __init__(self, root: Optional[Path] = None):
        self.root = (root or Path.cwd()).resolve()

    async def execute(self, tool_name: str, raw_args: dict[str, Any]) -> FunctionResponse:
        match tool_name:
            case "write":
                return self._write(raw_args)
            case "ls":
                return self._ls(raw_args)
            case "edit":
                return self._edit(raw_args)
            case "bash":
                return self._bash(raw_args)
            case _:
                return FunctionResponse(
                    name=tool_name,
                    content=[TextPart(text=f"Unknown tool: {tool_name}")],
                )

    def _resolve_path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if self.root != resolved and self.root not in resolved.parents:
            raise ValueError(f"Path escapes runtime root: {path}")
        return resolved

    def _write(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = WriteTool.model_validate(raw_args)
        path = self._resolve_path(args.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return FunctionResponse(
            name="write",
            content=[TextPart(text=f"Wrote content to {path}")],
        )

    def _ls(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = LsTool.model_validate(raw_args)
        path = self._resolve_path(args.path)
        if not path.exists():
            return FunctionResponse(
                name="ls",
                content=[TextPart(text=f"Path does not exist: {path}")],
            )
        if not path.is_dir():
            return FunctionResponse(
                name="ls",
                content=[TextPart(text=f"Not a directory: {path}")],
            )
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
        output = "\n".join(
            [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
        )
        return FunctionResponse(
            name="ls",
            content=[TextPart(text=output or "(empty directory)")],
        )

    def _edit(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = EditTool.model_validate(raw_args)
        path = self._resolve_path(args.path)
        if not path.exists():
            return FunctionResponse(
                name="edit",
                content=[TextPart(text=f"File does not exist: {path}")],
            )

        original = path.read_text(encoding="utf-8")
        if args.old_str not in original:
            return FunctionResponse(
                name="edit",
                content=[TextPart(text=f"Target string not found in {path}")],
            )

        occurrences = original.count(args.old_str)
        if not args.replace_all and occurrences > 1:
            return FunctionResponse(
                name="edit",
                content=[
                    TextPart(
                        text=(
                            "old_str appears multiple times; set replace_all=true "
                            "or provide a more specific old_str"
                        )
                    )
                ],
            )

        if args.replace_all:
            updated = original.replace(args.old_str, args.new_str)
            replacements = occurrences
        else:
            updated = original.replace(args.old_str, args.new_str, 1)
            replacements = 1

        path.write_text(updated, encoding="utf-8")
        return FunctionResponse(
            name="edit",
            content=[TextPart(text=f"Applied {replacements} edit(s) to {path}")],
        )

    def _bash(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = BashTool.model_validate(raw_args)
        cwd = self._resolve_path(args.working_dir)
        if not cwd.exists() or not cwd.is_dir():
            return FunctionResponse(
                name="bash",
                content=[TextPart(text=f"Invalid working directory: {cwd}")],
            )

        try:
            completed = subprocess.run(
                args.command,
                cwd=cwd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return FunctionResponse(
                name="bash",
                content=[
                    TextPart(
                        text=f"Command timed out after {args.timeout_seconds}s: {args.command}"
                    )
                ],
            )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        lines: list[str] = [f"exit_code: {completed.returncode}"]
        if stdout:
            lines.append("stdout:")
            lines.append(stdout)
        if stderr:
            lines.append("stderr:")
            lines.append(stderr)
        if not stdout and not stderr:
            lines.append("(no output)")

        return FunctionResponse(name="bash", content=[TextPart(text="\n".join(lines))])


class ModalSandboxToolRuntime(ToolRuntime):
    def __init__(
        self,
        *,
        app_name: str = "build-your-own-ai-assistant",
        sandbox_name: str = "agent-runtime",
        sandbox_timeout_seconds: int = 600,
        idle_timeout_seconds: int = 120,
        remote_root: str = "/root",
    ):
        self.app_name = app_name
        self.sandbox_name = sandbox_name
        self.sandbox_timeout_seconds = sandbox_timeout_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.remote_root = posixpath.normpath(remote_root)
        self._app: Optional[Any] = None
        self._sandbox: Optional[Any] = None
        self._image = None

    async def execute(self, tool_name: str, raw_args: dict[str, Any]) -> FunctionResponse:
        match tool_name:
            case "write":
                return await self._write(raw_args)
            case "ls":
                return await self._ls(raw_args)
            case "edit":
                return await self._edit(raw_args)
            case "bash":
                return await self._bash(raw_args)
            case _:
                return FunctionResponse(
                    name=tool_name,
                    content=[TextPart(text=f"Unknown tool: {tool_name}")],
                )

    async def _ensure_sandbox(self) -> Any:
        if modal is None:
            raise RuntimeError(
                "modal is not installed. Install it with `uv add modal` or `pip install modal`."
            )

        if self._sandbox is not None and await self._sandbox.poll.aio() is None:
            return self._sandbox

        if self._app is None:
            self._app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        if self._image is None:
            self._image = modal.Image.debian_slim(python_version="3.12")

        # Prefer reattaching to a named running sandbox so process restarts reuse state.
        try:
            candidate = await modal.Sandbox.from_name.aio(self.app_name, self.sandbox_name)
            await candidate.hydrate.aio()
            if await candidate.poll.aio() is None:
                self._sandbox = candidate
                return self._sandbox
        except modal.exception.NotFoundError:
            pass

        self._sandbox = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=self._app,
            name=self.sandbox_name,
            image=self._image,
            timeout=self.sandbox_timeout_seconds,
            idle_timeout=self.idle_timeout_seconds,
            workdir=self.remote_root,
        )
        return self._sandbox

    def _resolve_remote_path(self, path: str) -> str:
        if path.startswith("/"):
            candidate = path
        else:
            candidate = posixpath.join(self.remote_root, path)
        normalized = posixpath.normpath(candidate)
        root = self.remote_root.rstrip("/") or "/"
        if normalized != root and not normalized.startswith(f"{root}/"):
            raise ValueError(f"Path escapes runtime root: {path}")
        return normalized

    async def _run_command(
        self,
        command: str,
        *,
        timeout_seconds: int,
        workdir: Optional[str] = None,
    ) -> tuple[int, str, str]:
        sandbox = await self._ensure_sandbox()
        process = await sandbox.exec.aio(
            "bash",
            "-lc",
            command,
            timeout=timeout_seconds,
            workdir=workdir or self.remote_root,
            text=True,
        )
        exit_code = await process.wait.aio()
        stdout = (await process.stdout.read.aio()).strip() if process.stdout else ""
        stderr = (await process.stderr.read.aio()).strip() if process.stderr else ""
        return exit_code, stdout, stderr

    async def _write(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = WriteTool.model_validate(raw_args)
        remote_path = self._resolve_remote_path(args.path)
        try:
            sandbox = await self._ensure_sandbox()
            parent = posixpath.dirname(remote_path) or self.remote_root
            await sandbox.mkdir.aio(parent, parents=True)
            file_handle = await sandbox.open.aio(remote_path, "w")
            await file_handle.write.aio(args.content)
            await file_handle.close.aio()
        except Exception as exc:
            return FunctionResponse(
                name="write",
                content=[TextPart(text=f"Modal sandbox write failed: {exc}")],
            )
        return FunctionResponse(
            name="write",
            content=[TextPart(text=f"Wrote content to {remote_path}")],
        )

    async def _ls(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = LsTool.model_validate(raw_args)
        remote_path = self._resolve_remote_path(args.path)
        try:
            sandbox = await self._ensure_sandbox()
            entries = await sandbox.ls.aio(remote_path)
        except Exception as exc:
            message = str(exc)
            if "No such file or directory" in message:
                return FunctionResponse(
                    name="ls",
                    content=[TextPart(text=f"Path does not exist: {remote_path}")],
                )
            if "Not a directory" in message:
                return FunctionResponse(
                    name="ls",
                    content=[TextPart(text=f"Not a directory: {remote_path}")],
                )
            return FunctionResponse(
                name="ls",
                content=[TextPart(text=f"Modal sandbox ls failed: {exc}")],
            )
        output = "\n".join(entries) if entries else "(empty directory)"
        return FunctionResponse(name="ls", content=[TextPart(text=output)])

    async def _edit(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = EditTool.model_validate(raw_args)
        remote_path = self._resolve_remote_path(args.path)
        try:
            sandbox = await self._ensure_sandbox()
            file_handle = await sandbox.open.aio(remote_path, "r")
            original = await file_handle.read.aio()
            await file_handle.close.aio()
        except Exception as exc:
            message = str(exc)
            if "No such file or directory" in message:
                return FunctionResponse(
                    name="edit",
                    content=[TextPart(text=f"File does not exist: {remote_path}")],
                )
            return FunctionResponse(
                name="edit",
                content=[TextPart(text=f"Modal sandbox edit failed: {exc}")],
            )

        if args.old_str not in original:
            return FunctionResponse(
                name="edit",
                content=[TextPart(text=f"Target string not found in {remote_path}")],
            )

        occurrences = original.count(args.old_str)
        if not args.replace_all and occurrences > 1:
            return FunctionResponse(
                name="edit",
                content=[
                    TextPart(
                        text=(
                            "old_str appears multiple times; set replace_all=true "
                            "or provide a more specific old_str"
                        )
                    )
                ],
            )

        if args.replace_all:
            updated = original.replace(args.old_str, args.new_str)
            replacements = occurrences
        else:
            updated = original.replace(args.old_str, args.new_str, 1)
            replacements = 1

        try:
            file_handle = await sandbox.open.aio(remote_path, "w")
            await file_handle.write.aio(updated)
            await file_handle.close.aio()
        except Exception as exc:
            return FunctionResponse(
                name="edit",
                content=[TextPart(text=f"Modal sandbox edit failed: {exc}")],
            )
        return FunctionResponse(
            name="edit",
            content=[TextPart(text=f"Applied {replacements} edit(s) to {remote_path}")],
        )

    async def _bash(self, raw_args: dict[str, Any]) -> FunctionResponse:
        args = BashTool.model_validate(raw_args)
        remote_workdir = self._resolve_remote_path(args.working_dir)

        try:
            exit_code, stdout, stderr = await self._run_command(
                args.command,
                timeout_seconds=args.timeout_seconds,
                workdir=remote_workdir,
            )
        except Exception as exc:
            return FunctionResponse(
                name="bash",
                content=[TextPart(text=f"Modal sandbox command failed: {exc}")],
            )

        lines: list[str] = [f"exit_code: {exit_code}"]
        if stdout:
            lines.append("stdout:")
            lines.append(stdout)
        if stderr:
            lines.append("stderr:")
            lines.append(stderr)
        if not stdout and not stderr:
            lines.append("(no output)")

        return FunctionResponse(name="bash", content=[TextPart(text="\n".join(lines))])

    async def terminate(self) -> None:
        if self._sandbox is not None and await self._sandbox.poll.aio() is None:
            await self._sandbox.terminate.aio()
