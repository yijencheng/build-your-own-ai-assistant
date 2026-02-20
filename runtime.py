import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from agent_types import EditTool, FunctionResponse, LsTool, TextPart, WriteTool


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
