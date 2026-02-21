import json

from agent_types import (
    FunctionPart,
    FunctionResponse,
    HistoryItem,
    ImagePart,
    MessageAdaptor,
    TextPart,
    ThinkingPart,
)
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class ConsoleMemoryMessageAdaptor(MessageAdaptor):
    def __init__(self):
        self._history: list[HistoryItem] = []

    def print_new_message(self, message: HistoryItem) -> None:
        if isinstance(message, FunctionResponse):
            self._print_function_response(message)
            return

        visible_parts: list[str] = []
        for part in message.content:
            if isinstance(part, FunctionPart):
                visible_parts.append(f"[tool_call] {part.name}")
            elif isinstance(part, ThinkingPart):
                visible_parts.append(f"[Thinking] {part.text}")
            elif isinstance(part, TextPart):
                visible_parts.append(part.text)

        if not visible_parts:
            return

        print(f"{message.role}:")
        print("\n".join(visible_parts))

    def _print_function_response(self, response: FunctionResponse) -> None:
        lines: list[str] = []
        for part in response.content:
            if isinstance(part, TextPart):
                lines.append(f"[text] {part.text}")
            elif isinstance(part, ImagePart):
                lines.append(f"[image] {part.display_name} ({part.mime_type})")

        if not lines:
            return

        print("[Tool Call]")
        print("\n".join(lines))
        print("[Tool Call]")

    def save_message(self, message: HistoryItem) -> None:
        self._history.append(message)

    def get_history(self) -> list[HistoryItem]:
        return self._history


class TextualMemoryMessageAdaptor(MessageAdaptor):
    """Pretty console output using Rich renderables (bundled with Textual)."""

    def __init__(self, console: Console | None = None):
        self._history: list[HistoryItem] = []
        self._console = console or Console()

    def print_new_message(self, message: HistoryItem) -> None:
        if isinstance(message, FunctionResponse):
            self._print_function_response(message)
            return

        body: list[object] = []
        for part in message.content:
            if isinstance(part, FunctionPart):
                args = json.dumps(part.args, indent=2, sort_keys=True)
                body.append(
                    Panel(
                        Group(
                            Text(f"tool {part.name}", style="bold #87d7ff"),
                            Syntax(
                                args,
                                "json",
                                line_numbers=False,
                                word_wrap=True,
                                theme="monokai",
                            ),
                        ),
                        title="[bold #87d7ff]CALL[/]",
                        border_style="#4a5568",
                        box=box.ROUNDED,
                    )
                )
            elif isinstance(part, ThinkingPart):
                body.append(
                    Panel(
                        Text(part.text, style="italic #6b7280"),
                        title="[bold #6b7280]THINKING[/]",
                        border_style="#374151",
                        box=box.ROUNDED,
                    )
                )
            elif isinstance(part, TextPart):
                body.append(part.text)

        if not body:
            return

        role = message.role.upper()
        role_style = "#34d399" if message.role in ("assistant", "model") else "#60a5fa"
        self._console.print(Rule(f"[bold {role_style}]{role}[/]", style="#1f2937"))
        self._console.print(Group(*body))

    def _print_function_response(self, response: FunctionResponse) -> None:
        table = Table(show_header=True, header_style="bold #7dd3fc", box=box.SIMPLE)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Content", overflow="fold")

        has_rows = False
        for part in response.content:
            if isinstance(part, TextPart):
                table.add_row("text", part.text)
                has_rows = True
            elif isinstance(part, ImagePart):
                table.add_row(
                    "image",
                    f"{part.display_name} ({part.mime_type})",
                )
                has_rows = True

        if not has_rows:
            return

        self._console.print(
            Panel(
                table,
                title=f"[bold #7dd3fc]RESULT[/] [#7dd3fc]{response.name}[/]",
                border_style="#4a5568",
                box=box.ROUNDED,
            )
        )

    def save_message(self, message: HistoryItem) -> None:
        self._history.append(message)

    def get_history(self) -> list[HistoryItem]:
        return self._history
