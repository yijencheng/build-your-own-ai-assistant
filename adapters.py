from agent_types import (
    FunctionPart,
    FunctionResponse,
    HistoryItem,
    ImagePart,
    MessageAdaptor,
    TextPart,
    ThinkingPart,
)


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
