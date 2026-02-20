from typing import Any, Protocol


class MessageAdaptor(Protocol):
    def print_new_message(self, message: Any) -> None: ...

    def save_message(self, message: Any) -> None: ...

    def get_history(self) -> list[Any]: ...


class ConsoleMemoryMessageAdaptor:
    def __init__(self):
        self._history: list[Any] = []

    def print_new_message(self, message: Any) -> None:
        if not hasattr(message, "role"):
            self._print_function_response(message)
            return

        visible_parts: list[str] = []
        for part in message.content:
            if hasattr(part, "args") and hasattr(part, "name"):
                visible_parts.append(f"[tool_call] {part.name}")
            elif (
                hasattr(part, "text")
                and not hasattr(part, "args")
                and (getattr(part, "thought", None) or getattr(part, "thought_signature", None))
            ):
                visible_parts.append(f"[Thinking] {part.text}")
            elif hasattr(part, "text"):
                visible_parts.append(part.text)

        if not visible_parts:
            return

        print(f"{message.role}:")
        print("\n".join(visible_parts))

    def _print_function_response(self, response: Any) -> None:
        lines: list[str] = []
        for part in response.content:
            if hasattr(part, "display_name") and hasattr(part, "mime_type"):
                lines.append(f"[image] {part.display_name} ({part.mime_type})")
            elif hasattr(part, "text"):
                lines.append(f"[text] {part.text}")

        if not lines:
            return

        print("[Tool Call]")
        print("\n".join(lines))
        print("[Tool Call]")

    def save_message(self, message: Any) -> None:
        self._history.append(message)

    def get_history(self) -> list[Any]:
        return self._history
