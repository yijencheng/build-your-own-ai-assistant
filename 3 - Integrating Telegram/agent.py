import importlib
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias, TypedDict, overload

import agent_tools
from google.genai import Client, types

HookType: TypeAlias = Literal[
    "user_message_received",
    "model_response",
    "tool_executed",
]
UserMessageHook: TypeAlias = Callable[[list[types.Content]], None]
ModelResponseHook: TypeAlias = Callable[[types.Content], None]


class ToolExecution(TypedDict, total=False):
    success: bool
    result: Any
    error: str


class ToolExecutionPayload(TypedDict):
    name: str
    args: dict[str, Any]
    execution: ToolExecution


ToolExecutedHook: TypeAlias = Callable[[ToolExecutionPayload], None]


class Agent:
    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.model = model
        self.client = Client()

        self.tools_module = agent_tools
        self.tools_file = Path(self.tools_module.__file__).resolve()
        self.runtime = self.tools_module.get_default_runtime()
        self.last_modified = self._mtime(self.tools_file)
        self._hooks: dict[HookType, list[Callable[..., None]]] = {
            "user_message_received": [],
            "model_response": [],
            "tool_executed": [],
        }

    @staticmethod
    def _mtime(path: str | Path) -> float:
        return Path(path).stat().st_mtime

    def maybe_reload_runtime(self) -> bool:
        current = self._mtime(self.tools_file)
        if current == self.last_modified:
            return False

        try:
            self.tools_module = importlib.reload(self.tools_module)
            self.tools_file = Path(self.tools_module.__file__).resolve()
            self.runtime = self.tools_module.get_default_runtime()
            self.last_modified = self._mtime(self.tools_file)
            print(f"[Tool Reload] Loaded tools from {self.tools_file.name}")
            return True
        except Exception as exc:
            print(f"[Tool Reload] Failed, keeping previous runtime: {exc}")
            return False

    @overload
    def on(
        self, event: Literal["user_message_received"], handler: UserMessageHook
    ) -> "Agent":
        pass

    @overload
    def on(
        self, event: Literal["model_response"], handler: ModelResponseHook
    ) -> "Agent":
        pass

    @overload
    def on(self, event: Literal["tool_executed"], handler: ToolExecutedHook) -> "Agent":
        pass

    def on(self, event: HookType, handler: Callable[..., None]) -> "Agent":
        self._hooks[event].append(handler)
        return self

    def emit(self, event: HookType, *args: Any) -> None:
        for handler in self._hooks[event]:
            (handler)(*args)

    def run(self, conversation: list[types.Content]) -> types.Content | None:
        """Run one assistant step. Returns tool-response message, or None when done."""
        self.maybe_reload_runtime()

        completion = self.client.models.generate_content(
            model=self.model,
            contents=conversation,
            config=types.GenerateContentConfig(tools=self.runtime.get_tools()),
        )

        message = completion.candidates[0].content
        conversation.append(message)
        self.emit("model_response", message)

        function_calls = [
            part.function_call for part in message.parts if part.function_call
        ]

        if not function_calls:
            return None

        tool_responses: list[types.Part] = []
        for call in function_calls:
            call_args = call.args or {}
            try:
                tool_result = self.runtime.execute_tool(call.name, call_args)
                execution = {"success": True, "result": tool_result}
                error: Exception | None = None
            except Exception as exc:
                execution = {"success": False, "error": str(exc)}
                error = exc
            self.emit(
                "tool_executed",
                {"name": call.name, "args": call_args, "execution": execution},
            )
            tool_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"execution": execution},
                )
            )

        return types.Content(role="user", parts=tool_responses)


def print_llm_response(response: types.Content) -> None:
    for part in response.parts:
        if part.text:
            print(f"* {part.text}")


def print_llm_tool(response: ToolExecutionPayload) -> None:
    execution = response["execution"]
    if not execution.get("success"):
        print("X [Error Encountered]")
        print(response)
        return

    print(f"✓ {response['name']} : {response['args']}")
    result = execution.get("result")
    if result is not None:
        print(result)


def main() -> None:
    print("Welcome to Amie! (Type 'exit' to quit)")
    conversation: list[types.Content] = []
    agent = Agent()
    agent.on("model_response", print_llm_response)
    agent.on("tool_executed", print_llm_tool)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        conversation.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        while True:
            next_message = agent.run(conversation)
            if next_message is None:
                break
            conversation.append(next_message)


if __name__ == "__main__":
    main()
