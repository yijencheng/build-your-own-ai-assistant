import asyncio
import base64
from datetime import datetime
import importlib
import json
import time
from pathlib import Path
from typing import Annotated, Any, Awaitable, Literal, Protocol, TypeAlias, overload

import aiosqlite
from google.genai import Client, types
from pydantic import BaseModel, Field, TypeAdapter

import agent_tools
from agent_tools import AgentContext, AgentTool, ToolResult
from rich import print as rprint


class FCMetadata(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class FunctionCallData(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class FunctionResponseData(BaseModel):
    name: str
    response: dict[str, Any] = Field(default_factory=dict)


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    thought_signature_b64: str | None = None
    thought: bool | None = None


class FunctionCallPart(BaseModel):
    kind: Literal["function_call"] = "function_call"
    function_call: FunctionCallData
    thought_signature_b64: str | None = None


class FunctionResponsePart(BaseModel):
    kind: Literal["function_response"] = "function_response"
    function_response: FunctionResponseData
    thought_signature_b64: str | None = None
    fc_metadata: FCMetadata | None = None


SessionPart = Annotated[
    TextPart | FunctionCallPart | FunctionResponsePart,
    Field(discriminator="kind"),
]


class SessionEvent(BaseModel):
    event_type: Literal["session"] = "session"
    role: str
    parts: list[SessionPart] = Field(default_factory=list)

    @staticmethod
    def _encode_thought_signature(signature: bytes | None) -> str | None:
        if not signature:
            return None
        return base64.b64encode(signature).decode("ascii")

    @staticmethod
    def _decode_thought_signature(signature_b64: str | None) -> bytes | None:
        if not signature_b64:
            return None
        return base64.b64decode(signature_b64.encode("ascii"))

    @classmethod
    def from_content(
        cls,
        content: types.Content,
        part_metadata: list[FCMetadata | None] | None = None,
    ) -> "SessionEvent":
        metadata = part_metadata or [None] * len(content.parts)
        if len(metadata) != len(content.parts):
            raise ValueError("part_metadata length must match content.parts length")
        return cls(
            role=content.role or "user",
            parts=cls._parts_from_content(content.parts, metadata),
        )

    @classmethod
    def _parts_from_content(
        cls,
        parts: list[types.Part],
        metadata: list[FCMetadata | None],
    ) -> list[SessionPart]:
        out: list[SessionPart] = []
        for part, meta in zip(parts, metadata):
            thought_signature_b64 = cls._encode_thought_signature(
                getattr(part, "thought_signature", None)
            )
            if part.text is not None:
                out.append(
                    TextPart(
                        text=part.text,
                        thought_signature_b64=thought_signature_b64,
                        thought=part.thought,
                    )
                )
            elif part.function_call is not None:
                out.append(
                    FunctionCallPart(
                        function_call=FunctionCallData(
                            name=part.function_call.name,
                            args=dict(part.function_call.args or {}),
                        ),
                        thought_signature_b64=thought_signature_b64,
                    )
                )
            elif part.function_response is not None:
                out.append(
                    FunctionResponsePart(
                        function_response=FunctionResponseData(
                            name=part.function_response.name,
                            response=dict(part.function_response.response or {}),
                        ),
                        thought_signature_b64=thought_signature_b64,
                        fc_metadata=meta,
                    )
                )
            else:
                raise ValueError("Unsupported GenAI part")
        return out

    def to_content(self) -> types.Content:
        return types.Content(
            role=self.role,
            parts=[self._part_to_genai(part) for part in self.parts],
        )

    @classmethod
    def _part_to_genai(cls, part: SessionPart) -> types.Part:
        thought_signature = cls._decode_thought_signature(part.thought_signature_b64)
        if isinstance(part, TextPart):
            return types.Part(
                text=part.text,
                thought=part.thought,
                thought_signature=thought_signature,
            )
        if isinstance(part, FunctionCallPart):
            return types.Part(
                function_call=types.FunctionCall(
                    name=part.function_call.name,
                    args=part.function_call.args,
                ),
                thought_signature=thought_signature,
            )
        return types.Part(
            function_response=types.FunctionResponse(
                name=part.function_response.name,
                response=part.function_response.response,
            ),
            thought_signature=thought_signature,
        )


class CompactionEvent(BaseModel):
    event_type: Literal["compaction"] = "compaction"
    parts: list[SessionPart] = Field(default_factory=list)

    def to_content(self) -> types.Content:
        return types.UserContent(
            parts=[SessionEvent._part_to_genai(part) for part in self.parts]
        )


StoredEvent = Annotated[
    SessionEvent | CompactionEvent,
    Field(discriminator="event_type"),
]

StoredEventAdapter = TypeAdapter(StoredEvent)


INIT_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL DEFAULT 'session',
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class SessionManager:
    def __init__(self, db_path: str = "agent.db"):
        self.db_path = db_path
        self.client = Client()
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    def should_summarise(self, events: list[StoredEvent]) -> bool:
        max_events_before_summary = 50
        used_ratio = min(len(events) / max_events_before_summary, 1.0)
        remaining_pct = round((1.0 - used_ratio) * 100, 1)
        print(f"[Logging]: {remaining_pct}% context remaining")
        return len(events) > max_events_before_summary

    @staticmethod
    def _append_summary_to_memory(summary_text: str) -> None:
        now = datetime.now()
        memory_dir = Path("memory")
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_file = memory_dir / f"{now.strftime('%d-%m-%Y')}.md"
        line = f"[{now.strftime('%H:%M')}]: {summary_text}\n"
        with memory_file.open("a", encoding="utf-8") as f:
            f.write(line)

    @staticmethod
    def _map_events_for_compaction(events: list[StoredEvent]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, SessionEvent):
                continue

            for part in event.parts:
                if isinstance(part, TextPart):
                    if event.role == "user" and part.text.strip():
                        mapped.append(
                            {
                                "type": "user_text",
                                "text": part.text,
                            }
                        )
                    continue

                if isinstance(part, FunctionCallPart):
                    mapped.append(
                        {
                            "type": "tool_call",
                            "name": part.function_call.name,
                            "args": part.function_call.args,
                        }
                    )
                    continue

                if isinstance(part, FunctionResponsePart):
                    call_name = (
                        part.fc_metadata.name
                        if part.fc_metadata is not None
                        else part.function_response.name
                    )
                    call_args = (
                        part.fc_metadata.args if part.fc_metadata is not None else {}
                    )
                    mapped.append(
                        {
                            "type": "tool_response",
                            "name": call_name,
                            "args": call_args,
                            "response": part.function_response.response,
                        }
                    )

        return mapped

    @staticmethod
    def _mapped_events_to_conversation_text(
        mapped_events: list[dict[str, Any]],
    ) -> str:
        lines = ["<conversation>"]
        for entry in mapped_events:
            if entry["type"] == "user_text":
                lines.append(f"[user]: {entry['text']}")
                continue

            if entry["type"] == "tool_call":
                lines.append(
                    f"[model]: Tool call `{entry['name']}` with args {entry['args']}"
                )
                continue

            if entry["type"] == "tool_response":
                lines.append(
                    f"[model]: Tool response `{entry['name']}` with args {entry['args']} returned {entry['response']}"
                )

        lines.append("</conversation>")
        return "\n".join(lines)

    async def generate_summary(self, events: list[StoredEvent]) -> list[StoredEvent]:
        if not events:
            return events

        mapped_events = self._map_events_for_compaction(events)
        if not mapped_events:
            return events
        mapped_conversation = self._mapped_events_to_conversation_text(mapped_events)

        print("[Logging] Compaction Starting")
        response = await self.client.aio.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[
                types.UserContent(
                    parts=[
                        types.Part.from_text(
                            text=(
                                f"Summarize this conversation:\n{mapped_conversation}"
                            )
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                tools=[],
                system_instruction=(
                    """
You're about to be given mapped interactions from a conversation. Your job is to generate a summary of the conversation in at most 4 paragraphs.

Required structure:
1) Start with the key user objective first.
2) Then describe what was accomplished across the lifecycle based on the mapped user/tool interactions.
3) Then briefly cover blockers/loops and what was learned.
4) End with the current outcome and immediate next step.

Style constraints:
- Return natural language text only.
- Do not call tools or emit function calls.
- Do not return JSON, XML, code blocks, or markdown lists.
- Keep it brief, concrete, and factual.


"""
                ),
            ),
        )

        summary_parts = [
            part.text
            for part in response.candidates[0].content.parts
            if not part.thought and part.text
        ]
        summary = "\n".join(summary_parts).strip()
        print(summary)
        print("[Logging] Compaction Ended")

        self._append_summary_to_memory(summary)
        await self.add_message(CompactionEvent(parts=[TextPart(text=summary)]))
        return [CompactionEvent(parts=[TextPart(text=summary)])]

    async def initialize(self) -> None:
        if self._initialized and self._conn is not None:
            return

        async with self._init_lock:
            if self._initialized and self._conn is not None:
                return

            print(f"[SessionManager] Connecting to database at: {self.db_path}")
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.executescript(INIT_SQL)
            await self._conn.commit()
            self._initialized = True
            print(f"[SessionManager] Database initialized successfully: {self.db_path}")

    async def db(self) -> aiosqlite.Connection:
        if not self._initialized or self._conn is None:
            raise RuntimeError(
                "SessionManager is not initialized. Call initialize() first."
            )
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False

    async def delete(self) -> None:
        conn = await self.db()
        await conn.execute("DELETE FROM messages")
        await conn.commit()

    async def add_message(
        self, event: StoredEvent, created_at: float | None = None
    ) -> None:
        conn = await self.db()
        created_at = created_at or time.time()
        await conn.execute(
            """
            INSERT INTO messages (kind, event_json, created_at)
            VALUES (?, ?, ?)
            """,
            (event.event_type, event.model_dump_json(), created_at),
        )
        await conn.commit()

    async def _latest_compaction_id(self) -> int | None:
        conn = await self.db()
        cursor = await conn.execute(
            """
            SELECT id
            FROM messages
            WHERE kind = 'compaction'
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return int(row[0])

    async def load_messages(self) -> list[StoredEvent]:
        conn = await self.db()
        latest_compaction_id = await self._latest_compaction_id()
        if latest_compaction_id is None:
            cursor = await conn.execute(
                """
                SELECT kind, event_json
                FROM messages
                ORDER BY id ASC
                """
            )
        else:
            cursor = await conn.execute(
                """
                SELECT kind, event_json
                FROM messages
                WHERE id >= ?
                ORDER BY id ASC
                """,
                (latest_compaction_id,),
            )
        rows = await cursor.fetchall()
        await cursor.close()
        events: list[StoredEvent] = []
        for row in rows:
            kind, raw_event_json = row
            payload = json.loads(raw_event_json)
            if "event_type" not in payload:
                payload["event_type"] = kind or "session"
            events.append(StoredEventAdapter.validate_python(payload))

        if self.should_summarise(events):
            events = await self.generate_summary(events)

        return events


HookType: TypeAlias = Literal[
    "on_model_response",
    "on_tool_call",
    "on_tool_result",
]


class ModelResponseHook(Protocol):
    def __call__(
        self, *, message: types.Content, context: AgentContext
    ) -> Awaitable[None]: ...


class ToolCallHook(Protocol):
    def __call__(
        self, *, call: types.FunctionCall, context: AgentContext
    ) -> Awaitable[None]: ...


class ToolResultHook(Protocol):
    def __call__(
        self,
        *,
        result: ToolResult,
        call_name: str,
        call_args: dict[str, Any],
        context: AgentContext,
    ) -> Awaitable[None]: ...


class ReplayHook(Protocol):
    def __call__(self, *, event: StoredEvent) -> Awaitable[None]: ...


AnyHook: TypeAlias = ModelResponseHook | ToolCallHook | ToolResultHook

AGENT_SYSTEM_INSTRUCTION = """
You are Koroku, a helpful AI assistant.

Memories for each day will be stored in a memory/dd-mm-yyyy markdown file inside a /memory folder. When it doesn't exist, assume that there's no data on that specific day

Use this memory context when it is relevant to the current task.
"""

TOOL_CALL_GUARDRAIL_INSTRUCTION = (
    "You've reached the maximum number of tool calls allowed within a single turn, "
    "summarise what you've done so far in a single paragraph and ask the user for feedback."
)


class Agent:
    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        context: AgentContext | None = None,
        session_manager: SessionManager | None = None,
    ):
        self.model = model
        self.client = Client()
        self.session_manager = session_manager or SessionManager()
        self.context = context or AgentContext()

        self.tools_module = agent_tools
        self.tools_file = Path(self.tools_module.__file__).resolve()
        self.last_modified = self._mtime(self.tools_file)
        self.tools = self._load_tools(self.tools_module)
        self.max_tool_calls_per_turn = 10
        self._hooks: dict[HookType, list[AnyHook]] = {
            "on_model_response": [],
            "on_tool_call": [],
            "on_tool_result": [],
        }

    @staticmethod
    def _mtime(path: str | Path) -> float:
        return Path(path).stat().st_mtime

    @staticmethod
    def _load_tools(module: Any) -> dict[str, type[AgentTool]]:
        return {tool.__name__: tool for tool in module.TOOLS}

    @staticmethod
    def _is_real_user_turn(event: StoredEvent) -> bool:
        if not isinstance(event, SessionEvent) or event.role != "user":
            return False
        return not any(isinstance(part, FunctionResponsePart) for part in event.parts)

    async def initialize(
        self,
        *,
        replay_handler: ReplayHook | None = None,
    ) -> None:
        await self.session_manager.initialize()
        if replay_handler is None:
            return

        # for event in events:
        #     await replay_handler(event=event)

    def maybe_reload_runtime(self) -> bool:
        current = self._mtime(self.tools_file)
        if current == self.last_modified:
            return False

        try:
            self.tools_module = importlib.reload(self.tools_module)
            self.tools_file = Path(self.tools_module.__file__).resolve()
            self.tools = self._load_tools(self.tools_module)
            self.last_modified = self._mtime(self.tools_file)
            loaded_tools = ", ".join(sorted(self.tools.keys())) or "(none)"
            print(
                f"[Tool Reload] Loaded tools from {self.tools_file.name}: {loaded_tools}"
            )
            return True
        except Exception as exc:
            print(f"[Tool Reload] Failed, keeping previous runtime: {exc}")
            return False

    def get_tools(self) -> list[types.Tool]:
        self.maybe_reload_runtime()
        return [tool_cls.to_genai_schema() for tool_cls in self.tools.values()]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool_cls = self.tools.get(tool_name)
        if tool_cls is None:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": f"Unknown tool: {tool_name}"},
            )

        try:
            tool_input = tool_cls.model_validate(args or {})
            return await tool_input.execute(self.context)
        except Exception as exc:
            return ToolResult(
                error=True,
                name=tool_name,
                response={"error": str(exc)},
            )

    @overload
    def on(
        self, event: Literal["on_model_response"], handler: ModelResponseHook
    ) -> "Agent": ...

    @overload
    def on(self, event: Literal["on_tool_call"], handler: ToolCallHook) -> "Agent": ...

    @overload
    def on(
        self, event: Literal["on_tool_result"], handler: ToolResultHook
    ) -> "Agent": ...

    def on(self, event: HookType, handler: AnyHook) -> "Agent":
        self._hooks[event].append(handler)
        return self

    async def emit(self, event: HookType, **kwargs: Any) -> None:
        for handler in self._hooks[event]:
            try:
                await handler(**kwargs)
            except Exception as exc:
                print(f"[Hook Error] Event '{event}' failed: {exc}")

    async def run(self, message: types.Content) -> types.Content | None:
        """Run one assistant step. Returns tool-response message, or None when done."""
        has_function_response_input = any(
            part.function_response is not None for part in message.parts
        )
        conversation = await self.session_manager.load_messages()

        if not has_function_response_input:
            session_message = SessionEvent.from_content(message)
            await self.session_manager.add_message(session_message)
            conversation = await self.session_manager.load_messages()

        last_user_idx = -1
        for idx in range(len(conversation) - 1, -1, -1):
            event = conversation[idx]
            if self._is_real_user_turn(event):
                last_user_idx = idx
                break

        session_events_since_last_user = (
            len(conversation)
            if last_user_idx < 0
            else len(conversation) - last_user_idx - 1
        )
        has_tool_budget = session_events_since_last_user < self.max_tool_calls_per_turn

        tools = self.get_tools() if has_tool_budget else []
        events_for_model = list(conversation)
        if not has_tool_budget:
            events_for_model.append(
                SessionEvent(
                    role="user",
                    parts=[TextPart(text=TOOL_CALL_GUARDRAIL_INSTRUCTION)],
                )
            )

        completion = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[event.to_content() for event in events_for_model],
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=AGENT_SYSTEM_INSTRUCTION,
            ),
        )

        message = completion.candidates[0].content
        await self.session_manager.add_message(SessionEvent.from_content(message))
        await self.emit("on_model_response", message=message, context=self.context)

        function_calls = [
            part.function_call for part in message.parts if part.function_call
        ]

        if not function_calls:
            return None

        tool_responses: list[types.Part] = []
        for call in function_calls:
            call_args = call.args or {}
            await self.emit("on_tool_call", call=call, context=self.context)
            result = await self.execute_tool(call.name, call_args)
            await self.emit(
                "on_tool_result",
                result=result,
                call_name=call.name,
                call_args=call_args,
                context=self.context,
            )
            tool_responses.append(result.to_genai_message())

        final_response = types.UserContent(parts=tool_responses)
        response_metadata = [
            FCMetadata(name=call.name, args=(call.args or {}))
            for call in function_calls
        ]
        await self.session_manager.add_message(
            SessionEvent.from_content(final_response, part_metadata=response_metadata)
        )
        return final_response


async def print_llm_response(message: types.Content, context: AgentContext):
    for part in message.parts:
        if part.text:
            print(f"* {part.text}")


async def print_tool_result(
    *,
    result: ToolResult,
    call_name: str,
    call_args: dict[str, Any],
    context: AgentContext,
) -> None:
    status = "[green]✓[/green]" if not result.error else "[red]✗[/red]"
    rprint(f"{status} [bold]{call_name}[/bold] {call_args}")


async def render_history_event(*, event: StoredEvent) -> None:
    if isinstance(event, CompactionEvent):
        for part in event.parts:
            if isinstance(part, TextPart):
                print(f"[Compaction] {part.text}")
        return

    for part in event.parts:
        if isinstance(part, TextPart):
            label = (
                "You"
                if event.role == "user"
                else "Assistant"
                if event.role == "model"
                else event.role
            )
            print(f"{label}: {part.text}")
            continue

        if isinstance(part, FunctionCallPart):
            continue

        call_name = (
            part.fc_metadata.name
            if part.fc_metadata is not None
            else part.function_response.name
        )
        call_args = part.fc_metadata.args if part.fc_metadata is not None else {}
        error = "error" in part.function_response.response
        status = "[green]✓[/green]" if not error else "[red]✗[/red]"
        rprint(f"{status} [bold]{call_name}[/bold] {call_args}")


async def main() -> None:
    print("Type 'exit' or 'quit' to stop. Type 'clear' to clear session history.")
    agent = Agent()
    agent.on("on_tool_result", print_tool_result)
    agent.on("on_model_response", print_llm_response)

    print("[History]")
    await agent.initialize(replay_handler=render_history_event)
    print("[/History]")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "clear":
            await agent.session_manager.delete()
            print("[History cleared]")
            continue
        if not user_input:
            continue

        next_message = types.UserContent(parts=[types.Part.from_text(text=user_input)])

        while True:
            next_message = await agent.run(next_message)
            if next_message is None:
                break


if __name__ == "__main__":
    asyncio.run(main())
