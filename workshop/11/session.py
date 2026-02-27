import base64
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Awaitable, Literal, Protocol
from pydantic import BaseModel, Field, TypeAdapter
from google.genai import types, Client
import json
import asyncio
import time
import aiosqlite


class FunctionCallData(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class FunctionResponseData(BaseModel):
    name: str
    response: dict[str, Any] = Field(default_factory=dict)


class FCMetadata(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


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


class ReplayHook(Protocol):
    def __call__(self, *, event: SessionEvent) -> Awaitable[None]: ...


INIT_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL DEFAULT 'session',
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class SessionManager:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = (
            Path(base_dir).resolve()
            if base_dir is not None
            else Path(__file__).resolve().parent
        )
        self.db_path = self.base_dir / "agent.db"
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self.client = Client()

    async def initialize(self) -> None:
        if self._initialized and self._conn is not None:
            return

        async with self._init_lock:
            if self._initialized and self._conn is not None:
                return

            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.executescript(INIT_SQL)
            await self._conn.commit()
            self._initialized = True

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

    def should_summarise(self, events: list[StoredEvent]) -> bool:
        max_events_before_summary = 40
        used_ratio = min(len(events) / max_events_before_summary, 1.0)
        remaining_pct = round((1.0 - used_ratio) * 100, 1)
        print(f"[Logging]: {remaining_pct}% context remaining")
        return len(events) > max_events_before_summary

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

    async def load_messages(self) -> list[StoredEvent]:
        conn = await self.db()

        # Get most recent events
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

        # Now we need to determine if we need to compact it
        if self.should_summarise(events):
            events = await self.generate_summary(events)

        return events

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

    def _append_summary_to_memory(self, summary_text: str) -> None:
        now = datetime.now()
        file_name = Path("memory") / f"{now.strftime('%d-%m-%Y')}.md"
        memory_file = self.base_dir / file_name
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        line = f"[{now.strftime('%H:%M')}]: {summary_text}\n"
        with memory_file.open("a", encoding="utf-8") as f:
            f.write(line)

    async def generate_summary(self, events: list[StoredEvent]) -> list[StoredEvent]:
        if not events:
            return events

        mapped_events = self._map_events_for_compaction(events)
        if not mapped_events:
            return events
        mapped_conversation = self._mapped_events_to_conversation_text(mapped_events)

        print("[Logging] Compaction Starting")
        response = await self.client.aio.models.generate_content(
            model="gemini-3-flash-preview",
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

Only summarise what has happened in the conversatio. snippet that you've been given.

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
