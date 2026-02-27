import base64
from typing import Annotated, Any, Awaitable, Literal, Protocol
from pydantic import BaseModel, Field
from google.genai import types

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


class ReplayHook(Protocol):
    def __call__(self, *, event: SessionEvent) -> Awaitable[None]: ...


INIT_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class SessionManager:
    def __init__(self, db_path: str = "agent.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

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
        self, event: SessionEvent, created_at: float | None = None
    ) -> None:
        conn = await self.db()
        created_at = created_at or time.time()
        await conn.execute(
            "INSERT INTO messages (event_json, created_at) VALUES (?, ?)",
            (event.model_dump_json(), created_at),
        )
        await conn.commit()

    async def load_messages(self) -> list[SessionEvent]:
        conn = await self.db()
        cursor = await conn.execute("SELECT event_json FROM messages ORDER BY id ASC")
        rows = await cursor.fetchall()
        await cursor.close()
        return [SessionEvent.model_validate_json(row[0]) for row in rows]
