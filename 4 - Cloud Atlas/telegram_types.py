from pydantic import BaseModel


class TelegramChat(BaseModel):
    id: int


class TelegramPhotoSize(BaseModel):
    file_id: str
    file_unique_id: str
    width: int
    height: int
    file_size: int | None = None


class TelegramDocument(BaseModel):
    file_id: str
    file_unique_id: str
    file_name: str | None = None
    mime_type: str | None = None
    file_size: int | None = None


class TelegramAudio(BaseModel):
    file_id: str
    file_unique_id: str
    duration: int
    file_name: str | None = None
    mime_type: str | None = None
    file_size: int | None = None


class TelegramMessage(BaseModel):
    chat: TelegramChat
    text: str = ""
    caption: str | None = None
    photo: list[TelegramPhotoSize] | None = None
    document: TelegramDocument | None = None
    audio: TelegramAudio | None = None
    voice: TelegramAudio | None = None


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage
