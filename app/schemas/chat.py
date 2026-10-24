from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    # The configurable limits in app/core/config.py are authoritative; keeping a
    # second hard cap here would silently override a larger APP_MAX_* setting.
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Câu hỏi không được để trống.")
        return value


class Source(BaseModel):
    source_id: str
    chunk_id: str
    title: str
    url: AnyHttpUrl
    year: int | None = None
    category: str | None = None
    score: float = Field(ge=0)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    request_id: str


class HealthResponse(BaseModel):
    """Cheap liveness payload; safe for the existing frontend.

    The three original fields keep their meaning. Additions are diagnostics that
    contain no path, URL, credential, or exception text.
    """

    status: Literal["ready", "degraded"]
    rag_loaded: bool
    model_configured: bool
    index_ready: bool = False
    chunk_count: int = 0
    provider: str = ""
    model: str = ""
    app_version: str = ""


class ReadinessChecks(BaseModel):
    chunks_loaded: bool
    index_ready: bool
    provider_reachable: bool


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: ReadinessChecks
    chunk_count: int = 0
    provider: str = ""
    model: str = ""
    app_version: str = ""
    request_id: str = ""
