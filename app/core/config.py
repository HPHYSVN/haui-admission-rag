from __future__ import annotations

import os
from dataclasses import dataclass

from rag.config import DEFAULT_CHUNKS_DIR, DEFAULT_TOP_K, DEFAULT_VECTOR_INDEX_DIR

DEFAULT_MAX_REQUEST_BYTES = 64 * 1024
DEFAULT_RATE_LIMIT_PER_MINUTE = 0


def _csv(value: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    return entries or fallback


def _int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    allowed_origins: tuple[str, ...]
    chunks_dir: str
    index_dir: str
    max_question_length: int
    max_history_messages: int
    top_k: int
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    api_key: str | None = None


def get_settings() -> Settings:
    raw_top_k = _int("RAG_TOP_K", DEFAULT_TOP_K, minimum=1, maximum=10)
    return Settings(
        allowed_origins=_csv(
            os.getenv("APP_ALLOWED_ORIGINS", "http://localhost:5173"),
            ("http://localhost:5173",),
        ),
        chunks_dir=os.getenv("RAG_CHUNKS_DIR", str(DEFAULT_CHUNKS_DIR)),
        index_dir=os.getenv("RAG_INDEX_DIR", str(DEFAULT_VECTOR_INDEX_DIR)),
        max_question_length=_int("APP_MAX_QUESTION_LENGTH", 2000, minimum=1),
        max_history_messages=_int("APP_MAX_HISTORY_MESSAGES", 8, minimum=0),
        top_k=raw_top_k,
        max_request_bytes=_int(
            "APP_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES, minimum=1
        ),
        rate_limit_per_minute=_int("APP_RATE_LIMIT_PER_MINUTE", 0, minimum=0),
        api_key=(os.getenv("APP_API_KEY") or "").strip() or None,
    )
