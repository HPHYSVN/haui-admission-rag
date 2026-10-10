from __future__ import annotations

import os
from dataclasses import dataclass

from rag.config import DEFAULT_CHUNKS_DIR, DEFAULT_TOP_K, DEFAULT_VECTOR_INDEX_DIR


def _csv(value: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    return entries or fallback


@dataclass(frozen=True)
class Settings:
    allowed_origins: tuple[str, ...]
    chunks_dir: str
    index_dir: str
    max_question_length: int
    max_history_messages: int
    top_k: int


def get_settings() -> Settings:
    raw_top_k = int(os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K)))
    if not 1 <= raw_top_k <= 10:
        raise ValueError("RAG_TOP_K must be between 1 and 10")
    return Settings(
        allowed_origins=_csv(
            os.getenv("APP_ALLOWED_ORIGINS", "http://localhost:5173"),
            ("http://localhost:5173",),
        ),
        chunks_dir=os.getenv("RAG_CHUNKS_DIR", str(DEFAULT_CHUNKS_DIR)),
        index_dir=os.getenv("RAG_INDEX_DIR", str(DEFAULT_VECTOR_INDEX_DIR)),
        max_question_length=int(os.getenv("APP_MAX_QUESTION_LENGTH", "2000")),
        max_history_messages=int(os.getenv("APP_MAX_HISTORY_MESSAGES", "8")),
        top_k=raw_top_k,
    )
