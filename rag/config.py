"""Shared defaults for chunk retrieval and local indexing."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_DIR = ROOT_DIR / "data" / "chunks" / "final"
DEFAULT_PROPOSALS_DIR = ROOT_DIR / "data" / "chunks" / "proposals"
DEFAULT_VECTOR_INDEX_DIR = ROOT_DIR / "data" / "indexes" / "vector"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.5
DEFAULT_SCORE_THRESHOLD = 0.08
DEFAULT_MIN_TERM_COVERAGE = 0.3

load_dotenv(ROOT_DIR / ".env", override=False)


def embedding_model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def retrieval_alpha() -> float:
    return _read_float("RAG_ALPHA", DEFAULT_ALPHA, minimum=0.0, maximum=1.0)


def retrieval_score_threshold() -> float:
    return _read_float(
        "RAG_SCORE_THRESHOLD",
        DEFAULT_SCORE_THRESHOLD,
        minimum=0.0,
        maximum=1.0,
    )


def retrieval_min_term_coverage() -> float:
    return _read_float("RAG_MIN_TERM_COVERAGE", DEFAULT_MIN_TERM_COVERAGE, minimum=0.0, maximum=1.0)


def _read_float(name: str, default: float, *, minimum: float, maximum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        allowed = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be {allowed}, got {value}")
    return value
