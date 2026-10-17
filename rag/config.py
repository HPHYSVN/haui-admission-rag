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

DEFAULT_TOP_K = 4
DEFAULT_ALPHA = 0.5
DEFAULT_SCORE_THRESHOLD = 0.08
DEFAULT_MIN_TERM_COVERAGE = 0.3
DEFAULT_CONTEXT_TOKENS = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 800
DEFAULT_CONTEXT_CHAR_BUDGET = 15000
DEFAULT_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
DEFAULT_RERANKER_CANDIDATES = 12
DEFAULT_RERANKER_RETRIEVAL_WEIGHT = 4.0

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


def reranker_enabled() -> bool:
    raw = os.getenv("RAG_RERANKER_ENABLED", "true").strip().casefold()
    if raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("RAG_RERANKER_ENABLED must be true or false")
    return raw in {"true", "1", "yes"}


def reranker_model_name() -> str:
    return os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)


def reranker_candidate_count() -> int:
    raw = os.getenv("RAG_RERANKER_CANDIDATES", str(DEFAULT_RERANKER_CANDIDATES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"RAG_RERANKER_CANDIDATES must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"RAG_RERANKER_CANDIDATES must be at least 1, got {value}")
    return value


def reranker_retrieval_weight() -> float:
    return _read_float(
        "RAG_RERANKER_RETRIEVAL_WEIGHT",
        DEFAULT_RERANKER_RETRIEVAL_WEIGHT,
        minimum=0.0,
    )


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
