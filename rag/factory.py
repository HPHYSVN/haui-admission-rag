"""The single construction path shared by the API and the CLI.

Before this module existed each entry point built its own pipeline, which is how
the app could drift from the CLI. ``build_pipeline`` is now the only place that
joins corpus loading, retrieval configuration, the prompt profile, and the LLM
provider. Both modes call it, so a change applies to both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import DEFAULT_CHUNKS_DIR, DEFAULT_VECTOR_INDEX_DIR
from rag.embeddings import FAISS_INDEX_FILE, INDEX_FILE
from rag.llm import build_provider
from rag.pipeline import RAGPipeline, load_chunks
from rag.prompts import PromptBuilder, build_prompt_builder
from rag.reranker import CrossEncoderReranker


def index_ready(index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR) -> bool:
    """Cheap on-disk check: does a usable vector index exist?

    No model load and no embedding, so this is safe to call from ``/health``.
    """
    directory = Path(index_dir)
    return (directory / FAISS_INDEX_FILE).is_file() or (directory / INDEX_FILE).is_file()


def build_pipeline(
    chunks_dir: str | Path = DEFAULT_CHUNKS_DIR,
    index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    *,
    provider: Any | None = None,
    prompt_builder: PromptBuilder | None = None,
    alpha: float | None = None,
    score_threshold: float | None = None,
    embedder: Any | None = None,
    use_reranker: bool | None = None,
    reranker: CrossEncoderReranker | None = None,
    reranker_candidates: int | None = None,
) -> RAGPipeline:
    """Load final chunks and build a pipeline with the configured profile/provider.

    ``provider`` and ``prompt_builder`` default to the environment-selected
    implementations, so app and CLI get identical behaviour unless a caller
    injects a test double.
    """
    chunks = load_chunks(Path(chunks_dir))
    return RAGPipeline(
        chunks,
        provider=provider or build_provider(),
        prompt_builder=prompt_builder or build_prompt_builder(),
        alpha=alpha,
        score_threshold=score_threshold,
        index_dir=Path(index_dir),
        embedder=embedder,
        use_reranker=use_reranker,
        reranker=reranker,
        reranker_candidates=reranker_candidates,
    )


__all__ = ["build_pipeline", "index_ready"]
