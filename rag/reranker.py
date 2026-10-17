"""Optional cross-encoder reranking for the hybrid retrieval candidates."""

from __future__ import annotations

import math
from typing import Any

from rag.config import reranker_model_name
from rag.embeddings import chunk_embedding_text


class CrossEncoderReranker:
    """Lazy-load a fastembed cross-encoder and score query/chunk pairs."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or reranker_model_name()
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Reranking requires fastembed with TextCrossEncoder support. Run `uv sync`."
                ) from exc
            try:
                self._model = TextCrossEncoder(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load reranker model {self.model_name!r}. "
                    "Check network access for the first download or set RAG_RERANKER_MODEL."
                ) from exc
        return self._model

    def score(self, query: str, chunks: list[dict[str, Any]]) -> list[float]:
        if not chunks:
            return []
        try:
            raw_scores = list(
                self._get_model().rerank(
                    query,
                    [chunk_embedding_text(chunk) for chunk in chunks],
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Reranker model {self.model_name!r} failed: {exc}"
            ) from exc
        if len(raw_scores) != len(chunks):
            raise RuntimeError(
                f"Reranker model {self.model_name!r} returned {len(raw_scores)} scores "
                f"for {len(chunks)} chunks"
            )
        scores = [float(score) for score in raw_scores]
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError(
                f"Reranker model {self.model_name!r} returned a non-finite score"
            )
        # Cross-encoder logits are only used for ordering. Applying sigmoid here
        # saturates common positive scores and destroys useful rank separation.
        return scores
