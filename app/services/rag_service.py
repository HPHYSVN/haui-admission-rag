from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import DEFAULT_TOP_K
from rag.factory import build_pipeline
from rag.pipeline import PreparedRAGResponse, RAGPipeline, RAGResponse
from rag.prompts import PromptBuilder


class RagService:
    """API-facing wrapper around one shared RAG pipeline."""

    def __init__(
        self,
        chunks_dir: str,
        index_dir: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        provider: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.top_k = top_k
        self._pipeline: RAGPipeline = build_pipeline(
            Path(chunks_dir),
            Path(index_dir),
            provider=provider,
            prompt_builder=prompt_builder,
        )

    def ask(self, question: str, history: list[dict[str, str]]) -> RAGResponse:
        return self._pipeline.ask(question, top_k=self.top_k, history=history)

    def prepare(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> PreparedRAGResponse:
        return self._pipeline.prepare(question, top_k=self.top_k, history=history)

    def stream_prepared(self, prepared: PreparedRAGResponse):
        return self._pipeline.stream_prepared(prepared)

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    @property
    def chunk_count(self) -> int:
        return self._pipeline.chunk_count

    @property
    def provider_name(self) -> str:
        return self._pipeline.provider_name

    @property
    def model_name(self) -> str:
        return self._pipeline.model_name

    def probe(self, timeout: float = 2.0) -> bool:
        """Short provider reachability probe used by ``/api/v1/ready``."""
        probe = getattr(self._pipeline.provider, "probe", None)
        if probe is None:
            return False
        try:
            return bool(probe(timeout=timeout))
        except Exception:
            return False
