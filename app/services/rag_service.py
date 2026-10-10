from __future__ import annotations

from pathlib import Path

from rag.pipeline import RAGPipeline, RAGResponse, VLLMChatClient, load_chunks


class RagService:
    def __init__(self, chunks_dir: str, index_dir: str) -> None:
        self._pipeline = RAGPipeline(
            load_chunks(Path(chunks_dir)),
            VLLMChatClient(),
            index_dir=Path(index_dir),
        )

    def ask(self, question: str, history: list[dict[str, str]]) -> RAGResponse:
        return self._pipeline.ask(question, history=history)
