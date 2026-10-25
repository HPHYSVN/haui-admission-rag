"""Corpus loading, retrieval orchestration, and the legacy generation shim.

Prompt wording lives in :mod:`rag.prompts`; HTTP generation lives in
:mod:`rag.llm`. This module keeps the public names the rest of the project
imported before the split, so ``create_messages`` and ``VLLMChatClient`` keep
working while the internals are provider-agnostic.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.bm25 import HybridRetriever
from rag.config import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_INDEX_DIR,
    reranker_candidate_count,
    reranker_enabled,
)
from rag.document_workflow import DEFAULT_STATE_FILE, load_states
from rag.eligibility import (
    DEFAULT_ELIGIBILITY_FILE,
    EligibilityError,
    apply_eligibility,
    has_current_evidence,
    load_eligibility_policy,
    resolve_query_scope,
)
from rag.llm import OpenAICompatibleProvider
from rag.llm.config import DEFAULT_MODEL as DEFAULT_VLLM_MODEL
from rag.prompts import PromptBuilder, build_prompt_builder
from rag.prompts.haui_admission import _chunk_context, create_messages
from rag.reranker import CrossEncoderReranker

ROOT_DIR = Path(__file__).resolve().parents[1]

#: Document metadata copied onto chunks when the chunk itself does not carry it.
DOCUMENT_METADATA_FIELDS = (
    "year",
    "category",
    "published_date",
    "effective_date",
    "expiry_date",
    "document_number",
    "document_type",
    "status",
)


def load_document_metadata(documents_dir: Path = DEFAULT_DOCUMENTS_DIR) -> dict[str, dict[str, Any]]:
    """Load source-document metadata keyed by id, without changing any file."""
    metadata: dict[str, dict[str, Any]] = {}
    if not documents_dir.exists():
        return metadata
    for path in sorted(documents_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                document_id = document.get("id")
                if document_id:
                    metadata[str(document_id)] = document
    return metadata


def _merge_document_metadata(
    chunk: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    document = documents.get(str(chunk.get("document_id") or ""))
    if not document:
        return
    for field in DOCUMENT_METADATA_FIELDS:
        if chunk.get(field) in (None, "") and document.get(field) not in (None, ""):
            chunk[field] = document[field]
    chunk.setdefault("document_status", document.get("status"))


def load_chunks(
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    *,
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
    eligibility_file: Path | None = None,
) -> list[dict[str, Any]]:
    chunk_files = sorted(chunks_dir.glob("*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No final chunk JSONL files found in {chunks_dir}. "
            "Approve a document, generate and review its chunk proposals, then finalize them "
            "with `python -m rag.cli document-review approve --document <id>`, "
            "`python -m rag.cli chunk --document <id>`, and `finalize-chunks`."
        )
    chunks: list[dict[str, Any]] = []
    workflow_states = load_states(DEFAULT_STATE_FILE)
    documents = load_document_metadata(documents_dir)
    for path in chunk_files:
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                for key in ("chunk_id", "document_id", "title", "text", "source_url"):
                    if not chunk.get(key):
                        raise ValueError(f"{path}:{line_no}: chunk has no {key!r}")
                if chunk.get("status") != "final":
                    continue
                if workflow_states.get(chunk["document_id"], {}).get("status") not in {"final", "indexed"}:
                    continue
                _merge_document_metadata(chunk, documents)
                chunks.append(chunk)
    if not chunks:
        raise ValueError(f"Chunk files in {chunks_dir} contain no records")
    try:
        policy = load_eligibility_policy(
            eligibility_file if eligibility_file is not None else DEFAULT_ELIGIBILITY_FILE
        )
    except EligibilityError:
        # A missing policy must not make the service silently fabricate current
        # evidence. Callers that need a corpus can validate coverage explicitly;
        # the API stays unavailable until the reviewed policy is present.
        raise
    return apply_eligibility(chunks, policy)


class VLLMChatClient:
    """Backward-compatible vLLM client.

    Kept with its original constructor and ``answer(question, results, history)``
    signature, but implemented as a thin shim over the provider adapter. New code
    should use :func:`rag.llm.build_provider` and :class:`RAGPipeline(provider=...)`.
    """

    name = "vllm"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120,
    ):
        load_dotenv(ROOT_DIR / ".env", override=False)
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL") or DEFAULT_VLLM_MODEL
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.timeout = timeout
        self._provider = OpenAICompatibleProvider(
            provider="vllm",
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            timeout=timeout,
        )

    def answer(
        self,
        question: str,
        results: list[tuple[float, dict[str, Any]]],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages = create_messages(question, results, history)
        return self._provider.complete(messages)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._provider.complete(messages)

    def probe(self, timeout: float = 2.0) -> bool:
        return self._provider.probe(timeout=timeout)


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    retrieved: list[tuple[float, dict[str, Any]]]


@dataclass(frozen=True)
class PreparedRAGResponse:
    """Retrieval result plus prompt, ready for one-shot or streamed generation.

    Streaming callers need sources and the prompt before the first token, so the
    retrieval/prompt phase is separated from generation. ``direct_answer`` is set
    for abstentions; no provider call is made in that case.
    """

    question: str
    results: list[tuple[float, dict[str, Any]]]
    sources: list[dict[str, Any]]
    messages: list[dict[str, str]] | None
    direct_answer: str | None = None
    target_year: int | None = None


class RAGPipeline:
    """Retrieval plus generation, shared by the API and the CLI.

    Two generation paths are accepted:
      * ``provider`` — a :class:`rag.llm.ChatProvider`; the pipeline builds messages
        with ``prompt_builder`` and calls ``provider.complete()``.
      * ``chat_client`` — the legacy object with ``answer(question, results,
        history)``. Used by existing callers and tests; the default is
        :class:`VLLMChatClient`, so ``RAGPipeline(chunks, client)`` is unchanged.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        chat_client: Any | None = None,
        *,
        provider: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
        alpha: float | None = None,
        score_threshold: float | None = None,
        index_dir: Path = DEFAULT_VECTOR_INDEX_DIR,
        embedder: Any | None = None,
        use_reranker: bool | None = None,
        reranker: CrossEncoderReranker | None = None,
        reranker_candidates: int | None = None,
    ):
        if chat_client is not None and provider is not None:
            raise ValueError("Pass either chat_client or provider to RAGPipeline, not both")
        enabled = reranker_enabled() if use_reranker is None else use_reranker
        reranker_instance = reranker if enabled else None
        if enabled and reranker_instance is None:
            reranker_instance = CrossEncoderReranker()
        self.retriever = HybridRetriever(
            chunks,
            alpha=alpha,
            score_threshold=score_threshold,
            index_dir=index_dir,
            embedder=embedder,
            reranker=reranker_instance,
            reranker_candidates=(
                reranker_candidate_count()
                if reranker_candidates is None
                else reranker_candidates
            ),
        )
        self.provider = provider
        self.chat_client = (
            chat_client
            if chat_client is not None or provider is not None
            else VLLMChatClient()
        )
        self.prompt_builder = prompt_builder or build_prompt_builder()
        self.chunk_count = len(chunks)
        self._corpus_years = {
            chunk.get("year")
            for chunk in chunks
            if isinstance(chunk.get("year"), int) and not isinstance(chunk.get("year"), bool)
        }

    @property
    def provider_name(self) -> str:
        target = self.provider if self.provider is not None else self.chat_client
        return str(getattr(target, "name", type(target).__name__))

    @property
    def model_name(self) -> str:
        target = self.provider if self.provider is not None else self.chat_client
        return str(getattr(target, "model", "") or "")

    @staticmethod
    def _sources(results: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {
                "source_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "url": chunk["source_url"],
                "year": chunk.get("year"),
                "category": chunk.get("category"),
                "score": score,
            }
            for score, chunk in results
        ]

    def _generate_prepared(
        self,
        prepared: PreparedRAGResponse,
        history: list[dict[str, str]] | None,
    ) -> str:
        if prepared.direct_answer is not None:
            return prepared.direct_answer
        if self.provider is not None:
            if prepared.messages is None:
                raise RuntimeError("Prepared prompt is missing")
            return self.provider.complete(prepared.messages)
        if history is None:
            return self.chat_client.answer(prepared.question, prepared.results)
        return self.chat_client.answer(prepared.question, prepared.results, history)

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[tuple[float, dict[str, Any]]]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        return self.retriever.search(question, top_k=top_k)

    def prepare(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict[str, str]] | None = None,
    ) -> PreparedRAGResponse:
        """Retrieve evidence, apply the year policy, and build the prompt."""
        results = self.retrieve(question, top_k=top_k)
        scope = resolve_query_scope(question)
        if not results:
            return PreparedRAGResponse(
                question=question,
                results=[],
                sources=[],
                messages=None,
                direct_answer=(
                    "Chưa tìm thấy tài liệu liên quan để trả lời câu hỏi này."
                ),
                target_year=scope.target_year,
            )

        target_year = scope.target_year
        if (
            target_year is not None
            and scope.year_sensitive
            and (scope.explicit_year is not None or target_year in self._corpus_years)
            and not has_current_evidence(results, target_year)
        ):
            # Retrieval found evergreen/out-of-year material, but no positive
            # evidence for the cycle the question is really about. Do not let a
            # 2025 cutoff or a draft answer as 2026 fact.
            return PreparedRAGResponse(
                question=question,
                results=[],
                sources=[],
                messages=None,
                direct_answer=(
                    f"Chưa tìm thấy tài liệu phù hợp cho năm {target_year} "
                    "để trả lời câu hỏi này."
                ),
                target_year=target_year,
            )

        messages = self.prompt_builder.build(question, results, history)
        return PreparedRAGResponse(
            question=question,
            results=results,
            sources=self._sources(results),
            messages=messages,
            target_year=target_year,
        )

    def stream_prepared(
        self,
        prepared: PreparedRAGResponse,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Yield answer fragments for a prepared request.

        The provider path never joins deltas. The legacy ``chat_client`` path
        cannot stream, so it yields its single completed answer as one fragment;
        new code should inject a provider with ``stream()``.
        """
        if prepared.direct_answer is not None:
            yield prepared.direct_answer
            return
        if self.provider is not None:
            stream = getattr(self.provider, "stream", None)
            if callable(stream) and prepared.messages is not None:
                yield from stream(prepared.messages)
                return
            if prepared.messages is None:
                raise RuntimeError("Prepared prompt is missing")
            yield self.provider.complete(prepared.messages)
            return
        yield self._generate_prepared(prepared, history)

    def stream(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        prepared = self.prepare(question, top_k=top_k, history=history)
        yield from self.stream_prepared(prepared, history)

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict[str, str]] | None = None,
    ) -> RAGResponse:
        prepared = self.prepare(question, top_k=top_k, history=history)
        answer = self._generate_prepared(prepared, history)
        return RAGResponse(
            answer=answer,
            sources=prepared.sources,
            retrieved=prepared.results,
        )
