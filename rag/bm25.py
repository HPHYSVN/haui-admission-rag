"""BM25 and multilingual semantic retrieval primitives for HaUI RAG."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag.config import (
    DEFAULT_VECTOR_INDEX_DIR,
    retrieval_alpha,
    retrieval_min_term_coverage,
    retrieval_score_threshold,
)
from rag.embeddings import LocalEmbedder, VectorIndex
from rag.filters import matching_documents, query_filters, query_term_coverage

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.casefold())
    bigrams = [f"__bi__{left}_{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


def normalize_document_text(document: dict[str, Any]) -> str:
    title = document.get("title", "")
    section_path = " ".join(document.get("section_path", []) or [])
    text = document.get("text", "")
    return " ".join(part for part in (title, section_path, text) if part)


class BM25Retriever:
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 needs at least one document chunk")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [
            Counter(tokenize(normalize_document_text(doc)))
            for doc in documents
        ]
        self.lengths = [sum(counts.values()) for counts in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        total = len(documents)
        self.inverse_document_frequency = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict[str, Any]]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_factor = 1 - self.b + self.b * self.lengths[index] / self.average_length
            for term in query_terms:
                count = frequencies.get(term, 0)
                if count:
                    numerator = count * (self.k1 + 1)
                    denominator = count + self.k1 * length_factor
                    score += self.inverse_document_frequency.get(term, 0.0) * numerator / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[tuple[float, dict[str, Any]]] = []
        seen_chunks: set[str] = set()
        for score, index in scored:
            chunk_id = self.documents[index].get("chunk_id", str(index))
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            results.append((score, self.documents[index]))
            if len(results) == top_k:
                break
        return results


class SemanticRetriever:
    """Multilingual embedding search backed by an incremental local JSONL index."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        index_dir: Path = DEFAULT_VECTOR_INDEX_DIR,
        embedder: LocalEmbedder | None = None,
    ):
        if not documents:
            raise ValueError("Semantic retriever needs at least one document chunk")
        self.documents = documents
        self.index = VectorIndex(index_dir=index_dir, embedder=embedder)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict[str, Any]]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip():
            return []
        return self.index.search(query, self.documents, top_k)


class HybridRetriever:
    """Combines BM25 lexical matching with the local semantic vector search."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        alpha: float | None = None,
        score_threshold: float | None = None,
        index_dir: Path = DEFAULT_VECTOR_INDEX_DIR,
        embedder: LocalEmbedder | None = None,
    ):
        if not documents:
            raise ValueError("Hybrid retriever needs at least one document chunk")
        alpha = retrieval_alpha() if alpha is None else alpha
        score_threshold = (
            retrieval_score_threshold() if score_threshold is None else score_threshold
        )
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        self.documents = documents
        self.alpha = alpha
        self.score_threshold = score_threshold
        self.min_term_coverage = retrieval_min_term_coverage()
        self.index_dir = index_dir
        self.bm25 = BM25Retriever(documents)
        self.semantic = SemanticRetriever(documents, index_dir, embedder)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict[str, Any]]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = query.strip()
        if not query:
            return []
        candidates = matching_documents(self.documents, query_filters(query))
        if not candidates:
            return []
        self.semantic.index.ensure_current(self.documents)
        candidate_ids = {chunk.get("chunk_id") for chunk in candidates}
        lexical = [
            (score, chunk)
            for score, chunk in self.bm25.search(query, top_k=len(self.documents))
            if chunk.get("chunk_id") in candidate_ids
        ]
        lexical_by_id = {chunk.get("chunk_id"): score for score, chunk in lexical}
        max_lexical = max(lexical_by_id.values(), default=0.0)
        semantic = self.semantic.index.search(
            query,
            candidates,
            top_k=len(candidates),
        )
        semantic_by_id = {chunk.get("chunk_id"): score for score, chunk in semantic}
        fused: list[tuple[float, dict[str, Any]]] = []
        for chunk in candidates:
            chunk_id = chunk.get("chunk_id")
            semantic_score = max(0.0, semantic_by_id.get(chunk_id, 0.0))
            bm25_score = lexical_by_id.get(chunk_id, 0.0)
            normalized_bm25 = bm25_score / max_lexical if max_lexical else 0.0
            score = self.alpha * semantic_score + (1 - self.alpha) * normalized_bm25
            coverage = query_term_coverage(query, chunk)
            if score >= self.score_threshold and (
                coverage >= self.min_term_coverage or semantic_score >= 0.78
            ):
                fused.append((score, chunk))
        fused.sort(key=lambda item: (-item[0], item[1].get("chunk_id", "")))
        return fused[:top_k]
