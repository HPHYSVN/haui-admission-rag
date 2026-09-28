"""Local multilingual text embeddings and a persistent, document-updatable index."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from rag.config import DEFAULT_VECTOR_INDEX_DIR, embedding_model_name

INDEX_FILE = "vectors.jsonl"
METADATA_FILE = "metadata.json"


class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or embedding_model_name()
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval requires the project dependencies. Run `uv sync` "
                "to install fastembed."
            ) from exc
        try:
            self._model = TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model {self.model_name!r}. "
                "Check network access for the first download or set RAG_EMBEDDING_MODEL."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = [vector.tolist() for vector in self._model.embed(texts)]
        except Exception as exc:
            raise RuntimeError(f"Embedding model {self.model_name!r} failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding model {self.model_name!r} returned {len(vectors)} vectors for {len(texts)} texts"
            )
        if any(not vector or any(not isinstance(value, (int, float)) for value in vector) for vector in vectors):
            raise RuntimeError(f"Embedding model {self.model_name!r} returned an invalid vector")
        return vectors


def chunk_embedding_text(chunk: dict[str, Any]) -> str:
    heading_path = chunk.get("heading_path", chunk.get("section_path", [])) or []
    return "\n".join(
        part
        for part in (
            str(chunk.get("title", "")),
            " > ".join(str(heading) for heading in heading_path),
            str(chunk.get("text", "")),
        )
        if part
    )


class VectorIndex:
    """JSONL-backed cosine index; only changed or selected documents are embedded."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_VECTOR_INDEX_DIR,
        model_name: str | None = None,
        embedder: LocalEmbedder | None = None,
    ):
        self.index_dir = index_dir
        self.model_name = model_name or embedding_model_name()
        self.embedder = embedder

    def _get_embedder(self) -> LocalEmbedder:
        if self.embedder is None:
            self.embedder = LocalEmbedder(self.model_name)
        return self.embedder

    @property
    def vectors_path(self) -> Path:
        return self.index_dir / INDEX_FILE

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.vectors_path.exists():
            return {}
        vectors: dict[str, dict[str, Any]] = {}
        with self.vectors_path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.vectors_path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{self.vectors_path}:{line_no}: vector record must be an object")
                chunk_id = record.get("chunk_id")
                vector = record.get("vector")
                if (
                    not isinstance(chunk_id, str)
                    or not isinstance(vector, list)
                    or not vector
                    or any(
                        not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in vector
                    )
                    or not isinstance(record.get("document_id"), str)
                    or not isinstance(record.get("content_hash"), str)
                ):
                    raise ValueError(f"{self.vectors_path}:{line_no}: invalid vector record")
                vectors[chunk_id] = record
        return vectors

    @staticmethod
    def _content_hash(chunk: dict[str, Any]) -> str:
        return hashlib.sha256(chunk_embedding_text(chunk).encode("utf-8")).hexdigest()

    def ensure_current(self, chunks: list[dict[str, Any]]) -> None:
        if not self.vectors_path.exists():
            self.build(chunks)
            return
        records = self._load()
        if not records:
            self.build(chunks)
            return
        metadata_path = self.index_dir / METADATA_FILE
        if records and not metadata_path.exists():
            raise ValueError(
                f"Embedding index metadata is missing: {metadata_path}. Rebuild with `rag.cli index --all`."
            )
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid embedding index metadata in {metadata_path}: {exc}") from exc
            if records and metadata.get("model") != self.model_name:
                raise ValueError(
                    f"Index uses embedding model {metadata.get('model')!r}, not {self.model_name!r}; "
                    "choose a new index directory or rebuild all chunks."
                )
        changed_documents = {
            chunk.get("document_id")
            for chunk in chunks
            if (
                chunk.get("chunk_id") not in records
                or records[chunk["chunk_id"]].get("content_hash") != self._content_hash(chunk)
                or records[chunk["chunk_id"]].get("document_id") != chunk.get("document_id")
            )
        }
        for document_id in sorted(item for item in changed_documents if isinstance(item, str)):
            self.build(chunks, document_id=document_id)

    def build(self, chunks: list[dict[str, Any]], document_id: str | None = None) -> int:
        if not chunks:
            raise ValueError("Cannot build an embedding index without final chunks")
        if document_id is not None and not any(
            chunk.get("document_id") == document_id for chunk in chunks
        ):
            raise ValueError(f"No final chunks found for document {document_id!r}")
        existing = self._load()
        metadata_path = self.index_dir / METADATA_FILE
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            existing_model = metadata.get("model")
            if existing and existing_model != self.model_name:
                raise ValueError(
                    f"Index uses embedding model {existing_model!r}, not {self.model_name!r}; "
                    "choose a new index directory or rebuild all chunks."
                )

        if document_id is not None and existing:
            selected = [chunk for chunk in chunks if chunk.get("document_id") == document_id]
            if not selected:
                raise ValueError(f"No final chunks found for document {document_id!r}")
            for chunk_id, record in list(existing.items()):
                if record.get("document_id") == document_id:
                    del existing[chunk_id]
        else:
            selected = chunks
            existing = {}

        pending = [chunk for chunk in selected if chunk.get("chunk_id") not in existing]
        if pending:
            embedder = self._get_embedder()
            vectors = embedder.embed([chunk_embedding_text(chunk) for chunk in pending])
            if len(vectors) != len(pending):
                raise RuntimeError("Embedding model returned a different number of vectors than inputs")
            for chunk, vector in zip(pending, vectors):
                existing[chunk["chunk_id"]] = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "content_hash": self._content_hash(chunk),
                    "vector": vector,
                }

        self.index_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.vectors_path.with_suffix(".jsonl.tmp")
        with temp_path.open("w", encoding="utf-8") as output:
            for chunk_id in sorted(existing):
                output.write(json.dumps(existing[chunk_id], ensure_ascii=False, sort_keys=True) + "\n")
        temp_path.replace(self.vectors_path)
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        metadata_temp.write_text(
            json.dumps(
                {"model": self.model_name, "chunk_count": len(existing)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata_temp.replace(metadata_path)
        return len(pending)

    def search(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
        *,
        document_ids: set[str] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.ensure_current(chunks)
        records = self._load()
        query_vector = self._get_embedder().embed([query])[0]
        norm = sum(value * value for value in query_vector) ** 0.5
        if norm == 0:
            return []
        query_vector = [value / norm for value in query_vector]

        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk_id, record in records.items():
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or (document_ids is not None and chunk.get("document_id") not in document_ids):
                continue
            vector = record["vector"]
            if len(vector) != len(query_vector):
                raise ValueError(f"Embedding dimension mismatch for chunk {chunk_id!r}")
            vector_norm = sum(value * value for value in vector) ** 0.5
            score = (
                sum(left * right for left, right in zip(query_vector, vector)) / vector_norm
                if vector_norm
                else 0.0
            )
            scored.append((float(score), chunk))
        scored.sort(key=lambda item: (-item[0], item[1].get("chunk_id", "")))
        return scored[:top_k]
