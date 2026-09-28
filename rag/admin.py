"""Team-facing commands for independent document and chunk workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.chunk_review import finalize_chunks, review_chunks
from rag.config import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_INDEX_DIR,
    ROOT_DIR,
)
from rag.document_workflow import get_state, review_document, set_state
from rag.embeddings import VectorIndex
from rag.pipeline import RAGPipeline, VLLMChatClient, load_chunks

DEFAULT_DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
DEFAULT_REVIEW_DIR = ROOT_DIR / "data" / "review"
DEFAULT_PROPOSALS_DIR = ROOT_DIR / "data" / "chunks" / "proposals"
DEFAULT_REVIEWED_DIR = ROOT_DIR / "data" / "chunks" / "reviewed"


def find_document(document_id: str) -> tuple[Path, dict[str, Any]]:
    for path in sorted(DEFAULT_DOCUMENTS_DIR.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if document.get("id") == document_id:
                return path, document
    raise ValueError(f"Document ID {document_id!r} was not found in {DEFAULT_DOCUMENTS_DIR}")


def _load_chunk_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render")
    render.add_argument("--document", required=True)
    render.add_argument("--output-dir", type=Path)

    document_review = commands.add_parser("document-review")
    document_review.add_argument("action", choices=["status", "pending", "approve", "reject"])
    document_review.add_argument("--document", required=True)
    document_review.add_argument("--note", default="")

    chunk = commands.add_parser("chunk")
    chunk.add_argument("--document", required=True)
    chunk.add_argument("--output-dir", type=Path, default=DEFAULT_PROPOSALS_DIR)
    chunk.add_argument(
        "--replace-review",
        action="store_true",
        help="Discard this document's old chunk decisions when regenerating proposals",
    )

    chunk_review = commands.add_parser("chunk-review")
    chunk_review.add_argument("action", choices=["approve", "edit", "split", "merge", "reject"])
    chunk_review.add_argument("--input", type=Path, required=True)
    chunk_review.add_argument("--chunk-id", required=True)
    chunk_review.add_argument("--other-chunk-id")
    chunk_review.add_argument("--text-file", type=Path)
    chunk_review.add_argument("--note", default="")

    finalize = commands.add_parser("finalize-chunks")
    finalize.add_argument("--input", type=Path, required=True)
    finalize.add_argument("--document", required=True)

    index = commands.add_parser("index")
    index_target = index.add_mutually_exclusive_group(required=True)
    index_target.add_argument("--document")
    index_target.add_argument("--all", action="store_true")
    index.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    index.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)

    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    retrieve.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    retrieve.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)

    query = commands.add_parser("query")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    query.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    query.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    query.add_argument("--show-context", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            source_file, _ = find_document(args.document)
            from scripts.render_review import render as render_documents

            output_dir = args.output_dir or DEFAULT_REVIEW_DIR / source_file.stem
            count = render_documents(source_file, output_dir, args.document)
            print(f"Rendered {count} document to {output_dir}")
            return 0

        if args.command == "document-review":
            find_document(args.document)
            if args.action == "status":
                print(f"{args.document}: {get_state(args.document)}")
                return 0
            state = review_document(args.document, args.action, note=args.note)
            print(f"{args.document}: {state}")
            return 0

        if args.command == "chunk":
            source_file, document = find_document(args.document)
            if get_state(args.document) != "approved":
                raise ValueError(
                    f"Document {args.document!r} is {get_state(args.document)!r}; approve it before chunking."
                )
            from scripts.chunk import remove_reviewed_document, write_proposals

            proposal_file = args.output_dir / source_file.name
            reviewed_file = DEFAULT_REVIEWED_DIR / source_file.name
            if args.replace_review:
                remove_reviewed_document(reviewed_file, args.document)
            proposal_count = write_proposals([document], proposal_file)
            set_state(args.document, "chunk_proposed")
            print(f"Wrote {proposal_count} proposals for {args.document} to {proposal_file}")
            return 0

        if args.command == "chunk-review":
            proposal_file = args.input
            reviewed_file = DEFAULT_REVIEWED_DIR / proposal_file.name
            contents = _load_chunk_records(reviewed_file if reviewed_file.exists() else proposal_file)
            matching = next((chunk for chunk in contents if chunk.get("chunk_id") == args.chunk_id), None)
            if matching is None:
                raise ValueError(f"Chunk {args.chunk_id!r} not found in {proposal_file}")
            text = args.text_file.read_text(encoding="utf-8") if args.text_file else None
            review_chunks(
                proposal_file,
                reviewed_file,
                action=args.action,
                chunk_id=args.chunk_id,
                text=text,
                other_chunk_id=args.other_chunk_id,
                note=args.note,
            )
            set_state(matching["document_id"], "chunk_review")
            print(f"Reviewed chunk {args.chunk_id}: {args.action}; output={reviewed_file}")
            return 0

        if args.command == "finalize-chunks":
            proposal_file = args.input
            reviewed_file = DEFAULT_REVIEWED_DIR / proposal_file.name
            final_file = DEFAULT_CHUNKS_DIR / proposal_file.name
            count = finalize_chunks(
                proposal_file,
                reviewed_file,
                final_file,
                document_id=args.document,
            )
            print(f"Promoted {count} final chunks for {args.document} to {final_file}")
            return 0

        if args.command == "index":
            chunks = load_chunks(args.chunks_dir)
            count = VectorIndex(index_dir=args.index_dir).build(
                chunks,
                document_id=args.document,
            )
            if args.document:
                set_state(args.document, "indexed")
            else:
                for document_id in {chunk["document_id"] for chunk in chunks}:
                    set_state(document_id, "indexed")
            print(f"Embedded {count} chunks; index contains {args.index_dir}")
            return 0
        if args.command == "retrieve":
            if args.top_k < 1:
                raise ValueError("top_k must be at least 1")
            chunks = load_chunks(args.chunks_dir)
            from rag.bm25 import HybridRetriever

            retriever = HybridRetriever(chunks, index_dir=args.index_dir)
            results = retriever.search(args.query, args.top_k)
            print(json.dumps(
                [
                    {"score": score, "chunk": chunk}
                    for score, chunk in results
                ],
                ensure_ascii=False,
                indent=2,
            ))
            return 0
        if args.command == "query":
            if args.top_k < 1:
                raise ValueError("top_k must be at least 1")
            pipeline = RAGPipeline(
                load_chunks(args.chunks_dir),
                VLLMChatClient(),
                index_dir=args.index_dir,
            )
            response = pipeline.ask(args.query, top_k=args.top_k)
            print(response.answer)
            print("\nNguồn:")
            print(json.dumps(response.sources, ensure_ascii=False, indent=2))
            if args.show_context:
                for score, chunk_record in response.retrieved:
                    print(f"\n[{score:.3f}] {chunk_record['chunk_id']}\n{chunk_record['text']}")
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Workflow error: {exc}\n")
    return 0
