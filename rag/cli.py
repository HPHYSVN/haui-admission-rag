"""Interactive command-line entry point for the basic HaUI RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.pipeline import DEFAULT_CHUNKS_DIR, RAGPipeline, VLLMChatClient, load_chunks
from rag.config import DEFAULT_TOP_K


WORKFLOW_COMMANDS = {
    "render",
    "document-review",
    "chunk",
    "chunk-review",
    "finalize-chunks",
    "index",
    "retrieve",
    "query",
    "review",
}


def display_result(response, show_context: bool) -> None:
    print(f"\n{response.answer}\n")
    if response.sources:
        print("Nguồn:")
        for source in response.sources:
            print(
                f"- {source['title']} | {source['url']} | "
                f"document={source['source_id']} chunk={source['chunk_id']} "
                f"score={source['score']:.3f}"
            )
    if show_context:
        for score, chunk in response.retrieved:
            print(f"\n[{score:.3f}] {chunk['chunk_id']}\n{chunk['text']}")


def _legacy_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions against the HaUI corpus using vLLM.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--query", help="Ask one question and exit; omit for interactive mode")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks for learning/debugging")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        pipeline = RAGPipeline(load_chunks(args.chunks_dir), VLLMChatClient())
        if args.query:
            display_result(pipeline.ask(args.query, top_k=args.top_k), args.show_context)
            return 0
        print("HaUI RAG demo. Gõ /quit để thoát.")
        while True:
            try:
                question = input("\nBạn: ").strip()
            except EOFError:
                print()
                return 0
            if question.casefold() in {"/quit", "/exit"}:
                return 0
            if not question:
                continue
            try:
                response = pipeline.ask(question, top_k=args.top_k)
            except (RuntimeError, ValueError) as exc:
                print(f"Lỗi: {exc}", file=sys.stderr)
                continue
            display_result(response, args.show_context)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Lỗi khởi chạy RAG: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "review":
        from rag.review import main as review_main

        return review_main(args[1:])
    if args and args[0] in WORKFLOW_COMMANDS:
        from rag.admin import main as admin_main

        return admin_main(args)
    return _legacy_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
