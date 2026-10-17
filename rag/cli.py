"""Interactive command-line entry point for the basic HaUI RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from rag.config import DEFAULT_TOP_K, DEFAULT_VECTOR_INDEX_DIR
from rag.pipeline import (
    DEFAULT_CHUNKS_DIR,
    RAGPipeline,
    RAGResponse,
    VLLMChatClient,
    load_chunks,
)


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


def display_result(
    response: RAGResponse,
    show_context: bool,
    output_fn: Callable[[str], None] = print,
) -> None:
    output_fn(f"\n{response.answer}\n")
    if response.sources:
        output_fn("Nguồn:")
        for source in response.sources:
            output_fn(
                f"- {source['title']} | {source['url']} | "
                f"document={source['source_id']} chunk={source['chunk_id']} "
                f"score={source['score']:.3f}"
            )
    if show_context:
        for score, chunk in response.retrieved:
            output_fn(f"\n[{score:.3f}] {chunk['chunk_id']}\n{chunk['text']}")


def _chat_loop(
    pipeline: RAGPipeline,
    *,
    top_k: int,
    show_context: bool,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    history: list[dict[str, str]] = []
    output_fn("HaUI RAG Chat. Gõ /clear để xóa hội thoại, /quit để thoát.")
    while True:
        try:
            question = input_fn("\nBạn: ").strip()
        except EOFError:
            output_fn("")
            return 0
        except KeyboardInterrupt:
            output_fn("\nTạm biệt.")
            return 0
        command = question.casefold()
        if command in {"/quit", "/exit"}:
            return 0
        if command == "/clear":
            history.clear()
            output_fn("Đã xóa hội thoại.")
            continue
        if not question:
            continue
        try:
            response = pipeline.ask(question, top_k=top_k, history=history[-8:])
        except (RuntimeError, ValueError) as exc:
            output_fn(f"Lỗi: {exc}")
            continue
        except KeyboardInterrupt:
            output_fn("\nTạm biệt.")
            return 0
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response.answer},
            ]
        )
        display_result(response, show_context, output_fn)


def _chat_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat trực tiếp với HaUI RAG.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        pipeline = RAGPipeline(
            load_chunks(args.chunks_dir),
            VLLMChatClient(),
            index_dir=args.index_dir,
        )
        return _chat_loop(
            pipeline,
            top_k=args.top_k,
            show_context=args.show_context,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Lỗi khởi chạy RAG: {exc}", file=sys.stderr)
        return 1


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
        return _chat_loop(
            pipeline,
            top_k=args.top_k,
            show_context=args.show_context,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Lỗi khởi chạy RAG: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "review":
        from rag.review import main as review_main

        return review_main(args[1:])
    if args and args[0] == "chat":
        return _chat_main(args[1:])
    if args and args[0] in WORKFLOW_COMMANDS:
        from rag.admin import main as admin_main

        return admin_main(args)
    return _legacy_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
