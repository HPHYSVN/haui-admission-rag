"""Interactive terminal workflow over the existing document/chunk review services."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag.chunk_review import SPLIT_DELIMITER, load_review_chunks, review_chunks
from rag.config import ROOT_DIR
from rag.document_workflow import (
    DEFAULT_STATE_FILE,
    get_state,
    review_document,
    set_state,
)

DEFAULT_DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
DEFAULT_REVIEW_DIR = ROOT_DIR / "data" / "review"
DEFAULT_PROPOSALS_DIR = ROOT_DIR / "data" / "chunks" / "proposals"
DEFAULT_REVIEWED_DIR = ROOT_DIR / "data" / "chunks" / "reviewed"
DEFAULT_SESSION_FILE = ROOT_DIR / "data" / "workflow" / "review-session.json"
PROCESSED_DOCUMENT_STATES = {"approved", "rejected", "chunk_proposed", "chunk_review", "final", "indexed"}


@dataclass(frozen=True)
class ReviewPaths:
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR
    review_dir: Path = DEFAULT_REVIEW_DIR
    proposals_dir: Path = DEFAULT_PROPOSALS_DIR
    reviewed_dir: Path = DEFAULT_REVIEWED_DIR
    state_file: Path = DEFAULT_STATE_FILE
    session_file: Path = DEFAULT_SESSION_FILE


def _read_documents(documents_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(documents_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(document, dict) or not isinstance(document.get("id"), str):
                    raise ValueError(f"{path}:{line_no}: expected a document object with an id")
                documents.append((path, document))
    return documents


def pending_documents(paths: ReviewPaths) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, document)
        for path, document in _read_documents(paths.documents_dir)
        if get_state(document["id"], paths.state_file) not in PROCESSED_DOCUMENT_STATES
    ]


def progress_label(index: int, total: int) -> str:
    if total < 0 or index < 0 or index > total:
        raise ValueError("Review progress must satisfy 0 <= index <= total")
    return f"{index} / {total}"


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
                raise ValueError(f"{path}:{line_no}: expected a chunk object with chunk_id")
            chunks.append(chunk)
    return chunks


def pending_chunks(paths: ReviewPaths) -> list[tuple[Path, dict[str, Any]]]:
    pending: list[tuple[Path, dict[str, Any]]] = []
    for proposal_file in sorted(paths.proposals_dir.glob("*.jsonl")):
        reviewed_file = paths.reviewed_dir / proposal_file.name
        chunks = load_review_chunks(proposal_file, reviewed_file)
        pending.extend(
            (proposal_file, chunk)
            for chunk in chunks
            if chunk.get("status", "proposal") == "proposal"
        )
    pending.sort(
        key=lambda item: (
            str(item[1].get("category", "")),
            str(item[1].get("document_id", "")),
            str(item[1].get("chunk_id", "")),
        )
    )
    return pending


def _print_document(
    output_fn: Callable[[str], None],
    document: dict[str, Any],
    index: int,
    total: int,
) -> None:
    output_fn("\nHaUI RAG — Document Review")
    output_fn(f"Category: {document.get('category', 'unknown')}")
    output_fn(f"Progress: {progress_label(index, total)}")
    output_fn(f"Document: {document['id']}")
    output_fn(f"Title: {document.get('title', '')}")
    output_fn(f"Year: {document.get('year', 'unknown')}")
    output_fn(f"Source: {document.get('source_url', '')}")
    output_fn("Review the rendered Markdown in the editor.")
    output_fn("[a] Approve & save  [e] Save edits  [r] Reject  [n] Next  [q] Quit")


def _launch_editor(path: Path, editor: str | None = None) -> None:
    command = editor or os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if command:
        argv = [*shlex.split(command), str(path)]
    elif shutil.which("code"):
        argv = ["code", "--reuse-window", "--wait", str(path)]
    elif os.name == "nt":
        argv = ["notepad", str(path)]
    else:
        argv = ["vi", str(path)]
    try:
        subprocess.run(argv, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not open review file with {argv[0]!r}: {exc}") from exc


def _select_item(
    items: list[tuple[Path, dict[str, Any]]],
    *,
    kind: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> tuple[Path, dict[str, Any]] | None:
    all_items = items
    while items:
        output_fn(f"\nPending {kind}s: {len(items)}")
        for index, (_, item) in enumerate(items[:20], 1):
            identifier = item.get("id", item.get("chunk_id", ""))
            title = str(item.get("title", item.get("text", ""))).replace("\n", " ")
            output_fn(
                f"{index:>2}. {identifier} | {item.get('category', 'unknown')} | "
                f"{title[:90]}"
            )
        if len(items) > 20:
            output_fn("Showing first 20; search by ID, title, category, or chunk text.")
        try:
            selection = input_fn(
                f"Select {kind} number or search text (Enter = first, q = quit): "
            ).strip()
        except EOFError:
            return None
        if selection.casefold() in {"q", "quit"}:
            return None
        if not selection:
            return items[0]
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= min(20, len(items)):
                return items[index - 1]
            output_fn("Selection number is outside the displayed list.")
            continue

        query = selection.casefold()
        exact = [
            item
            for item in items
            if str(item[1].get("id", item[1].get("chunk_id", ""))).casefold() == query
        ]
        if exact:
            return exact[0]
        matches = [
            item
            for item in all_items
            if query
            in " ".join(
                str(item[1].get(field, ""))
                for field in ("id", "chunk_id", "title", "category", "text", "document_id")
            ).casefold()
        ]
        if not matches:
            output_fn(f"No pending {kind} matches {selection!r}.")
            continue
        if len(matches) == 1:
            return matches[0]
        items = matches
    return None


def _read_multiline(input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> str:
    output_fn("Enter text; finish with a line containing only '.'")
    lines: list[str] = []
    while True:
        line = input_fn("")
        if line == ".":
            return "\n".join(lines).strip()
        lines.append(line)


def review_documents(
    paths: ReviewPaths = ReviewPaths(),
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
    editor: str | None = None,
) -> None:
    read_input = input_fn or input
    all_documents = _read_documents(paths.documents_dir)
    category_counts: dict[str, int] = {}
    category_processed: dict[str, int] = {}
    for _, document in all_documents:
        category = str(document.get("category", "unknown"))
        category_counts[category] = category_counts.get(category, 0) + 1
        if get_state(document["id"], paths.state_file) in PROCESSED_DOCUMENT_STATES:
            category_processed[category] = category_processed.get(category, 0) + 1

    while True:
        queue = pending_documents(paths)
        if not queue:
            output_fn("✓ No pending document review items.")
            output_fn("All documents have been processed.")
            return

        selected = _select_item(
            queue,
            kind="document",
            input_fn=read_input,
            output_fn=output_fn,
        )
        if selected is None:
            output_fn("Leaving review. Saved decisions are available to resume.")
            return

        path, document = selected
        category = str(document.get("category", "unknown"))
        index = category_processed.get(category, 0) + 1
        total = category_counts.get(category, 1)
        _print_document(output_fn, document, min(index, total), total)
        from scripts.apply_review import apply_review_file
        from scripts.render_review import render

        review_file = paths.review_dir / category / f"{document['id']}.md"
        render(path, review_file.parent, document["id"], state_file=paths.state_file)
        _launch_editor(review_file, editor)
        try:
            action = read_input("Action: ").strip().casefold()
        except EOFError:
            output_fn("\nLeaving review. Saved decisions are available to resume.")
            return

        if action in {"q", "quit"}:
            output_fn("Leaving review. Saved decisions are available to resume.")
            return
        if action in {"n", "next"}:
            continue
        if action in {"a", "approve"}:
            apply_review_file(path, review_file, state_file=paths.state_file)
            review_document(document["id"], "approve", state_file=paths.state_file)
            category_processed[category] = category_processed.get(category, 0) + 1
            output_fn(f"Approved {document['id']}.")
            continue
        if action in {"e", "edit", "s", "save"}:
            apply_review_file(path, review_file, state_file=paths.state_file)
            output_fn(f"Applied content edit to {document['id']}; it remains pending approval.")
            continue
        if action in {"r", "reject"}:
            try:
                confirm = read_input(f"Reject {document['id']}? [y/N] ").strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. No rejection was saved.")
                return
            if confirm != "y":
                continue
            try:
                note = read_input("Reason (optional): ").strip()
            except EOFError:
                output_fn("\nLeaving review. No rejection was saved.")
                return
            review_document(document["id"], "reject", note=note, state_file=paths.state_file)
            category_processed[category] = category_processed.get(category, 0) + 1
            output_fn(f"Rejected {document['id']}.")
            continue
        output_fn("Choose a, e, r, n, or q.")


def _print_chunk(
    output_fn: Callable[[str], None],
    chunk: dict[str, Any],
    index: int,
    total: int,
) -> None:
    heading_path = chunk.get("heading_path", chunk.get("section_path", [])) or []
    output_fn("\nHaUI RAG — Chunk Review")
    output_fn(f"Document: {chunk.get('document_id', '')}")
    output_fn(f"Progress: {progress_label(index, total)}")
    output_fn(f"Chunk: {chunk['chunk_id']}")
    output_fn("-" * 72)
    output_fn(str(chunk.get("text", "")))
    output_fn("-" * 72)
    output_fn("Metadata:")
    output_fn(f"Title: {chunk.get('title', '')}")
    output_fn(f"Category: {chunk.get('category', 'unknown')}")
    output_fn(f"Year: {chunk.get('year', 'unknown')}")
    output_fn(f"Section: {' > '.join(str(item) for item in heading_path)}")
    output_fn(f"Content type: {chunk.get('content_type', 'unknown')}")
    output_fn(f"Source URL: {chunk.get('source_url', '')}")
    output_fn("-" * 72)
    output_fn("[a] Approve  [e] Edit  [s] Split  [m] Merge  [r] Reject  [n] Next  [q] Quit")


def review_chunk_queue(
    paths: ReviewPaths = ReviewPaths(),
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> None:
    read_input = input_fn or input
    processed_by_document: dict[str, int] = {}
    skipped: set[str] = set()

    while True:
        queue = [
            item for item in pending_chunks(paths)
            if item[1]["chunk_id"] not in skipped
        ]
        if not queue:
            output_fn("✓ No pending chunk review items.")
            output_fn("All chunks have been processed.")
            return
        selected = _select_item(
            queue,
            kind="chunk",
            input_fn=read_input,
            output_fn=output_fn,
        )
        if selected is None:
            output_fn("Leaving review. Saved decisions are available to resume.")
            return
        proposal_file, chunk = selected
        document_id = str(chunk.get("document_id", ""))
        all_for_document = [
            item for item in pending_chunks(paths)
            if item[1].get("document_id") == document_id
        ]
        total = len(all_for_document) + processed_by_document.get(document_id, 0)
        index = processed_by_document.get(document_id, 0) + 1
        _print_chunk(output_fn, chunk, min(index, total), total)
        try:
            action = read_input("Action: ").strip().casefold()
        except EOFError:
            output_fn("\nLeaving review. Saved decisions are available to resume.")
            return

        if action in {"q", "quit"}:
            output_fn("Leaving review. Saved decisions are available to resume.")
            return
        if action in {"n", "next"}:
            skipped.add(chunk["chunk_id"])
            processed_by_document[document_id] = processed_by_document.get(document_id, 0) + 1
            continue
        if action in {"r", "reject"}:
            try:
                confirm = read_input(f"Reject {chunk['chunk_id']}? [y/N] ").strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. No rejection was saved.")
                return
            if confirm != "y":
                continue
            try:
                note = read_input("Reason (optional): ").strip()
            except EOFError:
                output_fn("\nLeaving review. No rejection was saved.")
                return
            review_chunks(
                proposal_file,
                paths.reviewed_dir / proposal_file.name,
                action="reject",
                chunk_id=chunk["chunk_id"],
                note=note,
            )
        elif action in {"a", "approve"}:
            review_chunks(
                proposal_file,
                paths.reviewed_dir / proposal_file.name,
                action="approve",
                chunk_id=chunk["chunk_id"],
            )
        elif action in {"e", "edit"}:
            text = _read_multiline(read_input, output_fn)
            try:
                review_chunks(
                    proposal_file,
                    paths.reviewed_dir / proposal_file.name,
                    action="edit",
                    chunk_id=chunk["chunk_id"],
                    text=text,
                )
            except ValueError as exc:
                output_fn(f"Not saved: {exc}")
                continue
        elif action in {"s", "split"}:
            text = _read_multiline(read_input, output_fn)
            try:
                review_chunks(
                    proposal_file,
                    paths.reviewed_dir / proposal_file.name,
                    action="split",
                    chunk_id=chunk["chunk_id"],
                    text=text,
                )
            except ValueError as exc:
                output_fn(f"Not saved: {exc}")
                continue
        elif action in {"m", "merge"}:
            candidates = [
                candidate
                for candidate_file, candidate in queue[1:]
                if candidate_file.name == proposal_file.name
                and candidate.get("document_id") == document_id
            ]
            if not candidates:
                output_fn("No other pending chunk from this document is available to merge.")
                continue
            for candidate_index, candidate in enumerate(candidates, 1):
                output_fn(
                    f"{candidate_index}. {candidate['chunk_id']}: "
                    f"{str(candidate.get('text', '')).replace(chr(10), ' ')[:120]}"
                )
            try:
                selection = read_input("Merge with item number (blank cancels): ").strip()
            except EOFError:
                output_fn("\nLeaving review. No merge was saved.")
                return
            if not selection:
                continue
            if not selection.isdigit() or not 1 <= int(selection) <= len(candidates):
                output_fn("Invalid selection; no merge was saved.")
                continue
            other = candidates[int(selection) - 1]
            try:
                confirm = read_input(
                    f"Merge {chunk['chunk_id']} with {other['chunk_id']}? [y/N] "
                ).strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. No merge was saved.")
                return
            if confirm != "y":
                continue
            review_chunks(
                proposal_file,
                paths.reviewed_dir / proposal_file.name,
                action="merge",
                chunk_id=chunk["chunk_id"],
                other_chunk_id=other["chunk_id"],
            )
        else:
            output_fn("Choose a, e, s, m, r, n, or q.")
            continue

        set_state(document_id, "chunk_review", state_file=paths.state_file)
        skipped.discard(chunk["chunk_id"])
        processed_by_document[document_id] = processed_by_document.get(document_id, 0) + 1
        output_fn(f"Saved decision for {chunk['chunk_id']}.")


def _load_last_mode(session_file: Path) -> str | None:
    if not session_file.exists():
        return None
    try:
        session = json.loads(session_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid review session file {session_file}: {exc}") from exc
    mode = session.get("mode") if isinstance(session, dict) else None
    return mode if mode in {"documents", "chunks"} else None


def _save_last_mode(session_file: Path, mode: str) -> None:
    session_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = session_file.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps({"mode": mode}, indent=2) + "\n", encoding="utf-8")
    temp_file.replace(session_file)


def run_review(
    paths: ReviewPaths = ReviewPaths(),
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
    editor: str | None = None,
) -> int:
    read_input = input_fn or input
    output_fn("HaUI RAG Review\n\n1. Review documents\n2. Review chunks\n3. Continue previous review\n4. Exit")
    try:
        while True:
            try:
                choice = read_input("Select: ").strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. Saved decisions are available to resume.")
                return 0
            if choice in {"4", "q", "quit"}:
                return 0
            if choice in {"1", "documents", "document"}:
                _save_last_mode(paths.session_file, "documents")
                review_documents(paths, input_fn=read_input, output_fn=output_fn, editor=editor)
                return 0
            if choice in {"2", "chunks", "chunk"}:
                _save_last_mode(paths.session_file, "chunks")
                review_chunk_queue(paths, input_fn=read_input, output_fn=output_fn)
                return 0
            if choice in {"3", "resume"}:
                mode = _load_last_mode(paths.session_file)
                if mode == "documents":
                    review_documents(paths, input_fn=read_input, output_fn=output_fn, editor=editor)
                    return 0
                if mode == "chunks":
                    review_chunk_queue(paths, input_fn=read_input, output_fn=output_fn)
                    return 0
                output_fn("No previous review session found. Choose documents or chunks.")
                continue
            output_fn("Choose 1, 2, 3, or 4.")
    except KeyboardInterrupt:
        output_fn("\nReview interrupted. Saved decisions are available to resume.")
        return 0
    except EOFError:
        output_fn("\nLeaving review. Saved decisions are available to resume.")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        output_fn(f"Review error: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.cli review")
    parser.parse_args(argv)
    return run_review()
