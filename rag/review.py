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

from rag.chunk_review import (
    finalize_chunks,
    load_review_chunks,
    revise_final_chunk,
    review_chunks,
)
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
DEFAULT_FINAL_DIR = ROOT_DIR / "data" / "chunks" / "final"
DEFAULT_SESSION_FILE = ROOT_DIR / "data" / "workflow" / "review-session.json"
PICKER_PAGE_SIZE = 10
PROCESSED_DOCUMENT_STATES = {"approved", "rejected", "chunk_proposed", "chunk_review", "final", "indexed"}
_EDITOR_PROCESSES: list[subprocess.Popen] = []


@dataclass(frozen=True)
class ReviewPaths:
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR
    review_dir: Path = DEFAULT_REVIEW_DIR
    proposals_dir: Path = DEFAULT_PROPOSALS_DIR
    reviewed_dir: Path = DEFAULT_REVIEWED_DIR
    final_dir: Path = DEFAULT_FINAL_DIR
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


def amendable_chunks(paths: ReviewPaths) -> list[tuple[Path, dict[str, Any]]]:
    amendable: list[tuple[Path, dict[str, Any]]] = []
    for proposal_file in sorted(paths.proposals_dir.glob("*.jsonl")):
        reviewed_file = paths.reviewed_dir / proposal_file.name
        final_file = paths.final_dir / proposal_file.name
        if not final_file.exists():
            continue
        final_ids = {
            chunk["chunk_id"]
            for chunk in _read_chunks(final_file)
            if chunk.get("status") == "final"
        }
        amendable.extend(
            (proposal_file, chunk)
            for chunk in load_review_chunks(proposal_file, reviewed_file)
            if chunk.get("status") in {"approved", "edited"}
            and chunk.get("chunk_id") in final_ids
        )
    return sorted(
        amendable,
        key=lambda item: (
            str(item[1].get("category", "")),
            str(item[1].get("document_id", "")),
            str(item[1].get("chunk_id", "")),
        ),
    )


def finalize_ready_documents(
    paths: ReviewPaths = ReviewPaths(),
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> None:
    read_input = input_fn or input
    ready: list[tuple[Path, dict[str, Any]]] = []
    resolved_statuses = {"approved", "edited", "rejected", "split", "merged"}
    allowed_document_states = {"approved", "chunk_proposed", "chunk_review"}
    for proposal_file in sorted(paths.proposals_dir.glob("*.jsonl")):
        reviewed_file = paths.reviewed_dir / proposal_file.name
        chunks = load_review_chunks(proposal_file, reviewed_file)
        by_document: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            by_document.setdefault(str(chunk.get("document_id", "")), []).append(chunk)
        for document_id, document_chunks in by_document.items():
            if (
                document_id
                and get_state(document_id, paths.state_file) in allowed_document_states
                and document_chunks
                and all(chunk.get("status") in resolved_statuses for chunk in document_chunks)
            ):
                exemplar = document_chunks[0]
                ready.append(
                    (
                        proposal_file,
                        {
                            "id": document_id,
                            "document_id": document_id,
                            "title": exemplar.get("title", ""),
                            "category": exemplar.get("category", "unknown"),
                        },
                    )
                )
    if not ready:
        output_fn("No fully reviewed documents are waiting to be finalized.")
        return

    counts: dict[str, int] = {}
    for _, document in ready:
        category = str(document.get("category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    category = _select_category(
        counts,
        kind="finalization",
        count_label="ready",
        input_fn=read_input,
        output_fn=output_fn,
    )
    if category is None:
        output_fn("Finalization cancelled.")
        return
    selected = _select_item(
        [item for item in ready if str(item[1].get("category", "unknown")) == category],
        kind="document",
        input_fn=read_input,
        output_fn=output_fn,
    )
    if selected is None:
        output_fn("Finalization cancelled.")
        return
    proposal_file, document = selected
    reviewed_file = paths.reviewed_dir / proposal_file.name
    final_file = paths.final_dir / proposal_file.name
    count = finalize_chunks(
        proposal_file,
        reviewed_file,
        final_file,
        document_id=str(document["id"]),
        state_file=paths.state_file,
    )
    output_fn(
        f"Finalized {count} chunks for {document['id']}.\n"
        f"Final chunks: {final_file}\n"
        f"Review decisions: {reviewed_file}\n"
        "The proposals file remains unchanged as the generated baseline."
    )


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


def _launch_editor(path: Path, editor: str | None = None) -> bool:
    command = editor or os.environ.get("VISUAL") or os.environ.get("EDITOR")
    command_parts = shlex.split(command) if command else []
    is_vscode = bool(command_parts) and Path(command_parts[0]).name in {
        "code",
        "code-insiders",
        "codium",
    }
    if is_vscode:
        argv = [part for part in command_parts if part != "--wait"] + [str(path)]
    elif not command and shutil.which("code"):
        argv = ["code", "--reuse-window", str(path)]
        is_vscode = True
    else:
        argv = []
    if is_vscode:
        try:
            _EDITOR_PROCESSES[:] = [
                process for process in _EDITOR_PROCESSES if process.poll() is None
            ]
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _EDITOR_PROCESSES.append(process)
            return True
        except OSError as exc:
            raise RuntimeError(f"Could not open review file with {argv[0]!r}: {exc}") from exc
    if command:
        argv = [*command_parts, str(path)]
    elif os.name == "nt":
        argv = ["notepad", str(path)]
    elif shutil.which("xdg-open"):
        argv = ["xdg-open", str(path)]
    else:
        argv = ["vi", str(path)]
    try:
        subprocess.run(argv, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not open review file with {argv[0]!r}: {exc}") from exc
    return False


def _item_matches(item: dict[str, Any], query: str) -> bool:
    searchable = " ".join(
        str(item.get(field, ""))
        for field in ("id", "chunk_id", "title", "category", "text", "document_id", "year")
    )
    return query in searchable.casefold()


def _select_item(
    items: list[tuple[Path, dict[str, Any]]],
    *,
    kind: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> tuple[Path, dict[str, Any]] | None:
    search = ""
    page = 0
    while True:
        matches = [
            pair
            for pair in items
            if _item_matches(pair[1], search)
        ]
        if not matches:
            output_fn(f"No {kind}s match {search!r}.")
            search = ""
            page = 0
            continue
        page_count = (len(matches) + PICKER_PAGE_SIZE - 1) // PICKER_PAGE_SIZE
        page = min(page, page_count - 1)
        start = page * PICKER_PAGE_SIZE
        visible = matches[start : start + PICKER_PAGE_SIZE]
        output_fn(f"\n{kind.title()}s: {len(matches)} match(es) | page {page + 1}/{page_count}")
        for index, (_, item) in enumerate(visible, start + 1):
            identifier = item.get("id", item.get("chunk_id", ""))
            title = str(item.get("title", item.get("text", ""))).replace("\n", " ")
            output_fn(
                f"{index:>2}. {identifier} | {item.get('category', 'unknown')} | "
                f"{title[:90]}"
            )
        if search:
            output_fn(f"Filter: {search}")
        try:
            selection = input_fn(
                "Number = open; text = search; >/< = next/previous page; "
                "Enter = first match; q = back: "
            ).strip()
        except EOFError:
            return None
        if selection.casefold() in {"q", "quit"}:
            return None
        if not selection:
            return matches[0]
        if selection == ">":
            page = (page + 1) % page_count
            continue
        if selection == "<":
            page = (page - 1) % page_count
            continue
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(matches):
                return matches[index - 1]
            search = selection.casefold()
            page = 0
            filtered = [
                pair
                for pair in items
                if _item_matches(pair[1], search)
            ]
            if len(filtered) == 1:
                return filtered[0]
            continue
        search = selection.casefold()
        page = 0
        filtered = [
            pair
            for pair in items
            if _item_matches(pair[1], search)
        ]
        if len(filtered) == 1:
            return filtered[0]


def _select_category(
    counts: dict[str, int],
    *,
    kind: str,
    count_label: str = "pending",
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str | None:
    all_categories = sorted(counts)
    categories = all_categories
    while categories:
        output_fn(f"\nSelect a {kind} category:")
        for index, category in enumerate(categories, 1):
            output_fn(f"{index:>2}. {category} ({counts[category]} {count_label})")
        try:
            selection = input_fn("Category number or name (q = quit): ").strip()
        except EOFError:
            return None
        if selection.casefold() in {"q", "quit"}:
            return None
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(categories):
                return categories[index - 1]
            output_fn(f"Choose a category number from 1 to {len(categories)}.")
            continue
        exact = next((category for category in categories if category.casefold() == selection.casefold()), None)
        if exact:
            return exact
        matches = [
            category
            for category in all_categories
            if selection.casefold() in category.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            categories = matches
        else:
            output_fn(f"No {kind} category matches {selection!r}.")


def _read_multiline(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str | None:
    output_fn("Paste text; finish with :save on its own line (:cancel discards it).")
    lines: list[str] = []
    while True:
        line = input_fn("")
        if line == ":save":
            return "\n".join(lines).strip()
        if line == ":cancel":
            return None
        lines.append(line)


def _finalize_document_if_ready(
    proposal_file: Path,
    document_id: str,
    paths: ReviewPaths,
    output_fn: Callable[[str], None],
) -> None:
    chunks = [
        chunk
        for chunk in load_review_chunks(
            proposal_file,
            paths.reviewed_dir / proposal_file.name,
        )
        if chunk.get("document_id") == document_id
    ]
    resolved = {"approved", "edited", "rejected", "split", "merged"}
    if not chunks or any(chunk.get("status") not in resolved for chunk in chunks):
        return
    set_state(document_id, "chunk_review", state_file=paths.state_file)
    final_file = paths.final_dir / proposal_file.name
    count = finalize_chunks(
        proposal_file,
        paths.reviewed_dir / proposal_file.name,
        final_file,
        document_id=document_id,
        state_file=paths.state_file,
    )
    output_fn(
        f"All proposals resolved. Finalized {count} chunks for {document_id}.\n"
        f"Final chunks: {final_file}"
    )


def review_documents(
    paths: ReviewPaths = ReviewPaths(),
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
    editor: str | None = None,
) -> None:
    read_input = input_fn or input
    all_documents = _read_documents(paths.documents_dir)
    categories: set[str] = set()
    category_processed: dict[str, int] = {}
    for _, document in all_documents:
        category = str(document.get("category", "unknown"))
        categories.add(category)
        if get_state(document["id"], paths.state_file) in PROCESSED_DOCUMENT_STATES:
            category_processed[category] = category_processed.get(category, 0) + 1

    while True:
        queue = pending_documents(paths)
        counts = {category: 0 for category in categories}
        for _, document in queue:
            category = str(document.get("category", "unknown"))
            counts[category] = counts.get(category, 0) + 1
        if not queue:
            output_fn("✓ No pending document review items.")
            return
        category = _select_category(
            counts,
            kind="document",
            input_fn=read_input,
            output_fn=output_fn,
        )
        if category is None:
            output_fn("Leaving review. Saved decisions are available to resume.")
            return
        skipped: set[str] = set()
        while True:
            category_queue = [
                (path, document)
                for path, document in pending_documents(paths)
                if str(document.get("category", "unknown")) == category
                and document["id"] not in skipped
            ]
            if not category_queue:
                output_fn(f"No more pending documents in {category}.")
                break
            selected = _select_item(
                category_queue,
                kind="document",
                input_fn=read_input,
                output_fn=output_fn,
            )
            if selected is None:
                break
            path, document = selected
            index = category_processed.get(category, 0) + 1
            total = sum(
                str(record.get("category", "unknown")) == category
                for _, record in all_documents
            )
            _print_document(output_fn, document, min(index, total), total)
            from scripts.apply_review import apply_review_file
            from scripts.render_review import render

            review_file = paths.review_dir / category / f"{document['id']}.md"
            render(path, review_file.parent, document["id"], state_file=paths.state_file)
            opened_in_background = _launch_editor(review_file, editor)
            if opened_in_background:
                output_fn("VS Code opened in the background. Save the Markdown, then return here.")
            try:
                action = read_input("Action: ").strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. Saved decisions are available to resume.")
                return
            if action in {"q", "quit"}:
                output_fn("Leaving review. Saved decisions are available to resume.")
                return
            if action in {"n", "next"}:
                skipped.add(document["id"])
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
    amend: bool = False,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> None:
    read_input = input_fn or input
    processed_by_document: dict[str, int] = {}
    skipped: set[str] = set()

    while True:
        source_queue = amendable_chunks(paths) if amend else pending_chunks(paths)
        queue = [item for item in source_queue if item[1]["chunk_id"] not in skipped]
        if not queue:
            message = "approved/final chunks to edit" if amend else "pending chunk review items"
            output_fn(f"✓ No {message} remain.")
            return
        categories = {
            str(document.get("category", "unknown"))
            for _, document in _read_documents(paths.documents_dir)
        }
        categories.update(str(chunk.get("category", "unknown")) for _, chunk in source_queue)
        counts = {category: 0 for category in categories}
        for _, chunk in queue:
            category = str(chunk.get("category", "unknown"))
            counts[category] = counts.get(category, 0) + 1
        category = _select_category(
            counts,
            kind="chunk",
            count_label="approved/final" if amend else "pending",
            input_fn=read_input,
            output_fn=output_fn,
        )
        if category is None:
            output_fn("Leaving review. Saved decisions are available to resume.")
            return
        skipped_in_category: set[str] = set()
        while True:
            category_queue = [
                item
                for item in (amendable_chunks(paths) if amend else pending_chunks(paths))
                if str(item[1].get("category", "unknown")) == category
                and item[1]["chunk_id"] not in skipped
                and item[1]["chunk_id"] not in skipped_in_category
            ]
            if not category_queue:
                output_fn(f"No more chunks to review in {category}.")
                break
            selected = _select_item(
                category_queue,
                kind="chunk",
                input_fn=read_input,
                output_fn=output_fn,
            )
            if selected is None:
                break
            proposal_file, chunk = selected
            document_id = str(chunk.get("document_id", ""))
            all_for_document = [
                item for item in (amendable_chunks(paths) if amend else pending_chunks(paths))
                if item[1].get("document_id") == document_id
            ]
            total = len(all_for_document) + processed_by_document.get(document_id, 0)
            index = processed_by_document.get(document_id, 0) + 1
            _print_chunk(output_fn, chunk, min(index, max(total, 1)), max(total, 1))
            if amend:
                output_fn("[e] Edit and update finalized chunks  [q] Back to category list")
            try:
                action = read_input("Action: ").strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. Saved decisions are available to resume.")
                return

            if action in {"q", "quit"}:
                if amend:
                    break
                output_fn("Leaving review. Saved decisions are available to resume.")
                return
            if amend:
                if action not in {"e", "edit"}:
                    output_fn("Choose e to edit, or q to return to categories.")
                    continue
                text = _read_multiline(read_input, output_fn)
                if text is None:
                    output_fn("Edit cancelled; finalized chunk was not changed.")
                    continue
                try:
                    count = revise_final_chunk(
                        proposal_file,
                        paths.reviewed_dir / proposal_file.name,
                        paths.final_dir / proposal_file.name,
                        document_id=document_id,
                        chunk_id=chunk["chunk_id"],
                        text=text,
                    )
                except ValueError as exc:
                    output_fn(f"Not saved: {exc}")
                    continue
                output_fn(
                    f"Updated final chunk {chunk['chunk_id']} for {document_id}.\n"
                    f"Finalized chunks for this document: {count}\n"
                    f"Final file: {paths.final_dir / proposal_file.name}\n"
                    "The embedding index refreshes on the next retrieval."
                )
                skipped.add(chunk["chunk_id"])
                continue

            if action in {"n", "next"}:
                skipped_in_category.add(chunk["chunk_id"])
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
                if text is None:
                    output_fn("Edit cancelled; proposal was not changed.")
                    continue
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
                if text is None:
                    output_fn("Split cancelled; proposal was not changed.")
                    continue
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
                    for candidate_file, candidate in category_queue
                    if candidate_file.name == proposal_file.name
                    and candidate.get("document_id") == document_id
                    and candidate.get("chunk_id") != chunk["chunk_id"]
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
            processed_by_document[document_id] = processed_by_document.get(document_id, 0) + 1
            output_fn(
                f"Saved {chunk['chunk_id']} as {action} in "
                f"{paths.reviewed_dir / proposal_file.name}."
            )
            _finalize_document_if_ready(proposal_file, document_id, paths, output_fn)


def _load_last_mode(session_file: Path) -> str | None:
    if not session_file.exists():
        return None
    try:
        session = json.loads(session_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid review session file {session_file}: {exc}") from exc
    mode = session.get("mode") if isinstance(session, dict) else None
    return mode if mode in {"documents", "chunks", "finalize", "amend"} else None


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
    output_fn(
        "HaUI RAG Review\n\n"
        "1. Review documents\n"
        "2. Review pending chunks\n"
        "3. Finalize fully reviewed documents\n"
        "4. Edit final chunks\n"
        "5. Continue previous review\n"
        "6. Exit"
    )
    try:
        while True:
            try:
                choice = read_input("Select: ").strip().casefold()
            except EOFError:
                output_fn("\nLeaving review. Saved decisions are available to resume.")
                return 0
            if choice in {"6", "q", "quit"}:
                return 0
            if choice in {"1", "documents", "document"}:
                _save_last_mode(paths.session_file, "documents")
                review_documents(paths, input_fn=read_input, output_fn=output_fn, editor=editor)
                return 0
            if choice in {"2", "chunks", "chunk"}:
                _save_last_mode(paths.session_file, "chunks")
                review_chunk_queue(paths, input_fn=read_input, output_fn=output_fn)
                return 0
            if choice in {"3", "finalize"}:
                _save_last_mode(paths.session_file, "finalize")
                finalize_ready_documents(paths, input_fn=read_input, output_fn=output_fn)
                return 0
            if choice in {"4", "amend", "edit-final"}:
                _save_last_mode(paths.session_file, "amend")
                review_chunk_queue(paths, amend=True, input_fn=read_input, output_fn=output_fn)
                return 0
            if choice in {"5", "resume"}:
                mode = _load_last_mode(paths.session_file)
                if mode == "documents":
                    review_documents(paths, input_fn=read_input, output_fn=output_fn, editor=editor)
                    return 0
                if mode == "chunks":
                    review_chunk_queue(paths, input_fn=read_input, output_fn=output_fn)
                    return 0
                if mode == "finalize":
                    finalize_ready_documents(paths, input_fn=read_input, output_fn=output_fn)
                    return 0
                if mode == "amend":
                    review_chunk_queue(paths, amend=True, input_fn=read_input, output_fn=output_fn)
                    return 0
                output_fn("No previous review session found. Choose documents or chunks.")
                continue
            output_fn("Choose 1, 2, 3, 4, 5, or 6.")
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
