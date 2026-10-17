"""Human review operations for chunk proposals and final chunk promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.document_workflow import DEFAULT_STATE_FILE, get_state, set_state

SPLIT_DELIMITER = "\n---CHUNK-SPLIT---\n"
REVIEWABLE_STATUSES = {"proposal", "approved", "edited", "rejected", "split", "merged"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(chunk, dict) or not chunk.get("chunk_id"):
                raise ValueError(f"{path}:{line_no}: expected a chunk object with chunk_id")
            chunks.append(chunk)
    return chunks


def _write_jsonl(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _merge_proposals_with_reviews(
    proposal_file: Path,
    reviewed_file: Path,
) -> list[dict[str, Any]]:
    proposals = _read_jsonl(proposal_file)
    if not reviewed_file.exists():
        return proposals
    reviews = _read_jsonl(reviewed_file)
    proposal_by_id = {chunk["chunk_id"]: chunk for chunk in proposals}
    review_by_id = {chunk["chunk_id"]: chunk for chunk in reviews}
    if len(proposal_by_id) != len(proposals) or len(review_by_id) != len(reviews):
        raise ValueError(f"Duplicate chunk IDs in proposal/review files for {proposal_file.name}")

    combined: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        chunk_id = proposal["chunk_id"]
        reviewed = review_by_id.get(chunk_id)
        if reviewed is not None:
            if reviewed.get("proposal_revision") != proposal.get("proposal_revision"):
                raise ValueError(
                    f"Review for {chunk_id!r} is stale. Regenerate proposals with "
                    "`--replace-review` before reviewing/finalizing again."
                )
            combined[chunk_id] = reviewed
        else:
            combined[chunk_id] = proposal

    extras = [chunk for chunk in reviews if chunk["chunk_id"] not in proposal_by_id]
    for extra in extras:
        combined[extra["chunk_id"]] = extra
    for extra in extras:
        parent_id = extra.get("parent_chunk_id")
        merged_from = extra.get("merged_from")
        is_split_child = (
            isinstance(parent_id, str)
            and parent_id in combined
            and combined[parent_id].get("status") == "split"
        )
        is_merged_chunk = (
            isinstance(merged_from, list)
            and len(merged_from) >= 2
            and all(
                source_id in combined and combined[source_id].get("status") == "merged"
                for source_id in merged_from
            )
        )
        if not (is_split_child or is_merged_chunk):
            raise ValueError(
                f"Reviewed chunk {extra['chunk_id']!r} no longer has a current proposal. "
                "Regenerate proposals with `--replace-review` to remove obsolete decisions."
            )
    return [
        *[
            combined[proposal["chunk_id"]]
            for proposal in proposals
        ],
        *extras,
    ]


def load_review_chunks(proposal_file: Path, reviewed_file: Path) -> list[dict[str, Any]]:
    """Return current proposals combined with valid saved review decisions."""
    return _merge_proposals_with_reviews(proposal_file, reviewed_file)


def review_chunks(
    proposal_file: Path,
    reviewed_file: Path,
    *,
    action: str,
    chunk_id: str,
    text: str | None = None,
    other_chunk_id: str | None = None,
    note: str = "",
) -> int:
    chunks = _merge_proposals_with_reviews(proposal_file, reviewed_file)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError(f"Duplicate chunk IDs in {proposal_file}")
    if chunk_id not in by_id:
        raise ValueError(f"Chunk {chunk_id!r} not found in {proposal_file}")
    chunk = by_id[chunk_id]
    status = chunk.get("status", "proposal")
    if status not in REVIEWABLE_STATUSES:
        raise ValueError(f"Chunk {chunk_id!r} has unsupported workflow status")
    if status in {"split", "merged"}:
        raise ValueError(f"Chunk {chunk_id!r} was superseded and cannot be reviewed again")

    if action == "approve":
        chunk["status"] = "approved"
    elif action == "edit":
        if not text or not text.strip():
            raise ValueError("Editing a chunk requires non-empty replacement text")
        chunk["text_override"] = text.strip()
        chunk["status"] = "edited"
    elif action == "reject":
        chunk["status"] = "rejected"
    elif action == "split":
        if not text or len([part for part in text.split(SPLIT_DELIMITER) if part.strip()]) != 2:
            raise ValueError(f"Split text must contain exactly two parts separated by {SPLIT_DELIMITER!r}")
        parts = [part.strip() for part in text.split(SPLIT_DELIMITER)]
        index = chunks.index(chunk)
        chunk["status"] = "split"
        chunk["review_note"] = note or "Replaced by two reviewed proposals"
        children: list[dict[str, Any]] = []
        for suffix, part in zip(("a", "b"), parts):
            child = dict(chunk)
            child.update(
                chunk_id=f"{chunk_id}__{suffix}",
                text=part,
                text_override=None,
                status="proposal",
                review_note="",
                parent_chunk_id=chunk_id,
                structured_content=None,
                content_type="paragraph",
            )
            children.append(child)
        chunks[index + 1 : index + 1] = children
    elif action == "merge":
        if not other_chunk_id or other_chunk_id == chunk_id or other_chunk_id not in by_id:
            raise ValueError("Merge requires a different existing --other-chunk-id")
        other = by_id[other_chunk_id]
        if other.get("document_id") != chunk.get("document_id"):
            raise ValueError("Chunks from different documents cannot be merged")
        if other.get("status", "proposal") in {"rejected", "split", "merged"}:
            raise ValueError(f"Chunk {other_chunk_id!r} is not available to merge")
        structured_parts = [
            value for value in (chunk.get("structured_content"), other.get("structured_content"))
            if value is not None
        ]
        source_positions = [
            value
            for value in (
                *(chunk.get("source_positions") or [chunk.get("source_position")]),
                *(other.get("source_positions") or [other.get("source_position")]),
            )
            if value is not None
        ]
        merged = dict(chunk)
        merged.update(
            chunk_id=f"{chunk_id}__merged__{other_chunk_id}",
            text=f"{chunk.get('text', '').rstrip()}\n\n{other.get('text', '').lstrip()}",
            text_override=None,
            status="proposal",
            review_note=note,
            heading_path=list(dict.fromkeys(
                (chunk.get("heading_path") or chunk.get("section_path") or [])
                + (other.get("heading_path") or other.get("section_path") or [])
            )),
            merged_from=[chunk_id, other_chunk_id],
            content_type=(
                chunk.get("content_type")
                if chunk.get("content_type") == other.get("content_type")
                else "mixed"
            ),
            structured_content={"format": "merged", "parts": structured_parts} if structured_parts else None,
            source_positions=source_positions,
        )
        chunk["status"] = "merged"
        other["status"] = "merged"
        chunks.append(merged)
    else:
        raise ValueError(f"Unsupported chunk review action: {action!r}")

    if note and action not in {"split"}:
        chunk["review_note"] = note
    _write_jsonl(reviewed_file, chunks)
    return len(chunks)


def revise_final_chunk(
    proposal_file: Path,
    reviewed_file: Path,
    final_file: Path,
    *,
    document_id: str,
    chunk_id: str,
    text: str,
) -> int:
    replacement = text.strip()
    if not replacement:
        raise ValueError("Editing a chunk requires non-empty replacement text")

    chunks = _merge_proposals_with_reviews(proposal_file, reviewed_file)
    reviewed = next(
        (
            chunk
            for chunk in chunks
            if chunk.get("chunk_id") == chunk_id
            and chunk.get("document_id") == document_id
        ),
        None,
    )
    if reviewed is None or reviewed.get("status") not in {"approved", "edited"}:
        raise ValueError(f"Chunk {chunk_id!r} is not an approved chunk for {document_id!r}")

    final_chunks = _read_jsonl(final_file) if final_file.exists() else []
    final = next(
        (
            chunk
            for chunk in final_chunks
            if chunk.get("chunk_id") == chunk_id
            and chunk.get("document_id") == document_id
            and chunk.get("status") == "final"
        ),
        None,
    )
    if final is None:
        raise ValueError(f"Final chunk {chunk_id!r} was not found in {final_file}")

    reviewed["status"] = "edited"
    reviewed["text_override"] = replacement
    final["text"] = replacement
    final["text_override"] = replacement
    _write_jsonl(reviewed_file, chunks)
    _write_jsonl(final_file, final_chunks)
    return sum(chunk.get("document_id") == document_id for chunk in final_chunks)


def finalize_chunks(
    proposal_file: Path,
    reviewed_file: Path,
    final_file: Path,
    *,
    document_id: str,
    state_file: Path = DEFAULT_STATE_FILE,
) -> int:
    state = get_state(document_id, state_file)
    if state not in {"approved", "chunk_proposed", "chunk_review"}:
        raise ValueError(f"Document {document_id!r} is {state!r}; approve it before finalization")
    chunks = _merge_proposals_with_reviews(proposal_file, reviewed_file)
    selected = [chunk for chunk in chunks if chunk.get("document_id") == document_id]
    if not selected:
        raise ValueError(f"No chunk proposals found for document {document_id!r}")
    unresolved = [
        chunk["chunk_id"]
        for chunk in selected
        if chunk.get("status", "proposal") not in {"approved", "edited", "rejected", "split", "merged"}
    ]
    if unresolved:
        raise ValueError(f"Resolve every chunk proposal before finalizing; pending: {unresolved}")
    final_chunks = [
        {
            **chunk,
            "text": chunk.get("text_override") or chunk["text"],
            "status": "final",
        }
        for chunk in selected
        if chunk.get("status") in {"approved", "edited"}
    ]
    existing = _read_jsonl(final_file) if final_file.exists() else []
    updated = [chunk for chunk in existing if chunk.get("document_id") != document_id]
    updated.extend(final_chunks)
    _write_jsonl(final_file, updated)
    set_state(document_id, "final", state_file=state_file)
    return len(final_chunks)
