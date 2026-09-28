#!/usr/bin/env python3
"""Apply edited review Markdown files back to their source JSONL documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag.document_workflow import set_state

ID_RE = re.compile(r"(?m)^<!-- document-id: (\S+) -->$")
BEGIN = "<!-- content-begin -->"
END = "<!-- content-end -->"


def read_review(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    match = ID_RE.search(text)
    if not match:
        raise ValueError(f"{path}: document-id marker is missing")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError(f"{path}: expected exactly one content marker pair")
    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    if end < start:
        raise ValueError(f"{path}: content markers are out of order")
    content = text[start:end].strip()
    if not content:
        raise ValueError(f"{path}: refusing to replace content with empty text")
    return match.group(1), content


def _load_records(source_file: Path) -> list[dict]:
    records: list[dict] = []
    for line_no, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source_file}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _write_records(source_file: Path, records: list[dict]) -> None:
    temp_file = source_file.with_suffix(source_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temp_file.replace(source_file)


def apply_review_file(
    source_file: Path,
    review_file: Path,
    *,
    state_file: Path | None = None,
) -> str:
    doc_id, content = read_review(review_file)
    records = _load_records(source_file)
    matches = [record for record in records if record.get("id") == doc_id]
    if not matches:
        raise ValueError(f"{review_file}: document ID {doc_id!r} is not present in {source_file}")
    if len(matches) > 1:
        raise ValueError(f"{source_file}: duplicate source document ID {doc_id!r}")
    matches[0]["content"] = content
    _write_records(source_file, records)
    state_options = {"state_file": state_file} if state_file is not None else {}
    set_state(
        doc_id,
        "review_pending",
        note="Review Markdown was applied to the source JSONL",
        **state_options,
    )
    return doc_id


def apply_reviews(source_file: Path, review_dir: Path) -> int:
    records = _load_records(source_file)
    by_id = {record["id"]: record for record in records}
    changed = 0
    seen: set[str] = set()
    for path in sorted(review_dir.glob("*.md")):
        doc_id, content = read_review(path)
        if doc_id not in by_id:
            raise ValueError(f"{path}: document ID {doc_id!r} is not present in {source_file}")
        if doc_id in seen:
            raise ValueError(f"{path}: duplicate review for {doc_id}")
        seen.add(doc_id)
        by_id[doc_id]["content"] = content
        changed += 1

    if changed == 0:
        raise ValueError(f"No review Markdown files found in {review_dir}")

    _write_records(source_file, records)
    for doc_id in seen:
        set_state(doc_id, "review_pending", note="Review Markdown was applied to the source JSONL")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Source JSONL file to update")
    parser.add_argument("review_dir", type=Path, help="Directory containing edited document Markdown")
    args = parser.parse_args()
    count = apply_reviews(args.source, args.review_dir)
    print(f"Applied {count} reviewed documents to {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
