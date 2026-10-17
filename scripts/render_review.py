#!/usr/bin/env python3
"""Render JSONL documents as readable Markdown for source comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.document_workflow import set_state

ROOT_DIR = Path(__file__).resolve().parents[1]


def render(
    input_file: Path,
    output_dir: Path,
    document_id: str | None = None,
    *,
    state_file: Path | None = None,
) -> int:
    documents = [
        json.loads(line)
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if document_id is not None:
        documents = [document for document in documents if document.get("id") == document_id]
        if not documents:
            raise ValueError(f"Document {document_id!r} was not found in {input_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        output_file = output_dir / f"{document['id']}.md"
        with output_file.open("w", encoding="utf-8") as output:
            output.write(f"<!-- document-id: {document['id']} -->\n")
            output.write(f"<!-- source-url: {document['source_url']} -->\n")
            output.write(f"<!-- content-begin -->\n")
            output.write(document["content"].rstrip() + "\n")
            output.write("<!-- content-end -->\n")
        state_options = {"state_file": state_file} if state_file is not None else {}
        set_state(document["id"], "review_pending", **state_options)
    return len(documents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--document", help="Render only this document ID")
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT_DIR / "data" / "review" / args.input.stem
    count = render(args.input, output_dir, args.document)
    print(f"Rendered {count} documents to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
