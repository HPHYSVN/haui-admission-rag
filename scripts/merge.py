#!/usr/bin/env python3
"""Merge and verify script for HaUI Admission RAG corpus.

Utility for team members to safely merge category JSONL files,
detect ID or content collisions, and sort documents deterministically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = ROOT_DIR / "data" / "documents"


def merge_jsonl_files(
    input_files: list[Path],
    output_file: Path | None = None,
    sort_by: str = "id",
) -> int:
    all_docs: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    errors = 0

    for fpath in input_files:
        if not fpath.exists():
            print(f"Error: File not found: {fpath}", file=sys.stderr)
            errors += 1
            continue

        with fpath.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    doc = json.loads(clean_line)
                except json.JSONDecodeError as exc:
                    print(f"[{fpath.name}:{line_no}] JSON decode error: {exc}", file=sys.stderr)
                    errors += 1
                    continue

                doc_id = doc.get("id")
                if doc_id in seen_ids:
                    print(
                        f"ID collision: '{doc_id}' in {fpath.name}:{line_no} conflicts with {seen_ids[doc_id]}",
                        file=sys.stderr,
                    )
                    errors += 1
                else:
                    seen_ids[doc_id] = f"{fpath.name}:{line_no}"
                    all_docs.append(doc)

    if errors:
        print(f"\nMerge failed with {errors} errors. No files were written.", file=sys.stderr)
        return 1

    # Deterministic sorting
    if sort_by == "year":
        all_docs.sort(key=lambda x: (x.get("year") or 0, x.get("id", "")), reverse=True)
    else:
        all_docs.sort(key=lambda x: x.get("id", ""))

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as out:
            for doc in all_docs:
                out.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"Successfully merged {len(all_docs)} documents into {output_file}")
    else:
        print(f"Verification successful. {len(all_docs)} documents across {len(input_files)} files have no ID collisions.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and verify JSONL document files.")
    parser.add_argument("inputs", nargs="*", type=Path, help="Input JSONL files to merge or verify")
    parser.add_argument("-o", "--output", type=Path, help="Optional output JSONL file")
    parser.add_argument("--sort", choices=["id", "year"], default="id", help="Sort order (id or year)")

    args = parser.parse_args()

    input_files = args.inputs
    if not input_files:
        input_files = sorted(DEFAULT_DOCS_DIR.glob("*.jsonl"))

    if not input_files:
        print(f"No JSONL files found in {DEFAULT_DOCS_DIR}", file=sys.stderr)
        return 1

    return merge_jsonl_files(input_files, args.output, args.sort)


if __name__ == "__main__":
    raise SystemExit(main())
